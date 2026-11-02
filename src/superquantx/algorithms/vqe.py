"""Variational Quantum Eigensolver (VQE) implementation.

This module provides a VQE implementation for finding ground state energies
and eigenvalues of quantum systems using parameterized quantum circuits.
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import minimize

from ..backends.base_backend import hamiltonian_to_matrix
from .base_algorithm import OptimizationQuantumAlgorithm


logger = logging.getLogger(__name__)

class VQE(OptimizationQuantumAlgorithm):
    """Variational Quantum Eigensolver for finding ground states.

    VQE is a hybrid quantum-classical algorithm that uses a parameterized
    quantum circuit (ansatz) to find the ground state energy of a given
    Hamiltonian by minimizing the expectation value.

    The algorithm works by:
    1. Preparing a parameterized quantum state |ψ(θ)⟩
    2. Measuring the expectation value ⟨ψ(θ)|H|ψ(θ)⟩
    3. Classically optimizing parameters θ to minimize energy
    4. Iterating until convergence

    Args:
        backend: Quantum backend for circuit execution
        hamiltonian: Target Hamiltonian (matrix or operator)
        ansatz: Parameterized circuit ansatz ('UCCSD', 'RealAmplitudes', etc.)
        optimizer: Classical optimizer ('COBYLA', 'L-BFGS-B', etc.)
        shots: Number of measurement shots
        maxiter: Maximum optimization iterations
        initial_params: Initial parameter values
        **kwargs: Additional parameters

    Example:
        >>> # Define H2 molecule Hamiltonian
        >>> H2_hamiltonian = create_h2_hamiltonian(bond_distance=0.74)
        >>> vqe = VQE(backend='pennylane', hamiltonian=H2_hamiltonian, ansatz='UCCSD')
        >>> result = vqe.optimize()
        >>> ground_energy = result.result['optimal_value']

    """

    def __init__(
        self,
        hamiltonian: np.ndarray | Any,
        ansatz: str | Callable = 'RealAmplitudes',
        backend: str | Any = 'simulator',
        optimizer: str = 'COBYLA',
        shots: int = 1024,
        maxiter: int = 1000,
        initial_params: np.ndarray | None = None,
        include_custom_gates: bool = False,
        client = None,
        reps: int = 1,
        **kwargs
    ) -> None:
        super().__init__(backend=backend, shots=shots, **kwargs)
        self.reps = reps

        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.maxiter = maxiter
        self.initial_params = initial_params
        self.include_custom_gates = include_custom_gates
        self.client = client

        # VQE-specific attributes
        self.n_qubits = None
        self.n_params = None
        self.ansatz_circuit = None
        self.hamiltonian_terms = None
        self.hamiltonian_matrix = None

        # Convergence tracking
        self.energy_history = []
        self.gradient_history = []
        self.convergence_threshold = 1e-6

        self._initialize_hamiltonian()

        logger.info(f"Initialized VQE with ansatz={ansatz}, optimizer={optimizer}")

    def _initialize_hamiltonian(self) -> None:
        """Convert the Hamiltonian to a dense matrix and infer the qubit count.

        Accepted formats: square numpy matrix, ``superquantx.gates.Hamiltonian``,
        a dict such as ``{"ZZ": 1.0, "XI": 0.5}``, a list of ``(coeff, "ZZ")``
        pairs, or a PennyLane / Qiskit operator. Pauli label character i acts
        on qubit i. ``None`` defers setup until ``fit(hamiltonian)``.
        """
        if self.hamiltonian is None:
            return
        matrix = hamiltonian_to_matrix(self.hamiltonian)
        dim = matrix.shape[0]
        self.n_qubits = int(round(np.log2(dim)))
        if 2 ** self.n_qubits != dim:
            raise ValueError("Hamiltonian dimension must be a power of 2")
        if not np.allclose(matrix, matrix.conj().T, atol=1e-10):
            raise ValueError("Hamiltonian must be Hermitian")
        self.hamiltonian_matrix = matrix
        self.hamiltonian_terms = matrix

    def exact_ground_energy(self) -> float:
        """Classical baseline: lowest eigenvalue from exact diagonalisation."""
        return float(np.linalg.eigvalsh(self.hamiltonian_matrix)[0])

    def _create_ansatz_circuit(self, params: np.ndarray) -> Any:
        """Build the ansatz circuit on the backend for the given parameters."""
        if callable(self.ansatz):
            return self.ansatz(self.backend, self.n_qubits, params)
        return self.backend.create_ansatz(
            ansatz_type=self.ansatz,
            n_qubits=self.n_qubits,
            params=params,
            include_custom_gates=self.include_custom_gates,
            reps=self.reps,
        )

    def _compute_expectation_value(self, params: np.ndarray) -> float:
        """Exact expectation value <psi(theta)|H|psi(theta)> on the backend's statevector."""
        circuit = self._create_ansatz_circuit(params)
        expectation = float(self.backend.compute_expectation(
            circuit=circuit, hamiltonian=self.hamiltonian_matrix, shots=self.shots
        ))
        self.energy_history.append(expectation)
        return expectation

    def _compute_gradient(self, params: np.ndarray) -> np.ndarray:
        """Compute parameter gradients using parameter-shift rule.

        Args:
            params: Current parameters

        Returns:
            Gradient vector

        """
        gradients = np.zeros_like(params)
        shift = np.pi / 2  # Parameter-shift rule

        for i in range(len(params)):
            # Forward shift
            params_plus = params.copy()
            params_plus[i] += shift
            energy_plus = self._compute_expectation_value(params_plus)

            # Backward shift
            params_minus = params.copy()
            params_minus[i] -= shift
            energy_minus = self._compute_expectation_value(params_minus)

            # Gradient via parameter-shift rule
            gradients[i] = 0.5 * (energy_plus - energy_minus)

        self.gradient_history.append(np.linalg.norm(gradients))
        return gradients

    def fit(self, X: np.ndarray | None = None, y: np.ndarray | None = None, **kwargs) -> 'VQE':
        """Fit VQE (setup for optimization).

        Args:
            X: Optional Hamiltonian, if none was given to the constructor
            y: Not used in VQE
            **kwargs: Additional parameters

        Returns:
            Self for method chaining

        """
        if X is not None:
            self.hamiltonian = X
            self._initialize_hamiltonian()
        if self.hamiltonian_matrix is None:
            raise ValueError("VQE needs a Hamiltonian: pass it to the constructor or fit()")
        logger.info(f"Setting up VQE for {self.n_qubits} qubits")

        # Determine number of parameters based on ansatz
        self.n_params = self._get_ansatz_param_count()

        # Initialize parameters if not provided
        if self.initial_params is None:
            self.initial_params = self._generate_initial_params()

        # Reset histories
        self.energy_history = []
        self.gradient_history = []
        self.optimization_history_ = []

        self.is_fitted = True
        return self

    def _get_ansatz_param_count(self) -> int:
        """Number of parameters for the ansatz."""
        if callable(self.ansatz):
            if self.initial_params is None:
                raise ValueError("A callable ansatz needs initial_params to fix the parameter count")
            return len(self.initial_params)
        return self.backend.ansatz_param_count(self.ansatz, self.n_qubits, self.reps)

    def _generate_initial_params(self) -> np.ndarray:
        """Generate random initial parameters (seeded when ``seed`` is set)."""
        rng = np.random.default_rng(self.seed)
        return rng.uniform(-np.pi, np.pi, self.n_params)

    def predict(self, X: np.ndarray | None = None, **kwargs) -> np.ndarray:
        """Return the optimised state vector (qubit 0 most significant)."""
        if self.optimal_params_ is None:
            raise ValueError("VQE must be optimized before prediction")
        circuit = self._create_ansatz_circuit(self.optimal_params_)
        return self.backend.statevector_big_endian(circuit)

    def _run_optimization(self, objective_function=None, initial_params: np.ndarray | None = None, **kwargs):
        """Run VQE optimization.

        Args:
            objective_function: Not used (VQE has its own objective)
            initial_params: Initial parameter guess
            **kwargs: Additional optimization parameters

        Returns:
            Optimization result

        """
        if not self.is_fitted:
            raise ValueError("VQE must be fitted before optimization")

        # Use provided initial parameters or default
        if initial_params is None:
            initial_params = self.initial_params

        logger.info(f"Starting VQE optimization with {len(initial_params)} parameters")

        # Define objective function for minimization
        def objective(params):
            energy = self._compute_expectation_value(params)

            # Store optimization history
            self.optimization_history_.append({
                'params': params.copy(),
                'energy': energy,
                'iteration': len(self.optimization_history_)
            })

            return energy

        # Run classical optimization
        try:
            result = minimize(
                fun=objective,
                x0=initial_params,
                method=self.optimizer,
                options={
                    'maxiter': self.maxiter,
                    'disp': False
                },
                jac=self._compute_gradient if self.optimizer in ['L-BFGS-B', 'SLSQP'] else None
            )

            self.optimal_params_ = result.x
            self.optimal_value_ = result.fun

            logger.info(f"VQE optimization completed. Ground energy: {self.optimal_value_:.6f}")

            return {
                'optimal_params': self.optimal_params_,
                'ground_energy': self.optimal_value_,
                'exact_energy': self.exact_ground_energy(),
                'success': result.success,
                'message': result.message,
                'n_iterations': result.nfev,
            }

        except Exception as e:
            logger.error(f"VQE optimization failed: {e}")
            raise

    def get_energy_landscape(self, param_indices: list[int], param_ranges: list[tuple[float, float]],
                           resolution: int = 20) -> dict[str, Any]:
        """Compute energy landscape for visualization.

        Args:
            param_indices: Indices of parameters to vary
            param_ranges: Ranges for each parameter
            resolution: Number of points per dimension

        Returns:
            Dictionary with landscape data

        """
        if len(param_indices) != 2:
            raise ValueError("Energy landscape visualization supports only 2 parameters")

        if self.optimal_params_ is None:
            raise ValueError("VQE must be optimized to compute landscape")

        param1_range = np.linspace(*param_ranges[0], resolution)
        param2_range = np.linspace(*param_ranges[1], resolution)

        landscape = np.zeros((resolution, resolution))
        base_params = self.optimal_params_.copy()

        for i, p1 in enumerate(param1_range):
            for j, p2 in enumerate(param2_range):
                params = base_params.copy()
                params[param_indices[0]] = p1
                params[param_indices[1]] = p2
                landscape[i, j] = self._compute_expectation_value(params)

        return {
            'param1_range': param1_range,
            'param2_range': param2_range,
            'landscape': landscape,
            'optimal_params': self.optimal_params_[param_indices],
            'param_indices': param_indices
        }

    def analyze_convergence(self) -> dict[str, Any]:
        """Analyze VQE convergence properties.

        Returns:
            Convergence analysis results

        """
        if not self.energy_history:
            raise ValueError("No optimization history available")

        energies = np.array(self.energy_history)
        gradients = np.array(self.gradient_history) if self.gradient_history else None

        # Basic convergence metrics
        analysis = {
            'final_energy': energies[-1],
            'energy_variance': np.var(energies[-10:]) if len(energies) >= 10 else np.var(energies),
            'total_iterations': len(energies),
            'energy_change': abs(energies[-1] - energies[0]) if len(energies) > 1 else 0,
        }

        # Convergence detection
        if len(energies) >= 10:
            recent_change = abs(energies[-1] - energies[-10])
            analysis['converged'] = recent_change < self.convergence_threshold
        else:
            analysis['converged'] = False

        # Gradient analysis
        if gradients is not None and len(gradients) > 0:
            analysis.update({
                'final_gradient_norm': gradients[-1],
                'gradient_trend': 'decreasing' if gradients[-1] < gradients[0] else 'increasing',
                'min_gradient_norm': np.min(gradients),
            })

        # Identify plateaus and oscillations
        if len(energies) >= 20:
            # Check for plateaus (little change over many iterations)
            plateau_threshold = self.convergence_threshold * 10
            recent_energies = energies[-20:]
            energy_std = np.std(recent_energies)
            analysis['plateau_detected'] = energy_std < plateau_threshold

            # Check for oscillations
            energy_diff = np.diff(energies[-20:])
            sign_changes = np.sum(np.diff(np.sign(energy_diff)) != 0)
            analysis['oscillation_detected'] = sign_changes > len(energy_diff) * 0.7

        return analysis

    def compare_with_exact(self, exact_ground_energy: float) -> dict[str, Any]:
        """Compare VQE result with exact ground state energy.

        Args:
            exact_ground_energy: Known exact ground state energy

        Returns:
            Comparison analysis

        """
        if self.optimal_value_ is None:
            raise ValueError("VQE must be optimized for comparison")

        error = abs(self.optimal_value_ - exact_ground_energy)
        relative_error = error / abs(exact_ground_energy) if exact_ground_energy != 0 else float('inf')

        return {
            'vqe_energy': self.optimal_value_,
            'exact_energy': exact_ground_energy,
            'absolute_error': error,
            'relative_error': relative_error,
            'chemical_accuracy': error < 1.6e-3,  # 1 kcal/mol in Hartree
            'energy_above_ground': max(0, self.optimal_value_ - exact_ground_energy)
        }

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get VQE parameters."""
        params = super().get_params(deep)
        params.update({
            'ansatz': self.ansatz,
            'optimizer': self.optimizer,
            'maxiter': self.maxiter,
            'include_custom_gates': self.include_custom_gates,
            'convergence_threshold': self.convergence_threshold,
        })
        return params

    def set_params(self, **params) -> 'VQE':
        """Set VQE parameters."""
        if self.is_fitted and any(key in params for key in ['ansatz', 'hamiltonian']):
            logger.warning("Changing core parameters requires refitting the model")
            self.is_fitted = False

        return super().set_params(**params)

    def find_ground_state(self) -> float:
        """Find the ground state energy of the Hamiltonian.
        
        This is a convenience method that combines fit() and _run_optimization()
        to find the ground state energy in a single call.
        
        Returns:
            Ground state energy
            
        Example:
            >>> vqe = VQE(backend='simulator', hamiltonian=hamiltonian)
            >>> ground_energy = vqe.find_ground_state()
        """
        # Fit if not already fitted
        if not self.is_fitted:
            self.fit()
            
        # Run optimization
        self._run_optimization()
        
        # Return the optimal energy
        return self.optimal_value_


# Two-qubit H2 Hamiltonian in STO-3G at 0.735 Angstrom after parity mapping
# with two-qubit reduction (electronic energy only; the exact ground energy is
# -1.857275 Ha, add 0.719969 Ha nuclear repulsion for the total -1.137306 Ha).
H2_STO3G_0735 = {
    "II": -1.052373245772859,
    "IZ": 0.39793742484318045,
    "ZI": -0.39793742484318045,
    "ZZ": -0.01128010425623538,
    "XX": 0.18093119978423156,
}


def create_vqe_for_molecule(
    molecule_name: str,
    bond_distance: float | None = None,
    backend: str = 'simulator',
    ansatz: str = 'RealAmplitudes',
    optimizer: str = 'COBYLA',
    client = None,
    **kwargs,
) -> VQE:
    """Create a VQE instance for a small molecule.

    Only H2 in STO-3G at its equilibrium bond length (0.735 Angstrom) ships
    with the library. Other molecules or bond lengths need a Hamiltonian from
    a chemistry package (e.g. PySCF + OpenFermion or Qiskit Nature) passed to
    :class:`VQE` directly.
    """
    if molecule_name.upper() != 'H2':
        raise NotImplementedError(
            f"No built-in Hamiltonian for {molecule_name}. Build one with a chemistry "
            "package and pass it to VQE(hamiltonian=...)."
        )
    if bond_distance is not None and abs(bond_distance - 0.735) > 0.02:
        raise NotImplementedError(
            "Only the H2 equilibrium geometry (0.735 Angstrom) is built in."
        )
    return VQE(
        hamiltonian=dict(H2_STO3G_0735),
        ansatz=ansatz,
        backend=backend,
        optimizer=optimizer,
        client=client,
        **kwargs,
    )
