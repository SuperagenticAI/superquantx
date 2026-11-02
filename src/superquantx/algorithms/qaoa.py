"""Quantum Approximate Optimization Algorithm (QAOA) implementation.

This module provides a QAOA implementation for solving combinatorial optimization
problems using quantum circuits with parameterized gates.
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import minimize

from ..backends.base_backend import maxcut_values
from .base_algorithm import OptimizationQuantumAlgorithm


logger = logging.getLogger(__name__)


def brute_force_maxcut(adjacency: np.ndarray) -> tuple[np.ndarray, float]:
    """Classical exact MaxCut by enumerating all 2^n partitions.

    Returns the best bitstring (entry i is the side of node i) and its cut value.
    Practical up to roughly 20 nodes.
    """
    W = np.asarray(adjacency, dtype=float)
    n = W.shape[0]
    values = maxcut_values(W)
    best = int(np.argmax(values))
    bits = np.array([(best >> (n - 1 - i)) & 1 for i in range(n)])
    return bits, float(values[best])


def cut_value(adjacency: np.ndarray, bits: np.ndarray) -> float:
    """Cut value of a partition given as a 0/1 array."""
    W = np.asarray(adjacency, dtype=float)
    bits = np.asarray(bits).astype(int)
    diff = bits[:, None] != bits[None, :]
    return float(np.sum(np.triu(W, 1) * diff))


class QAOA(OptimizationQuantumAlgorithm):
    """Quantum Approximate Optimization Algorithm for combinatorial optimization.

    QAOA is a hybrid quantum-classical algorithm that alternates between
    quantum evolution and classical parameter optimization to find approximate
    solutions to combinatorial optimization problems.

    The algorithm works by:
    1. Preparing an initial superposition state
    2. Applying alternating problem and mixer Hamiltonians
    3. Measuring the quantum state
    4. Classically optimizing the parameters

    Args:
        backend: Quantum backend for circuit execution
        p: Number of QAOA layers (depth)
        problem_hamiltonian: Problem Hamiltonian function
        mixer_hamiltonian: Mixer Hamiltonian function
        initial_state: Initial quantum state preparation
        optimizer: Classical optimizer ('COBYLA', 'L-BFGS-B', etc.)
        shots: Number of measurement shots
        maxiter: Maximum optimization iterations
        **kwargs: Additional parameters

    Example:
        >>> # Define Max-Cut problem
        >>> def problem_ham(gamma, graph):
        ...     return create_maxcut_hamiltonian(gamma, graph)
        >>> qaoa = QAOA(backend='pennylane', p=2, problem_hamiltonian=problem_ham)
        >>> result = qaoa.optimize(graph_data)

    """

    def __init__(
        self,
        backend: str | Any,
        p: int = 1,
        problem_hamiltonian: Callable | None = None,
        mixer_hamiltonian: Callable | None = None,
        initial_state: str = 'uniform_superposition',
        optimizer: str = 'COBYLA',
        shots: int = 1024,
        maxiter: int = 1000,
        n_restarts: int = 3,
        **kwargs
    ) -> None:
        super().__init__(backend=backend, shots=shots, **kwargs)

        self.p = p
        self.n_restarts = n_restarts
        self.problem_hamiltonian = problem_hamiltonian
        self.mixer_hamiltonian = mixer_hamiltonian or self._default_mixer
        self.initial_state = initial_state
        self.optimizer = optimizer
        self.maxiter = maxiter

        # QAOA-specific parameters
        self.n_qubits = None
        self.problem_instance = None
        self.circuit = None

        # Parameter bounds
        self.gamma_bounds = (0, 2*np.pi)
        self.beta_bounds = (0, np.pi)

        logger.info(f"Initialized QAOA with p={p}, optimizer={optimizer}")

    def _default_mixer(self, beta: float) -> Any:
        """Marker for the standard transverse-field (RX) mixer built by the backend."""
        return 'x_mixer'

    def _create_qaoa_circuit(self, params: np.ndarray) -> Any:
        """Create the QAOA circuit for params ``[gamma_1..gamma_p, beta_1..beta_p]``."""
        params = np.asarray(params, dtype=float)
        if len(params) != 2 * self.p:
            raise ValueError(f"Expected {2*self.p} parameters, got {len(params)}")
        return self.backend.create_qaoa_circuit(
            n_qubits=self.n_qubits,
            gammas=params[:self.p],
            betas=params[self.p:],
            problem_hamiltonian=self.problem_hamiltonian,
            mixer_hamiltonian=self.mixer_hamiltonian,
            initial_state=self.initial_state,
            problem_instance=self.problem_instance,
        )

    def _objective_function(self, params: np.ndarray) -> float:
        """Negative expected cut value of the QAOA state (minimised by scipy)."""
        circuit = self._create_qaoa_circuit(params)
        expectation = float(self.backend.execute_qaoa(
            circuit, self.problem_hamiltonian, self.problem_instance, shots=self.shots
        ))
        self.optimization_history_.append({
            'params': np.array(params, copy=True),
            'cost': expectation,
            'iteration': len(self.optimization_history_),
        })
        return -expectation

    def fit(self, X: np.ndarray, y: np.ndarray | None = None, **kwargs) -> 'QAOA':
        """Fit QAOA to problem instance.

        Args:
            X: Problem instance data (e.g., adjacency matrix for Max-Cut)
            y: Not used in QAOA
            **kwargs: Additional parameters

        Returns:
            Self for method chaining

        """
        logger.info(f"Fitting QAOA to problem instance of shape {X.shape}")

        self.problem_instance = X
        self.n_qubits = self._infer_qubits(X)

        # Reset optimization history
        self.optimization_history_ = []

        self.is_fitted = True
        return self

    def _infer_qubits(self, problem_instance: np.ndarray) -> int:
        """One qubit per node of the (square) adjacency matrix."""
        problem_instance = np.asarray(problem_instance)
        if problem_instance.ndim != 2 or problem_instance.shape[0] != problem_instance.shape[1]:
            raise ValueError(
                "QAOA currently solves weighted MaxCut: pass a square adjacency matrix"
            )
        return problem_instance.shape[0]

    def predict(self, X: np.ndarray | None = None, **kwargs) -> np.ndarray:
        """Sample the optimised circuit and return the best sampled partition.

        Entry i of the returned 0/1 array is the side of node i.
        """
        if not self.is_fitted or self.optimal_params_ is None:
            raise ValueError("QAOA must be fitted and optimized before prediction")
        circuit = self._create_qaoa_circuit(self.optimal_params_)
        samples = self.backend.sample_circuit(circuit, shots=self.shots)
        unique = np.unique(samples, axis=0)
        values = [cut_value(self.problem_instance, row) for row in unique]
        return unique[int(np.argmax(values))]

    def solve(self, adjacency: np.ndarray) -> dict[str, Any]:
        """Fit, optimise and sample in one call, with an exact classical check.

        Returns the QAOA partition and cut, the brute-force optimum, and the
        approximation ratio of the expected QAOA cut.
        """
        self.fit(np.asarray(adjacency, dtype=float))
        result = self._run_optimization(None)
        bits = self.predict()
        best_bits, best_cut = brute_force_maxcut(self.problem_instance)
        return {
            'bitstring': ''.join(str(int(b)) for b in bits),
            'cut_value': cut_value(self.problem_instance, bits),
            'expected_cut': self.optimal_value_,
            'classical_bitstring': ''.join(str(int(b)) for b in best_bits),
            'classical_optimum': best_cut,
            'approximation_ratio': self.optimal_value_ / best_cut if best_cut else float('nan'),
            'optimal_params': self.optimal_params_,
            'n_evaluations': result['n_iterations'],
        }

    def _run_optimization(self, objective_function=None, initial_params: np.ndarray | None = None, **kwargs):
        """Run QAOA optimization.

        Args:
            objective_function: Function to optimize (ignored, uses internal)
            initial_params: Initial parameter guess
            **kwargs: Additional optimization parameters

        Returns:
            Optimization result

        """
        if not self.is_fitted:
            raise ValueError("QAOA must be fitted before optimization")

        bounds = [self.gamma_bounds] * self.p + [self.beta_bounds] * self.p
        rng = np.random.default_rng(self.seed)
        starts = [np.asarray(initial_params, dtype=float)] if initial_params is not None else []
        while len(starts) < max(1, self.n_restarts):
            starts.append(self._generate_initial_params(rng))

        best = None
        total_evals = 0
        for x0 in starts:
            result = minimize(
                fun=self._objective_function,
                x0=x0,
                method=self.optimizer,
                bounds=bounds,
                options={'maxiter': self.maxiter, 'disp': False},
            )
            total_evals += result.nfev
            if best is None or result.fun < best.fun:
                best = result

        self.optimal_params_ = best.x
        self.optimal_value_ = -float(best.fun)
        logger.info("QAOA optimization completed. Expected cut: %.6f", self.optimal_value_)
        return {
            'optimal_params': self.optimal_params_,
            'optimal_value': self.optimal_value_,
            'success': bool(best.success),
            'message': best.message,
            'n_iterations': total_evals,
        }

    def _generate_initial_params(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Generate random initial parameters."""
        rng = rng or np.random.default_rng(self.seed)
        gammas = rng.uniform(*self.gamma_bounds, self.p)
        betas = rng.uniform(*self.beta_bounds, self.p)
        return np.concatenate([gammas, betas])

    def get_optimization_landscape(self, param_range: tuple[float, float], resolution: int = 50) -> dict[str, Any]:
        """Compute optimization landscape for visualization.

        Args:
            param_range: Range of parameters to explore
            resolution: Number of points per dimension

        Returns:
            Dictionary with landscape data

        """
        if self.p != 1:
            logger.warning("Landscape visualization only supported for p=1")
            return {}

        gamma_range = np.linspace(*param_range, resolution)
        beta_range = np.linspace(*param_range, resolution)

        landscape = np.zeros((resolution, resolution))

        for i, gamma in enumerate(gamma_range):
            for j, beta in enumerate(beta_range):
                params = np.array([gamma, beta])
                landscape[i, j] = -self._objective_function(params)

        return {
            'gamma_range': gamma_range,
            'beta_range': beta_range,
            'landscape': landscape,
            'optimal_params': self.optimal_params_ if hasattr(self, 'optimal_params_') else None
        }

    def analyze_solution_quality(self, true_optimum: float | None = None) -> dict[str, Any]:
        """Analyze quality of QAOA solution.

        Args:
            true_optimum: Known optimal value for comparison

        Returns:
            Analysis results

        """
        if self.optimal_value_ is None:
            raise ValueError("No optimal solution available")

        analysis = {
            'qaoa_value': self.optimal_value_,
            'n_layers': self.p,
            'n_parameters': 2 * self.p,
            'optimization_iterations': len(self.optimization_history_),
        }

        if true_optimum is not None:
            approximation_ratio = self.optimal_value_ / true_optimum
            analysis.update({
                'true_optimum': true_optimum,
                'approximation_ratio': approximation_ratio,
                'relative_error': abs(1 - approximation_ratio),
            })

        # Analyze convergence
        if len(self.optimization_history_) > 1:
            costs = [-entry['cost'] for entry in self.optimization_history_]
            analysis.update({
                'convergence_achieved': costs[-1] == max(costs),
                'improvement_over_random': self.optimal_value_ - np.mean(costs[:5]) if len(costs) >= 5 else 0,
                'final_cost_variance': np.var(costs[-10:]) if len(costs) >= 10 else 0,
            })

        return analysis

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get QAOA parameters."""
        params = super().get_params(deep)
        params.update({
            'p': self.p,
            'optimizer': self.optimizer,
            'initial_state': self.initial_state,
            'maxiter': self.maxiter,
            'gamma_bounds': self.gamma_bounds,
            'beta_bounds': self.beta_bounds,
        })
        return params

    def set_params(self, **params) -> 'QAOA':
        """Set QAOA parameters."""
        if self.is_fitted and any(key in params for key in ['p', 'problem_hamiltonian', 'mixer_hamiltonian']):
            logger.warning("Changing core parameters requires refitting the model")
            self.is_fitted = False

        return super().set_params(**params)
