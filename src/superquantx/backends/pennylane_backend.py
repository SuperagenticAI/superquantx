"""PennyLane backend implementation for SuperQuantX.

This module provides integration with PennyLane for quantum machine learning
and variational quantum algorithms.
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from .base_backend import BaseBackend


logger = logging.getLogger(__name__)

# Try to import PennyLane
try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    qml = None
    pnp = None

class PennyLaneBackend(BaseBackend):
    """PennyLane backend for quantum computing operations.

    This backend provides access to PennyLane's quantum devices and
    automatic differentiation capabilities for variational quantum
    algorithms.

    Args:
        device: PennyLane device name ('default.qubit', 'qiskit.aer', etc.)
        wires: Number of qubits/wires
        shots: Number of measurement shots
        **kwargs: Additional device parameters

    """

    def __init__(self, device: str = 'default.qubit', wires: int = 4, shots: int = 1024, **kwargs):
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane is required for PennyLaneBackend. Install with: pip install pennylane")

        self.wires = wires
        super().__init__(device=device, shots=shots, **kwargs)
        self.dev = None
        self.capabilities = {
            'supports_gradient': True,
            'supports_parameter_shift': True,
            'supports_finite_diff': True,
            'supports_backprop': device in ['default.qubit'],
            'supports_adjoint': device in ['default.qubit'],
        }

    def _initialize_backend(self) -> None:
        """Initialize PennyLane device."""
        try:
            device_kwargs = self.config.copy()
            device_kwargs.pop('wires', None)  # Remove wires from kwargs if present

            # Create PennyLane device
            if self.shots is not None and self.shots > 0:
                self.dev = qml.device(self.device, wires=self.wires, shots=self.shots, **device_kwargs)
            else:
                # Exact simulation (no shots)
                self.dev = qml.device(self.device, wires=self.wires, **device_kwargs)

            logger.info(f"Initialized PennyLane device: {self.device} with {self.wires} wires")

        except Exception as e:
            logger.error(f"Failed to initialize PennyLane device: {e}")
            raise

    def create_circuit(self, n_qubits: int) -> Callable:
        """Create a PennyLane quantum function template."""
        if n_qubits > self.wires:
            logger.warning(f"Requested {n_qubits} qubits, but device only has {self.wires} wires")
            n_qubits = self.wires

        def circuit_template():
            """Empty circuit template."""
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        return circuit_template

    def add_gate(self, circuit: Callable, gate: str, qubits: int | list[int],
                 params: list[float] | None = None) -> Callable:
        """Add a quantum gate to the circuit (conceptual - PennyLane uses functions)."""
        # In PennyLane, gates are added within quantum functions
        # This method is more for compatibility with the base interface
        logger.debug(f"Gate {gate} would be added to qubits {qubits} with params {params}")
        return circuit

    def add_measurement(self, circuit: Callable, qubits: list[int] | None = None) -> Callable:
        """Add measurement operations (conceptual in PennyLane)."""
        return circuit

    def execute_circuit(self, circuit: Callable, shots: int | None = None) -> dict[str, Any]:
        """Execute quantum circuit and return results."""
        try:
            # Create QNode
            if shots is not None:
                dev = qml.device(self.device, wires=self.wires, shots=shots)
                qnode = qml.QNode(circuit, dev)
            else:
                qnode = qml.QNode(circuit, self.dev)

            # Execute
            result = qnode()

            return {
                'result': result,
                'shots': shots or self.shots,
                'device': self.device,
            }

        except Exception as e:
            logger.error(f"Circuit execution failed: {e}")
            raise

    def get_statevector(self, circuit: Callable) -> np.ndarray:
        """Get the statevector from a quantum circuit."""
        # Create a statevector device
        dev_statevector = qml.device('default.qubit', wires=self.wires)

        @qml.qnode(dev_statevector)
        def statevector_circuit():
            circuit()
            return qml.state()

        return np.array(statevector_circuit())

    def _get_n_qubits(self, circuit: Any) -> int:
        """Get number of qubits in circuit."""
        return self.wires

    # ========================================================================
    # Enhanced PennyLane-specific implementations
    # ========================================================================

    def create_feature_map(self, n_features: int, feature_map: str, reps: int = 1) -> Callable:
        """Create quantum feature map for data encoding."""
        if feature_map == 'ZZFeatureMap':
            return self._create_zz_feature_map(n_features, reps)
        elif feature_map == 'PauliFeatureMap':
            return self._create_pauli_feature_map(n_features, reps)
        elif feature_map == 'AmplitudeMap':
            return self._create_amplitude_map(n_features)
        else:
            logger.warning(f"Unknown feature map '{feature_map}', using angle encoding")
            return self._create_angle_encoding_map(n_features)

    def _create_zz_feature_map(self, n_features: int, reps: int) -> Callable:
        """Create ZZ feature map circuit."""
        def zz_feature_map(x):
            for r in range(reps):
                # Single-qubit rotations
                for i in range(min(n_features, self.wires)):
                    qml.Hadamard(wires=i)
                    qml.RZ(x[i], wires=i)

                # Two-qubit interactions
                for i in range(min(n_features - 1, self.wires - 1)):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(x[i] * x[i + 1], wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])

            return [qml.expval(qml.PauliZ(i)) for i in range(min(n_features, self.wires))]

        return zz_feature_map

    def _create_pauli_feature_map(self, n_features: int, reps: int) -> Callable:
        """Create Pauli feature map circuit."""
        def pauli_feature_map(x):
            for r in range(reps):
                for i in range(min(n_features, self.wires)):
                    qml.RX(x[i], wires=i)
                    qml.RY(x[i], wires=i)
                    qml.RZ(x[i], wires=i)

                # Entangling layer
                for i in range(min(n_features - 1, self.wires - 1)):
                    qml.CNOT(wires=[i, i + 1])

            return [qml.expval(qml.PauliZ(i)) for i in range(min(n_features, self.wires))]

        return pauli_feature_map

    def _create_amplitude_map(self, n_features: int) -> Callable:
        """Create amplitude encoding map."""
        def amplitude_map(x):
            # Normalize input
            norm = np.linalg.norm(x)
            if norm > 0:
                normalized_x = x / norm
            else:
                normalized_x = x

            # Pad to power of 2
            n_qubits = int(np.ceil(np.log2(len(normalized_x))))
            padded_x = np.pad(normalized_x, (0, 2**n_qubits - len(normalized_x)))

            # Amplitude encoding (simplified)
            qml.QubitStateVector(padded_x, wires=range(min(n_qubits, self.wires)))

            return [qml.expval(qml.PauliZ(i)) for i in range(min(n_qubits, self.wires))]

        return amplitude_map

    def _create_angle_encoding_map(self, n_features: int) -> Callable:
        """Create angle encoding map."""
        def angle_encoding_map(x):
            for i in range(min(n_features, self.wires)):
                qml.RY(x[i], wires=i)

            return [qml.expval(qml.PauliZ(i)) for i in range(min(n_features, self.wires))]

        return angle_encoding_map

    def compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray,
                            feature_map: Callable, shots: int | None = None) -> np.ndarray:
        """Compute quantum kernel matrix using PennyLane."""
        n1, n2 = len(X1), len(X2)
        kernel_matrix = np.zeros((n1, n2))

        # Create kernel evaluation circuit
        def kernel_circuit(x1, x2):
            # Encode first data point
            for i in range(min(len(x1), self.wires)):
                qml.RY(x1[i], wires=i)

            # Encode second data point with inverse
            for i in range(min(len(x2), self.wires)):
                qml.RY(-x2[i], wires=i)

            # Measure probability of |00...0⟩ state
            return qml.probs(wires=range(min(len(x1), self.wires)))

        # Create QNode
        dev = qml.device(self.device, wires=self.wires, shots=shots) if shots else self.dev
        kernel_qnode = qml.QNode(kernel_circuit, dev)

        # Compute kernel matrix
        for i in range(n1):
            for j in range(n2):
                try:
                    probs = kernel_qnode(X1[i], X2[j])
                    kernel_matrix[i, j] = probs[0]  # Probability of |00...0⟩
                except Exception as e:
                    logger.warning(f"Kernel computation failed for ({i},{j}): {e}")
                    kernel_matrix[i, j] = 0.0

        return kernel_matrix

    def create_ansatz(self, ansatz_type: str, n_qubits: int, params: np.ndarray,
                     include_custom_gates: bool = False) -> Callable:
        """Create parameterized ansatz circuit."""
        if ansatz_type == 'RealAmplitudes':
            return self._create_real_amplitudes_ansatz(n_qubits, params)
        elif ansatz_type == 'EfficientSU2':
            return self._create_efficient_su2_ansatz(n_qubits, params)
        elif ansatz_type == 'TwoLocal':
            return self._create_two_local_ansatz(n_qubits, params)
        elif ansatz_type == 'UCCSD':
            return self._create_uccsd_ansatz(n_qubits, params)
        else:
            logger.warning(f"Unknown ansatz '{ansatz_type}', using RealAmplitudes")
            return self._create_real_amplitudes_ansatz(n_qubits, params)

    def _create_real_amplitudes_ansatz(self, n_qubits: int, params: np.ndarray) -> Callable:
        """Create RealAmplitudes ansatz."""
        def real_amplitudes_circuit():
            n_layers = len(params) // (n_qubits * 2)
            param_idx = 0

            for layer in range(n_layers):
                # Rotation layer
                for i in range(min(n_qubits, self.wires)):
                    if param_idx < len(params):
                        qml.RY(params[param_idx], wires=i)
                        param_idx += 1

                # Entangling layer
                for i in range(min(n_qubits - 1, self.wires - 1)):
                    qml.CNOT(wires=[i, i + 1])

                # Second rotation layer
                for i in range(min(n_qubits, self.wires)):
                    if param_idx < len(params):
                        qml.RZ(params[param_idx], wires=i)
                        param_idx += 1

            return [qml.expval(qml.PauliZ(i)) for i in range(min(n_qubits, self.wires))]

        return real_amplitudes_circuit

    def _create_efficient_su2_ansatz(self, n_qubits: int, params: np.ndarray) -> Callable:
        """Create EfficientSU2 ansatz."""
        def efficient_su2_circuit():
            n_layers = len(params) // (n_qubits * 3)
            param_idx = 0

            for layer in range(n_layers):
                # SU(2) rotations
                for i in range(min(n_qubits, self.wires)):
                    if param_idx + 2 < len(params):
                        qml.RY(params[param_idx], wires=i)
                        qml.RZ(params[param_idx + 1], wires=i)
                        qml.RY(params[param_idx + 2], wires=i)
                        param_idx += 3

                # Entangling layer
                for i in range(min(n_qubits - 1, self.wires - 1)):
                    qml.CNOT(wires=[i, i + 1])

            return [qml.expval(qml.PauliZ(i)) for i in range(min(n_qubits, self.wires))]

        return efficient_su2_circuit

    def _create_two_local_ansatz(self, n_qubits: int, params: np.ndarray) -> Callable:
        """Create TwoLocal ansatz."""
        def two_local_circuit():
            n_layers = len(params) // (n_qubits * 2)
            param_idx = 0

            for layer in range(n_layers):
                # Local rotations
                for i in range(min(n_qubits, self.wires)):
                    if param_idx + 1 < len(params):
                        qml.RY(params[param_idx], wires=i)
                        qml.RZ(params[param_idx + 1], wires=i)
                        param_idx += 2

                # Two-qubit gates
                for i in range(0, min(n_qubits - 1, self.wires - 1), 2):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(1, min(n_qubits - 1, self.wires - 1), 2):
                    qml.CNOT(wires=[i, i + 1])

            return [qml.expval(qml.PauliZ(i)) for i in range(min(n_qubits, self.wires))]

        return two_local_circuit

    def _create_uccsd_ansatz(self, n_qubits: int, params: np.ndarray) -> Callable:
        """Create UCCSD ansatz (simplified)."""
        def uccsd_circuit():
            # Simplified UCCSD - full implementation would require molecular geometry
            param_idx = 0

            # Singles excitations
            for i in range(0, min(n_qubits, self.wires), 2):
                for j in range(1, min(n_qubits, self.wires), 2):
                    if param_idx < len(params) and i != j:
                        # Simplified single excitation
                        qml.CNOT(wires=[i, j])
                        qml.RY(params[param_idx], wires=j)
                        qml.CNOT(wires=[i, j])
                        param_idx += 1

            # Doubles excitations (simplified)
            for i in range(0, min(n_qubits - 3, self.wires - 3), 2):
                if param_idx < len(params):
                    # Simplified double excitation
                    qml.CNOT(wires=[i, i + 1])
                    qml.CNOT(wires=[i + 2, i + 3])
                    qml.RY(params[param_idx], wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])
                    qml.CNOT(wires=[i + 2, i + 3])
                    param_idx += 1

            return [qml.expval(qml.PauliZ(i)) for i in range(min(n_qubits, self.wires))]

        return uccsd_circuit

    def compute_expectation(self, circuit: Callable, hamiltonian: Any,
                          shots: int | None = None) -> float:
        """Compute expectation value of Hamiltonian."""
        try:
            # If hamiltonian is a PennyLane Hamiltonian
            if hasattr(hamiltonian, 'coeffs') and hasattr(hamiltonian, 'ops'):
                @qml.qnode(self.dev if shots is None else qml.device(self.device, wires=self.wires, shots=shots))
                def expectation_circuit():
                    circuit()
                    return qml.expval(hamiltonian)

                return float(expectation_circuit())

            # If hamiltonian is a matrix, decompose it
            elif isinstance(hamiltonian, np.ndarray):
                return self._compute_matrix_expectation(circuit, hamiltonian, shots)

            else:
                logger.warning("Unknown Hamiltonian format, returning 0")
                return 0.0

        except Exception as e:
            logger.error(f"Expectation computation failed: {e}")
            return 0.0

    def _compute_matrix_expectation(self, circuit: Callable, H: np.ndarray, shots: int | None) -> float:
        """Compute expectation value for matrix Hamiltonian."""
        # Get statevector
        statevector = self.get_statevector(circuit)

        # Compute ⟨ψ|H|ψ⟩
        expectation = np.real(np.conj(statevector) @ H @ statevector)
        return float(expectation)

    def get_version_info(self) -> dict[str, Any]:
        """Get PennyLane version information."""
        info = super().get_version_info()
        info.update({
            'pennylane_version': qml.version() if qml else 'Not available',
            'available_devices': qml.plugin_devices if qml else [],
        })
        return info
