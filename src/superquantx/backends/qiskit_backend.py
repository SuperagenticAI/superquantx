"""Qiskit backend implementation for SuperQuantX.

This module provides integration with IBM's Qiskit for quantum computing
operations and access to IBM Quantum hardware.
"""

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .base_backend import BaseBackend


if TYPE_CHECKING:
    from qiskit import QuantumCircuit

logger = logging.getLogger(__name__)

# Try to import Qiskit with modern structure
try:
    from qiskit import (
        ClassicalRegister,
        QuantumCircuit,
        QuantumRegister,
        transpile,
    )
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector
    QISKIT_AVAILABLE = True

    # Try to import algorithms and circuit library
    try:
        from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B
        QISKIT_ALGORITHMS_AVAILABLE = True
    except ImportError:
        QISKIT_ALGORITHMS_AVAILABLE = False
        COBYLA = None
        L_BFGS_B = None

    try:
        from qiskit.circuit.library import EfficientSU2, RealAmplitudes, ZZFeatureMap
        QISKIT_CIRCUIT_LIBRARY_AVAILABLE = True
    except ImportError:
        QISKIT_CIRCUIT_LIBRARY_AVAILABLE = False
        EfficientSU2 = None
        RealAmplitudes = None
        ZZFeatureMap = None

    # Try to import IBM Quantum provider (deprecated)
    try:
        from qiskit_ibm_provider import IBMProvider
        IBM_PROVIDER_AVAILABLE = True
        IBMQ = None  # IBMQ is deprecated
        IBMQBackend = None
    except ImportError:
        IBM_PROVIDER_AVAILABLE = False
        IBMProvider = None
        IBMQ = None
        IBMQBackend = None

except ImportError:
    QISKIT_AVAILABLE = False
    QuantumCircuit = None
    AerSimulator = None
    Statevector = None
    QISKIT_ALGORITHMS_AVAILABLE = False
    QISKIT_CIRCUIT_LIBRARY_AVAILABLE = False
    IBM_PROVIDER_AVAILABLE = False

class QiskitBackend(BaseBackend):
    """Qiskit backend for quantum computing operations.

    This backend provides access to Qiskit simulators and IBM Quantum
    hardware for quantum algorithm execution.

    Args:
        device: Qiskit backend name ('aer_simulator', 'ibmq_qasm_simulator', etc.)
        provider: IBM Quantum provider (if using IBMQ)
        shots: Number of measurement shots
        **kwargs: Additional backend parameters

    """

    def __init__(self, device: str = 'aer_simulator', provider: str | None = None,
                 shots: int = 1024, **kwargs):
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for QiskitBackend. Install with: pip install qiskit")

        super().__init__(device=device, shots=shots, **kwargs)

        self.provider = provider
        self.backend = None
        self.capabilities = {
            'supports_gradient': False,
            'supports_parameter_shift': True,
            'supports_finite_diff': True,
            'supports_hardware': IBM_PROVIDER_AVAILABLE,
            'supports_noise_models': True,
        }
        
        # Initialize the backend
        self._initialize_backend()

    def _initialize_backend(self) -> None:
        """Initialize Qiskit backend."""
        try:
            if self.device == 'aer_simulator':
                self.backend = AerSimulator()
                logger.info("Initialized Qiskit AerSimulator")

            elif self.device.startswith('ibmq_') or self.provider:
                if not IBM_PROVIDER_AVAILABLE:
                    raise ImportError("IBM Quantum provider not available")

                # Load IBMQ account
                if not IBMQ.active_account():
                    try:
                        IBMQ.load_account()
                    except Exception as e:
                        logger.error(f"Failed to load IBMQ account: {e}")
                        raise

                # Get provider
                if self.provider:
                    provider = IBMQ.get_provider(hub=self.provider.get('hub', 'ibm-q'),
                                               group=self.provider.get('group', 'open'),
                                               project=self.provider.get('project', 'main'))
                else:
                    provider = IBMQ.get_provider()

                # Get backend
                self.backend = provider.get_backend(self.device)
                logger.info(f"Initialized IBM Quantum backend: {self.device}")

            else:
                # Try to get backend from Aer
                self.backend = Aer.get_backend(self.device)
                logger.info(f"Initialized Aer backend: {self.device}")

        except Exception as e:
            logger.error(f"Failed to initialize Qiskit backend: {e}")
            # Fallback to simulator
            self.backend = AerSimulator()
            logger.warning("Falling back to AerSimulator")

    def create_circuit(self, n_qubits: int) -> Any:
        """Create a quantum circuit with n qubits."""
        qreg = QuantumRegister(n_qubits, 'q')
        creg = ClassicalRegister(n_qubits, 'c')
        circuit = QuantumCircuit(qreg, creg)
        return circuit

    def add_gate(self, circuit: Any, gate: str, qubits: int | list[int],
                 params: list[float] | None = None) -> Any:
        """Add a quantum gate to the circuit."""
        if isinstance(qubits, int):
            qubits = [qubits]

        params = params or []

        try:
            if gate.upper() == 'H' or gate.upper() == 'HADAMARD':
                circuit.h(qubits[0])
            elif gate.upper() == 'X' or gate.upper() == 'PAULI_X':
                circuit.x(qubits[0])
            elif gate.upper() == 'Y' or gate.upper() == 'PAULI_Y':
                circuit.y(qubits[0])
            elif gate.upper() == 'Z' or gate.upper() == 'PAULI_Z':
                circuit.z(qubits[0])
            elif gate.upper() == 'RX':
                circuit.rx(params[0] if params else 0, qubits[0])
            elif gate.upper() == 'RY':
                circuit.ry(params[0] if params else 0, qubits[0])
            elif gate.upper() == 'RZ':
                circuit.rz(params[0] if params else 0, qubits[0])
            elif gate.upper() == 'CNOT' or gate.upper() == 'CX':
                circuit.cx(qubits[0], qubits[1])
            elif gate.upper() == 'CZ':
                circuit.cz(qubits[0], qubits[1])
            elif gate.upper() == 'SWAP':
                circuit.swap(qubits[0], qubits[1])
            elif gate.upper() == 'CCX' or gate.upper() == 'TOFFOLI':
                circuit.ccx(qubits[0], qubits[1], qubits[2])
            else:
                logger.warning(f"Unknown gate: {gate}")

        except Exception as e:
            logger.error(f"Failed to add gate {gate}: {e}")

        return circuit

    def add_measurement(self, circuit: Any, qubits: list[int] | None = None) -> Any:
        """Add measurement operations to the circuit."""
        if qubits is None:
            qubits = list(range(circuit.num_qubits))

        for i, qubit in enumerate(qubits):
            if i < circuit.num_clbits:
                circuit.measure(qubit, i)

        return circuit

    def execute_circuit(self, circuit: Any, shots: int | None = None) -> dict[str, Any]:
        """Execute quantum circuit and return results."""
        shots = shots or self.shots

        try:
            # Add measurements if not present
            has_measurements = circuit.num_clbits > 0 and any(
                hasattr(instr.operation, 'name') and instr.operation.name == 'measure'
                for instr in circuit.data
            )
            
            if not has_measurements:
                circuit.add_register(ClassicalRegister(circuit.num_qubits, 'c'))
                circuit.measure_all()

            # Transpile circuit
            transpiled = transpile(circuit, self.backend)

            # Execute using modern Qiskit API
            job = self.backend.run(transpiled, shots=shots)
            result = job.result()

            # Get counts
            counts = result.get_counts()

            return {
                'counts': counts,
                'shots': shots,
                'backend': self.device,
                'job_id': job.job_id(),
            }

        except Exception as e:
            logger.error(f"Circuit execution failed: {e}")
            raise

    def get_statevector(self, circuit: Any) -> np.ndarray:
        """Get the statevector from a quantum circuit."""
        try:
            # Use statevector simulator
            backend = AerSimulator(method='statevector')

            # Create circuit copy without measurements
            circuit_copy = circuit.copy()
            circuit_copy.remove_final_measurements()

            # Execute
            job = execute(circuit_copy, backend, shots=1)
            result = job.result()

            # Get statevector
            statevector = result.get_statevector()
            return np.array(statevector.data)

        except Exception as e:
            logger.error(f"Statevector computation failed: {e}")
            return np.array([1.0] + [0.0] * (2**circuit.num_qubits - 1))

    def _get_n_qubits(self, circuit: Any) -> int:
        """Get number of qubits in circuit."""
        return circuit.num_qubits

    # ========================================================================
    # Enhanced Qiskit-specific implementations
    # ========================================================================

    def create_feature_map(self, n_features: int, feature_map: str, reps: int = 1) -> Any:
        """Create quantum feature map for data encoding."""
        if feature_map == 'ZZFeatureMap':
            return ZZFeatureMap(n_features, reps=reps)
        elif feature_map == 'PauliFeatureMap':
            return self._create_pauli_feature_map(n_features, reps)
        elif feature_map == 'AmplitudeMap':
            return self._create_amplitude_map(n_features)
        else:
            logger.warning(f"Unknown feature map '{feature_map}', using angle encoding")
            return self._create_angle_encoding_map(n_features)

    def _create_pauli_feature_map(self, n_features: int, reps: int) -> Any:
        """Create Pauli feature map circuit."""
        circuit = QuantumCircuit(n_features)

        for r in range(reps):
            # Pauli rotations
            for i in range(n_features):
                circuit.rx(circuit.parameters[0], i)  # Placeholder parameter
                circuit.ry(circuit.parameters[0], i)
                circuit.rz(circuit.parameters[0], i)

            # Entangling layer
            for i in range(n_features - 1):
                circuit.cx(i, i + 1)

        return circuit

    def _create_amplitude_map(self, n_features: int) -> Any:
        """Create amplitude encoding map."""
        n_qubits = int(np.ceil(np.log2(n_features)))
        circuit = QuantumCircuit(n_qubits)

        # Amplitude encoding would require state initialization
        # This is a simplified placeholder
        logger.warning("Amplitude encoding not fully implemented")

        return circuit

    def _create_angle_encoding_map(self, n_features: int) -> Any:
        """Create angle encoding map."""
        circuit = QuantumCircuit(n_features)

        for i in range(n_features):
            circuit.ry(circuit.parameters[0], i)  # Placeholder parameter

        return circuit

    def compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray,
                            feature_map: QuantumCircuit, shots: int | None = None) -> np.ndarray:
        """Compute quantum kernel matrix using Qiskit."""
        n1, n2 = len(X1), len(X2)
        kernel_matrix = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                try:
                    # Create kernel evaluation circuit
                    circuit = QuantumCircuit(feature_map.num_qubits, feature_map.num_qubits)

                    # Encode first data point
                    circuit.compose(feature_map.bind_parameters(X1[i]), inplace=True)

                    # Encode second data point with inverse
                    inverse_map = feature_map.bind_parameters(-X2[j]).inverse()
                    circuit.compose(inverse_map, inplace=True)

                    # Measure
                    circuit.measure_all()

                    # Execute
                    result = self.execute_circuit(circuit, shots)
                    counts = result['counts']

                    # Kernel value is probability of measuring |00...0⟩
                    zero_state = '0' * feature_map.num_qubits
                    kernel_matrix[i, j] = counts.get(zero_state, 0) / sum(counts.values())

                except Exception as e:
                    logger.warning(f"Kernel computation failed for ({i},{j}): {e}")
                    kernel_matrix[i, j] = 0.0

        return kernel_matrix

    def create_ansatz(self, ansatz_type: str, n_qubits: int, params: np.ndarray,
                     include_custom_gates: bool = False) -> Any:
        """Create parameterized ansatz circuit."""
        if ansatz_type == 'RealAmplitudes':
            return RealAmplitudes(n_qubits, reps=len(params) // (n_qubits * 2))
        elif ansatz_type == 'EfficientSU2':
            return EfficientSU2(n_qubits, reps=len(params) // (n_qubits * 3))
        elif ansatz_type == 'TwoLocal':
            return self._create_two_local_ansatz(n_qubits, params)
        elif ansatz_type == 'UCCSD':
            return self._create_uccsd_ansatz(n_qubits, params)
        else:
            logger.warning(f"Unknown ansatz '{ansatz_type}', using RealAmplitudes")
            return RealAmplitudes(n_qubits, reps=len(params) // (n_qubits * 2))

    def _create_two_local_ansatz(self, n_qubits: int, params: np.ndarray) -> Any:
        """Create TwoLocal ansatz."""
        from qiskit.circuit.library import TwoLocal
        return TwoLocal(n_qubits, 'ry', 'cx', 'linear', reps=len(params) // (n_qubits * 2))

    def _create_uccsd_ansatz(self, n_qubits: int, params: np.ndarray) -> Any:
        """Create UCCSD ansatz (simplified)."""
        circuit = QuantumCircuit(n_qubits)

        # Simplified UCCSD implementation
        param_idx = 0

        # Single excitations
        for i in range(0, n_qubits, 2):
            for j in range(1, n_qubits, 2):
                if param_idx < len(params) and i != j:
                    circuit.cx(i, j)
                    circuit.ry(params[param_idx], j)
                    circuit.cx(i, j)
                    param_idx += 1

        # Double excitations (simplified)
        for i in range(0, n_qubits - 3, 2):
            if param_idx < len(params):
                circuit.cx(i, i + 1)
                circuit.cx(i + 2, i + 3)
                circuit.ry(params[param_idx], i + 1)
                circuit.cx(i, i + 1)
                circuit.cx(i + 2, i + 3)
                param_idx += 1

        return circuit

    def compute_expectation(self, circuit: Any, hamiltonian: Any,
                          shots: int | None = None) -> float:
        """Compute expectation value of Hamiltonian."""
        try:
            # For Qiskit, we need to decompose Hamiltonian into Pauli terms
            if isinstance(hamiltonian, np.ndarray):
                # Compute expectation using statevector
                statevector = self.get_statevector(circuit)
                expectation = np.real(np.conj(statevector) @ hamiltonian @ statevector)
                return float(expectation)
            else:
                logger.warning("Hamiltonian expectation not fully implemented")
                return 0.0

        except Exception as e:
            logger.error(f"Expectation computation failed: {e}")
            return 0.0

    def create_qaoa_circuit(self, n_qubits: int, gammas: np.ndarray, betas: np.ndarray,
                          problem_hamiltonian: Any, mixer_hamiltonian: Any,
                          initial_state: Any, problem_instance: Any) -> Any:
        """Create QAOA circuit with given parameters."""
        circuit = QuantumCircuit(n_qubits)

        # Initial state (uniform superposition)
        for i in range(n_qubits):
            circuit.h(i)

        # QAOA layers
        for gamma, beta in zip(gammas, betas, strict=False):
            # Problem Hamiltonian layer
            self._apply_problem_hamiltonian(circuit, gamma, problem_instance)

            # Mixer Hamiltonian layer
            self._apply_mixer_hamiltonian(circuit, beta)

        return circuit

    def _apply_problem_hamiltonian(self, circuit: Any, gamma: float, problem_instance: Any) -> None:
        """Apply problem Hamiltonian to circuit."""
        # Simplified implementation for Max-Cut type problems
        n_qubits = circuit.num_qubits

        if hasattr(problem_instance, 'shape') and len(problem_instance.shape) == 2:
            # Assume adjacency matrix
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    if problem_instance[i, j] != 0:
                        circuit.cx(i, j)
                        circuit.rz(gamma * problem_instance[i, j], j)
                        circuit.cx(i, j)

    def _apply_mixer_hamiltonian(self, circuit: Any, beta: float) -> None:
        """Apply mixer Hamiltonian to circuit."""
        # Standard X-mixer
        for i in range(circuit.num_qubits):
            circuit.rx(2 * beta, i)

    def get_version_info(self) -> dict[str, Any]:
        """Get Qiskit version information."""
        info = super().get_version_info()

        try:
            import qiskit
            info.update({
                'qiskit_version': qiskit.__version__,
                'backend_name': self.backend.name() if self.backend else 'Unknown',
                'backend_version': getattr(self.backend, 'version', 'Unknown'),
            })
        except Exception as e:
            info['version_error'] = str(e)

        return info
