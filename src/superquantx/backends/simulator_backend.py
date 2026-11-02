"""Pure Python quantum simulator backend for SuperQuantX.

This module provides a simple quantum simulator implementation that doesn't
require external quantum computing libraries, useful for testing and fallback.
"""

import logging
from typing import Any, Union

import numpy as np

from .base_backend import BaseBackend


logger = logging.getLogger(__name__)

class QuantumResult:
    """Container for quantum circuit execution results."""
    
    def __init__(self, result_dict: dict[str, Any]):
        self._result_dict = result_dict
    
    def get_counts(self) -> dict[str, int]:
        """Get measurement counts."""
        return self._result_dict.get('counts', {})

class QuantumState:
    """Simple quantum state representation."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.state = np.zeros(2**n_qubits, dtype=complex)
        self.state[0] = 1.0  # |00...0⟩ state

    def apply_gate(self, gate_matrix: np.ndarray, qubits: list[int]) -> None:
        """Apply a k-qubit gate matrix to ``qubits``.

        The gate matrix is written in the basis |q_0 q_1 ...> of the listed
        qubits (first listed qubit is the most significant), e.g. CNOT with
        ``qubits=[control, target]``. The state index uses little-endian
        ordering: qubit q is bit q of the index.
        """
        k = len(qubits)
        if len(set(qubits)) != k:
            raise ValueError(f"Gate qubits must be distinct, got {qubits}")
        if gate_matrix.shape != (2 ** k, 2 ** k):
            raise ValueError(
                f"Gate of shape {gate_matrix.shape} cannot act on {k} qubit(s)"
            )
        n = self.n_qubits
        for q in qubits:
            if not 0 <= q < n:
                raise ValueError(f"Qubit {q} out of range for {n}-qubit circuit")
        axes = [n - 1 - q for q in qubits]
        psi = self.state.reshape([2] * n)
        gate = gate_matrix.reshape([2] * (2 * k))
        psi = np.tensordot(gate, psi, axes=(list(range(k, 2 * k)), axes))
        psi = np.moveaxis(psi, list(range(k)), axes)
        self.state = psi.reshape(-1)

    def measure(self, qubit_or_shots=None, classical_bit=None, shots: int = 1024) -> Union[dict[str, int], "QuantumState"]:
        """Measure quantum state or add measurement to circuit."""
        # If called with two arguments, it's measure(qubit, classical_bit) - for circuit building
        if qubit_or_shots is not None and classical_bit is not None:
            # This is a circuit building operation, just return self for chaining
            # Actual measurement happens during execution
            return self
        
        # If called with one argument that's an int < 100, treat as shots
        # If called with no arguments, use default shots
        if qubit_or_shots is None or (isinstance(qubit_or_shots, int) and qubit_or_shots >= 100):
            actual_shots = qubit_or_shots if qubit_or_shots is not None else shots
            
            probabilities = np.abs(self.state)**2

            # Sample outcomes
            outcomes = np.random.choice(len(self.state), size=actual_shots, p=probabilities)

            # Convert to bit strings and count
            counts = {}
            for outcome in outcomes:
                bit_string = format(outcome, f'0{self.n_qubits}b')
                counts[bit_string] = counts.get(bit_string, 0) + 1

            return counts
        else:
            # Called as measure(qubit, classical_bit) format
            return self

    def get_statevector(self) -> np.ndarray:
        """Get the current statevector."""
        return self.state.copy()

    def measure_all(self) -> "QuantumState":
        """Measure all qubits (for API compatibility)."""
        # This is a no-op since measurements are handled in the measure() method
        return self

    def h(self, qubit: int) -> "QuantumState":
        """Apply Hadamard gate to specified qubit."""
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        self.apply_gate(H, [qubit])
        return self

    def x(self, qubit: int) -> "QuantumState":
        """Apply X (Pauli-X) gate to specified qubit."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        self.apply_gate(X, [qubit])
        return self

    def y(self, qubit: int) -> "QuantumState":
        """Apply Y (Pauli-Y) gate to specified qubit."""
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.apply_gate(Y, [qubit])
        return self

    def z(self, qubit: int) -> "QuantumState":
        """Apply Z (Pauli-Z) gate to specified qubit."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.apply_gate(Z, [qubit])
        return self

    def cnot(self, control: int, target: int) -> "QuantumState":
        """Apply CNOT gate between control and target qubits."""
        CNOT = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        self.apply_gate(CNOT, [control, target])
        return self

    def cx(self, control: int, target: int) -> "QuantumState":
        """Apply CX (CNOT) gate between control and target qubits. Alias for cnot."""
        return self.cnot(control, target)

    def cz(self, control: int, target: int) -> "QuantumState":
        """Apply CZ (controlled-Z) gate between control and target qubits."""
        CZ = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=complex)
        self.apply_gate(CZ, [control, target])
        return self

    def rx(self, angle: float, qubit: int) -> "QuantumState":
        """Apply RX rotation gate around X-axis."""
        cos_half = np.cos(angle / 2)
        sin_half = np.sin(angle / 2)
        RX = np.array([
            [cos_half, -1j * sin_half],
            [-1j * sin_half, cos_half]
        ], dtype=complex)
        self.apply_gate(RX, [qubit])
        return self

    def ry(self, angle: float, qubit: int) -> "QuantumState":
        """Apply RY rotation gate around Y-axis."""
        cos_half = np.cos(angle / 2)
        sin_half = np.sin(angle / 2)
        RY = np.array([
            [cos_half, -sin_half],
            [sin_half, cos_half]
        ], dtype=complex)
        self.apply_gate(RY, [qubit])
        return self

    def rz(self, angle: float, qubit: int) -> "QuantumState":
        """Apply RZ rotation gate around Z-axis."""
        RZ = np.array([
            [np.exp(-1j * angle / 2), 0],
            [0, np.exp(1j * angle / 2)]
        ], dtype=complex)
        self.apply_gate(RZ, [qubit])
        return self

class SimulatorBackend(BaseBackend):
    """Pure Python quantum simulator backend.

    This backend provides a simple quantum simulator implementation
    that doesn't require external libraries. Useful for testing,
    development, and as a fallback when other backends are unavailable.

    Args:
        device: Device name (always 'simulator')
        max_qubits: Maximum number of qubits to simulate
        shots: Default number of measurement shots
        **kwargs: Additional parameters

    """

    def __init__(self, device: str = 'simulator', max_qubits: int = 10,
                 shots: int = 1024, **kwargs):
        self.max_qubits = max_qubits
        super().__init__(device=device, shots=shots, **kwargs)
        self.capabilities = {
            'supports_gradient': True,  # Can compute numerical gradients
            'supports_parameter_shift': True,
            'supports_finite_diff': True,
            'supports_backprop': False,
            'supports_hardware': False,
            'supports_noise_models': False,
            'supports_measurements': True,  # Can perform measurements
            'supports_parameterized_circuits': True,  # Can handle parameterized circuits
        }

        # Gate matrices
        self.gates = self._define_gates()

    def _initialize_backend(self) -> None:
        """Initialize simulator backend."""
        logger.info(f"Initialized Python quantum simulator with max {self.max_qubits} qubits")

    def _define_gates(self) -> dict[str, np.ndarray]:
        """Define quantum gate matrices."""
        # Pauli matrices
        identity = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        # Hadamard
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

        # Phase gates
        S = np.array([[1, 0], [0, 1j]], dtype=complex)
        T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

        # CNOT gate
        CNOT = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)

        # CZ gate
        CZ = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=complex)

        # SWAP gate
        SWAP = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)

        return {
            'I': identity, 'X': X, 'Y': Y, 'Z': Z, 'H': H, 'S': S, 'T': T,
            'SDG': S.conj().T, 'TDG': T.conj().T,
            'CNOT': CNOT, 'CX': CNOT, 'CZ': CZ, 'SWAP': SWAP,
            'PAULI_X': X, 'PAULI_Y': Y, 'PAULI_Z': Z, 'HADAMARD': H
        }

    def _rotation_gate(self, axis: str, angle: float) -> np.ndarray:
        """Create rotation gate matrix."""
        c, s_ = np.cos(angle / 2), np.sin(angle / 2)
        if axis == 'X':
            return np.array([[c, -1j * s_], [-1j * s_, c]], dtype=complex)
        if axis == 'Y':
            return np.array([[c, -s_], [s_, c]], dtype=complex)
        if axis == 'Z':
            return np.array([[np.exp(-1j * angle / 2), 0],
                             [0, np.exp(1j * angle / 2)]], dtype=complex)
        raise ValueError(f"Unknown rotation axis: {axis}")

    @staticmethod
    def _controlled(u: np.ndarray) -> np.ndarray:
        d = u.shape[0]
        m = np.eye(2 * d, dtype=complex)
        m[d:, d:] = u
        return m

    def create_circuit(self, n_qubits: int) -> QuantumState:
        """Create a quantum circuit with n qubits."""
        if n_qubits > self.max_qubits:
            raise ValueError(
                f"Requested {n_qubits} qubits, but this simulator is limited to "
                f"{self.max_qubits} (pass max_qubits=... to raise the limit)"
            )
        return QuantumState(n_qubits)

    def add_gate(self, circuit: QuantumState, gate: str, qubits: int | list[int],
                 params: list[float] | None = None) -> QuantumState:
        """Apply a quantum gate to the circuit state.

        Raises:
            ValueError: for unknown gates or invalid qubits.

        """
        if isinstance(qubits, int):
            qubits = [qubits]
        qubits = [int(q) for q in qubits]
        params = list(params) if params is not None else []
        name = gate.upper()

        if name in ('RX', 'RY', 'RZ'):
            matrix = self._rotation_gate(name[1], float(params[0]))
        elif name in ('P', 'PHASE', 'U1'):
            matrix = np.diag([1, np.exp(1j * float(params[0]))]).astype(complex)
        elif name in ('CRX', 'CRY', 'CRZ'):
            matrix = self._controlled(self._rotation_gate(name[2], float(params[0])))
        elif name in ('CCX', 'TOFFOLI'):
            matrix = self._controlled(self.gates['CNOT'])
        elif name in self.gates:
            matrix = self.gates[name]
        else:
            raise ValueError(f"Unsupported gate for SimulatorBackend: {gate}")

        circuit.apply_gate(matrix, qubits)
        return circuit

    def add_measurement(self, circuit: QuantumState, qubits: list[int] | None = None) -> QuantumState:
        """Add measurement operations (no-op in this implementation)."""
        # Measurements are handled in execute_circuit
        return circuit

    def execute_circuit(self, circuit: QuantumState, shots: int | None = None) -> dict[str, Any]:
        """Sample all qubits. Bitstrings are little-endian (qubit 0 rightmost), like Qiskit."""
        shots = shots or self.shots
        probabilities = np.abs(circuit.state) ** 2
        probabilities = probabilities / probabilities.sum()
        outcomes = np.random.choice(len(circuit.state), size=shots, p=probabilities)
        values, freq = np.unique(outcomes, return_counts=True)
        counts = {format(int(v), f'0{circuit.n_qubits}b'): int(c) for v, c in zip(values, freq, strict=True)}
        return {
            'counts': counts,
            'shots': shots,
            'backend': 'python_simulator',
        }

    def get_statevector(self, circuit: QuantumState) -> np.ndarray:
        """Get the statevector from a quantum circuit."""
        return circuit.get_statevector()

    def _get_n_qubits(self, circuit: QuantumState) -> int:
        """Get number of qubits in circuit."""
        return circuit.n_qubits

    def get_version_info(self) -> dict[str, Any]:
        """Get simulator version information."""
        info = super().get_version_info()
        info.update({
            'simulator_type': 'pure_python',
            'max_qubits': self.max_qubits,
            'available_gates': list(self.gates.keys()),
        })
        return info

    def get_backend_info(self) -> dict[str, Any]:
        """Get backend information."""
        return {
            'backend_name': self.__class__.__name__,
            'device': self.device,
            'capabilities': self.capabilities,
            'simulator_type': 'pure_python',
            'max_qubits': self.max_qubits,
            'available_gates': list(self.gates.keys()),
        }

    def run(self, circuit: QuantumState, shots: int | None = None) -> "QuantumResult":
        """Run a quantum circuit and return results."""
        result_dict = self.execute_circuit(circuit, shots)
        return QuantumResult(result_dict)

    def is_available(self) -> bool:
        """Check if the backend is available."""
        return True
