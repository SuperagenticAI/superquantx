"""Base backend interface for quantum computing platforms.

This module defines the abstract interface that all quantum backends
must implement to work with SuperQuantX algorithms.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)

FEATURE_MAPS = ('ZZFeatureMap', 'PauliFeatureMap', 'AngleEncoding')

_PAULI = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}


@dataclass(frozen=True)
class FeatureMapSpec:
    """Backend-independent description of a data-encoding feature map."""

    name: str
    n_features: int
    reps: int = 2


def _pauli_label_to_dense(label: str, n_qubits: int) -> str:
    """Normalise 'ZI' or indexed forms like 'Z0 X2' to a dense label."""
    label = label.strip().upper()
    if any(ch.isdigit() for ch in label):
        dense = ['I'] * n_qubits
        token = ''
        for ch in label.replace('*', ' ') + ' ':
            if ch.isalpha():
                if token:
                    dense[int(token[1:])] = token[0]
                token = ch
            elif ch.isdigit():
                token += ch
            elif token:
                dense[int(token[1:])] = token[0]
                token = ''
        return ''.join(dense)
    return label


def pauli_string_matrix(label: str) -> np.ndarray:
    """Dense matrix of a Pauli label; character i acts on qubit i (np.kron order)."""
    return reduce(np.kron, [_PAULI[c] for c in label.upper()])


def hamiltonian_to_matrix(hamiltonian: Any, n_qubits: int | None = None) -> np.ndarray:
    """Convert supported Hamiltonian formats to a dense matrix.

    The convention is big-endian: in a label such as ``"ZI"`` the first
    character acts on qubit 0, which is the most significant bit.
    """
    if isinstance(hamiltonian, np.ndarray):
        H = np.asarray(hamiltonian, dtype=complex)
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ValueError("Hamiltonian matrix must be square")
        return H

    terms: list[tuple[complex, str]] | None = None
    if isinstance(hamiltonian, dict):
        terms = [(c, k) for k, c in hamiltonian.items()]
    elif isinstance(hamiltonian, (list, tuple)) and hamiltonian and all(
        isinstance(t, (list, tuple)) and len(t) == 2 for t in hamiltonian
    ):
        terms = [(a, b) if isinstance(b, str) else (b, a) for a, b in hamiltonian]
    elif hasattr(hamiltonian, 'pauli_strings'):  # superquantx.gates.Hamiltonian
        terms = [(p.coefficient, p.pauli_ops) for p in hamiltonian.pauli_strings]
    elif hasattr(hamiltonian, 'to_list') and hasattr(hamiltonian, 'num_qubits'):
        # qiskit SparsePauliOp labels are little-endian; reverse them
        terms = [(c, lbl[::-1]) for lbl, c in hamiltonian.to_list()]
    elif hasattr(hamiltonian, 'wires') and hasattr(hamiltonian, 'terms'):
        import pennylane as qml  # PennyLane operator
        n = n_qubits if n_qubits is not None else len(hamiltonian.wires)
        return np.asarray(qml.matrix(hamiltonian, wire_order=list(range(n))), dtype=complex)

    if terms is None:
        raise TypeError(
            f"Unsupported Hamiltonian type {type(hamiltonian).__name__}. Use a matrix, "
            "a dict of Pauli labels, a list of (coeff, label) pairs or gates.Hamiltonian."
        )

    if n_qubits is None:
        import re
        if any(ch.isdigit() for _, lbl in terms for ch in lbl):
            n_qubits = 1 + max(int(m) for _, lbl in terms for m in re.findall(r'\d+', lbl))
        else:
            n_qubits = len(terms[0][1].strip())
    H = np.zeros((2 ** n_qubits, 2 ** n_qubits), dtype=complex)
    for coeff, lbl in terms:
        dense = _pauli_label_to_dense(lbl, n_qubits)
        if len(dense) != n_qubits:
            raise ValueError(f"Pauli label '{lbl}' does not act on {n_qubits} qubits")
        H += complex(coeff) * pauli_string_matrix(dense)
    return H


def hamiltonian_num_qubits(hamiltonian: Any) -> int:
    """Infer the number of qubits a Hamiltonian acts on."""
    if isinstance(hamiltonian, np.ndarray):
        n = int(round(np.log2(hamiltonian.shape[0])))
        if 2 ** n != hamiltonian.shape[0]:
            raise ValueError("Hamiltonian dimension must be a power of 2")
        return n
    return int(round(np.log2(hamiltonian_to_matrix(hamiltonian).shape[0])))


def _as_weight_matrix(problem_instance: Any, n_qubits: int) -> np.ndarray:
    W = np.asarray(problem_instance, dtype=float)
    if W.ndim != 2 or W.shape != (n_qubits, n_qubits):
        raise NotImplementedError(
            "The built-in QAOA circuit supports weighted MaxCut only: pass an "
            f"{n_qubits}x{n_qubits} adjacency matrix as the problem instance."
        )
    return W


def maxcut_values(W: np.ndarray) -> np.ndarray:
    """Cut value of every bitstring (big-endian index) for weight matrix W."""
    n = W.shape[0]
    idx = np.arange(2 ** n)
    bits = (idx[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1
    diff = bits[:, :, None] != bits[:, None, :]
    return 0.5 * np.einsum('kij,ij->k', diff, np.triu(W, 1) + np.triu(W, 1).T)

class BaseBackend(ABC):
    """Abstract base class for quantum computing backends.

    This class defines the interface that all quantum backends must implement
    to provide quantum circuit execution, measurement, and algorithm-specific
    operations for SuperQuantX.

    Args:
        device: Device or simulator to use
        shots: Default number of measurement shots
        **kwargs: Backend-specific configuration

    """

    #: False for backends that are not gate-model devices (e.g. annealers).
    gate_model: bool = True

    def __init__(self, device: str | None = None, shots: int = 1024, **kwargs):
        self.device = device
        self.shots = shots
        self.config = kwargs
        self.capabilities = {}

        self._initialize_backend()

    @abstractmethod
    def _initialize_backend(self) -> None:
        """Initialize backend-specific components."""
        pass

    # ========================================================================
    # Core Circuit Operations
    # ========================================================================

    @abstractmethod
    def create_circuit(self, n_qubits: int) -> Any:
        """Create a quantum circuit with n qubits."""
        pass

    @abstractmethod
    def add_gate(self, circuit: Any, gate: str, qubits: int | list[int],
                 params: list[float] | None = None) -> Any:
        """Add a quantum gate to the circuit."""
        pass

    @abstractmethod
    def add_measurement(self, circuit: Any, qubits: list[int] | None = None) -> Any:
        """Add measurement operations to the circuit."""
        pass

    @abstractmethod
    def execute_circuit(self, circuit: Any, shots: int | None = None) -> dict[str, Any]:
        """Execute quantum circuit and return results."""
        pass

    @abstractmethod
    def get_statevector(self, circuit: Any) -> np.ndarray:
        """Get the statevector from a quantum circuit."""
        pass

    # ========================================================================
    # Qubit ordering helpers
    # ========================================================================
    #
    # Different SDKs order qubits differently in bitstrings and statevectors.
    # Qiskit and the built-in simulator are little-endian (qubit 0 is the
    # rightmost bit); Cirq, PennyLane, Braket and TKET are big-endian (qubit 0
    # is the leftmost bit). The generic algorithm code below always works in a
    # big-endian convention (qubit i is bit i from the left, matching
    # np.kron ordering of Pauli strings such as "ZI"). The convention of each
    # backend is detected once, empirically, by preparing |1> on qubit 0.

    def _bit_order(self, kind: str) -> str:
        """Return 'big' or 'little' for this backend's counts or statevectors."""
        cache = self.__dict__.setdefault('_sqx_bit_order', {})
        if kind in cache:
            return cache[kind]

        circuit = self.create_circuit(2)
        circuit = self.add_gate(circuit, 'X', 0)
        if kind == 'statevector':
            sv = np.asarray(self.get_statevector(circuit)).ravel()
            idx = int(np.argmax(np.abs(sv)))
            order = 'little' if idx == 1 else 'big'
        elif kind == 'counts':
            circuit = self.add_measurement(circuit)
            counts = self.execute_circuit(circuit, shots=8).get('counts', {})
            if not counts:
                raise RuntimeError(
                    f"{self.__class__.__name__} returned no counts while probing qubit order"
                )
            key = str(max(counts, key=counts.get)).replace(' ', '')
            order = 'little' if key == '01' else 'big'
        else:
            raise ValueError(f"Unknown ordering kind: {kind}")

        cache[kind] = order
        return order

    def counts_big_endian(self, counts: dict[str, int]) -> dict[str, int]:
        """Return counts re-keyed so that character i is the value of qubit i."""
        if self._bit_order('counts') == 'big':
            return {str(k).replace(' ', ''): int(v) for k, v in counts.items()}
        return {str(k).replace(' ', '')[::-1]: int(v) for k, v in counts.items()}

    def statevector_big_endian(self, circuit: Any) -> np.ndarray:
        """Statevector of ``circuit`` with qubit 0 as the most significant bit."""
        sv = np.asarray(self.get_statevector(circuit), dtype=complex).ravel()
        n = int(round(np.log2(len(sv))))
        if 2 ** n != len(sv):
            raise RuntimeError(f"Statevector length {len(sv)} is not a power of two")
        if self._bit_order('statevector') == 'little' and n > 1:
            sv = sv.reshape([2] * n).transpose(list(range(n))[::-1]).reshape(-1)
        return sv

    # ========================================================================
    # Quantum Algorithm Support (generic, built on the primitives above)
    # ========================================================================

    def create_feature_map(self, n_features: int, feature_map: str = 'ZZFeatureMap',
                           reps: int = 2) -> 'FeatureMapSpec':
        """Return a feature-map specification used by :meth:`compute_kernel_matrix`.

        Supported maps: ``ZZFeatureMap`` (Havlicek et al. 2019, linear
        entanglement), ``PauliFeatureMap`` (RX/RY/RZ plus CNOT chain) and
        ``AngleEncoding`` (one RY per feature).
        """
        if feature_map not in FEATURE_MAPS:
            raise ValueError(
                f"Unknown feature map '{feature_map}'. Supported: {sorted(FEATURE_MAPS)}"
            )
        return FeatureMapSpec(name=feature_map, n_features=n_features, reps=reps)

    def build_feature_map_circuit(self, x: np.ndarray, spec: 'FeatureMapSpec') -> Any:
        """Build the backend-native circuit that encodes ``x`` with ``spec``."""
        n = spec.n_features
        x = np.asarray(x, dtype=float).ravel()
        if len(x) != n:
            raise ValueError(f"Feature map expects {n} features, got {len(x)}")
        circuit = self.create_circuit(n)
        for _ in range(spec.reps):
            if spec.name == 'ZZFeatureMap':
                for i in range(n):
                    circuit = self.add_gate(circuit, 'H', i)
                    circuit = self.add_gate(circuit, 'RZ', i, [2.0 * x[i]])
                for i in range(n - 1):
                    phi = 2.0 * (np.pi - x[i]) * (np.pi - x[i + 1])
                    circuit = self.add_gate(circuit, 'CNOT', [i, i + 1])
                    circuit = self.add_gate(circuit, 'RZ', i + 1, [phi])
                    circuit = self.add_gate(circuit, 'CNOT', [i, i + 1])
            elif spec.name == 'PauliFeatureMap':
                for i in range(n):
                    circuit = self.add_gate(circuit, 'RX', i, [x[i]])
                    circuit = self.add_gate(circuit, 'RY', i, [x[i]])
                    circuit = self.add_gate(circuit, 'RZ', i, [x[i]])
                for i in range(n - 1):
                    circuit = self.add_gate(circuit, 'CNOT', [i, i + 1])
            else:  # AngleEncoding
                for i in range(n):
                    circuit = self.add_gate(circuit, 'RY', i, [x[i]])
        return circuit

    def compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray,
                              feature_map: Any = None, shots: int | None = None) -> np.ndarray:
        """Fidelity quantum kernel K[i, j] = |<phi(x1_i)|phi(x2_j)>|^2.

        Each data point is encoded with the feature map and simulated on this
        backend; the kernel is computed exactly from the resulting statevectors
        (``shots`` is accepted for API compatibility and ignored). This is a
        real quantum-circuit kernel, but it is classically simulated, so it says
        nothing about quantum advantage.
        """
        X1 = np.atleast_2d(np.asarray(X1, dtype=float))
        X2 = np.atleast_2d(np.asarray(X2, dtype=float))
        if feature_map is None:
            spec = self.create_feature_map(X1.shape[1])
        elif isinstance(feature_map, str):
            spec = self.create_feature_map(X1.shape[1], feature_map)
        elif isinstance(feature_map, FeatureMapSpec):
            spec = feature_map
        else:
            raise TypeError(
                "feature_map must be a FeatureMapSpec, a feature map name, or None; "
                f"got {type(feature_map).__name__}"
            )

        def states(X: np.ndarray) -> np.ndarray:
            return np.array([
                self.statevector_big_endian(self.build_feature_map_circuit(x, spec))
                for x in X
            ])

        S1 = states(X1)
        S2 = S1 if X2 is X1 or (X2.shape == X1.shape and np.array_equal(X1, X2)) else states(X2)
        return np.abs(S1.conj() @ S2.T) ** 2

    def ansatz_param_count(self, ansatz_type: str, n_qubits: int, reps: int = 1) -> int:
        """Number of parameters used by :meth:`create_ansatz`."""
        if ansatz_type == 'EfficientSU2':
            return 2 * n_qubits * (reps + 1)
        if ansatz_type in ('RealAmplitudes', 'TwoLocal'):
            return n_qubits * (reps + 1)
        raise ValueError(
            f"Ansatz '{ansatz_type}' is not implemented. Use 'RealAmplitudes', "
            "'EfficientSU2' or 'TwoLocal', or pass a callable ansatz."
        )

    def create_ansatz(self, ansatz_type: str, n_qubits: int, params: np.ndarray,
                      include_custom_gates: bool = False, reps: int | None = None) -> Any:
        """Build a hardware-efficient ansatz with this backend's own gates.

        ``RealAmplitudes``/``TwoLocal``: ``reps`` layers of RY on every qubit
        followed by a linear CNOT chain, plus a final RY layer.
        ``EfficientSU2``: the same structure with RY and RZ rotations.
        """
        params = np.asarray(params, dtype=float).ravel()
        per_layer = 2 * n_qubits if ansatz_type == 'EfficientSU2' else n_qubits
        if reps is None:
            reps = max(len(params) // per_layer - 1, 0)
        expected = self.ansatz_param_count(ansatz_type, n_qubits, reps)
        if len(params) != expected:
            raise ValueError(
                f"{ansatz_type} with {n_qubits} qubits and reps={reps} needs "
                f"{expected} parameters, got {len(params)}"
            )

        circuit = self.create_circuit(n_qubits)
        idx = 0
        for layer in range(reps + 1):
            for q in range(n_qubits):
                circuit = self.add_gate(circuit, 'RY', q, [params[idx]])
                idx += 1
                if ansatz_type == 'EfficientSU2':
                    circuit = self.add_gate(circuit, 'RZ', q, [params[idx]])
                    idx += 1
            if layer < reps:
                for q in range(n_qubits - 1):
                    circuit = self.add_gate(circuit, 'CNOT', [q, q + 1])
        return circuit

    def compute_expectation(self, circuit: Any, hamiltonian: Any,
                            shots: int | None = None) -> float:
        """Exact expectation value <psi|H|psi> from this backend's statevector.

        ``hamiltonian`` may be a dense matrix (qubit 0 most significant), a
        ``superquantx.gates.Hamiltonian``, a dict like ``{"ZI": -0.4}``, a list
        of ``(coeff, "ZI")`` pairs, or a PennyLane / Qiskit operator.
        ``shots`` is accepted for API compatibility and ignored.
        """
        sv = self.statevector_big_endian(circuit)
        n = int(round(np.log2(len(sv))))
        H = hamiltonian_to_matrix(hamiltonian, n)
        return float(np.real(np.vdot(sv, H @ sv)))

    def create_qaoa_circuit(self, n_qubits: int, gammas: np.ndarray, betas: np.ndarray,
                            problem_hamiltonian: Any, mixer_hamiltonian: Any,
                            initial_state: Any, problem_instance: Any) -> Any:
        """QAOA circuit for weighted MaxCut given an adjacency matrix.

        Cost layer: exp(-i gamma w_ij Z_i Z_j / 2) per edge (CNOT, RZ, CNOT);
        mixer layer: RX(2 beta) on every qubit; initial state |+>^n.
        """
        W = _as_weight_matrix(problem_instance, n_qubits)
        circuit = self.create_circuit(n_qubits)
        for q in range(n_qubits):
            circuit = self.add_gate(circuit, 'H', q)
        for gamma, beta in zip(gammas, betas, strict=True):
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    if W[i, j] != 0:
                        circuit = self.add_gate(circuit, 'CNOT', [i, j])
                        circuit = self.add_gate(circuit, 'RZ', j, [float(gamma * W[i, j])])
                        circuit = self.add_gate(circuit, 'CNOT', [i, j])
            for q in range(n_qubits):
                circuit = self.add_gate(circuit, 'RX', q, [float(2 * beta)])
        return circuit

    def execute_qaoa(self, circuit: Any, problem_hamiltonian: Any, problem_instance: Any,
                     shots: int | None = None) -> float:
        """Expected MaxCut value of the QAOA state (exact, from the statevector)."""
        probs = np.abs(self.statevector_big_endian(circuit)) ** 2
        n = int(round(np.log2(len(probs))))
        W = _as_weight_matrix(problem_instance, n)
        return float(probs @ maxcut_values(W))

    def sample_circuit(self, circuit: Any, shots: int | None = None) -> np.ndarray:
        """Sample bitstrings; row k column i is the measured value of qubit i."""
        circuit = self.add_measurement(circuit)
        result = self.execute_circuit(circuit, shots)
        if not result.get('counts'):
            raise RuntimeError(f"{self.__class__.__name__} returned no counts: {result}")
        counts = self.counts_big_endian(result['counts'])
        rows = []
        for bitstring, count in counts.items():
            rows.extend([[int(b) for b in bitstring]] * int(count))
        return np.array(rows, dtype=int)

    # ========================================================================
    # Data Encoding and Processing
    # ========================================================================

    def encode_data_point(self, data: np.ndarray, encoding: str, n_qubits: int) -> Any:
        """Encode classical data into quantum state."""
        circuit = self.create_circuit(n_qubits)

        if encoding == 'amplitude':
            return self._amplitude_encoding(circuit, data)
        elif encoding == 'angle':
            return self._angle_encoding(circuit, data)
        elif encoding == 'basis':
            return self._basis_encoding(circuit, data)
        else:
            logger.warning(f"Encoding '{encoding}' not implemented, using angle encoding")
            return self._angle_encoding(circuit, data)

    def _amplitude_encoding(self, circuit: Any, data: np.ndarray) -> Any:
        """Amplitude encoding of data."""
        raise NotImplementedError(
            "Amplitude encoding is not implemented; use 'angle' or 'basis' encoding"
        )

    def _angle_encoding(self, circuit: Any, data: np.ndarray) -> Any:
        """Angle encoding of data into rotation gates."""
        for i, value in enumerate(data):
            if i < self._get_n_qubits(circuit):
                self.add_gate(circuit, 'RY', i, [value])
        return circuit

    def _basis_encoding(self, circuit: Any, data: np.ndarray) -> Any:
        """Basis encoding of data."""
        # Convert to binary and encode
        binary_data = np.unpackbits(data.astype(np.uint8))
        for i, bit in enumerate(binary_data):
            if i < self._get_n_qubits(circuit) and bit:
                self.add_gate(circuit, 'X', i)
        return circuit

    # ========================================================================
    # Machine Learning Specific Operations
    # ========================================================================

    def compute_quantum_distance(self, x1: np.ndarray, x2: np.ndarray, metric: str,
                                encoding: str, n_qubits: int, shots: int) -> float:
        """Distance between two data points.

        Only ``metric='fidelity'`` uses quantum states: it angle-encodes both
        points and returns sqrt(1 - |<x1|x2>|^2) computed from statevectors.
        ``'euclidean'`` and ``'manhattan'`` are plain classical distances.
        """
        if metric == 'fidelity':
            spec = FeatureMapSpec(name='AngleEncoding', n_features=len(x1), reps=1)
            s1 = self.statevector_big_endian(self.build_feature_map_circuit(x1, spec))
            s2 = self.statevector_big_endian(self.build_feature_map_circuit(x2, spec))
            return float(np.sqrt(max(0.0, 1.0 - abs(np.vdot(s1, s2)) ** 2)))
        if metric == 'euclidean':
            return np.linalg.norm(x1 - x2)
        elif metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            return np.linalg.norm(x1 - x2)

    def execute_qnn(self, input_data: np.ndarray, weights: np.ndarray,
                   quantum_layers: list[dict], classical_layers: list[dict],
                   encoding: str, measurement: str, shots: int) -> np.ndarray:
        """Execute quantum neural network."""
        # Each variational layer holds an RY and an RZ angle per qubit.
        n_qubits = len(weights) // (2 * len(quantum_layers)) if quantum_layers else input_data.shape[1]

        # Get expected output size from classical layers
        expected_output_size = None
        if classical_layers:
            for layer in classical_layers:
                if 'units' in layer:
                    expected_output_size = layer['units']
                    break


        outputs = []
        for sample in input_data:
            # Encode input data into a fresh circuit
            circuit = self.encode_data_point(sample, encoding, n_qubits)

            # Add variational layers
            param_idx = 0
            for layer in quantum_layers:
                layer_params = weights[param_idx:param_idx + n_qubits * 2]
                circuit = self._add_variational_layer(circuit, layer_params)
                param_idx += n_qubits * 2

            # Measure
            if measurement == 'expectation':
                # Compute expectation values
                result = self._compute_pauli_expectation(circuit, ['Z'] * n_qubits)
            else:
                # Sample and get probabilities
                result = self.execute_circuit(circuit, shots)
                result = self._result_to_probabilities(result)

            # Apply classical layers if present
            if classical_layers and expected_output_size:
                # Simple linear transformation to get desired output size
                if len(result) != expected_output_size:
                    # Reshape/transform the quantum output to match expected size
                    if len(result) > expected_output_size:
                        # Take first n elements or average
                        result = result[:expected_output_size]
                    else:
                        # Pad with zeros or repeat
                        result = np.pad(result, (0, expected_output_size - len(result)), mode='constant')

            outputs.append(result)

        return np.array(outputs)

    def _add_variational_layer(self, circuit: Any, params: np.ndarray) -> Any:
        """Add a variational layer to circuit."""
        n_qubits = self._get_n_qubits(circuit)

        # RY rotations
        for i in range(n_qubits):
            if i < len(params) // 2:
                self.add_gate(circuit, 'RY', i, [params[i]])

        # Entangling gates
        for i in range(n_qubits - 1):
            self.add_gate(circuit, 'CNOT', [i, i + 1])

        # RZ rotations
        for i in range(n_qubits):
            if i + n_qubits < len(params):
                self.add_gate(circuit, 'RZ', i, [params[i + n_qubits]])

        return circuit

    # ========================================================================
    # Utility Methods
    # ========================================================================

    @abstractmethod
    def _get_n_qubits(self, circuit: Any) -> int:
        """Get number of qubits in circuit."""
        pass

    def _result_to_bitstrings(self, result: dict[str, Any]) -> np.ndarray:
        """Convert execution result to bit string array."""
        # This is backend-specific and should be overridden
        if 'counts' in result:
            counts = result['counts']
            bitstrings = []
            for bitstring, count in counts.items():
                for _ in range(count):
                    bitstrings.append([int(b) for b in bitstring])
            return np.array(bitstrings)
        raise ValueError(f"Result has no 'counts' entry: {sorted(result)}")

    def _result_to_probabilities(self, result: dict[str, Any]) -> np.ndarray:
        """Convert execution result to probability array."""
        if 'counts' in result:
            counts = result['counts']
            total_shots = sum(counts.values())
            probs = []
            for bitstring in sorted(counts.keys()):
                probs.append(counts[bitstring] / total_shots)
            return np.array(probs)
        raise ValueError(f"Result has no 'counts' entry: {sorted(result)}")

    def _compute_pauli_expectation(self, circuit: Any, pauli_strings: list[str]) -> np.ndarray:
        """Exact single-qubit Pauli expectations; ``pauli_strings[i]`` acts on qubit i."""
        sv = self.statevector_big_endian(circuit)
        n = int(round(np.log2(len(sv))))
        values = []
        for i, p in enumerate(pauli_strings[:n]):
            label = ['I'] * n
            label[i] = p.upper()
            H = hamiltonian_to_matrix({''.join(label): 1.0}, n)
            values.append(float(np.real(np.vdot(sv, H @ sv))))
        return np.array(values)

    def get_version_info(self) -> dict[str, Any]:
        """Get backend version information."""
        return {
            'backend_name': self.__class__.__name__,
            'device': self.device,
            'capabilities': self.capabilities,
            'backend_version': '1.0.0',  # Default backend version
        }

    def get_device_info(self) -> dict[str, Any]:
        """Get information about the quantum device."""
        return {
            'device': self.device,
            'n_qubits': getattr(self, 'n_qubits', 'Unknown'),
            'topology': getattr(self, 'topology', 'Unknown'),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device='{self.device}', shots={self.shots})"
