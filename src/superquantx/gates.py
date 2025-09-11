"""Advanced quantum gate implementations and utilities for SuperQuantX
"""

import math
from typing import Union

import numpy as np
from scipy.linalg import expm


class GateMatrix:
    """Quantum gate matrix representations and operations
    """

    # Pauli matrices
    identity = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Common single-qubit gates
    H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
    S = np.array([[1, 0], [0, 1j]], dtype=complex)
    T = np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex)

    @staticmethod
    def rx(theta: float) -> np.ndarray:
        """Rotation around X-axis"""
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

    @staticmethod
    def ry(theta: float) -> np.ndarray:
        """Rotation around Y-axis"""
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    @staticmethod
    def rz(theta: float) -> np.ndarray:
        """Rotation around Z-axis"""
        return np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=complex)

    @staticmethod
    def u(theta: float, phi: float, lam: float) -> np.ndarray:
        """General single-qubit unitary gate U(θ,φ,λ)"""
        return np.array([
            [math.cos(theta / 2), -np.exp(1j * lam) * math.sin(theta / 2)],
            [np.exp(1j * phi) * math.sin(theta / 2),
             np.exp(1j * (phi + lam)) * math.cos(theta / 2)]
        ], dtype=complex)

    @staticmethod
    def phase(phi: float) -> np.ndarray:
        """Phase gate"""
        return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)

    # Two-qubit gates
    @staticmethod
    def cnot() -> np.ndarray:
        """CNOT gate matrix"""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)

    @staticmethod
    def cz() -> np.ndarray:
        """Controlled-Z gate matrix"""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=complex)

    @staticmethod
    def swap() -> np.ndarray:
        """SWAP gate matrix"""
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)

    @staticmethod
    def iswap() -> np.ndarray:
        """ISWAP gate matrix"""
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)

    @staticmethod
    def crx(theta: float) -> np.ndarray:
        """Controlled rotation around X-axis"""
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, c, -1j * s],
            [0, 0, -1j * s, c]
        ], dtype=complex)

    @staticmethod
    def cry(theta: float) -> np.ndarray:
        """Controlled rotation around Y-axis"""
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, c, -s],
            [0, 0, s, c]
        ], dtype=complex)

    @staticmethod
    def crz(theta: float) -> np.ndarray:
        """Controlled rotation around Z-axis"""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, np.exp(-1j * theta / 2), 0],
            [0, 0, 0, np.exp(1j * theta / 2)]
        ], dtype=complex)

    @staticmethod
    def xx(theta: float) -> np.ndarray:
        """XX interaction gate"""
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array([
            [c, 0, 0, -1j * s],
            [0, c, -1j * s, 0],
            [0, -1j * s, c, 0],
            [-1j * s, 0, 0, c]
        ], dtype=complex)

    @staticmethod
    def yy(theta: float) -> np.ndarray:
        """YY interaction gate"""
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array([
            [c, 0, 0, 1j * s],
            [0, c, -1j * s, 0],
            [0, -1j * s, c, 0],
            [1j * s, 0, 0, c]
        ], dtype=complex)

    @staticmethod
    def zz(theta: float) -> np.ndarray:
        """ZZ interaction gate"""
        return np.array([
            [np.exp(-1j * theta / 2), 0, 0, 0],
            [0, np.exp(1j * theta / 2), 0, 0],
            [0, 0, np.exp(1j * theta / 2), 0],
            [0, 0, 0, np.exp(-1j * theta / 2)]
        ], dtype=complex)

    # Three-qubit gates
    @staticmethod
    def toffoli() -> np.ndarray:
        """Toffoli (CCNOT) gate matrix"""
        matrix = np.eye(8, dtype=complex)
        matrix[6, 6] = 0
        matrix[7, 7] = 0
        matrix[6, 7] = 1
        matrix[7, 6] = 1
        return matrix

    @staticmethod
    def fredkin() -> np.ndarray:
        """Fredkin (CSWAP) gate matrix"""
        matrix = np.eye(8, dtype=complex)
        matrix[5, 5] = 0
        matrix[6, 6] = 0
        matrix[5, 6] = 1
        matrix[6, 5] = 1
        return matrix


class ParametricGate:
    """Parametric quantum gate with symbolic parameters
    """

    def __init__(
        self,
        name: str,
        num_qubits: int,
        matrix_func: callable,
        parameters: list[str],
        description: str | None = None
    ):
        """Initialize parametric gate

        Args:
            name: Gate name
            num_qubits: Number of qubits the gate acts on
            matrix_func: Function that returns gate matrix given parameters
            parameters: List of parameter names
            description: Optional gate description

        """
        self.name = name
        self.num_qubits = num_qubits
        self.matrix_func = matrix_func
        self.parameters = parameters
        self.description = description or f"{name} gate"

    def matrix(self, *args) -> np.ndarray:
        """Get gate matrix for given parameter values"""
        if len(args) != len(self.parameters):
            raise ValueError(f"Expected {len(self.parameters)} parameters, got {len(args)}")
        return self.matrix_func(*args)

    def __call__(self, *args) -> np.ndarray:
        """Shorthand for matrix method"""
        return self.matrix(*args)

    def __repr__(self) -> str:
        params_str = ", ".join(self.parameters)
        return f"{self.name}({params_str})"


class GateDecomposer:
    """Utility class for decomposing quantum gates into basic gate sets
    """

    @staticmethod
    def decompose_arbitrary_single_qubit(matrix: np.ndarray) -> list[tuple[str, list[float]]]:
        """Decompose arbitrary single-qubit unitary into U3 gates

        Args:
            matrix: 2x2 unitary matrix

        Returns:
            List of (gate_name, parameters) tuples

        """
        if matrix.shape != (2, 2):
            raise ValueError("Matrix must be 2x2 for single-qubit decomposition")

        # Extract parameters from SU(2) matrix
        # U = e^(iα) * U3(θ, φ, λ)
        det = np.linalg.det(matrix)
        alpha = np.angle(det) / 2

        # Normalize to SU(2)
        su2_matrix = matrix / np.exp(1j * alpha)

        # Extract U3 parameters
        theta = 2 * np.arccos(np.abs(su2_matrix[0, 0]))

        if np.abs(np.sin(theta / 2)) < 1e-10:
            # θ ≈ 0, gate is just a phase
            phi = 0
            lam = np.angle(su2_matrix[0, 0]) * 2
        else:
            phi = np.angle(su2_matrix[1, 0]) - np.angle(su2_matrix[0, 1]) + np.pi
            lam = np.angle(su2_matrix[1, 0]) + np.angle(su2_matrix[0, 1])

        decomposition = []
        if abs(alpha) > 1e-10:
            decomposition.append(("global_phase", [alpha]))

        decomposition.append(("u3", [theta, phi, lam]))

        return decomposition

    @staticmethod
    def decompose_cnot_to_cz(control: int, target: int) -> list[tuple[str, list[int], list[float]]]:
        """Decompose CNOT to CZ using Hadamard gates

        Returns:
            List of (gate_name, qubits, parameters) tuples

        """
        return [
            ("h", [target], []),
            ("cz", [control, target], []),
            ("h", [target], [])
        ]

    @staticmethod
    def decompose_toffoli() -> list[tuple[str, list[int], list[float]]]:
        """Decompose Toffoli gate into CNOT and single-qubit gates

        Returns:
            List of (gate_name, qubits, parameters) for qubits [0, 1, 2]

        """
        return [
            ("h", [2], []),
            ("cnot", [1, 2], []),
            ("t", [2], []),
            ("cnot", [0, 2], []),
            ("tdg", [2], []),
            ("cnot", [1, 2], []),
            ("t", [2], []),
            ("cnot", [0, 2], []),
            ("tdg", [1], []),
            ("tdg", [2], []),
            ("cnot", [0, 1], []),
            ("h", [2], []),
            ("tdg", [0], []),
            ("t", [1], []),
            ("cnot", [0, 1], [])
        ]

    @staticmethod
    def decompose_fredkin() -> list[tuple[str, list[int], list[float]]]:
        """Decompose Fredkin gate into CNOT and Toffoli gates

        Returns:
            List of (gate_name, qubits, parameters) for qubits [0, 1, 2]

        """
        return [
            ("cnot", [2, 1], []),
            ("toffoli", [0, 1, 2], []),
            ("cnot", [2, 1], [])
        ]


class PauliString:
    """Represents a Pauli string for Hamiltonian construction
    """

    def __init__(self, pauli_ops: str, coefficient: complex = 1.0):
        """Initialize Pauli string

        Args:
            pauli_ops: String of Pauli operators (e.g., "IXZY")
            coefficient: Complex coefficient

        """
        self.pauli_ops = pauli_ops.upper()
        self.coefficient = coefficient
        self.num_qubits = len(pauli_ops)

        # Validate Pauli string
        valid_ops = set("IXYZ")
        if not all(op in valid_ops for op in self.pauli_ops):
            raise ValueError("Pauli string must contain only I, X, Y, Z operators")

    def matrix(self) -> np.ndarray:
        """Get the matrix representation of the Pauli string"""
        matrices = {
            'I': GateMatrix.identity,
            'X': GateMatrix.X,
            'Y': GateMatrix.Y,
            'Z': GateMatrix.Z
        }

        result = np.array([[1]], dtype=complex)
        for op in self.pauli_ops:
            result = np.kron(result, matrices[op])

        return self.coefficient * result

    def commutes_with(self, other: "PauliString") -> bool:
        """Check if this Pauli string commutes with another"""
        if len(self.pauli_ops) != len(other.pauli_ops):
            return False

        anti_commuting_pairs = {('X', 'Y'), ('Y', 'X'), ('X', 'Z'), ('Z', 'X'), ('Y', 'Z'), ('Z', 'Y')}
        anti_commutations = 0

        for op1, op2 in zip(self.pauli_ops, other.pauli_ops, strict=False):
            if (op1, op2) in anti_commuting_pairs:
                anti_commutations += 1

        return anti_commutations % 2 == 0

    def __mul__(self, other: Union[complex, "PauliString"]) -> "PauliString":
        """Multiply with scalar or another Pauli string"""
        if isinstance(other, int | float | complex):
            return PauliString(self.pauli_ops, self.coefficient * other)
        elif isinstance(other, PauliString):
            if len(self.pauli_ops) != len(other.pauli_ops):
                raise ValueError("Pauli strings must have same length")

            # Multiply Pauli operators
            result_ops = []
            phase = 1

            for op1, op2 in zip(self.pauli_ops, other.pauli_ops, strict=False):
                if op1 == 'I':
                    result_ops.append(op2)
                elif op2 == 'I':
                    result_ops.append(op1)
                elif op1 == op2:
                    result_ops.append('I')
                else:
                    # Anti-commuting case
                    pauli_order = {'X': 0, 'Y': 1, 'Z': 2}
                    ops = [op1, op2]
                    if pauli_order[ops[0]] > pauli_order[ops[1]]:
                        ops.reverse()
                        phase *= -1

                    if ops == ['X', 'Y']:
                        result_ops.append('Z')
                        phase *= 1j
                    elif ops == ['X', 'Z']:
                        result_ops.append('Y')
                        phase *= -1j
                    elif ops == ['Y', 'Z']:
                        result_ops.append('X')
                        phase *= 1j

            return PauliString(''.join(result_ops), self.coefficient * other.coefficient * phase)
        else:
            return NotImplemented

    def __rmul__(self, other: complex) -> "PauliString":
        """Right multiplication by scalar"""
        return self * other

    def __repr__(self) -> str:
        if self.coefficient == 1:
            return self.pauli_ops
        elif self.coefficient == -1:
            return f"-{self.pauli_ops}"
        else:
            return f"({self.coefficient})*{self.pauli_ops}"


class Hamiltonian:
    """Quantum Hamiltonian represented as sum of Pauli strings
    """

    def __init__(self, pauli_strings: list[PauliString]):
        """Initialize Hamiltonian

        Args:
            pauli_strings: List of Pauli strings that sum to form the Hamiltonian

        """
        self.pauli_strings = pauli_strings
        if pauli_strings:
            self.num_qubits = pauli_strings[0].num_qubits
            # Validate all strings have same length
            if not all(p.num_qubits == self.num_qubits for p in pauli_strings):
                raise ValueError("All Pauli strings must have same length")
        else:
            self.num_qubits = 0

    def matrix(self) -> np.ndarray:
        """Get matrix representation of the Hamiltonian"""
        if not self.pauli_strings:
            return np.zeros((1, 1), dtype=complex)

        dim = 2 ** self.num_qubits
        result = np.zeros((dim, dim), dtype=complex)

        for pauli_string in self.pauli_strings:
            result += pauli_string.matrix()

        return result

    def expectation_value(self, state: np.ndarray) -> complex:
        """Calculate expectation value ⟨ψ|H|ψ⟩"""
        H = self.matrix()
        return np.conj(state).T @ H @ state

    def ground_state_energy(self) -> float:
        """Calculate ground state energy (lowest eigenvalue)"""
        eigenvals = np.linalg.eigvals(self.matrix())
        return float(np.min(np.real(eigenvals)))

    def time_evolution(self, time: float) -> np.ndarray:
        """Generate time evolution operator U(t) = exp(-iHt)"""
        H = self.matrix()
        return expm(-1j * H * time)

    def __add__(self, other: "Hamiltonian") -> "Hamiltonian":
        """Add two Hamiltonians"""
        return Hamiltonian(self.pauli_strings + other.pauli_strings)

    def __mul__(self, scalar: complex) -> "Hamiltonian":
        """Multiply Hamiltonian by scalar"""
        return Hamiltonian([scalar * p for p in self.pauli_strings])

    def __rmul__(self, scalar: complex) -> "Hamiltonian":
        """Right multiplication by scalar"""
        return self * scalar

    @classmethod
    def from_dict(cls, hamiltonian_dict: dict[str, complex]) -> "Hamiltonian":
        """Create Hamiltonian from dictionary

        Args:
            hamiltonian_dict: Dictionary mapping Pauli strings to coefficients

        Example:
            {"IXZI": 0.5, "YZII": -0.3, "ZXYX": 1.2j}

        """
        pauli_strings = []
        for pauli_ops, coeff in hamiltonian_dict.items():
            pauli_strings.append(PauliString(pauli_ops, coeff))
        return cls(pauli_strings)

    @staticmethod
    def heisenberg_model(
        num_qubits: int,
        Jx: float = 1.0,
        Jy: float = 1.0,
        Jz: float = 1.0,
        periodic: bool = False
    ) -> "Hamiltonian":
        """Create Heisenberg model Hamiltonian

        H = Σᵢ (Jₓ XᵢXᵢ₊₁ + Jᵧ YᵢYᵢ₊₁ + Jᵤ ZᵢZᵢ₊₁)
        """
        pauli_strings = []

        max_i = num_qubits if periodic else num_qubits - 1

        for i in range(max_i):
            j = (i + 1) % num_qubits

            # XX term
            xx_ops = ['I'] * num_qubits
            xx_ops[i] = 'X'
            xx_ops[j] = 'X'
            pauli_strings.append(PauliString(''.join(xx_ops), Jx))

            # YY term
            yy_ops = ['I'] * num_qubits
            yy_ops[i] = 'Y'
            yy_ops[j] = 'Y'
            pauli_strings.append(PauliString(''.join(yy_ops), Jy))

            # ZZ term
            zz_ops = ['I'] * num_qubits
            zz_ops[i] = 'Z'
            zz_ops[j] = 'Z'
            pauli_strings.append(PauliString(''.join(zz_ops), Jz))

        return Hamiltonian(pauli_strings)

    @staticmethod
    def ising_model(
        num_qubits: int,
        J: float = 1.0,
        h: float = 0.0,
        periodic: bool = False
    ) -> "Hamiltonian":
        """Create transverse field Ising model Hamiltonian

        H = -J Σᵢ ZᵢZᵢ₊₁ - h Σᵢ Xᵢ
        """
        pauli_strings = []

        # ZZ interactions
        max_i = num_qubits if periodic else num_qubits - 1
        for i in range(max_i):
            j = (i + 1) % num_qubits
            zz_ops = ['I'] * num_qubits
            zz_ops[i] = 'Z'
            zz_ops[j] = 'Z'
            pauli_strings.append(PauliString(''.join(zz_ops), -J))

        # X fields
        for i in range(num_qubits):
            x_ops = ['I'] * num_qubits
            x_ops[i] = 'X'
            pauli_strings.append(PauliString(''.join(x_ops), -h))

        return Hamiltonian(pauli_strings)

    def __repr__(self) -> str:
        if not self.pauli_strings:
            return "0"
        return " + ".join(str(p) for p in self.pauli_strings)
