"""Quantum Circuit representation and manipulation for SuperQuantX
"""

import json
from copy import deepcopy
from typing import Any

from pydantic import BaseModel, Field


class QuantumGate(BaseModel):
    """Represents a quantum gate operation
    """

    name: str = Field(..., description="Gate name (e.g., 'H', 'CNOT', 'RZ')")
    qubits: list[int] = Field(..., description="Target qubit indices")
    parameters: list[float] = Field(default_factory=list, description="Gate parameters")
    classical_condition: tuple[str, int] | None = Field(
        default=None,
        description="Classical register condition (register_name, value)"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert gate to dictionary representation"""
        return {
            "name": self.name,
            "qubits": self.qubits,
            "parameters": self.parameters,
            "classical_condition": self.classical_condition
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuantumGate":
        """Create gate from dictionary representation"""
        return cls(**data)

    def __repr__(self) -> str:
        params_str = f"({', '.join(map(str, self.parameters))})" if self.parameters else ""
        qubits_str = f"q{', q'.join(map(str, self.qubits))}"
        return f"{self.name}{params_str} {qubits_str}"


class ClassicalRegister(BaseModel):
    """Represents a classical register for measurements"""

    name: str = Field(..., description="Register name")
    size: int = Field(..., description="Number of classical bits")

    def __repr__(self) -> str:
        return f"ClassicalRegister('{self.name}', {self.size})"


class QuantumRegister(BaseModel):
    """Represents a quantum register"""

    name: str = Field(..., description="Register name")
    size: int = Field(..., description="Number of qubits")

    def __repr__(self) -> str:
        return f"QuantumRegister('{self.name}', {self.size})"


class QuantumCircuit:
    """Quantum circuit representation with gate operations and measurements

    This class provides a high-level interface for building and manipulating
    quantum circuits compatible with multiple quantum computing frameworks.
    """

    def __init__(
        self,
        num_qubits: int,
        num_classical_bits: int | None = None,
        name: str | None = None
    ):
        """Initialize a quantum circuit

        Args:
            num_qubits: Number of qubits in the circuit
            num_classical_bits: Number of classical bits (defaults to num_qubits)
            name: Optional circuit name

        """
        self.num_qubits = num_qubits
        self.num_classical_bits = num_classical_bits or num_qubits
        self.name = name or f"circuit_{id(self)}"

        self.quantum_registers: list[QuantumRegister] = [
            QuantumRegister(name="q", size=num_qubits)
        ]
        self.classical_registers: list[ClassicalRegister] = [
            ClassicalRegister(name="c", size=self.num_classical_bits)
        ]

        self.gates: list[QuantumGate] = []
        self.measurements: list[tuple[int, int]] = []  # (qubit_index, classical_bit_index)
        self.barriers: list[list[int]] = []  # Barrier positions

    def __len__(self) -> int:
        """Return the number of gates in the circuit"""
        return len(self.gates)

    def __repr__(self) -> str:
        return f"QuantumCircuit({self.num_qubits} qubits, {len(self.gates)} gates)"

    def copy(self) -> "QuantumCircuit":
        """Create a deep copy of the circuit"""
        return deepcopy(self)

    def add_register(self, register: QuantumRegister | ClassicalRegister) -> None:
        """Add a quantum or classical register to the circuit"""
        if isinstance(register, QuantumRegister):
            self.quantum_registers.append(register)
            self.num_qubits += register.size
        elif isinstance(register, ClassicalRegister):
            self.classical_registers.append(register)
            self.num_classical_bits += register.size

    # Single-qubit gates
    def h(self, qubit: int) -> "QuantumCircuit":
        """Apply Hadamard gate"""
        self.gates.append(QuantumGate(name="H", qubits=[qubit]))
        return self

    def x(self, qubit: int) -> "QuantumCircuit":
        """Apply Pauli-X gate"""
        self.gates.append(QuantumGate(name="X", qubits=[qubit]))
        return self

    def y(self, qubit: int) -> "QuantumCircuit":
        """Apply Pauli-Y gate"""
        self.gates.append(QuantumGate(name="Y", qubits=[qubit]))
        return self

    def z(self, qubit: int) -> "QuantumCircuit":
        """Apply Pauli-Z gate"""
        self.gates.append(QuantumGate(name="Z", qubits=[qubit]))
        return self

    def s(self, qubit: int) -> "QuantumCircuit":
        """Apply S gate (phase gate)"""
        self.gates.append(QuantumGate(name="S", qubits=[qubit]))
        return self

    def sdg(self, qubit: int) -> "QuantumCircuit":
        """Apply S† gate (inverse phase gate)"""
        self.gates.append(QuantumGate(name="SDG", qubits=[qubit]))
        return self

    def t(self, qubit: int) -> "QuantumCircuit":
        """Apply T gate"""
        self.gates.append(QuantumGate(name="T", qubits=[qubit]))
        return self

    def tdg(self, qubit: int) -> "QuantumCircuit":
        """Apply T† gate"""
        self.gates.append(QuantumGate(name="TDG", qubits=[qubit]))
        return self

    def rx(self, theta: float, qubit: int) -> "QuantumCircuit":
        """Apply rotation around X-axis"""
        self.gates.append(QuantumGate(name="RX", qubits=[qubit], parameters=[theta]))
        return self

    def ry(self, theta: float, qubit: int) -> "QuantumCircuit":
        """Apply rotation around Y-axis"""
        self.gates.append(QuantumGate(name="RY", qubits=[qubit], parameters=[theta]))
        return self

    def rz(self, theta: float, qubit: int) -> "QuantumCircuit":
        """Apply rotation around Z-axis"""
        self.gates.append(QuantumGate(name="RZ", qubits=[qubit], parameters=[theta]))
        return self

    def u(self, theta: float, phi: float, lam: float, qubit: int) -> "QuantumCircuit":
        """Apply general unitary gate U(θ,φ,λ)"""
        self.gates.append(
            QuantumGate(name="U", qubits=[qubit], parameters=[theta, phi, lam])
        )
        return self

    # Two-qubit gates
    def cx(self, control: int, target: int) -> "QuantumCircuit":
        """Apply CNOT gate"""
        self.gates.append(QuantumGate(name="CNOT", qubits=[control, target]))
        return self

    def cnot(self, control: int, target: int) -> "QuantumCircuit":
        """Apply CNOT gate (alias for cx)"""
        return self.cx(control, target)

    def cy(self, control: int, target: int) -> "QuantumCircuit":
        """Apply controlled-Y gate"""
        self.gates.append(QuantumGate(name="CY", qubits=[control, target]))
        return self

    def cz(self, control: int, target: int) -> "QuantumCircuit":
        """Apply controlled-Z gate"""
        self.gates.append(QuantumGate(name="CZ", qubits=[control, target]))
        return self

    def swap(self, qubit1: int, qubit2: int) -> "QuantumCircuit":
        """Apply SWAP gate"""
        self.gates.append(QuantumGate(name="SWAP", qubits=[qubit1, qubit2]))
        return self

    def crx(self, theta: float, control: int, target: int) -> "QuantumCircuit":
        """Apply controlled rotation around X-axis"""
        self.gates.append(
            QuantumGate(name="CRX", qubits=[control, target], parameters=[theta])
        )
        return self

    def cry(self, theta: float, control: int, target: int) -> "QuantumCircuit":
        """Apply controlled rotation around Y-axis"""
        self.gates.append(
            QuantumGate(name="CRY", qubits=[control, target], parameters=[theta])
        )
        return self

    def crz(self, theta: float, control: int, target: int) -> "QuantumCircuit":
        """Apply controlled rotation around Z-axis"""
        self.gates.append(
            QuantumGate(name="CRZ", qubits=[control, target], parameters=[theta])
        )
        return self

    # Three-qubit gates
    def ccx(self, control1: int, control2: int, target: int) -> "QuantumCircuit":
        """Apply Toffoli (CCNOT) gate"""
        self.gates.append(
            QuantumGate(name="TOFFOLI", qubits=[control1, control2, target])
        )
        return self

    def toffoli(self, control1: int, control2: int, target: int) -> "QuantumCircuit":
        """Apply Toffoli gate (alias for ccx)"""
        return self.ccx(control1, control2, target)

    def cswap(self, control: int, target1: int, target2: int) -> "QuantumCircuit":
        """Apply controlled SWAP (Fredkin) gate"""
        self.gates.append(
            QuantumGate(name="FREDKIN", qubits=[control, target1, target2])
        )
        return self

    def fredkin(self, control: int, target1: int, target2: int) -> "QuantumCircuit":
        """Apply Fredkin gate (alias for cswap)"""
        return self.cswap(control, target1, target2)

    # Measurement operations
    def measure(self, qubit: int, classical_bit: int) -> "QuantumCircuit":
        """Measure a qubit into a classical bit"""
        self.measurements.append((qubit, classical_bit))
        return self

    def measure_all(self) -> "QuantumCircuit":
        """Measure all qubits into classical bits"""
        for i in range(min(self.num_qubits, self.num_classical_bits)):
            self.measure(i, i)
        return self

    def barrier(self, qubits: list[int] | None = None) -> "QuantumCircuit":
        """Add a barrier (prevents gate reordering across barrier)"""
        if qubits is None:
            qubits = list(range(self.num_qubits))
        self.barriers.append(qubits)
        return self

    # Circuit composition
    def compose(self, other: "QuantumCircuit", qubits: list[int] | None = None) -> "QuantumCircuit":
        """Compose this circuit with another circuit

        Args:
            other: Circuit to compose with
            qubits: Qubit mapping for the other circuit

        Returns:
            New composed circuit

        """
        if qubits is None:
            qubits = list(range(other.num_qubits))

        if len(qubits) != other.num_qubits:
            raise ValueError("Qubit mapping must match other circuit size")

        new_circuit = self.copy()

        # Map gates from other circuit
        for gate in other.gates:
            mapped_qubits = [qubits[q] for q in gate.qubits]
            new_gate = QuantumGate(
                name=gate.name,
                qubits=mapped_qubits,
                parameters=gate.parameters,
                classical_condition=gate.classical_condition
            )
            new_circuit.gates.append(new_gate)

        # Map measurements
        for qubit, cbit in other.measurements:
            new_circuit.measurements.append((qubits[qubit], cbit))

        return new_circuit

    def inverse(self) -> "QuantumCircuit":
        """Return the inverse (adjoint) of the circuit"""
        inv_circuit = QuantumCircuit(self.num_qubits, self.num_classical_bits)

        # Reverse gates and apply inverse
        for gate in reversed(self.gates):
            inv_gate = self._inverse_gate(gate)
            inv_circuit.gates.append(inv_gate)

        return inv_circuit

    def _inverse_gate(self, gate: QuantumGate) -> QuantumGate:
        """Get the inverse of a gate"""
        inverse_map = {
            "H": "H", "X": "X", "Y": "Y", "Z": "Z",
            "S": "SDG", "SDG": "S", "T": "TDG", "TDG": "T",
            "CNOT": "CNOT", "CZ": "CZ", "SWAP": "SWAP"
        }

        if gate.name in inverse_map:
            return QuantumGate(
                name=inverse_map[gate.name],
                qubits=gate.qubits,
                parameters=gate.parameters,
                classical_condition=gate.classical_condition
            )
        elif gate.name in ["RX", "RY", "RZ", "CRX", "CRY", "CRZ"]:
            # Rotation gates: negate the angle
            inv_params = [-p for p in gate.parameters]
            return QuantumGate(
                name=gate.name,
                qubits=gate.qubits,
                parameters=inv_params,
                classical_condition=gate.classical_condition
            )
        elif gate.name == "U":
            # U(θ,φ,λ)† = U(-θ,-λ,-φ)
            theta, phi, lam = gate.parameters
            inv_params = [-theta, -lam, -phi]
            return QuantumGate(
                name=gate.name,
                qubits=gate.qubits,
                parameters=inv_params,
                classical_condition=gate.classical_condition
            )
        else:
            # For unknown gates, assume self-inverse
            return gate

    # Export functions
    def to_dict(self) -> dict[str, Any]:
        """Convert circuit to dictionary representation"""
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "num_classical_bits": self.num_classical_bits,
            "quantum_registers": [reg.dict() for reg in self.quantum_registers],
            "classical_registers": [reg.dict() for reg in self.classical_registers],
            "gates": [gate.to_dict() for gate in self.gates],
            "measurements": self.measurements,
            "barriers": self.barriers
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert circuit to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuantumCircuit":
        """Create circuit from dictionary representation"""
        circuit = cls(
            num_qubits=data["num_qubits"],
            num_classical_bits=data["num_classical_bits"],
            name=data.get("name")
        )

        circuit.quantum_registers = [
            QuantumRegister(**reg) for reg in data.get("quantum_registers", [])
        ]
        circuit.classical_registers = [
            ClassicalRegister(**reg) for reg in data.get("classical_registers", [])
        ]
        circuit.gates = [QuantumGate.from_dict(gate) for gate in data.get("gates", [])]
        circuit.measurements = data.get("measurements", [])
        circuit.barriers = data.get("barriers", [])

        return circuit

    @classmethod
    def from_json(cls, json_str: str) -> "QuantumCircuit":
        """Create circuit from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def draw(self, output: str = "text") -> str:
        """Draw the circuit in text format

        Args:
            output: Output format ("text" only for now)

        Returns:
            String representation of the circuit

        """
        if output != "text":
            raise NotImplementedError("Only text output is currently supported")

        lines = []
        for i in range(self.num_qubits):
            line = f"q{i} |0⟩─"
            lines.append(line)

        for gate in self.gates:
            max(gate.qubits)
            gate_repr = gate.name

            if len(gate.qubits) == 1:
                # Single-qubit gate
                qubit = gate.qubits[0]
                lines[qubit] += f"─{gate_repr}─"
                for i in range(self.num_qubits):
                    if i != qubit:
                        lines[i] += "─" * (len(gate_repr) + 2)
            elif len(gate.qubits) == 2:
                # Two-qubit gate
                control, target = gate.qubits
                for i in range(self.num_qubits):
                    if i == control:
                        lines[i] += "─●─"
                    elif i == target:
                        lines[i] += "─⊕─"
                    elif min(control, target) < i < max(control, target):
                        lines[i] += "─│─"
                    else:
                        lines[i] += "───"

        # Add measurements
        for qubit, cbit in self.measurements:
            lines[qubit] += f"─M{cbit}"

        return "\n".join(lines)
