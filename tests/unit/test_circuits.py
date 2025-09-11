"""
Unit tests for quantum circuits
"""

import json

import numpy as np
import pytest

from superquantx.circuits import (
    ClassicalRegister,
    QuantumCircuit,
    QuantumGate,
    QuantumRegister,
)


class TestQuantumGate:
    """Test QuantumGate class"""

    def test_gate_creation(self):
        """Test basic gate creation"""
        gate = QuantumGate(name="H", qubits=[0])

        assert gate.name == "H"
        assert gate.qubits == [0]
        assert gate.parameters == []
        assert gate.classical_condition is None

    def test_parameterized_gate(self):
        """Test parameterized gate creation"""
        gate = QuantumGate(name="RY", qubits=[1], parameters=[np.pi/2])

        assert gate.name == "RY"
        assert gate.qubits == [1]
        assert gate.parameters == [np.pi/2]

    def test_two_qubit_gate(self):
        """Test two-qubit gate creation"""
        gate = QuantumGate(name="CNOT", qubits=[0, 1])

        assert gate.name == "CNOT"
        assert gate.qubits == [0, 1]

    def test_gate_with_classical_condition(self):
        """Test gate with classical condition"""
        gate = QuantumGate(
            name="X",
            qubits=[0],
            classical_condition=("c", 1)
        )

        assert gate.classical_condition == ("c", 1)

    def test_gate_serialization(self):
        """Test gate to_dict and from_dict"""
        original_gate = QuantumGate(
            name="RZ",
            qubits=[2],
            parameters=[np.pi/4],
            classical_condition=("c0", 0)
        )

        gate_dict = original_gate.to_dict()
        restored_gate = QuantumGate.from_dict(gate_dict)

        assert restored_gate.name == original_gate.name
        assert restored_gate.qubits == original_gate.qubits
        assert restored_gate.parameters == original_gate.parameters
        assert restored_gate.classical_condition == original_gate.classical_condition

    def test_gate_repr(self):
        """Test gate string representation"""
        gate1 = QuantumGate(name="H", qubits=[0])
        gate2 = QuantumGate(name="RY", qubits=[1], parameters=[np.pi/2])
        gate3 = QuantumGate(name="CNOT", qubits=[0, 1])

        assert str(gate1) == "H q0"
        assert str(gate2) == f"RY({np.pi/2}) q1"
        assert str(gate3) == "CNOT q0, q1"


class TestQuantumRegister:
    """Test QuantumRegister class"""

    def test_register_creation(self):
        """Test register creation"""
        qreg = QuantumRegister(name="qubits", size=5)

        assert qreg.name == "qubits"
        assert qreg.size == 5

    def test_register_repr(self):
        """Test register string representation"""
        qreg = QuantumRegister(name="q", size=3)
        assert str(qreg) == "QuantumRegister('q', 3)"


class TestClassicalRegister:
    """Test ClassicalRegister class"""

    def test_register_creation(self):
        """Test classical register creation"""
        creg = ClassicalRegister(name="bits", size=5)

        assert creg.name == "bits"
        assert creg.size == 5

    def test_register_repr(self):
        """Test classical register string representation"""
        creg = ClassicalRegister(name="c", size=3)
        assert str(creg) == "ClassicalRegister('c', 3)"


class TestQuantumCircuit:
    """Test QuantumCircuit class"""

    def test_circuit_creation(self):
        """Test basic circuit creation"""
        circuit = QuantumCircuit(3)

        assert circuit.num_qubits == 3
        assert circuit.num_classical_bits == 3
        assert len(circuit.gates) == 0
        assert len(circuit.measurements) == 0
        assert len(circuit.quantum_registers) == 1
        assert len(circuit.classical_registers) == 1

    def test_circuit_creation_with_different_classical_bits(self):
        """Test circuit with different number of classical bits"""
        circuit = QuantumCircuit(3, 5)

        assert circuit.num_qubits == 3
        assert circuit.num_classical_bits == 5

    def test_circuit_length(self):
        """Test circuit length (number of gates)"""
        circuit = QuantumCircuit(2)

        assert len(circuit) == 0

        circuit.h(0)
        circuit.cnot(0, 1)

        assert len(circuit) == 2

    def test_single_qubit_gates(self):
        """Test single-qubit gate operations"""
        circuit = QuantumCircuit(2)

        # Test all single-qubit gates
        circuit.h(0)
        circuit.x(0)
        circuit.y(0)
        circuit.z(0)
        circuit.s(0)
        circuit.sdg(0)
        circuit.t(0)
        circuit.tdg(0)

        assert len(circuit.gates) == 8
        assert circuit.gates[0].name == "H"
        assert circuit.gates[1].name == "X"
        assert circuit.gates[2].name == "Y"
        assert circuit.gates[3].name == "Z"
        assert circuit.gates[4].name == "S"
        assert circuit.gates[5].name == "SDG"
        assert circuit.gates[6].name == "T"
        assert circuit.gates[7].name == "TDG"

    def test_parameterized_gates(self):
        """Test parameterized gate operations"""
        circuit = QuantumCircuit(2)

        circuit.rx(np.pi/2, 0)
        circuit.ry(np.pi/4, 1)
        circuit.rz(np.pi/3, 0)
        circuit.u(np.pi/2, np.pi/4, np.pi/3, 1)

        assert len(circuit.gates) == 4
        assert circuit.gates[0].name == "RX"
        assert circuit.gates[0].parameters == [np.pi/2]
        assert circuit.gates[1].name == "RY"
        assert circuit.gates[1].parameters == [np.pi/4]
        assert circuit.gates[2].name == "RZ"
        assert circuit.gates[2].parameters == [np.pi/3]
        assert circuit.gates[3].name == "U"
        assert circuit.gates[3].parameters == [np.pi/2, np.pi/4, np.pi/3]

    def test_two_qubit_gates(self):
        """Test two-qubit gate operations"""
        circuit = QuantumCircuit(3)

        circuit.cx(0, 1)
        circuit.cnot(1, 2)
        circuit.cy(0, 2)
        circuit.cz(1, 0)
        circuit.swap(0, 2)

        assert len(circuit.gates) == 5
        assert circuit.gates[0].name == "CNOT"
        assert circuit.gates[0].qubits == [0, 1]
        assert circuit.gates[1].name == "CNOT"
        assert circuit.gates[1].qubits == [1, 2]
        assert circuit.gates[2].name == "CY"
        assert circuit.gates[3].name == "CZ"
        assert circuit.gates[4].name == "SWAP"

    def test_parameterized_two_qubit_gates(self):
        """Test parameterized two-qubit gates"""
        circuit = QuantumCircuit(2)

        circuit.crx(np.pi/2, 0, 1)
        circuit.cry(np.pi/4, 1, 0)
        circuit.crz(np.pi/3, 0, 1)

        assert len(circuit.gates) == 3
        assert circuit.gates[0].name == "CRX"
        assert circuit.gates[0].parameters == [np.pi/2]
        assert circuit.gates[1].name == "CRY"
        assert circuit.gates[1].parameters == [np.pi/4]
        assert circuit.gates[2].name == "CRZ"
        assert circuit.gates[2].parameters == [np.pi/3]

    def test_three_qubit_gates(self):
        """Test three-qubit gate operations"""
        circuit = QuantumCircuit(3)

        circuit.ccx(0, 1, 2)
        circuit.toffoli(0, 1, 2)
        circuit.cswap(0, 1, 2)
        circuit.fredkin(0, 1, 2)

        assert len(circuit.gates) == 4
        assert circuit.gates[0].name == "TOFFOLI"
        assert circuit.gates[0].qubits == [0, 1, 2]
        assert circuit.gates[1].name == "TOFFOLI"
        assert circuit.gates[2].name == "FREDKIN"
        assert circuit.gates[3].name == "FREDKIN"

    def test_measurements(self):
        """Test measurement operations"""
        circuit = QuantumCircuit(3, 3)

        circuit.measure(0, 0)
        circuit.measure(1, 1)

        assert len(circuit.measurements) == 2
        assert circuit.measurements[0] == (0, 0)
        assert circuit.measurements[1] == (1, 1)

        # Test measure_all
        circuit_all = QuantumCircuit(2, 2)
        circuit_all.measure_all()

        assert len(circuit_all.measurements) == 2
        assert circuit_all.measurements[0] == (0, 0)
        assert circuit_all.measurements[1] == (1, 1)

    def test_barriers(self):
        """Test barrier operations"""
        circuit = QuantumCircuit(3)

        circuit.h(0)
        circuit.barrier()
        circuit.cnot(0, 1)
        circuit.barrier([0, 1])

        assert len(circuit.barriers) == 2
        assert circuit.barriers[0] == [0, 1, 2]  # All qubits
        assert circuit.barriers[1] == [0, 1]     # Specific qubits

    def test_circuit_copy(self):
        """Test circuit copying"""
        original = QuantumCircuit(2)
        original.h(0)
        original.cnot(0, 1)
        original.measure(0, 0)

        copy = original.copy()

        assert len(copy.gates) == len(original.gates)
        assert len(copy.measurements) == len(original.measurements)
        assert copy.num_qubits == original.num_qubits

        # Modify copy to ensure it's independent
        copy.x(1)
        assert len(copy.gates) == len(original.gates) + 1

    def test_circuit_composition(self):
        """Test circuit composition"""
        circuit1 = QuantumCircuit(2)
        circuit1.h(0)
        circuit1.cnot(0, 1)

        circuit2 = QuantumCircuit(2)
        circuit2.x(0)
        circuit2.z(1)

        composed = circuit1.compose(circuit2)

        assert len(composed.gates) == 4
        assert composed.gates[0].name == "H"
        assert composed.gates[1].name == "CNOT"
        assert composed.gates[2].name == "X"
        assert composed.gates[3].name == "Z"

    def test_circuit_composition_with_qubit_mapping(self):
        """Test circuit composition with qubit mapping"""
        circuit1 = QuantumCircuit(3)
        circuit1.h(0)

        circuit2 = QuantumCircuit(2)
        circuit2.x(0)
        circuit2.cnot(0, 1)

        # Map circuit2 qubits to circuit1 qubits [1, 2]
        composed = circuit1.compose(circuit2, qubits=[1, 2])

        assert len(composed.gates) == 3
        assert composed.gates[0].name == "H"
        assert composed.gates[0].qubits == [0]
        assert composed.gates[1].name == "X"
        assert composed.gates[1].qubits == [1]  # Mapped from qubit 0
        assert composed.gates[2].name == "CNOT"
        assert composed.gates[2].qubits == [1, 2]  # Mapped from qubits [0, 1]

    def test_circuit_inverse(self):
        """Test circuit inverse"""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.rz(np.pi/2, 0)
        circuit.cnot(0, 1)
        circuit.s(1)

        inverse = circuit.inverse()

        # Check that gates are reversed and inverted
        assert len(inverse.gates) == 4
        assert inverse.gates[0].name == "SDG"  # S†
        assert inverse.gates[1].name == "CNOT"  # CNOT is self-inverse
        assert inverse.gates[2].name == "RZ"
        assert inverse.gates[2].parameters == [-np.pi/2]  # Negated parameter
        assert inverse.gates[3].name == "H"  # H is self-inverse

    def test_circuit_serialization(self):
        """Test circuit to_dict and from_dict"""
        original = QuantumCircuit(2, name="test_circuit")
        original.h(0)
        original.cnot(0, 1)
        original.measure(0, 0)
        original.measure(1, 1)
        original.barrier()

        # Test to_dict
        circuit_dict = original.to_dict()

        assert circuit_dict["name"] == "test_circuit"
        assert circuit_dict["num_qubits"] == 2
        assert circuit_dict["num_classical_bits"] == 2
        assert len(circuit_dict["gates"]) == 2
        assert len(circuit_dict["measurements"]) == 2
        assert len(circuit_dict["barriers"]) == 1

        # Test from_dict
        restored = QuantumCircuit.from_dict(circuit_dict)

        assert restored.name == original.name
        assert restored.num_qubits == original.num_qubits
        assert restored.num_classical_bits == original.num_classical_bits
        assert len(restored.gates) == len(original.gates)
        assert len(restored.measurements) == len(original.measurements)
        assert len(restored.barriers) == len(original.barriers)

    def test_circuit_json_serialization(self):
        """Test circuit JSON serialization"""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)

        # Test to_json
        json_str = circuit.to_json()
        json_data = json.loads(json_str)

        assert json_data["num_qubits"] == 2
        assert len(json_data["gates"]) == 2

        # Test from_json
        restored = QuantumCircuit.from_json(json_str)

        assert restored.num_qubits == circuit.num_qubits
        assert len(restored.gates) == len(circuit.gates)

    def test_circuit_draw_text(self):
        """Test circuit text drawing"""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)

        text_drawing = circuit.draw("text")

        assert isinstance(text_drawing, str)
        assert "q0" in text_drawing
        assert "q1" in text_drawing
        assert "H" in text_drawing

        # Test unsupported format
        with pytest.raises(NotImplementedError):
            circuit.draw("matplotlib")

    def test_add_register(self):
        """Test adding registers to circuit"""
        circuit = QuantumCircuit(2)

        # Add quantum register
        qreg = QuantumRegister("extra_q", 2)
        circuit.add_register(qreg)

        assert circuit.num_qubits == 4
        assert len(circuit.quantum_registers) == 2

        # Add classical register
        creg = ClassicalRegister("extra_c", 3)
        circuit.add_register(creg)

        assert circuit.num_classical_bits == 5
        assert len(circuit.classical_registers) == 2

    def test_method_chaining(self):
        """Test that circuit methods return self for chaining"""
        circuit = QuantumCircuit(2)

        result = (circuit
                 .h(0)
                 .cnot(0, 1)
                 .measure(0, 0)
                 .barrier())

        # Should return the same circuit object
        assert result is circuit
        assert len(circuit.gates) == 2
        assert len(circuit.measurements) == 1
        assert len(circuit.barriers) == 1
