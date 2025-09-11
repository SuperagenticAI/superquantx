"""
Basic quantum circuit example for SuperQuantX

This example demonstrates:
- Creating a simple quantum circuit
- Adding quantum gates
- Measuring qubits
- Visualizing the circuit
"""

import numpy as np

from superquantx import QuantumCircuit, SuperQuantXClient


def create_bell_state():
    """Create a Bell state (maximally entangled state)"""
    print("Creating Bell state circuit...")

    # Create 2-qubit circuit
    circuit = QuantumCircuit(2, name="Bell State")

    # Apply Hadamard to first qubit
    circuit.h(0)

    # Apply CNOT to create entanglement
    circuit.cnot(0, 1)

    # Measure both qubits
    circuit.measure_all()

    return circuit


def create_ghz_state(num_qubits=3):
    """Create a GHZ state (multi-qubit entangled state)"""
    print(f"Creating {num_qubits}-qubit GHZ state...")

    circuit = QuantumCircuit(num_qubits, name=f"GHZ-{num_qubits}")

    # Apply Hadamard to first qubit
    circuit.h(0)

    # Apply CNOT gates to create entanglement
    for i in range(1, num_qubits):
        circuit.cnot(0, i)

    # Measure all qubits
    circuit.measure_all()

    return circuit


def create_quantum_teleportation():
    """Create a quantum teleportation circuit"""
    print("Creating quantum teleportation circuit...")

    # 3 qubits: alice's qubit, entangled pair (alice + bob)
    circuit = QuantumCircuit(3, 3, name="Quantum Teleportation")

    # Prepare state to teleport (arbitrary state)
    circuit.ry(np.pi/3, 0)  # Alice's qubit

    # Create Bell pair between Alice and Bob
    circuit.h(1)
    circuit.cnot(1, 2)

    # Bell measurement on Alice's qubits
    circuit.cnot(0, 1)
    circuit.h(0)

    # Measure Alice's qubits
    circuit.measure(0, 0)
    circuit.measure(1, 1)

    # Apply corrections to Bob's qubit based on measurements
    circuit.gates.append(
        circuit.QuantumGate(name="CNOT", qubits=[1, 2],
                          classical_condition=("c", 1))
    )
    circuit.gates.append(
        circuit.QuantumGate(name="Z", qubits=[2],
                          classical_condition=("c", 0))
    )

    # Measure Bob's qubit
    circuit.measure(2, 2)

    return circuit


def demonstrate_circuit_operations():
    """Demonstrate various circuit operations"""
    print("Demonstrating circuit operations...")

    circuit = QuantumCircuit(3, name="Operations Demo")

    # Single-qubit gates
    circuit.h(0)      # Hadamard
    circuit.x(1)      # Pauli-X (bit flip)
    circuit.y(2)      # Pauli-Y
    circuit.z(0)      # Pauli-Z (phase flip)

    # Rotation gates
    circuit.rx(np.pi/4, 1)   # X-rotation
    circuit.ry(np.pi/3, 2)   # Y-rotation
    circuit.rz(np.pi/2, 0)   # Z-rotation

    # Two-qubit gates
    circuit.cnot(0, 1)       # CNOT
    circuit.cz(1, 2)         # Controlled-Z
    circuit.swap(0, 2)       # SWAP

    # Three-qubit gate
    circuit.toffoli(0, 1, 2) # Toffoli (CCNOT)

    # Add barrier for visual separation
    circuit.barrier()

    # Measurements
    circuit.measure_all()

    return circuit


def run_circuit_locally(circuit):
    """Run circuit using local simulation"""
    print(f"\nRunning {circuit.name} locally...")

    from superquantx.measurements import QuantumMeasurement

    # Create measurement system
    measurement = QuantumMeasurement(backend="simulator")

    # Execute circuit
    result = measurement.measure_circuit(circuit, shots=1024)

    # Display results
    print("Measurement results:")
    print(f"Total shots: {result.shots}")
    print(f"Outcomes: {result.counts}")
    print(f"Probabilities: {result.probabilities}")
    print(f"Most frequent outcome: {result.most_frequent}")
    print(f"Entropy: {result.entropy():.4f}")

    return result


def run_circuit_remotely(circuit, api_key):
    """Run circuit on SuperQuantX platform"""
    if not api_key:
        print("No API key provided, skipping remote execution")
        return None

    print(f"\nRunning {circuit.name} on SuperQuantX platform...")

    # Initialize client
    client = SuperQuantXClient(api_key)

    try:
        # Check platform status
        health = client.health_check_sync()
        print(f"Platform status: {health.get('status', 'unknown')}")

        # Submit job
        job = client.submit_job_sync(
            circuit_data=circuit.to_dict(),
            backend="quantum_simulator",
            shots=1024
        )

        print(f"Job submitted: {job.job_id}")
        print(f"Status: {job.status}")

        # Wait for completion
        completed_job = client.wait_for_job_sync(job.job_id, timeout=300)

        print(f"Job completed with status: {completed_job.status}")
        if completed_job.results:
            counts = completed_job.results.get("counts", {})
            print(f"Results: {counts}")

        return completed_job

    except Exception as e:
        print(f"Error running remote job: {e}")
        return None


def analyze_circuit(circuit):
    """Analyze circuit properties"""
    print(f"\nAnalyzing {circuit.name}...")
    print(f"Number of qubits: {circuit.num_qubits}")
    print(f"Number of classical bits: {circuit.num_classical_bits}")
    print(f"Circuit depth (gates): {len(circuit.gates)}")

    # Count gate types
    gate_counts = {}
    for gate in circuit.gates:
        gate_counts[gate.name] = gate_counts.get(gate.name, 0) + 1

    print("Gate composition:")
    for gate_name, count in sorted(gate_counts.items()):
        print(f"  {gate_name}: {count}")

    # Display circuit diagram
    print("\nCircuit diagram:")
    print(circuit.draw("text"))


def main():
    """Main example function"""
    print("SuperQuantX Basic Circuit Examples")
    print("=" * 40)

    # Get API key (optional)
    api_key = input("Enter your SuperQuantX API key (or press Enter to skip remote execution): ").strip()
    if not api_key:
        api_key = None

    # Example 1: Bell State
    print("\n1. Bell State Example")
    print("-" * 20)
    bell_circuit = create_bell_state()
    analyze_circuit(bell_circuit)
    run_circuit_locally(bell_circuit)
    run_circuit_remotely(bell_circuit, api_key)

    # Example 2: GHZ State
    print("\n\n2. GHZ State Example")
    print("-" * 20)
    ghz_circuit = create_ghz_state(3)
    analyze_circuit(ghz_circuit)
    run_circuit_locally(ghz_circuit)

    # Example 3: Circuit Operations
    print("\n\n3. Circuit Operations Example")
    print("-" * 30)
    ops_circuit = demonstrate_circuit_operations()
    analyze_circuit(ops_circuit)
    run_circuit_locally(ops_circuit)

    # Example 4: Circuit Serialization
    print("\n\n4. Circuit Serialization Example")
    print("-" * 33)
    print("Serializing Bell circuit to JSON...")
    json_str = bell_circuit.to_json()
    print(f"JSON length: {len(json_str)} characters")

    print("Deserializing from JSON...")
    restored_circuit = QuantumCircuit.from_json(json_str)
    print(f"Restored circuit: {restored_circuit.name}")
    print(f"Gates match: {len(restored_circuit.gates) == len(bell_circuit.gates)}")

    # Example 5: Circuit Composition
    print("\n\n5. Circuit Composition Example")
    print("-" * 32)
    circuit1 = QuantumCircuit(2, name="Circuit 1")
    circuit1.h(0)
    circuit1.cnot(0, 1)

    circuit2 = QuantumCircuit(2, name="Circuit 2")
    circuit2.x(0)
    circuit2.z(1)

    composed = circuit1.compose(circuit2)
    composed.name = "Composed Circuit"
    print(f"Original circuit 1 gates: {len(circuit1.gates)}")
    print(f"Original circuit 2 gates: {len(circuit2.gates)}")
    print(f"Composed circuit gates: {len(composed.gates)}")

    # Example 6: Circuit Inverse
    print("\n\n6. Circuit Inverse Example")
    print("-" * 28)
    original = QuantumCircuit(2, name="Original")
    original.h(0)
    original.rz(np.pi/2, 1)
    original.cnot(0, 1)

    inverse = original.inverse()
    inverse.name = "Inverse"

    print("Original circuit:")
    print(original.draw("text"))
    print("\nInverse circuit:")
    print(inverse.draw("text"))


if __name__ == "__main__":
    main()
