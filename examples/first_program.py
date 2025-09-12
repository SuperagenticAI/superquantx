import superquantx as sqx
import numpy as np
import matplotlib.pyplot as plt

# Verify your installation
print(f"SuperQuantX version: {sqx.__version__}")
print(f"Available backends: {sqx.list_available_backends()}")

# Get a quantum backend
backend = sqx.get_backend('simulator')

# Create a circuit with one qubit
circuit = backend.create_circuit(n_qubits=1)

# Initially, the qubit is in state |0⟩
print("Initial state: |0⟩")

# Apply a Hadamard gate to create superposition
circuit = backend.add_gate(circuit, 'H', 0)  # Now the qubit is in state (|0⟩ + |1⟩)/√2

# Measure the qubit
circuit = backend.add_measurement(circuit)

# Run the circuit multiple times
result = backend.execute_circuit(circuit, shots=1000)
counts = result['counts']

print(f"Measurement results: {counts}")
print("🎉 You've created quantum superposition!")

# Create a circuit with 2 qubits
circuit = backend.create_circuit(n_qubits=2)

# Step 1: Put first qubit in superposition
circuit = backend.add_gate(circuit, 'H', 0)

# Step 2: Entangle the qubits with CNOT gate
circuit = backend.add_gate(circuit, 'CNOT', [0, 1])  # Controlled-X gate (CNOT)

# Step 3: Measure both qubits
circuit = backend.add_measurement(circuit)

# Run the circuit
result = backend.execute_circuit(circuit, shots=1000)
counts = result['counts']

print(f"Bell state results: {counts}")

# Circuit information (visualization to be implemented)
print(f"\nCircuit created successfully!")
print(f"Circuit has {circuit.n_qubits} qubits")
print(f"Current state vector shape: {circuit.state.shape}")


# Let's verify the entanglement property
print("\n🔍 Analyzing entanglement:")
total_shots = sum(counts.values())

prob_00 = counts.get('00', 0) / total_shots
prob_11 = counts.get('11', 0) / total_shots
prob_01 = counts.get('01', 0) / total_shots
prob_10 = counts.get('10', 0) / total_shots

print(f"P(00) = {prob_00:.3f}")
print(f"P(11) = {prob_11:.3f}")
print(f"P(01) = {prob_01:.3f}")
print(f"P(10) = {prob_10:.3f}")

if prob_01 + prob_10 < 0.1:  # Less than 10% due to statistical noise
    print("✅ Qubits are entangled!")
else:
    print("❌ Something went wrong...")


def quantum_random_number(num_bits=8):
    """Generate a random number using quantum superposition."""

    # Create circuit with specified number of qubits
    circuit = backend.create_circuit(n_qubits=num_bits)

    # Put all qubits in superposition
    for i in range(num_bits):
        circuit = backend.add_gate(circuit, 'H', i)

    # Measure all qubits
    circuit = backend.add_measurement(circuit)

    # Run the circuit once
    result = backend.execute_circuit(circuit, shots=1)

    # Convert result to integer
    binary_result = list(result['counts'].keys())[0]
    random_number = int(binary_result, 2)

    return random_number, binary_result

# Generate some quantum random numbers
print("🎲 Quantum Random Numbers:")
for i in range(5):
    number, binary = quantum_random_number(8)  # 8-bit numbers (0-255)
    print(f"  {number:3d} (binary: {binary})")



def quantum_interference_demo():
    """Demonstrate quantum interference patterns."""

    circuit = backend.create_circuit(n_qubits=1)

    # Create superposition
    circuit = backend.add_gate(circuit, 'H', 0)

    # Add a phase (rotation around Z-axis)
    circuit = backend.add_gate(circuit, 'RZ', 0, [np.pi/4])  # 45-degree phase

    # Apply another Hadamard - this creates interference
    circuit = backend.add_gate(circuit, 'H', 0)

    circuit = backend.add_measurement(circuit)

    # Run multiple times to see the pattern
    result = backend.execute_circuit(circuit, shots=1000)
    counts = result['counts']

    return counts

# Test different phases
phases = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
results = []

print("🌊 Quantum Interference Patterns:")
print("Phase\t|0⟩ Count\t|1⟩ Count")
print("-" * 35)

for phase in phases:
    circuit = backend.create_circuit(n_qubits=1)
    circuit = backend.add_gate(circuit, 'H', 0)
    circuit = backend.add_gate(circuit, 'RZ', 0, [phase])
    circuit = backend.add_gate(circuit, 'H', 0)
    circuit = backend.add_measurement(circuit)

    result = backend.execute_circuit(circuit, shots=1000)
    counts = result['counts']

    count_0 = counts.get('0', 0)
    count_1 = counts.get('1', 0)

    print(f"{phase:.2f}\t{count_0}\t\t{count_1}")
    results.append((phase, count_0, count_1))


class QuantumCoin:
    """A quantum coin flipper with controllable bias."""

    def __init__(self, backend_name='simulator'):
        self.backend = sqx.get_backend(backend_name)

    def flip(self, bias=0.5, shots=1000):
        """
        Flip the quantum coin.

        Args:
            bias (float): Probability of getting 'heads' (0.0 to 1.0)
            shots (int): Number of measurements

        Returns:
            dict: Results with counts for heads and tails
        """
        # Calculate rotation angle for desired bias
        theta = 2 * np.arcsin(np.sqrt(bias))

        circuit = self.backend.create_circuit(n_qubits=1)

        # Start in |0⟩ (tails)
        # Rotate to achieve desired bias
        circuit = self.backend.add_gate(circuit, 'RY', 0, [theta])

        circuit = self.backend.add_measurement(circuit)

        result = self.backend.execute_circuit(circuit, shots=shots)
        counts = result['counts']

        # Map 0->tails, 1->heads
        heads = counts.get('1', 0)
        tails = counts.get('0', 0)

        return {
            'heads': heads,
            'tails': tails,
            'bias': heads / (heads + tails) if (heads + tails) > 0 else 0
        }

# Test the quantum coin
coin = QuantumCoin()

print("🪙 Quantum Coin Flipper Test:")
biases = [0.1, 0.3, 0.5, 0.7, 0.9]

for target_bias in biases:
    result = coin.flip(bias=target_bias, shots=1000)
    print(f"Target: {target_bias:.1f}, Actual: {result['bias']:.3f}, "
          f"Heads: {result['heads']}, Tails: {result['tails']}")


def plot_quantum_results(counts, title="Quantum Measurement Results"):
    """Plot quantum measurement results."""

    states = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(states, values, alpha=0.7)

    # Color bars differently for different states
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    for i, bar in enumerate(bars):
        bar.set_color(colors[i % len(colors)])

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Quantum State', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')

    plt.tight_layout()
    # Save instead of showing to avoid timeout in automated tests
    plt.savefig('/tmp/quantum_results.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close to free memory
    print(f"✅ Plot saved successfully with data: {dict(zip(states, values))}")

# Example: Visualize Bell state results
circuit = backend.create_circuit(n_qubits=2)
circuit = backend.add_gate(circuit, 'H', 0)
circuit = backend.add_gate(circuit, 'CNOT', [0, 1])
circuit = backend.add_measurement(circuit)

result = backend.execute_circuit(circuit, shots=1000)
counts = result['counts']

plot_quantum_results(counts, "Bell State |Φ⁺⟩ = (|00⟩ + |11⟩)/√2")


def compare_backends(algorithm_func, *args, **kwargs):
    """Run the same algorithm on different backends."""

    available_backends = sqx.list_available_backends()
    results = {}

    # Only test backends that are actually available and support gate-model circuits
    for backend_name in available_backends:
        # Skip quantum annealing backends (they use a different programming model)
        if backend_name in {'ocean', 'dwave'}:
            print(f"⏭️  Skipping {backend_name}: Quantum annealing backend (not compatible with gate circuits)")
            continue

        if available_backends[backend_name].get('available', False):
            try:
                print(f"🔄 Testing {backend_name}...")
                result = algorithm_func(backend_name, *args, **kwargs)
                results[backend_name] = result
                print(f"✅ {backend_name} completed successfully")

            except Exception as e:
                print(f"❌ {backend_name} failed: {e}")
                results[backend_name] = None
        else:
            reason = available_backends[backend_name].get('reason', 'Not available')
            print(f"⏭️  Skipping {backend_name}: {reason}")

    return results

def bell_state_test(backend_name):
    """Create Bell state on specified backend."""
    backend = sqx.get_backend(backend_name)
    circuit = backend.create_circuit(n_qubits=2)

    circuit = backend.add_gate(circuit, 'H', 0)
    circuit = backend.add_gate(circuit, 'CNOT', [0, 1])
    circuit = backend.add_measurement(circuit)

    result = backend.execute_circuit(circuit, shots=1000)
    return result['counts']

# Compare Bell state across backends
print("🔍 Cross-Backend Comparison:")
backend_results = compare_backends(bell_state_test)

print("\n📊 Results Summary:")
for backend_name, counts in backend_results.items():
    if counts:
        prob_entangled = (counts.get('00', 0) + counts.get('11', 0)) / 1000
        print(f"{backend_name}: Entanglement fidelity = {prob_entangled:.3f}")
        print(f"  Results: {counts}")

print("\n💡 Note: This example uses gate-model quantum computing.")
print("   Ocean/D-Wave backends are quantum annealers for optimization problems")
print("   and use a completely different programming model (QUBO/Ising).")



def phase_kickback_demo():
    """Demonstrate quantum phase kickback."""

    print("🔄 Quantum Phase Kickback Demonstration")
    print("This shows how a controlled operation can affect the control qubit")

    circuit = backend.create_circuit(n_qubits=2)

    # Put control qubit in superposition
    circuit = backend.add_gate(circuit, 'H', 0)

    # Put target qubit in |1⟩ state (important for kickback!)
    circuit = backend.add_gate(circuit, 'X', 1)

    # Apply controlled-Z gate
    circuit = backend.add_gate(circuit, 'CZ', [0, 1])

    # Measure in X-basis to see phase difference
    circuit = backend.add_gate(circuit, 'H', 0)  # H†|±⟩ = |0⟩/|1⟩
    circuit = backend.add_measurement(circuit)

    result = backend.execute_circuit(circuit, shots=1000)
    counts = result['counts']

    print(f"Results: {counts}")

    # Compare with reference (no kickback)
    circuit_ref = backend.create_circuit(n_qubits=2)
    circuit_ref = backend.add_gate(circuit_ref, 'H', 0)
    circuit_ref = backend.add_gate(circuit_ref, 'X', 1)
    # Skip the CZ gate
    circuit_ref = backend.add_gate(circuit_ref, 'H', 0)
    circuit_ref = backend.add_measurement(circuit_ref)

    result_ref = backend.execute_circuit(circuit_ref, shots=1000)
    counts_ref = result_ref['counts']

    print(f"Reference (no CZ): {counts_ref}")
    print("The difference shows the phase kickback effect!")

    return counts, counts_ref

phase_kickback_demo()
