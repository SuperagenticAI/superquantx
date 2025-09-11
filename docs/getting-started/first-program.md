# Your First Quantum Program

Welcome to your first quantum program with SuperQuantX! This comprehensive guide will walk you through building quantum applications from the ground up, explaining each concept along the way.

## 🎯 What You'll Learn

By the end of this tutorial, you'll understand:

- **Quantum Basics**: Qubits, superposition, and entanglement
- **Circuit Building**: Creating and manipulating quantum circuits
- **Quantum Algorithms**: Implementing basic quantum algorithms
- **Backend Integration**: Using different quantum computing frameworks
- **Result Analysis**: Interpreting quantum measurement results

## 🧪 Prerequisites

Make sure you have SuperQuantX installed:

```bash
pip install superquantx[pennylane]
```

## 🚀 Hello Quantum World

Let's start with the quantum equivalent of "Hello, World!" - creating a superposition state:

### Step 1: Setting Up Your Environment

```python
import superquantx as sqx
import numpy as np
import matplotlib.pyplot as plt

# Verify your installation
print(f"SuperQuantX version: {sqx.__version__}")
print(f"Available backends: {sqx.list_backends()}")
```

### Step 2: Understanding Qubits

Unlike classical bits that can only be 0 or 1, **qubits** can exist in a superposition of both states simultaneously.

```python
# Get a quantum backend
backend = sqx.get_backend('simulator')

# Create a circuit with one qubit
circuit = backend.create_circuit(1)

# Initially, the qubit is in state |0⟩
print("Initial state: |0⟩")

# Apply a Hadamard gate to create superposition
circuit.h(0)  # Now the qubit is in state (|0⟩ + |1⟩)/√2

# Measure the qubit
circuit.measure_all()

# Run the circuit multiple times
result = backend.run(circuit, shots=1000)
counts = result.get_counts()

print(f"Measurement results: {counts}")
print("🎉 You've created quantum superposition!")
```

**Expected output:**
```
Measurement results: {'0': 503, '1': 497}
```

Notice how we get roughly equal counts of 0 and 1! This is quantum superposition in action.

### Step 3: Understanding the Results

The Hadamard gate (H) puts a qubit in an equal superposition:

- **Before H gate**: Qubit is definitely in state |0⟩
- **After H gate**: Qubit is in state (|0⟩ + |1⟩)/√2
- **After measurement**: We get 0 or 1 with equal probability

## 🔗 Quantum Entanglement

Now let's create something more exotic - quantum entanglement between two qubits:

### Creating the Bell State

The Bell state is a famous quantum entangled state where two qubits are perfectly correlated:

```python
# Create a circuit with 2 qubits
circuit = backend.create_circuit(2)

# Step 1: Put first qubit in superposition
circuit.h(0)

# Step 2: Entangle the qubits with CNOT gate
circuit.cx(0, 1)  # Controlled-X gate (CNOT)

# Step 3: Measure both qubits
circuit.measure_all()

# Run the circuit
result = backend.run(circuit, shots=1000)
counts = result.get_counts()

print(f"Bell state results: {counts}")

# Visualize the circuit
print("\nCircuit diagram:")
circuit.draw()
```

**Expected output:**
```
Bell state results: {'00': 496, '11': 504}
```

🤔 **Notice something interesting?** We only get `00` and `11` - never `01` or `10`! This is quantum entanglement - the qubits are perfectly correlated.

### Understanding Entanglement

```python
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
```

## 🎲 Random Number Generator

Let's build a quantum random number generator:

```python
def quantum_random_number(num_bits=8):
    """Generate a random number using quantum superposition."""
    
    # Create circuit with specified number of qubits
    circuit = backend.create_circuit(num_bits)
    
    # Put all qubits in superposition
    for i in range(num_bits):
        circuit.h(i)
    
    # Measure all qubits
    circuit.measure_all()
    
    # Run the circuit once
    result = backend.run(circuit, shots=1)
    
    # Convert result to integer
    binary_result = list(result.get_counts().keys())[0]
    random_number = int(binary_result, 2)
    
    return random_number, binary_result

# Generate some quantum random numbers
print("🎲 Quantum Random Numbers:")
for i in range(5):
    number, binary = quantum_random_number(8)  # 8-bit numbers (0-255)
    print(f"  {number:3d} (binary: {binary})")
```

## 🧮 Quantum Interference

Let's explore quantum interference with a more complex example:

```python
def quantum_interference_demo():
    """Demonstrate quantum interference patterns."""
    
    circuit = backend.create_circuit(1)
    
    # Create superposition
    circuit.h(0)
    
    # Add a phase (rotation around Z-axis)
    circuit.rz(np.pi/4, 0)  # 45-degree phase
    
    # Apply another Hadamard - this creates interference
    circuit.h(0)
    
    circuit.measure_all()
    
    # Run multiple times to see the pattern
    result = backend.run(circuit, shots=1000)
    counts = result.get_counts()
    
    return counts

# Test different phases
phases = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
results = []

print("🌊 Quantum Interference Patterns:")
print("Phase\t|0⟩ Count\t|1⟩ Count")
print("-" * 35)

for phase in phases:
    circuit = backend.create_circuit(1)
    circuit.h(0)
    circuit.rz(phase, 0)
    circuit.h(0)
    circuit.measure_all()
    
    result = backend.run(circuit, shots=1000)
    counts = result.get_counts()
    
    count_0 = counts.get('0', 0)
    count_1 = counts.get('1', 0)
    
    print(f"{phase:.2f}\t{count_0}\t\t{count_1}")
    results.append((phase, count_0, count_1))
```

## 🎯 Building a Quantum Coin Flipper

Let's create a biased quantum coin that we can control:

```python
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
        theta = 2 * np.arccos(np.sqrt(bias))
        
        circuit = self.backend.create_circuit(1)
        
        # Start in |0⟩ (tails)
        # Rotate to achieve desired bias
        circuit.ry(theta, 0)
        
        circuit.measure_all()
        
        result = self.backend.run(circuit, shots=shots)
        counts = result.get_counts()
        
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
```

## 📊 Visualizing Your Results

Let's add some visualization to better understand our quantum programs:

```python
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
    plt.show()

# Example: Visualize Bell state results
circuit = backend.create_circuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

result = backend.run(circuit, shots=1000)
counts = result.get_counts()

plot_quantum_results(counts, "Bell State |Φ⁺⟩ = (|00⟩ + |11⟩)/√2")
```

## 🔧 Working with Different Backends

SuperQuantX's power lies in backend flexibility. Let's compare the same algorithm across different backends:

```python
def compare_backends(algorithm_func, *args, **kwargs):
    """Run the same algorithm on different backends."""
    
    available_backends = sqx.list_backends()
    results = {}
    
    for backend_name in available_backends:
        try:
            print(f"🔄 Testing {backend_name}...")
            result = algorithm_func(backend_name, *args, **kwargs)
            results[backend_name] = result
            print(f"✅ {backend_name} completed successfully")
            
        except Exception as e:
            print(f"❌ {backend_name} failed: {e}")
            results[backend_name] = None
    
    return results

def bell_state_test(backend_name):
    """Create Bell state on specified backend."""
    backend = sqx.get_backend(backend_name)
    circuit = backend.create_circuit(2)
    
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    
    result = backend.run(circuit, shots=1000)
    return result.get_counts()

# Compare Bell state across backends
print("🔍 Cross-Backend Comparison:")
backend_results = compare_backends(bell_state_test)

for backend_name, counts in backend_results.items():
    if counts:
        prob_entangled = (counts.get('00', 0) + counts.get('11', 0)) / 1000
        print(f"{backend_name}: Entanglement fidelity = {prob_entangled:.3f}")
```

## 🎓 Algorithm: Quantum Phase Kickback

Let's implement a more advanced concept - quantum phase kickback:

```python
def phase_kickback_demo():
    """Demonstrate quantum phase kickback."""
    
    print("🔄 Quantum Phase Kickback Demonstration")
    print("This shows how a controlled operation can affect the control qubit")
    
    circuit = backend.create_circuit(2)
    
    # Put control qubit in superposition
    circuit.h(0)
    
    # Put target qubit in |1⟩ state (important for kickback!)
    circuit.x(1)
    
    # Apply controlled-Z gate
    circuit.cz(0, 1)
    
    # Measure in X-basis to see phase difference
    circuit.h(0)  # H†|±⟩ = |0⟩/|1⟩
    circuit.measure_all()
    
    result = backend.run(circuit, shots=1000)
    counts = result.get_counts()
    
    print(f"Results: {counts}")
    
    # Compare with reference (no kickback)
    circuit_ref = backend.create_circuit(2)
    circuit_ref.h(0)
    circuit_ref.x(1)
    # Skip the CZ gate
    circuit_ref.h(0)
    circuit_ref.measure_all()
    
    result_ref = backend.run(circuit_ref, shots=1000)
    counts_ref = result_ref.get_counts()
    
    print(f"Reference (no CZ): {counts_ref}")
    print("The difference shows the phase kickback effect!")
    
    return counts, counts_ref

phase_kickback_demo()
```

## 🎯 Next Steps: Your Learning Path

Congratulations! You've built your first quantum programs. Here's what to explore next:

### Immediate Next Steps

1. **[Configuration Guide](configuration.md)**: Customize SuperQuantX for your needs
2. **[Basic Quantum Tutorial](../tutorials/basic-quantum.md)**: Deep dive into quantum concepts
3. **[Quantum Algorithms](../user-guide/algorithms.md)**: Explore built-in algorithms

### Intermediate Topics

4. **[Quantum Machine Learning](../tutorials/quantum-ml.md)**: Apply QML algorithms
5. **[Multi-Backend Usage](../tutorials/multi-backend.md)**: Compare frameworks
6. **[Circuit Optimization](../user-guide/circuits.md)**: Efficient circuit design

### Advanced Projects

7. **Build a Quantum Game**: Create quantum versions of classical games
8. **Quantum Cryptography**: Implement quantum key distribution
9. **Variational Algorithms**: Explore VQE and QAOA

## 🧪 Practice Challenges

Try these challenges to reinforce your learning:

### Challenge 1: Quantum Dice
Create a quantum 6-sided die that's perfectly fair:

```python
# Hint: Use multiple qubits and post-process results
def quantum_dice():
    # Your code here
    pass
```

### Challenge 2: Quantum Walk
Implement a simple quantum random walk:

```python
def quantum_walk(steps=10):
    # Use superposition to move left and right simultaneously
    pass
```

### Challenge 3: Grover's Algorithm
Try implementing a simple version of Grover's search:

```python
def mini_grover_search(target_item):
    # Search for an item in a small database
    pass
```

## 📞 Getting Help

If you get stuck or have questions:

- **[FAQ](../help/faq.md)**: Common questions and answers
- **[Troubleshooting](../help/troubleshooting.md)**: Solve technical issues
- **[GitHub Issues](https://github.com/SuperagenticAI/superquantx/issues)**: Report bugs or ask questions
- **Email**: [research@super-agentic.ai](mailto:research@super-agentic.ai)

## 📚 Additional Resources

- **[Quantum Computing Primer](../tutorials/basic-quantum.md)**: Understand the physics
- **[Algorithm Library](../user-guide/algorithms.md)**: Ready-to-use quantum algorithms
- **[Backend Comparison](../user-guide/backends.md)**: Choose the right framework

---

!!! success "Congratulations! 🎉"
    You've successfully created your first quantum programs with SuperQuantX! You now understand superposition, entanglement, and quantum measurement. Keep experimenting and exploring the quantum world!

!!! tip "Pro Tip"
    The best way to learn quantum computing is through hands-on experimentation. Try modifying the examples above and see what happens!