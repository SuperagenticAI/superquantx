# Cirq Backend Integration

Cirq is Google's open-source quantum computing framework that focuses on near-term quantum devices and NISQ algorithms. SuperQuantX integrates with Cirq to provide access to Google's quantum ecosystem and advanced circuit optimization features.

## Overview

The Cirq backend in SuperQuantX offers:

- **Google Quantum Hardware**: Access to Google's quantum processors
- **Advanced Circuit Optimization**: NISQ-optimized circuit compilation
- **Flexible Circuit Construction**: Pythonic circuit building with moments
- **Noise Modeling**: Comprehensive noise simulation capabilities
- **Custom Gates**: Easy creation of custom quantum operations

## Installation

```bash
# Install SuperQuantX with Cirq support
pip install superquantx[cirq]

# Or install Cirq directly
pip install cirq cirq-google cirq-web
```

## Quick Start

### Basic Usage

```python
import superquantx as sqx

# Get Cirq backend
backend = sqx.get_backend('cirq')
print(f"Backend: {backend}")

# Create Bell state circuit
circuit = backend.create_circuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

# Execute circuit
result = backend.run(circuit, shots=1000)
print(f"Bell state results: {result.get_counts()}")
```

### Google Quantum Hardware

```python
# Note: Requires Google Quantum AI access
# This is for demonstration - actual access requires approval

# Connect to Google quantum processor
try:
    import cirq_google
    
    # Get Google quantum engine service
    engine = cirq_google.get_engine()
    
    # Use Google processor
    backend = sqx.CirqBackend(
        processor_id='rainbow',  # Example processor
        shots=1000
    )
    
    # Execute on Google hardware
    result = backend.run(circuit, shots=1000)
    print(f"Google quantum result: {result.get_counts()}")
    
except Exception as e:
    print(f"Google Quantum access requires setup: {e}")
```

## Configuration Options

### Simulator Configuration

```python
# Different Cirq simulators
simulators = {
    'simulator': 'Basic state vector simulator',
    'density_matrix_simulator': 'Density matrix with noise support',
    'sparse_simulator': 'Sparse state vector simulator',
    'clifford_simulator': 'Efficient Clifford simulation',
}

# Create backend with specific simulator
backend = sqx.CirqBackend(
    simulator='density_matrix_simulator',
    shots=2048,
    seed=42
)

# Custom simulator configuration
backend = sqx.CirqBackend(
    simulator='simulator',
    shots=1000,
    noise_model=None,      # Custom noise model
    initial_state=None,    # Custom initial state
    qubit_order=None       # Custom qubit ordering
)
```

### Advanced Configuration

```python
import cirq
import numpy as np

# Custom qubit layout (Google Sycamore-like)
def create_sycamore_qubits(rows=5, cols=6):
    """Create Sycamore-like qubit layout."""
    qubits = []
    for row in range(rows):
        for col in range(cols):
            # Skip some qubits to match Sycamore pattern
            if (row % 2 == 0 and col % 2 == 1) or (row % 2 == 1 and col % 2 == 0):
                continue
            qubits.append(cirq.GridQubit(row, col))
    return qubits

# Create backend with custom qubits
custom_qubits = create_sycamore_qubits(3, 4)
backend = sqx.CirqBackend(
    qubits=custom_qubits,
    simulator='simulator',
    shots=1000
)

print(f"Custom qubit layout: {len(custom_qubits)} qubits")
```

## Working with Circuits

### Circuit Construction

```python
def demonstrate_cirq_circuits():
    """Show Cirq circuit construction features."""
    
    backend = sqx.get_backend('cirq')
    
    # Create circuit with 4 qubits
    circuit = backend.create_circuit(4)
    
    # Basic single-qubit gates
    circuit.h(0)           # Hadamard
    circuit.x(1)           # Pauli-X
    circuit.y(2)           # Pauli-Y
    circuit.z(3)           # Pauli-Z
    
    # Rotation gates
    circuit.rx(np.pi/4, 0) # X rotation
    circuit.ry(np.pi/3, 1) # Y rotation
    circuit.rz(np.pi/6, 2) # Z rotation
    
    # Phase gates
    circuit.s(0)           # S gate
    circuit.t(1)           # T gate
    
    # Two-qubit gates
    circuit.cx(0, 1)       # CNOT
    circuit.cz(1, 2)       # Controlled-Z
    circuit.swap(2, 3)     # SWAP
    
    # Controlled operations
    circuit.cz(0, 3)       # Controlled-Z
    
    # Add measurement
    circuit.measure_all()
    
    # Execute circuit
    result = backend.run(circuit, shots=1000)
    print(f"Cirq circuit results: {result.get_counts()}")
    
    # Display circuit structure (if available)
    try:
        print("Circuit structure:")
        print(circuit)
    except:
        print("Circuit visualization not available")

demonstrate_cirq_circuits()
```

### Moment-Based Circuit Construction

```python
def cirq_moment_construction():
    """Demonstrate Cirq's moment-based circuit construction."""
    
    import cirq
    
    backend = sqx.get_backend('cirq')
    
    # Create qubits
    qubits = [cirq.GridQubit(0, i) for i in range(4)]
    
    # Build circuit using Cirq's native moment system
    cirq_circuit = cirq.Circuit()
    
    # Moment 1: Initialize all qubits in superposition
    cirq_circuit.append([
        cirq.H(qubits[0]),
        cirq.H(qubits[1]),
        cirq.H(qubits[2]),
        cirq.H(qubits[3])
    ])
    
    # Moment 2: Entangling layer
    cirq_circuit.append([
        cirq.CZ(qubits[0], qubits[1]),
        cirq.CZ(qubits[2], qubits[3])
    ])
    
    # Moment 3: Second entangling layer
    cirq_circuit.append([
        cirq.CZ(qubits[1], qubits[2])
    ])
    
    # Moment 4: Parameterized rotations
    angles = np.random.random(4) * 2 * np.pi
    cirq_circuit.append([
        cirq.rz(angles[0])(qubits[0]),
        cirq.rz(angles[1])(qubits[1]),
        cirq.rz(angles[2])(qubits[2]),
        cirq.rz(angles[3])(qubits[3])
    ])
    
    # Add measurements
    cirq_circuit.append([
        cirq.measure(qubit, key=f'q{i}')
        for i, qubit in enumerate(qubits)
    ])
    
    print("Moment-based circuit:")
    print(f"Circuit depth: {len(cirq_circuit)}")
    print(f"Number of moments: {len(cirq_circuit.moments)}")
    
    # Execute using Cirq simulator directly
    simulator = cirq.Simulator()
    result = simulator.run(cirq_circuit, repetitions=1000)
    
    # Convert to SuperQuantX format
    counts = {}
    for i in range(1000):
        measurement = ''.join([
            str(result.measurements[f'q{j}'][i][0])
            for j in range(4)
        ])
        counts[measurement] = counts.get(measurement, 0) + 1
    
    print(f"Execution results: {dict(list(counts.items())[:5])}...")
    
    # Analyze circuit structure
    print(f"\nCircuit Analysis:")
    print(f"Total operations: {len(list(cirq_circuit.all_operations()))}")
    print(f"Two-qubit gates: {len([op for op in cirq_circuit.all_operations() if len(op.qubits) == 2])}")
    print(f"Single-qubit gates: {len([op for op in cirq_circuit.all_operations() if len(op.qubits) == 1])}")

cirq_moment_construction()
```

### Custom Gate Implementation

```python
def cirq_custom_gates():
    """Demonstrate custom gate creation in Cirq."""
    
    import cirq
    
    # Define custom single-qubit gate
    class CustomRotation(cirq.Gate):
        """Custom parameterized rotation gate."""
        
        def __init__(self, theta, phi):
            self.theta = theta
            self.phi = phi
        
        def _num_qubits_(self):
            return 1
        
        def _unitary_(self):
            c = np.cos(self.theta / 2)
            s = np.sin(self.theta / 2)
            return np.array([
                [c, -s * np.exp(1j * self.phi)],
                [s, c * np.exp(1j * self.phi)]
            ])
        
        def _circuit_diagram_info_(self, args):
            return f'CustomRot(θ={self.theta:.2f}, φ={self.phi:.2f})'
    
    # Define custom two-qubit gate
    class CustomEntangling(cirq.Gate):
        """Custom entangling gate."""
        
        def __init__(self, angle):
            self.angle = angle
        
        def _num_qubits_(self):
            return 2
        
        def _unitary_(self):
            # Parametric entangling gate
            c = np.cos(self.angle)
            s = np.sin(self.angle)
            return np.array([
                [1, 0, 0, 0],
                [0, c, 1j*s, 0],
                [0, 1j*s, c, 0],
                [0, 0, 0, 1]
            ])
        
        def _circuit_diagram_info_(self, args):
            return ['CustomEnt'] * 2
    
    # Use custom gates in circuit
    qubits = [cirq.GridQubit(0, i) for i in range(3)]
    circuit = cirq.Circuit()
    
    # Apply custom single-qubit gates
    circuit.append([
        CustomRotation(np.pi/4, np.pi/3)(qubits[0]),
        CustomRotation(np.pi/6, np.pi/4)(qubits[1]),
        CustomRotation(np.pi/8, np.pi/2)(qubits[2])
    ])
    
    # Apply custom entangling gates
    circuit.append([
        CustomEntangling(np.pi/5)(qubits[0], qubits[1]),
        CustomEntangling(np.pi/7)(qubits[1], qubits[2])
    ])
    
    # Add measurements
    circuit.append(cirq.measure_each(*qubits))
    
    print("Custom Gate Circuit:")
    print(circuit)
    
    # Execute circuit
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=1000)
    
    # Analyze results
    measurement_counts = {}
    for i in range(1000):
        measurement = ''.join([str(result.measurements[f'{qubit}'][i][0]) for qubit in qubits])
        measurement_counts[measurement] = measurement_counts.get(measurement, 0) + 1
    
    print(f"\nCustom gate circuit results:")
    for outcome, count in sorted(measurement_counts.items()):
        probability = count / 1000
        print(f"  |{outcome}⟩: {count:3d} ({probability:.3f})")

cirq_custom_gates()
```

## Google Quantum Hardware Integration

### Sycamore Processor Emulation

```python
def sycamore_emulation():
    """Emulate Google's Sycamore quantum processor."""
    
    import cirq
    import cirq_google
    
    # Create Sycamore-like device
    def create_sycamore_device():
        """Create a device mimicking Google's Sycamore."""
        # Simplified Sycamore layout (5x5 subset)
        qubits = []
        for row in range(5):
            for col in range(5):
                # Create checkerboard pattern like Sycamore
                if (row + col) % 2 == 0:
                    qubits.append(cirq.GridQubit(row, col))
        
        # Define connectivity (nearest neighbors only)
        pairs = []
        for qubit in qubits:
            row, col = qubit.row, qubit.col
            # Check all four directions
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = cirq.GridQubit(row + dr, col + dc)
                if neighbor in qubits and (qubit, neighbor) not in pairs and (neighbor, qubit) not in pairs:
                    pairs.append((qubit, neighbor))
        
        return qubits, pairs
    
    qubits, connectivity = create_sycamore_device()
    
    print(f"Sycamore emulation:")
    print(f"  Qubits: {len(qubits)}")
    print(f"  Connections: {len(connectivity)}")
    
    # Create quantum supremacy-inspired circuit
    def create_supremacy_circuit(depth=12):
        """Create a quantum supremacy-style random circuit."""
        circuit = cirq.Circuit()
        
        # Single-qubit gates (Sycamore gateset)
        single_gates = [cirq.X**0.5, cirq.Y**0.5, cirq.T]
        
        for layer in range(depth):
            # Single-qubit layer
            operations = []
            for qubit in qubits:
                gate = np.random.choice(single_gates)
                operations.append(gate(qubit))
            circuit.append(operations)
            
            # Two-qubit layer (use subset of connections)
            two_qubit_ops = []
            used_qubits = set()
            
            # Randomly select connections that don't conflict
            available_pairs = [pair for pair in connectivity 
                             if pair[0] not in used_qubits and pair[1] not in used_qubits]
            
            np.random.shuffle(available_pairs)
            
            for pair in available_pairs[:len(available_pairs)//2]:  # Use half the connections
                # Use fSim gate (approximating Google's hardware)
                theta = np.pi/2  # Typical fSim parameter
                phi = np.pi/6
                
                # Approximate fSim with available gates
                fsim_ops = [
                    cirq.CZ(*pair),
                    cirq.rz(phi)(pair[0]),
                    cirq.rz(phi)(pair[1])
                ]
                two_qubit_ops.extend(fsim_ops)
                
                used_qubits.update(pair)
            
            if two_qubit_ops:
                circuit.append(two_qubit_ops)
        
        # Add measurements
        circuit.append(cirq.measure_each(*qubits))
        
        return circuit
    
    # Generate and execute supremacy-style circuit
    supremacy_circuit = create_supremacy_circuit(depth=8)  # Reduced depth for demo
    
    print(f"\nQuantum Supremacy Circuit:")
    print(f"  Depth: {len(supremacy_circuit)}")
    print(f"  Operations: {len(list(supremacy_circuit.all_operations()))}")
    
    # Execute with limited shots due to complexity
    simulator = cirq.Simulator()
    result = simulator.run(supremacy_circuit, repetitions=100)  # Reduced shots
    
    # Analyze output distribution
    measurement_counts = {}
    for i in range(100):
        # Get measurement result
        measurement = ''.join([
            str(result.measurements[f'{qubit}'][i][0])
            for qubit in qubits
        ])
        measurement_counts[measurement] = measurement_counts.get(measurement, 0) + 1
    
    print(f"\nOutput Analysis:")
    print(f"  Unique outcomes: {len(measurement_counts)}")
    print(f"  Max probability: {max(measurement_counts.values())/100:.3f}")
    print(f"  Min probability: {min(measurement_counts.values())/100:.3f}")
    
    # Check for quantum supremacy signature (Porter-Thomas distribution)
    probabilities = np.array(list(measurement_counts.values())) / 100
    mean_prob = np.mean(probabilities)
    variance = np.var(probabilities)
    
    print(f"  Mean probability: {mean_prob:.6f}")
    print(f"  Variance: {variance:.6f}")
    print(f"  Expected variance (Porter-Thomas): {mean_prob**2:.6f}")

sycamore_emulation()
```

## Noise Modeling

### Comprehensive Noise Models

```python
def cirq_noise_modeling():
    """Demonstrate comprehensive noise modeling in Cirq."""
    
    import cirq
    
    # Create noise models
    def create_device_noise_model():
        """Create realistic device noise model."""
        
        # Single-qubit depolarizing noise
        single_qubit_error = cirq.depolarize(p=0.001)  # 0.1% error rate
        
        # Two-qubit depolarizing noise  
        two_qubit_error = cirq.depolarize(p=0.01)      # 1% error rate
        
        # Readout errors
        readout_error = cirq.BitFlipChannel(p=0.02)    # 2% readout error
        
        # Amplitude damping (T1 decay)
        t1_error = cirq.AmplitudeDampingChannel(gamma=0.001)  # T1 ~ 100μs, gate ~ 25ns
        
        # Phase damping (T2 dephasing)  
        t2_error = cirq.PhaseDampingChannel(gamma=0.0005)     # T2 ~ 70μs
        
        return {
            'single_qubit_gates': [single_qubit_error, t1_error, t2_error],
            'two_qubit_gates': [two_qubit_error, t1_error, t2_error],
            'readout': [readout_error]
        }
    
    noise_model = create_device_noise_model()
    
    # Create test circuit
    qubits = [cirq.GridQubit(0, i) for i in range(3)]
    circuit = cirq.Circuit()
    
    # Bell state preparation
    circuit.append([
        cirq.H(qubits[0]),
        cirq.CZ(qubits[0], qubits[1]),
        cirq.CZ(qubits[1], qubits[2])
    ])
    
    circuit.append(cirq.measure_each(*qubits))
    
    print("Noise Model Testing")
    print("=" * 20)
    
    # Test without noise
    noiseless_simulator = cirq.Simulator()
    noiseless_result = noiseless_simulator.run(circuit, repetitions=1000)
    
    noiseless_counts = {}
    for i in range(1000):
        measurement = ''.join([str(noiseless_result.measurements[f'{qubit}'][i][0]) for qubit in qubits])
        noiseless_counts[measurement] = noiseless_counts.get(measurement, 0) + 1
    
    print("Noiseless simulation:")
    for outcome, count in sorted(noiseless_counts.items()):
        print(f"  |{outcome}⟩: {count:3d} ({count/1000:.3f})")
    
    # Test with noise
    noisy_circuit = circuit.copy()
    
    # Add noise after each gate
    def add_noise_to_circuit(circuit, noise_model):
        noisy_circuit = cirq.Circuit()
        
        for moment in circuit:
            # Add original operations
            noisy_circuit.append(moment)
            
            # Add noise after each operation
            noise_ops = []
            for op in moment:
                if len(op.qubits) == 1:  # Single-qubit gate
                    for noise_channel in noise_model['single_qubit_gates']:
                        noise_ops.append(noise_channel.on(*op.qubits))
                elif len(op.qubits) == 2:  # Two-qubit gate
                    for noise_channel in noise_model['two_qubit_gates']:
                        for qubit in op.qubits:
                            noise_ops.append(noise_channel.on(qubit))
            
            if noise_ops:
                noisy_circuit.append(noise_ops)
        
        return noisy_circuit
    
    # Apply noise (simplified approach)
    noisy_simulator = cirq.DensityMatrixSimulator()
    
    # Create noisy circuit manually for demonstration
    noisy_test_circuit = cirq.Circuit()
    
    # Bell preparation with noise
    noisy_test_circuit.append(cirq.H(qubits[0]))
    noisy_test_circuit.append(cirq.depolarize(0.001).on(qubits[0]))
    
    noisy_test_circuit.append(cirq.CZ(qubits[0], qubits[1]))
    noisy_test_circuit.append([
        cirq.depolarize(0.01).on(qubits[0]),
        cirq.depolarize(0.01).on(qubits[1])
    ])
    
    noisy_test_circuit.append(cirq.CZ(qubits[1], qubits[2]))
    noisy_test_circuit.append([
        cirq.depolarize(0.01).on(qubits[1]),
        cirq.depolarize(0.01).on(qubits[2])
    ])
    
    noisy_test_circuit.append(cirq.measure_each(*qubits))
    
    noisy_result = noisy_simulator.run(noisy_test_circuit, repetitions=1000)
    
    noisy_counts = {}
    for i in range(1000):
        measurement = ''.join([str(noisy_result.measurements[f'{qubit}'][i][0]) for qubit in qubits])
        noisy_counts[measurement] = noisy_counts.get(measurement, 0) + 1
    
    print("\nNoisy simulation:")
    for outcome, count in sorted(noisy_counts.items()):
        print(f"  |{outcome}⟩: {count:3d} ({count/1000:.3f})")
    
    # Compare fidelities
    def calculate_fidelity(counts1, counts2, total_shots):
        """Calculate fidelity between two probability distributions."""
        all_outcomes = set(counts1.keys()) | set(counts2.keys())
        fidelity = 0
        for outcome in all_outcomes:
            p1 = counts1.get(outcome, 0) / total_shots
            p2 = counts2.get(outcome, 0) / total_shots
            fidelity += np.sqrt(p1 * p2)
        return fidelity
    
    fidelity = calculate_fidelity(noiseless_counts, noisy_counts, 1000)
    print(f"\nFidelity between noiseless and noisy: {fidelity:.3f}")
    
    # Analyze error types
    error_states = ['001', '010', '100', '011', '101', '110']  # Non-computational states
    total_errors = sum(noisy_counts.get(state, 0) for state in error_states)
    error_rate = total_errors / 1000
    
    print(f"Total error rate: {error_rate:.3f}")

cirq_noise_modeling()
```

## Advanced Features

### Circuit Optimization

```python
def cirq_circuit_optimization():
    """Demonstrate Cirq's circuit optimization capabilities."""
    
    import cirq
    
    print("Circuit Optimization with Cirq")
    print("=" * 32)
    
    # Create unoptimized circuit
    qubits = [cirq.GridQubit(0, i) for i in range(4)]
    unoptimized_circuit = cirq.Circuit()
    
    # Add redundant operations that can be optimized
    unoptimized_circuit.append([
        cirq.H(qubits[0]),
        cirq.X(qubits[0]),  # H-X can be optimized
        cirq.H(qubits[0]),  # H-X-H = Z-H
        
        cirq.Y(qubits[1]),
        cirq.Y(qubits[1]),  # Y-Y = I (cancels out)
        
        cirq.CZ(qubits[0], qubits[1]),
        cirq.CZ(qubits[0], qubits[1]),  # CZ-CZ = I
        
        cirq.SWAP(qubits[2], qubits[3]),
        cirq.SWAP(qubits[2], qubits[3]),  # SWAP-SWAP = I
    ])
    
    unoptimized_circuit.append(cirq.measure_each(*qubits))
    
    print("Original circuit:")
    print(f"  Moments: {len(unoptimized_circuit)}")
    print(f"  Operations: {len(list(unoptimized_circuit.all_operations()))}")
    
    # Apply various optimization strategies
    optimizers = [
        ("Merge Single-Qubit Gates", cirq.MergeSingleQubitGates()),
        ("Drop Empty Moments", cirq.DropEmptyMoments()),
        ("Drop Negligible Operations", cirq.DropNegligible()),
        ("Eject Z", cirq.EjectZ()),
        ("Eject PhasedPaulis", cirq.EjectPhasedPaulis()),
    ]
    
    current_circuit = unoptimized_circuit.copy()
    
    for name, optimizer in optimizers:
        try:
            optimizer.optimize_circuit(current_circuit)
            print(f"\nAfter {name}:")
            print(f"  Moments: {len(current_circuit)}")
            print(f"  Operations: {len(list(current_circuit.all_operations()))}")
        except Exception as e:
            print(f"  {name} failed: {e}")
    
    # Compare execution results
    simulator = cirq.Simulator()
    
    # Execute original circuit
    original_result = simulator.run(unoptimized_circuit, repetitions=1000)
    original_counts = {}
    for i in range(1000):
        measurement = ''.join([str(original_result.measurements[f'{qubit}'][i][0]) for qubit in qubits])
        original_counts[measurement] = original_counts.get(measurement, 0) + 1
    
    # Execute optimized circuit
    optimized_result = simulator.run(current_circuit, repetitions=1000)
    optimized_counts = {}
    for i in range(1000):
        measurement = ''.join([str(optimized_result.measurements[f'{qubit}'][i][0]) for qubit in qubits])
        optimized_counts[measurement] = optimized_counts.get(measurement, 0) + 1
    
    print(f"\nExecution Results Comparison:")
    print(f"Original circuit outcomes: {len(original_counts)}")
    print(f"Optimized circuit outcomes: {len(optimized_counts)}")
    
    # Check if results are equivalent
    outcomes_match = (set(original_counts.keys()) == set(optimized_counts.keys()))
    print(f"Outcomes match: {outcomes_match}")
    
    if outcomes_match:
        max_diff = max(abs(original_counts.get(outcome, 0) - optimized_counts.get(outcome, 0)) 
                      for outcome in original_counts.keys())
        print(f"Maximum count difference: {max_diff}")

cirq_circuit_optimization()
```

### Quantum Algorithms Implementation

```python
def cirq_quantum_algorithms():
    """Implement quantum algorithms using Cirq backend."""
    
    import cirq
    
    print("Quantum Algorithms with Cirq")
    print("=" * 30)
    
    # 1. Quantum Fourier Transform
    def qft_cirq(qubits):
        """Implement QFT using Cirq."""
        n = len(qubits)
        circuit = cirq.Circuit()
        
        for i in range(n):
            circuit.append(cirq.H(qubits[i]))
            
            for j in range(i + 1, n):
                # Controlled phase rotation
                angle = np.pi / (2 ** (j - i))
                circuit.append(cirq.CZ(qubits[i], qubits[j]) ** (angle / np.pi))
        
        # Reverse qubit order
        for i in range(n // 2):
            circuit.append(cirq.SWAP(qubits[i], qubits[n - 1 - i]))
        
        return circuit
    
    # Test QFT
    qubits = [cirq.GridQubit(0, i) for i in range(3)]
    qft_circuit = qft_cirq(qubits)
    
    # Prepare input state |110⟩
    test_circuit = cirq.Circuit()
    test_circuit.append([cirq.X(qubits[0]), cirq.X(qubits[1])])  # |110⟩
    test_circuit += qft_circuit
    test_circuit.append(cirq.measure_each(*qubits))
    
    simulator = cirq.Simulator()
    result = simulator.run(test_circuit, repetitions=1000)
    
    qft_counts = {}
    for i in range(1000):
        measurement = ''.join([str(result.measurements[f'{qubit}'][i][0]) for qubit in qubits])
        qft_counts[measurement] = qft_counts.get(measurement, 0) + 1
    
    print("QFT Results:")
    for outcome, count in sorted(qft_counts.items(), key=lambda x: -x[1]):
        print(f"  |{outcome}⟩: {count:3d} ({count/1000:.3f})")
    
    # 2. Grover's Algorithm (simplified)
    def grovers_oracle_cirq(qubits, target_state):
        """Create oracle for Grover's algorithm."""
        circuit = cirq.Circuit()
        
        # Flip qubits that should be 0 in target state
        for i, bit in enumerate(target_state):
            if bit == '0':
                circuit.append(cirq.X(qubits[i]))
        
        # Multi-controlled Z gate
        if len(qubits) == 2:
            circuit.append(cirq.CZ(qubits[0], qubits[1]))
        elif len(qubits) == 3:
            # Implement CCZ using decomposition
            circuit.append([
                cirq.H(qubits[2]),
                cirq.CCX(qubits[0], qubits[1], qubits[2]),
                cirq.H(qubits[2])
            ])
        
        # Flip back
        for i, bit in enumerate(target_state):
            if bit == '0':
                circuit.append(cirq.X(qubits[i]))
        
        return circuit
    
    def grovers_diffusion_cirq(qubits):
        """Create diffusion operator for Grover's algorithm."""
        circuit = cirq.Circuit()
        
        # H gates
        circuit.append([cirq.H(qubit) for qubit in qubits])
        
        # Apply oracle for |000...0⟩ state
        target_zero = '0' * len(qubits)
        oracle = grovers_oracle_cirq(qubits, target_zero)
        circuit += oracle
        
        # H gates
        circuit.append([cirq.H(qubit) for qubit in qubits])
        
        return circuit
    
    # Implement Grover's search for |11⟩ in 2-qubit space
    search_qubits = qubits[:2]
    target = '11'
    
    grovers_circuit = cirq.Circuit()
    
    # Initial superposition
    grovers_circuit.append([cirq.H(qubit) for qubit in search_qubits])
    
    # Optimal number of iterations for 2 qubits: π/4 * √4 ≈ 1
    for _ in range(1):
        # Apply oracle
        oracle = grovers_oracle_cirq(search_qubits, target)
        grovers_circuit += oracle
        
        # Apply diffusion
        diffusion = grovers_diffusion_cirq(search_qubits)
        grovers_circuit += diffusion
    
    grovers_circuit.append(cirq.measure_each(*search_qubits))
    
    grover_result = simulator.run(grovers_circuit, repetitions=1000)
    
    grover_counts = {}
    for i in range(1000):
        measurement = ''.join([str(grover_result.measurements[f'{qubit}'][i][0]) for qubit in search_qubits])
        grover_counts[measurement] = grover_counts.get(measurement, 0) + 1
    
    print(f"\nGrover's Search for |{target}⟩:")
    for outcome, count in sorted(grover_counts.items(), key=lambda x: -x[1]):
        print(f"  |{outcome}⟩: {count:3d} ({count/1000:.3f})")
    
    success_prob = grover_counts.get(target, 0) / 1000
    print(f"Success probability: {success_prob:.3f}")

cirq_quantum_algorithms()
```

## Performance Optimization

### Benchmarking and Profiling

```python
def cirq_performance_benchmarking():
    """Benchmark Cirq backend performance."""
    
    import time
    import cirq
    
    print("Cirq Performance Benchmarking")
    print("=" * 32)
    
    # Test different simulators
    simulators = {
        'Simulator': cirq.Simulator(),
        'DensityMatrixSimulator': cirq.DensityMatrixSimulator(),
        # 'SparseSimulator': cirq.SparseSimulator(),  # If available
    }
    
    # Test different circuit sizes
    circuit_sizes = [2, 4, 6, 8]  # Number of qubits
    
    results = {}
    
    for sim_name, simulator in simulators.items():
        print(f"\nTesting {sim_name}:")
        results[sim_name] = {}
        
        for n_qubits in circuit_sizes:
            # Create test circuit
            qubits = [cirq.GridQubit(0, i) for i in range(n_qubits)]
            circuit = cirq.Circuit()
            
            # Add parameterized layers
            depth = 3
            for layer in range(depth):
                # Single-qubit layer
                circuit.append([
                    cirq.ry(np.random.random() * 2 * np.pi)(qubit)
                    for qubit in qubits
                ])
                
                # Two-qubit layer
                for i in range(n_qubits - 1):
                    circuit.append(cirq.CZ(qubits[i], qubits[i + 1]))
            
            circuit.append(cirq.measure_each(*qubits))
            
            try:
                # Time execution
                start_time = time.time()
                result = simulator.run(circuit, repetitions=1000)
                execution_time = time.time() - start_time
                
                # Count unique outcomes
                measurement_counts = {}
                for i in range(1000):
                    measurement = ''.join([
                        str(result.measurements[f'{qubit}'][i][0])
                        for qubit in qubits
                    ])
                    measurement_counts[measurement] = measurement_counts.get(measurement, 0) + 1
                
                results[sim_name][n_qubits] = {
                    'time': execution_time,
                    'unique_outcomes': len(measurement_counts)
                }
                
                print(f"  {n_qubits} qubits: {execution_time:.3f}s "
                      f"({len(measurement_counts)} unique outcomes)")
                
            except Exception as e:
                print(f"  {n_qubits} qubits: Failed - {e}")
                results[sim_name][n_qubits] = {'error': str(e)}
    
    # Analyze scaling
    print(f"\nScaling Analysis:")
    for sim_name in results:
        print(f"{sim_name}:")
        times = [results[sim_name][n]['time'] 
                for n in circuit_sizes 
                if 'time' in results[sim_name].get(n, {})]
        
        if len(times) >= 2:
            # Estimate scaling (very rough)
            scaling_factor = times[-1] / times[0] if times[0] > 0 else float('inf')
            theoretical_scaling = 2**(circuit_sizes[-1] - circuit_sizes[0])  # Exponential
            
            print(f"  Empirical scaling factor: {scaling_factor:.2f}")
            print(f"  Theoretical (exponential): {theoretical_scaling:.2f}")
            print(f"  Scaling efficiency: {theoretical_scaling/scaling_factor:.2f}x")
    
    # Memory usage estimation
    print(f"\nMemory Usage Estimates:")
    for n_qubits in circuit_sizes:
        state_vector_memory = 2**n_qubits * 16  # Complex128 = 16 bytes
        density_matrix_memory = (2**n_qubits)**2 * 16
        
        print(f"  {n_qubits} qubits:")
        print(f"    State vector: {state_vector_memory/1024:.1f} KB")
        print(f"    Density matrix: {density_matrix_memory/(1024**2):.1f} MB")

cirq_performance_benchmarking()
```

## Integration Examples

### Quantum Machine Learning with Cirq

```python
def cirq_quantum_ml_example():
    """Implement quantum ML using Cirq backend."""
    
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    import cirq
    
    print("Quantum ML with Cirq Backend")
    print("=" * 30)
    
    # Generate dataset
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features, 2 classes")
    
    # Create variational quantum classifier
    class CirqVariationalClassifier:
        def __init__(self, n_qubits=2, n_layers=2):
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            self.qubits = [cirq.GridQubit(0, i) for i in range(n_qubits)]
            self.n_params = n_qubits * n_layers * 2  # RY and RZ per qubit per layer
            self.params = np.random.random(self.n_params) * 2 * np.pi
        
        def create_ansatz(self, x_data, params):
            """Create variational ansatz circuit."""
            circuit = cirq.Circuit()
            
            # Data encoding
            for i, xi in enumerate(x_data[:self.n_qubits]):
                circuit.append(cirq.ry(np.pi * xi)(self.qubits[i]))
            
            # Variational layers
            param_idx = 0
            for layer in range(self.n_layers):
                # Parameterized rotations
                for i in range(self.n_qubits):
                    circuit.append(cirq.ry(params[param_idx])(self.qubits[i]))
                    param_idx += 1
                    circuit.append(cirq.rz(params[param_idx])(self.qubits[i]))
                    param_idx += 1
                
                # Entangling gates
                if layer < self.n_layers - 1:
                    for i in range(self.n_qubits - 1):
                        circuit.append(cirq.CZ(self.qubits[i], self.qubits[i + 1]))
            
            return circuit
        
        def measure_expectation(self, x_data, params):
            """Measure expectation value for classification."""
            circuit = self.create_ansatz(x_data, params)
            
            # Add measurement
            circuit.append(cirq.measure(self.qubits[0], key='measurement'))
            
            simulator = cirq.Simulator()
            result = simulator.run(circuit, repetitions=1000)
            
            # Calculate expectation value of first qubit
            measurements = result.measurements['measurement']
            expectation = np.mean(1 - 2 * measurements)  # Convert {0,1} to {1,-1}
            
            return expectation
        
        def predict_single(self, x_data, params=None):
            """Predict single sample."""
            if params is None:
                params = self.params
            
            expectation = self.measure_expectation(x_data, params)
            return 1 if expectation > 0 else 0
        
        def cost_function(self, params, X, y):
            """Cost function for training."""
            predictions = []
            for xi in X:
                pred = self.predict_single(xi, params)
                predictions.append(pred)
            
            # Binary cross-entropy loss
            predictions = np.array(predictions, dtype=float)
            y = np.array(y, dtype=float)
            
            # Add small epsilon to prevent log(0)
            epsilon = 1e-10
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            
            loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            return loss
        
        def train(self, X, y, epochs=30, learning_rate=0.1):
            """Train using parameter-shift rule."""
            print(f"Training VQC for {epochs} epochs...")
            
            cost_history = []
            
            for epoch in range(epochs):
                # Compute gradients using parameter-shift rule
                gradients = np.zeros_like(self.params)
                
                for i in range(len(self.params)):
                    # Forward shift
                    params_plus = self.params.copy()
                    params_plus[i] += np.pi / 2
                    cost_plus = self.cost_function(params_plus, X, y)
                    
                    # Backward shift
                    params_minus = self.params.copy()
                    params_minus[i] -= np.pi / 2
                    cost_minus = self.cost_function(params_minus, X, y)
                    
                    # Parameter-shift gradient
                    gradients[i] = (cost_plus - cost_minus) / 2
                
                # Update parameters
                self.params = self.params - learning_rate * gradients
                
                # Record cost
                current_cost = self.cost_function(self.params, X, y)
                cost_history.append(current_cost)
                
                if epoch % 10 == 0:
                    print(f"  Epoch {epoch}: Cost = {current_cost:.4f}")
            
            return cost_history
        
        def predict(self, X):
            """Predict multiple samples."""
            predictions = []
            for xi in X:
                pred = self.predict_single(xi)
                predictions.append(pred)
            return np.array(predictions)
    
    # Create and train classifier
    vqc = CirqVariationalClassifier(n_qubits=2, n_layers=2)
    cost_history = vqc.train(X_train_scaled, y_train, epochs=25, learning_rate=0.3)
    
    # Make predictions
    train_pred = vqc.predict(X_train_scaled)
    test_pred = vqc.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\nResults:")
    print(f"Training Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy: {test_acc:.3f}")
    
    # Compare with classical ML
    from sklearn.svm import SVC
    classical_svm = SVC(kernel='rbf')
    classical_svm.fit(X_train_scaled, y_train)
    classical_pred = classical_svm.predict(X_test_scaled)
    classical_acc = accuracy_score(y_test, classical_pred)
    
    print(f"Classical SVM Accuracy: {classical_acc:.3f}")
    
    # Performance comparison
    if test_acc > classical_acc:
        print("✅ Quantum classifier outperforms classical!")
    elif test_acc > classical_acc - 0.05:
        print("🤔 Quantum classifier is competitive")
    else:
        print("📈 Classical method currently better")

cirq_quantum_ml_example()
```

## Best Practices

### Code Organization

```python
# cirq_utils.py
import cirq
import numpy as np

class CirqUtils:
    """Utility functions for Cirq backend operations."""
    
    @staticmethod
    def create_grid_qubits(rows, cols):
        """Create grid qubit layout."""
        return [cirq.GridQubit(r, c) for r in range(rows) for c in range(cols)]
    
    @staticmethod
    def create_line_qubits(n):
        """Create line qubit layout."""
        return [cirq.LineQubit(i) for i in range(n)]
    
    @staticmethod
    def create_connectivity_map(qubits, topology='linear'):
        """Create qubit connectivity map."""
        connections = []
        
        if topology == 'linear':
            for i in range(len(qubits) - 1):
                connections.append((qubits[i], qubits[i + 1]))
        
        elif topology == 'circular':
            for i in range(len(qubits)):
                connections.append((qubits[i], qubits[(i + 1) % len(qubits)]))
        
        elif topology == 'grid':
            # Assumes qubits are GridQubits
            for qubit in qubits:
                if isinstance(qubit, cirq.GridQubit):
                    neighbors = [
                        cirq.GridQubit(qubit.row + dr, qubit.col + dc)
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                    ]
                    for neighbor in neighbors:
                        if neighbor in qubits and (qubit, neighbor) not in connections:
                            connections.append((qubit, neighbor))
        
        return connections
    
    @staticmethod
    def optimize_circuit_for_device(circuit, device_connectivity):
        """Optimize circuit for specific device connectivity."""
        # Apply basic optimizations
        optimizers = [
            cirq.MergeSingleQubitGates(),
            cirq.DropEmptyMoments(),
            cirq.DropNegligible()
        ]
        
        optimized_circuit = circuit.copy()
        for optimizer in optimizers:
            optimizer.optimize_circuit(optimized_circuit)
        
        return optimized_circuit
```

### Error Handling

```python
def robust_cirq_execution(circuit, simulator_type='Simulator', max_retries=3):
    """Execute Cirq circuit with error handling and fallbacks."""
    
    simulators = {
        'Simulator': cirq.Simulator,
        'DensityMatrixSimulator': cirq.DensityMatrixSimulator,
        'SparseSimulator': getattr(cirq, 'SparseSimulator', cirq.Simulator)
    }
    
    for attempt in range(max_retries):
        try:
            simulator_class = simulators.get(simulator_type, cirq.Simulator)
            simulator = simulator_class()
            
            result = simulator.run(circuit, repetitions=1000)
            return result
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with {simulator_type}: {e}")
            
            if attempt == max_retries - 1:
                # Final fallback to basic simulator
                if simulator_type != 'Simulator':
                    print("Falling back to basic Simulator...")
                    return robust_cirq_execution(circuit, 'Simulator', 1)
                else:
                    raise e
            
            # Try simpler simulator on next attempt
            if simulator_type == 'DensityMatrixSimulator':
                simulator_type = 'Simulator'
```

## Resources

### Documentation
- **Cirq Documentation**: https://quantumai.google/cirq
- **Cirq Tutorials**: https://quantumai.google/cirq/tutorials
- **Google Quantum AI**: https://quantumai.google/

### Hardware Access
- **Google Quantum AI**: Apply for access to quantum processors
- **Quantum AI Campus Program**: Academic partnerships
- **Cirq Community**: Open-source development and support

---

*Cirq's integration with SuperQuantX brings Google's quantum computing expertise to your applications. With its focus on NISQ algorithms and flexible circuit construction, Cirq is ideal for near-term quantum applications and research.*

!!! tip "Circuit Optimization"
    Use Cirq's moment structure and optimization passes to create efficient circuits. The `MergeSingleQubitGates` optimizer can significantly reduce circuit depth.

!!! warning "Memory Scaling"
    Density matrix simulation scales as O(4^n) in memory. For circuits with >10 qubits, prefer state vector simulation when possible.

!!! info "Custom Gates"
    Cirq makes it easy to create custom gates and operations. This is particularly useful for implementing domain-specific quantum algorithms.