# Python Simulator Backend

The Python Simulator backend provides a pure Python implementation of quantum circuit simulation that doesn't require external quantum computing libraries. This backend is ideal for testing, educational purposes, and as a fallback when other quantum backends are unavailable.

## Overview

The Simulator backend in SuperQuantX offers:

- **Pure Python Implementation**: No external quantum computing dependencies
- **Educational Value**: Clear, readable quantum simulation code
- **Rapid Development**: Fast prototyping without backend setup
- **Statevector Access**: Direct access to quantum state amplitudes
- **Feature Maps**: Built-in quantum machine learning encoding circuits
- **Variational Ansätze**: Common parameterized circuit templates
- **Gradient Support**: Numerical differentiation for optimization

!!! note "Performance Considerations"
    This backend is optimized for clarity and simplicity rather than performance. For larger quantum circuits, consider using specialized quantum simulators.

## Installation

The Simulator backend is included with SuperQuantX and requires no additional dependencies:

```bash
# Install SuperQuantX (simulator included)
pip install superquantx
```

### Verify Installation

```python
import superquantx as sqx

# Initialize simulator backend
backend = sqx.get_backend('simulator')
print(f"Simulator backend loaded: {backend.is_available()}")
print(f"Max qubits: {backend.max_qubits}")
```

## Quick Start

### Basic Quantum Circuit

```python
import superquantx as sqx
import numpy as np

# Initialize simulator backend
backend = sqx.get_backend('simulator', max_qubits=5, shots=1024)

# Create a simple Bell state circuit
circuit = backend.create_circuit(2)

# Add quantum gates
circuit = backend.add_gate(circuit, 'H', 0)  # Hadamard on qubit 0
circuit = backend.add_gate(circuit, 'CX', [0, 1])  # CNOT gate

# Execute the circuit
result = backend.execute_circuit(circuit, shots=1000)
print("Bell state results:", result['counts'])

# Expected output: roughly equal probabilities for '00' and '11'
```

### Parameterized Circuit

```python
# Create circuit with rotation gates
circuit = backend.create_circuit(3)

# Add parameterized gates
circuit = backend.add_gate(circuit, 'RY', 0, [np.pi/4])  # π/4 rotation
circuit = backend.add_gate(circuit, 'RX', 1, [np.pi/3])  # π/3 rotation
circuit = backend.add_gate(circuit, 'RZ', 2, [np.pi/2])  # π/2 rotation

# Add entangling gates
circuit = backend.add_gate(circuit, 'CX', [0, 1])
circuit = backend.add_gate(circuit, 'CX', [1, 2])

# Get statevector
statevector = backend.get_statevector(circuit)
print(f"Statevector shape: {statevector.shape}")
print(f"Statevector norm: {np.linalg.norm(statevector):.6f}")
```

## Configuration

### Backend Initialization

```python
# Basic configuration
backend = sqx.SimulatorBackend(
    device='simulator',
    max_qubits=10,     # Maximum qubits to simulate
    shots=1024         # Default measurement shots
)

# High-precision simulation
backend_precise = sqx.SimulatorBackend(
    max_qubits=8,
    shots=10000  # More shots for better statistics
)

# Development/testing configuration
backend_dev = sqx.SimulatorBackend(
    max_qubits=4,
    shots=100   # Fast execution for testing
)
```

### Simulator Capabilities

```python
# Check simulator capabilities
capabilities = backend.capabilities

print("Simulator capabilities:")
for capability, supported in capabilities.items():
    print(f"  {capability}: {supported}")

# Output includes:
# supports_gradient: True
# supports_parameter_shift: True  
# supports_finite_diff: True
# supports_backprop: False (not implemented)
# supports_hardware: False
# supports_noise_models: False
```

## Gate Operations

### Single-Qubit Gates

```python
circuit = backend.create_circuit(3)

# Pauli gates
circuit = backend.add_gate(circuit, 'X', 0)  # Pauli-X (NOT)
circuit = backend.add_gate(circuit, 'Y', 1)  # Pauli-Y
circuit = backend.add_gate(circuit, 'Z', 2)  # Pauli-Z

# Hadamard gate
circuit = backend.add_gate(circuit, 'H', 0)

# Phase gates
circuit = backend.add_gate(circuit, 'S', 1)  # S gate (√Z)
circuit = backend.add_gate(circuit, 'T', 2)  # T gate (⁴√Z)

# Rotation gates
circuit = backend.add_gate(circuit, 'RX', 0, [np.pi/2])  # X-rotation
circuit = backend.add_gate(circuit, 'RY', 1, [np.pi/3])  # Y-rotation  
circuit = backend.add_gate(circuit, 'RZ', 2, [np.pi/4])  # Z-rotation

# Available single-qubit gates
available_gates = backend.gates.keys()
print("Available gates:", list(available_gates))
```

### Two-Qubit Gates

```python
circuit = backend.create_circuit(4)

# CNOT gate (Controlled-X)
circuit = backend.add_gate(circuit, 'CNOT', [0, 1])
circuit = backend.add_gate(circuit, 'CX', [0, 1])  # Same as CNOT

# Controlled-Z gate
circuit = backend.add_gate(circuit, 'CZ', [1, 2])

# SWAP gate
circuit = backend.add_gate(circuit, 'SWAP', [2, 3])

# Execute and measure
result = backend.execute_circuit(circuit)
print("Two-qubit circuit results:", result['counts'])
```

### Multi-Qubit Gates

```python
# Toffoli gate (CCX - simplified implementation)
circuit = backend.create_circuit(4)
circuit = backend.add_gate(circuit, 'H', 0)
circuit = backend.add_gate(circuit, 'H', 1)  
circuit = backend.add_gate(circuit, 'CCX', [0, 1, 2])  # Control qubits 0,1, target 2

# Note: Toffoli implementation is simplified in the current backend
print("Toffoli circuit executed (simplified implementation)")
```

## Advanced Features

### Statevector Analysis

```python
def analyze_quantum_state(backend, circuit):
    """Analyze quantum state properties."""
    
    # Get statevector
    statevector = backend.get_statevector(circuit)
    n_qubits = backend._get_n_qubits(circuit)
    
    # Calculate probabilities
    probabilities = np.abs(statevector)**2
    
    # Find dominant basis states
    dominant_indices = np.where(probabilities > 0.01)[0]
    
    print(f"Quantum State Analysis ({n_qubits} qubits):")
    print(f"Statevector dimension: {len(statevector)}")
    print(f"Normalization: {np.linalg.norm(statevector):.6f}")
    print(f"Entropy: {-np.sum(probabilities * np.log2(probabilities + 1e-16)):.4f}")
    
    print("\nDominant basis states:")
    for idx in dominant_indices:
        basis_state = format(idx, f'0{n_qubits}b')
        amplitude = statevector[idx]
        probability = probabilities[idx]
        print(f"|{basis_state}⟩: amplitude = {amplitude:.4f}, probability = {probability:.4f}")
    
    return {
        'statevector': statevector,
        'probabilities': probabilities,
        'entropy': -np.sum(probabilities * np.log2(probabilities + 1e-16))
    }

# Example: Analyze GHZ state
circuit = backend.create_circuit(3)
circuit = backend.add_gate(circuit, 'H', 0)
circuit = backend.add_gate(circuit, 'CX', [0, 1])
circuit = backend.add_gate(circuit, 'CX', [1, 2])

analysis = analyze_quantum_state(backend, circuit)
```

### Quantum Feature Maps

```python
def demonstrate_feature_maps():
    """Demonstrate quantum feature maps for machine learning."""
    
    # Generate sample data
    X = np.random.random((5, 3))  # 5 samples, 3 features
    
    # Create ZZ feature map
    zz_map = backend.create_feature_map(
        n_features=3, 
        feature_map='ZZFeatureMap', 
        reps=2
    )
    
    # Apply feature map to first data point
    encoded_circuit = zz_map(X[0])
    statevector = backend.get_statevector(encoded_circuit)
    
    print("ZZ Feature Map Encoding:")
    print(f"Input data: {X[0]}")
    print(f"Encoded statevector norm: {np.linalg.norm(statevector):.6f}")
    
    # Pauli feature map
    pauli_map = backend.create_feature_map(
        n_features=3,
        feature_map='PauliFeatureMap',
        reps=1
    )
    
    pauli_circuit = pauli_map(X[0])
    pauli_statevector = backend.get_statevector(pauli_circuit)
    
    print(f"Pauli feature map statevector norm: {np.linalg.norm(pauli_statevector):.6f}")

demonstrate_feature_maps()
```

### Quantum Kernel Computation

```python
def compute_quantum_kernel_example():
    """Compute quantum kernel matrix."""
    
    # Sample training data
    X_train = np.array([
        [0.1, 0.2],
        [0.4, 0.8], 
        [0.7, 0.3],
        [0.9, 0.6]
    ])
    
    # Sample test data
    X_test = np.array([
        [0.2, 0.3],
        [0.5, 0.7]
    ])
    
    # Compute kernel matrix
    kernel_matrix = backend.compute_kernel_matrix(
        X_train, X_test, 
        feature_map='ZZFeatureMap',
        shots=1000
    )
    
    print("Quantum Kernel Matrix:")
    print(kernel_matrix)
    print(f"Matrix shape: {kernel_matrix.shape}")
    
    return kernel_matrix

kernel_matrix = compute_quantum_kernel_example()
```

### Variational Ansätze

```python
def demonstrate_ansatz_circuits():
    """Demonstrate different ansatz circuits for VQE/QAOA."""
    
    n_qubits = 4
    
    # Real Amplitudes ansatz
    params_ra = np.random.random(8)  # 2 layers × 4 qubits
    ra_ansatz = backend.create_ansatz('RealAmplitudes', n_qubits, params_ra)
    ra_circuit = ra_ansatz()
    
    print("Real Amplitudes Ansatz:")
    print(f"Parameters: {params_ra}")
    print(f"Circuit qubits: {backend._get_n_qubits(ra_circuit)}")
    
    # EfficientSU2 ansatz  
    params_su2 = np.random.random(12)  # 1 layer × 4 qubits × 3 params
    su2_ansatz = backend.create_ansatz('EfficientSU2', n_qubits, params_su2)
    su2_circuit = su2_ansatz()
    
    print("\nEfficientSU2 Ansatz:")
    print(f"Parameters: {params_su2}")
    print(f"Circuit qubits: {backend._get_n_qubits(su2_circuit)}")
    
    # Compare statevectors
    ra_state = backend.get_statevector(ra_circuit)
    su2_state = backend.get_statevector(su2_circuit)
    
    print(f"\nRA statevector entropy: {-np.sum(np.abs(ra_state)**2 * np.log2(np.abs(ra_state)**2 + 1e-16)):.4f}")
    print(f"SU2 statevector entropy: {-np.sum(np.abs(su2_state)**2 * np.log2(np.abs(su2_state)**2 + 1e-16)):.4f}")

demonstrate_ansatz_circuits()
```

### Expectation Value Calculations

```python
def hamiltonian_expectation_example():
    """Calculate expectation values with different Hamiltonians."""
    
    # Create test circuit (Bell state)
    circuit = backend.create_circuit(2)
    circuit = backend.add_gate(circuit, 'H', 0)
    circuit = backend.add_gate(circuit, 'CX', [0, 1])
    
    # Simple Hamiltonian: σ_z ⊗ I
    H_z = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0], 
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ])
    
    expectation_z = backend.compute_expectation(circuit, H_z)
    print(f"⟨ψ|σ_z ⊗ I|ψ⟩ = {expectation_z:.6f}")
    
    # Hamiltonian: σ_x ⊗ σ_x
    H_xx = np.array([
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0], 
        [1, 0, 0, 0]
    ])
    
    expectation_xx = backend.compute_expectation(circuit, H_xx)
    print(f"⟨ψ|σ_x ⊗ σ_x|ψ⟩ = {expectation_xx:.6f}")
    
    # For Bell state |00⟩ + |11⟩, expect:
    # ⟨σ_z ⊗ I⟩ = 0 (equal superposition)  
    # ⟨σ_x ⊗ σ_x⟩ = 1 (maximally correlated)
    
    return expectation_z, expectation_xx

exp_z, exp_xx = hamiltonian_expectation_example()
```

## Quantum Machine Learning Examples

### VQE Implementation

```python
def simple_vqe_h2():
    """Simple VQE implementation for H2 molecule."""
    
    # H2 Hamiltonian (simplified)
    # H = -1.0523 I + 0.3979 Z_0 - 0.3979 Z_1 - 0.0113 Z_0*Z_1 + 0.1809 X_0*X_1
    
    def h2_energy(params):
        """Calculate H2 energy for given parameters."""
        # Create ansatz circuit
        ansatz = backend.create_ansatz('RealAmplitudes', 2, params)
        circuit = ansatz()
        
        # Compute expectation values for each Pauli term
        statevector = backend.get_statevector(circuit)
        
        # Identity term
        energy = -1.0523
        
        # Z_0 term (σ_z ⊗ I)
        H_z0 = np.diag([1, -1, 1, -1])
        energy += 0.3979 * backend.compute_expectation(circuit, H_z0)
        
        # Z_1 term (I ⊗ σ_z)  
        H_z1 = np.diag([1, 1, -1, -1])
        energy -= 0.3979 * backend.compute_expectation(circuit, H_z1)
        
        # Z_0*Z_1 term
        H_zz = np.diag([1, -1, -1, 1])
        energy -= 0.0113 * backend.compute_expectation(circuit, H_zz)
        
        # X_0*X_1 term
        H_xx = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
        energy += 0.1809 * backend.compute_expectation(circuit, H_xx)
        
        return energy
    
    # Optimize parameters
    from scipy.optimize import minimize
    
    initial_params = np.random.random(4) * 2 * np.pi
    result = minimize(h2_energy, initial_params, method='COBYLA')
    
    print("VQE H2 Results:")
    print(f"Optimal energy: {result.fun:.6f} Ha")
    print(f"Optimal parameters: {result.x}")
    print(f"Expected ground state: ≈ -1.137 Ha")
    
    return result

# Run VQE example
vqe_result = simple_vqe_h2()
```

### QAOA Implementation

```python
def simple_qaoa_maxcut():
    """Simple QAOA for Max-Cut problem."""
    
    # Define graph edges (triangle + additional edge)
    edges = [(0, 1), (1, 2), (2, 0), (0, 3)]
    n_nodes = 4
    
    def qaoa_circuit(gamma, beta, p_layers=1):
        """Create QAOA circuit."""
        circuit = backend.create_circuit(n_nodes)
        
        # Initial superposition
        for i in range(n_nodes):
            circuit = backend.add_gate(circuit, 'H', i)
        
        # QAOA layers
        for layer in range(p_layers):
            # Cost Hamiltonian (ZZ interactions)
            for i, j in edges:
                circuit = backend.add_gate(circuit, 'CZ', [i, j])
                circuit = backend.add_gate(circuit, 'RZ', j, [2 * gamma[layer]])
            
            # Mixer Hamiltonian (X rotations)
            for i in range(n_nodes):
                circuit = backend.add_gate(circuit, 'RX', i, [2 * beta[layer]])
        
        return circuit
    
    def qaoa_cost(params):
        """QAOA cost function."""
        p_layers = len(params) // 2
        gamma = params[:p_layers]
        beta = params[p_layers:]
        
        circuit = qaoa_circuit(gamma, beta, p_layers)
        result = backend.execute_circuit(circuit, shots=1000)
        
        # Calculate cut value expectation
        total_cut = 0
        total_counts = sum(result['counts'].values())
        
        for bitstring, count in result['counts'].items():
            # Count edges crossing the cut
            cut_value = 0
            for i, j in edges:
                if bitstring[i] != bitstring[j]:
                    cut_value += 1
            
            total_cut += (count / total_counts) * cut_value
        
        return -total_cut  # Minimize negative cut (maximize cut)
    
    # Optimize QAOA
    from scipy.optimize import minimize
    
    p_layers = 2
    initial_params = np.random.random(2 * p_layers) * np.pi
    result = minimize(qaoa_cost, initial_params, method='COBYLA')
    
    print("QAOA Max-Cut Results:")
    print(f"Optimal cut value: {-result.fun:.4f}")
    print(f"Optimal parameters: {result.x}")
    print(f"Maximum possible cut: {len(edges)} edges")
    
    return result

# Run QAOA example  
qaoa_result = simple_qaoa_maxcut()
```

### Quantum Neural Network

```python
def simple_qnn_classifier():
    """Simple quantum neural network for binary classification."""
    
    # Generate synthetic 2D classification data
    np.random.seed(42)
    n_samples = 20
    X = np.random.random((n_samples, 2)) * 2 - 1  # [-1, 1] range
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple linear separation
    
    def qnn_circuit(x, params):
        """Quantum neural network circuit."""
        # Data encoding
        feature_map = backend.create_feature_map(2, 'ZZFeatureMap', reps=1)
        circuit = feature_map(x)
        
        # Variational layers
        ansatz = backend.create_ansatz('EfficientSU2', 2, params)
        variational_circuit = ansatz()
        
        # Combine encoding and variational parts (simplified)
        # In practice, you'd need to compose the circuits properly
        return circuit
    
    def qnn_predict(X_batch, params):
        """Make predictions with QNN."""
        predictions = []
        
        for x in X_batch:
            circuit = qnn_circuit(x, params)
            
            # Measure expectation value of Z_0 for classification
            statevector = backend.get_statevector(circuit)
            
            # Simple measurement probability
            prob_0 = np.abs(statevector[0])**2 + np.abs(statevector[2])**2
            prediction = 1 if prob_0 > 0.5 else 0
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def qnn_loss(params):
        """QNN training loss."""
        predictions = qnn_predict(X, params)
        
        # Binary cross-entropy loss (simplified)
        loss = np.mean((predictions - y)**2)
        return loss
    
    # Train QNN
    initial_params = np.random.random(6) * 2 * np.pi  # 6 parameters
    
    print("Training simple QNN...")
    print(f"Initial accuracy: {np.mean(qnn_predict(X, initial_params) == y):.4f}")
    
    # Note: For real training, you'd use proper gradient descent
    # This is a simplified demonstration
    
    return initial_params

# Run QNN example
qnn_params = simple_qnn_classifier()
```

## Performance and Limitations

### Performance Characteristics

```python
def performance_analysis():
    """Analyze simulator performance."""
    import time
    
    qubit_counts = [2, 4, 6, 8]
    execution_times = []
    
    print("Performance Analysis:")
    print("Qubits | Time (ms) | State Dimension")
    print("-------|-----------|----------------")
    
    for n_qubits in qubit_counts:
        if n_qubits <= backend.max_qubits:
            # Create test circuit
            start_time = time.time()
            
            circuit = backend.create_circuit(n_qubits)
            
            # Add gates
            for i in range(n_qubits):
                circuit = backend.add_gate(circuit, 'H', i)
            
            for i in range(n_qubits - 1):
                circuit = backend.add_gate(circuit, 'CX', [i, i + 1])
            
            # Execute
            result = backend.execute_circuit(circuit, shots=100)
            
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000
            
            execution_times.append(execution_time)
            state_dim = 2**n_qubits
            
            print(f"   {n_qubits}   |   {execution_time:.2f}   |    {state_dim}")
    
    print(f"\nSimulator limits:")
    print(f"Max qubits: {backend.max_qubits}")
    print(f"Max state dimension: {2**backend.max_qubits:,}")

performance_analysis()
```

### Memory Usage Estimation

```python
def memory_usage_analysis():
    """Estimate memory usage for different qubit counts."""
    
    print("Memory Usage Estimates:")
    print("Qubits | State Vector | Memory (MB)")
    print("-------|--------------|------------")
    
    for n_qubits in range(1, 16):
        state_dim = 2**n_qubits
        # Complex128: 16 bytes per complex number
        memory_bytes = state_dim * 16
        memory_mb = memory_bytes / (1024**2)
        
        if memory_mb < 1000:  # Only show reasonable sizes
            print(f"   {n_qubits:2d}  |  {state_dim:8,} |   {memory_mb:8.2f}")
        else:
            print(f"   {n_qubits:2d}  |  {state_dim:8,} |   {memory_mb:8.0f}")
            
        if n_qubits >= 20:  # Stop at very large sizes
            break

memory_usage_analysis()
```

## Error Handling and Debugging

### Comprehensive Error Handling

```python
def robust_simulation_example():
    """Demonstrate robust simulation with error handling."""
    
    import logging
    
    # Setup logging for debugging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    def safe_circuit_execution(backend, circuit_builder, max_retries=3):
        """Execute circuit with error handling."""
        
        for attempt in range(max_retries):
            try:
                # Build circuit
                circuit = circuit_builder()
                
                # Validate circuit
                n_qubits = backend._get_n_qubits(circuit)
                if n_qubits > backend.max_qubits:
                    raise ValueError(f"Circuit has {n_qubits} qubits, max is {backend.max_qubits}")
                
                # Execute circuit
                result = backend.execute_circuit(circuit)
                
                # Validate result
                if 'counts' not in result:
                    raise RuntimeError("Invalid execution result")
                
                logger.info(f"Circuit execution successful on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries - 1:
                    logger.error("All retry attempts failed")
                    return None
        
        return None
    
    # Example circuit builder that might fail
    def problematic_circuit():
        circuit = backend.create_circuit(3)
        
        # Add gates that might cause issues
        circuit = backend.add_gate(circuit, 'H', 0)
        circuit = backend.add_gate(circuit, 'UNKNOWN_GATE', 1)  # Will cause warning
        circuit = backend.add_gate(circuit, 'RX', 2, [np.pi/4])
        
        return circuit
    
    # Test robust execution
    result = safe_circuit_execution(backend, problematic_circuit)
    
    if result:
        print("Robust execution successful:")
        print(f"Results: {result['counts']}")
    else:
        print("Robust execution failed after all retries")

robust_simulation_example()
```

### Circuit Validation

```python
def validate_circuit_construction():
    """Validate circuit construction steps."""
    
    def validate_gate_application(backend, gate, qubits, params=None):
        """Validate individual gate application."""
        
        validation = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check qubit indices
        if isinstance(qubits, int):
            qubits = [qubits]
        
        max_qubit = max(qubits) if qubits else -1
        if max_qubit >= backend.max_qubits:
            validation['errors'].append(f"Qubit index {max_qubit} exceeds max {backend.max_qubits-1}")
            validation['valid'] = False
        
        # Check gate existence
        if gate.upper() not in backend.gates and not gate.upper().startswith('R'):
            validation['warnings'].append(f"Gate '{gate}' may not be recognized")
        
        # Check parameter requirements
        rotation_gates = ['RX', 'RY', 'RZ']
        if gate.upper() in rotation_gates:
            if not params or len(params) == 0:
                validation['errors'].append(f"Rotation gate '{gate}' requires angle parameter")
                validation['valid'] = False
        
        return validation
    
    # Test various gate validations
    test_cases = [
        ('H', [0], None),
        ('RX', [1], [np.pi/2]),
        ('RY', [2], []),  # Missing parameter - should warn
        ('CX', [0, 1], None),
        ('UNKNOWN', [0], None),  # Unknown gate
        ('H', [15], None),  # Invalid qubit index
    ]
    
    print("Gate Validation Results:")
    for gate, qubits, params in test_cases:
        validation = validate_gate_application(backend, gate, qubits, params)
        
        status = "✓" if validation['valid'] else "✗"
        print(f"{status} {gate} on {qubits} with params {params}")
        
        for warning in validation['warnings']:
            print(f"    ⚠ {warning}")
        for error in validation['errors']:
            print(f"    ✗ {error}")

validate_circuit_construction()
```

## Best Practices

### Simulation Guidelines

1. **Qubit Limits**: Keep circuits under 12 qubits for reasonable performance
2. **Gate Decomposition**: Use basic gates (H, X, Y, Z, CX) when possible
3. **Statevector Access**: Use statevector operations for debugging
4. **Parameter Optimization**: Use numerical gradients for parameter optimization

### Code Organization

```python
# Example: Well-structured quantum algorithm
class QuantumAlgorithmSimulator:
    """Template for quantum algorithm implementation."""
    
    def __init__(self, backend, n_qubits):
        self.backend = backend
        self.n_qubits = n_qubits
        self.circuit = None
    
    def initialize_circuit(self):
        """Initialize quantum circuit."""
        self.circuit = self.backend.create_circuit(self.n_qubits)
        return self.circuit
    
    def add_initialization(self):
        """Add state initialization."""
        for i in range(self.n_qubits):
            self.circuit = self.backend.add_gate(self.circuit, 'H', i)
    
    def add_algorithm_gates(self, parameters):
        """Add algorithm-specific gates."""
        # Override in subclasses
        pass
    
    def execute(self, parameters=None, shots=1000):
        """Execute the quantum algorithm."""
        self.initialize_circuit()
        self.add_initialization()
        
        if parameters:
            self.add_algorithm_gates(parameters)
        
        return self.backend.execute_circuit(self.circuit, shots=shots)
    
    def get_statevector(self):
        """Get current statevector."""
        if self.circuit is None:
            raise ValueError("Circuit not initialized")
        return self.backend.get_statevector(self.circuit)

# Example usage
algorithm = QuantumAlgorithmSimulator(backend, n_qubits=3)
result = algorithm.execute(shots=1000)
print("Algorithm result:", result['counts'])
```

## Troubleshooting

### Common Issues

**Circuit Size Too Large**
```python
# Check qubit limits
if n_qubits > backend.max_qubits:
    print(f"Reducing circuit size from {n_qubits} to {backend.max_qubits} qubits")
    n_qubits = backend.max_qubits
```

**Unknown Gate Error**
```python
# Check available gates
available_gates = list(backend.gates.keys())
print("Available gates:", available_gates)

# Use alternative gates
if 'CNOT' not in available_gates and 'CX' in available_gates:
    circuit = backend.add_gate(circuit, 'CX', [0, 1])  # Use CX instead of CNOT
```

**Memory Issues**
```python
# Reduce circuit size or use shot-based simulation
if n_qubits > 10:
    print("Large circuit detected - consider using shot-based approaches")
    shots = 10000  # More shots for better statistics with smaller circuits
```

**Numerical Precision**
```python
# Check statevector normalization
statevector = backend.get_statevector(circuit)
norm = np.linalg.norm(statevector)
if abs(norm - 1.0) > 1e-10:
    print(f"Warning: Statevector norm = {norm:.12f} (should be 1.0)")
```

### Performance Optimization

**Optimize for Small Circuits**
```python
# Use statevector operations for analysis
def efficient_small_circuit_analysis(circuit):
    """Efficient analysis for small circuits."""
    
    # Direct statevector computation
    statevector = backend.get_statevector(circuit)
    
    # Calculate all probabilities at once
    probabilities = np.abs(statevector)**2
    
    # Find most likely outcomes
    top_indices = np.argsort(probabilities)[-5:]  # Top 5
    
    n_qubits = backend._get_n_qubits(circuit)
    results = {}
    
    for idx in top_indices:
        bitstring = format(idx, f'0{n_qubits}b')
        results[bitstring] = float(probabilities[idx])
    
    return results

# Use for small circuits instead of shot-based simulation
small_circuit = backend.create_circuit(3)
small_circuit = backend.add_gate(small_circuit, 'H', 0)
small_circuit = backend.add_gate(small_circuit, 'CX', [0, 1])

efficient_results = efficient_small_circuit_analysis(small_circuit)
print("Efficient analysis:", efficient_results)
```

## API Reference

The Simulator backend provides these key methods:

- `create_circuit(n_qubits)`: Create quantum circuit
- `add_gate(circuit, gate, qubits, params)`: Add quantum gate  
- `execute_circuit(circuit, shots)`: Execute circuit with measurements
- `get_statevector(circuit)`: Get quantum statevector
- `create_feature_map(n_features, type, reps)`: Create quantum feature map
- `create_ansatz(type, n_qubits, params)`: Create parameterized ansatz
- `compute_expectation(circuit, hamiltonian)`: Calculate expectation values
- `compute_kernel_matrix(X1, X2, feature_map)`: Compute quantum kernel matrix

### Gate Library

Available quantum gates:
- Single-qubit: `I`, `X`, `Y`, `Z`, `H`, `S`, `T`
- Rotations: `RX`, `RY`, `RZ` (require angle parameter)
- Two-qubit: `CNOT`, `CX`, `CZ`, `SWAP`
- Multi-qubit: `CCX` (Toffoli, simplified implementation)

For complete API documentation, see the [API Reference](../api/backends.md#simulator-backend) section.

---

*The Python Simulator backend is ideal for learning quantum computing concepts, prototyping algorithms, and testing quantum code without external dependencies.*