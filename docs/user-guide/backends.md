# Quantum Backends

SuperQuantX's core strength lies in its ability to provide a unified interface across multiple quantum computing frameworks. This guide explains how to work with different backends, compare their capabilities, and choose the right one for your research.

## 🎯 Backend Overview

A **backend** in SuperQuantX is an abstraction layer that connects your quantum algorithms to specific quantum computing frameworks or hardware. Think of it as a translator that converts your high-level SuperQuantX code into the native operations of each framework.

```python
import superquantx as sqx

# Same algorithm, different backends
for backend_name in ['simulator', 'pennylane', 'qiskit', 'cirq']:
    backend = sqx.get_backend(backend_name)
    circuit = backend.create_circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    result = backend.run(circuit, shots=1000)
    print(f"{backend_name}: {result.get_counts()}")
```

## 🔍 Available Backends

### Built-in Simulator Backend

The simulator backend is SuperQuantX's native quantum simulator, optimized for educational purposes and rapid prototyping.

```python
# Get the simulator backend
backend = sqx.get_backend('simulator')

# Check capabilities
caps = backend.get_capabilities()
print(f"Max qubits: {caps.max_qubits}")
print(f"Supports noise: {caps.has_noise}")
print(f"Native gates: {caps.native_gates}")
```

**Characteristics:**
- ✅ Always available (no additional dependencies)
- ✅ Fast for small circuits (< 20 qubits)
- ✅ Perfect for learning and prototyping
- ❌ Limited scalability
- ❌ No real quantum hardware access

### PennyLane Backend

PennyLane integration provides excellent quantum machine learning capabilities with automatic differentiation.

```python
# PennyLane backend with specific device
backend = sqx.get_backend('pennylane', device='default.qubit')

# Or use lightning for speed
fast_backend = sqx.get_backend('pennylane', device='lightning.qubit')
```

**Characteristics:**
- ✅ Excellent for quantum machine learning
- ✅ Automatic differentiation support
- ✅ Multiple device options (CPU, GPU, hardware)
- ✅ Strong optimization capabilities
- ❌ Learning curve for PennyLane-specific concepts

### Qiskit Backend

IBM's Qiskit framework integration provides access to IBM quantum computers and advanced simulators.

```python
# Local Qiskit backend
backend = sqx.get_backend('qiskit', provider='local')

# IBM Quantum backend (requires setup)
# backend = sqx.get_backend('qiskit', provider='IBMQ')
```

**Characteristics:**
- ✅ Access to IBM quantum hardware
- ✅ Mature ecosystem and tools
- ✅ Advanced noise modeling
- ✅ Strong community support
- ❌ Complex setup for hardware access
- ❌ Can be slower for simulations

### Cirq Backend

Google's Cirq framework, optimized for NISQ (Noisy Intermediate-Scale Quantum) devices.

```python
# Cirq backend
backend = sqx.get_backend('cirq')

# With specific simulator
backend = sqx.get_backend('cirq', simulator='SparseSimulator')
```

**Characteristics:**
- ✅ Optimized for NISQ algorithms
- ✅ Good for quantum simulation
- ✅ Access to Google quantum hardware (with setup)
- ✅ Flexible circuit construction
- ❌ Google-centric ecosystem
- ❌ Limited quantum ML support

## 🚀 Working with Backends

### Basic Backend Operations

```python
import superquantx as sqx

# List all available backends
available = sqx.list_backends()
print(f"Available backends: {available}")

# Get a specific backend
backend = sqx.get_backend('pennylane')

# Backend information
print(f"Backend name: {backend.name}")
print(f"Backend type: {backend.backend_type}")
print(f"Version: {backend.version}")

# Check if backend is available
if backend.is_available():
    print("✅ Backend is ready to use")
else:
    print("❌ Backend is not available")
```

### Backend Configuration

```python
# Configure backend with specific options
backend = sqx.get_backend(
    'pennylane',
    device='lightning.qubit',
    shots=5000,
    seed=42
)

# Runtime configuration changes
backend.configure(
    optimization_level=2,
    noise_model=None,
    memory_limit='4GB'
)
```

### Backend Context Manager

```python
# Temporary backend switching
with sqx.backend_context('qiskit'):
    # All operations in this block use Qiskit
    circuit = sqx.create_circuit(2)
    circuit.h(0)
    result = sqx.run(circuit)

# Back to default backend outside context
```

## 🔄 Backend Comparison

### Performance Comparison

```python
import time
import numpy as np

def benchmark_backends(circuit_depth=10, num_qubits=4, shots=1000):
    """Compare backend performance."""
    
    backends = ['simulator', 'pennylane', 'qiskit']
    results = {}
    
    for backend_name in backends:
        try:
            backend = sqx.get_backend(backend_name)
            
            # Create test circuit
            circuit = backend.create_circuit(num_qubits)
            
            # Add random gates
            np.random.seed(42)
            for _ in range(circuit_depth):
                qubit = np.random.randint(0, num_qubits)
                gate = np.random.choice(['h', 'x', 'y', 'z'])
                getattr(circuit, gate)(qubit)
            
            # Add entangling gates
            for i in range(num_qubits - 1):
                circuit.cx(i, i + 1)
            
            circuit.measure_all()
            
            # Time execution
            start_time = time.time()
            result = backend.run(circuit, shots=shots)
            end_time = time.time()
            
            results[backend_name] = {
                'execution_time': end_time - start_time,
                'counts': result.get_counts(),
                'success': True
            }
            
        except Exception as e:
            results[backend_name] = {
                'error': str(e),
                'success': False
            }
    
    return results

# Run benchmark
benchmark_results = benchmark_backends()

print("🏃 Backend Performance Comparison:")
for backend, result in benchmark_results.items():
    if result['success']:
        print(f"{backend:12}: {result['execution_time']:.3f} seconds")
    else:
        print(f"{backend:12}: ❌ {result['error']}")
```

### Feature Comparison

```python
def compare_backend_features():
    """Compare capabilities across backends."""
    
    backends = sqx.list_backends()
    
    features = {
        'max_qubits': [],
        'has_noise': [],
        'supports_gpu': [],
        'native_gates': [],
        'measurement_types': []
    }
    
    comparison_data = []
    
    for backend_name in backends:
        try:
            backend = sqx.get_backend(backend_name)
            caps = backend.get_capabilities()
            
            row = {
                'Backend': backend_name,
                'Max Qubits': caps.max_qubits,
                'Noise Support': '✅' if caps.has_noise else '❌',
                'GPU Support': '✅' if caps.supports_gpu else '❌',
                'Gate Count': len(caps.native_gates),
                'Hardware Access': '✅' if caps.hardware_access else '❌'
            }
            comparison_data.append(row)
            
        except Exception as e:
            print(f"⚠️ Could not analyze {backend_name}: {e}")
    
    # Display comparison table
    if comparison_data:
        import pandas as pd
        df = pd.DataFrame(comparison_data)
        print("🔍 Backend Feature Comparison:")
        print(df.to_string(index=False))
    
    return comparison_data

# Run feature comparison
feature_comparison = compare_backend_features()
```

## 🎛️ Advanced Backend Usage

### Custom Backend Selection

```python
def smart_backend_selection(requirements):
    """Automatically select best backend based on requirements."""
    
    available_backends = sqx.list_backends()
    
    # Score backends based on requirements
    scores = {}
    
    for backend_name in available_backends:
        backend = sqx.get_backend(backend_name)
        caps = backend.get_capabilities()
        score = 0
        
        # Qubit requirement
        if caps.max_qubits >= requirements.get('min_qubits', 1):
            score += 10
        
        # GPU requirement
        if requirements.get('gpu_preferred', False) and caps.supports_gpu:
            score += 5
        
        # Noise requirement
        if requirements.get('noise_modeling', False) and caps.has_noise:
            score += 5
        
        # Speed preference
        if backend_name == 'simulator':  # Fastest for small circuits
            score += 3
        elif backend_name == 'pennylane' and 'lightning' in str(caps.device):
            score += 4
        
        scores[backend_name] = score
    
    # Return best backend
    best_backend = max(scores, key=scores.get)
    return best_backend, scores

# Example usage
requirements = {
    'min_qubits': 8,
    'gpu_preferred': True,
    'noise_modeling': False
}

best_backend, all_scores = smart_backend_selection(requirements)
print(f"🎯 Best backend for your requirements: {best_backend}")
print(f"📊 Backend scores: {all_scores}")
```

### Backend Pooling

```python
class BackendPool:
    """Manage multiple backends for parallel execution."""
    
    def __init__(self, backend_names=None):
        if backend_names is None:
            backend_names = sqx.list_backends()
        
        self.backends = {}
        for name in backend_names:
            try:
                self.backends[name] = sqx.get_backend(name)
            except Exception as e:
                print(f"⚠️ Could not load {name}: {e}")
    
    def run_parallel(self, circuit, shots=1000):
        """Run the same circuit on multiple backends in parallel."""
        
        import concurrent.futures
        
        def run_on_backend(backend_name):
            backend = self.backends[backend_name]
            return backend.run(circuit, shots=shots)
        
        results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit jobs to all backends
            futures = {
                executor.submit(run_on_backend, name): name 
                for name in self.backends.keys()
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                backend_name = futures[future]
                try:
                    result = future.result()
                    results[backend_name] = result.get_counts()
                except Exception as e:
                    results[backend_name] = f"Error: {e}"
        
        return results

# Example usage
pool = BackendPool(['simulator', 'pennylane'])

# Create test circuit
circuit = sqx.create_circuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

# Run on all backends
parallel_results = pool.run_parallel(circuit)
print("🔄 Parallel execution results:")
for backend, counts in parallel_results.items():
    print(f"{backend}: {counts}")
```

## 🔧 Backend-Specific Features

### PennyLane-Specific Features

```python
# Automatic differentiation with PennyLane
backend = sqx.get_backend('pennylane')

# Create parameterized circuit
circuit = backend.create_circuit(2)
theta = backend.create_parameter('theta')
phi = backend.create_parameter('phi')

circuit.ry(theta, 0)
circuit.rz(phi, 1)
circuit.cx(0, 1)

# Define cost function
def cost_function(params):
    return backend.expectation_value(circuit, observable='Z', params=params)

# Compute gradients
import jax
grad_fn = jax.grad(cost_function)
gradients = grad_fn([0.5, 0.8])
print(f"Gradients: {gradients}")
```

### Qiskit-Specific Features

```python
# Access Qiskit's transpiler
backend = sqx.get_backend('qiskit')

circuit = backend.create_circuit(3)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(1, 2)

# Transpile for specific hardware
transpiled = backend.transpile(
    circuit,
    optimization_level=3,
    target_device='ibm_manila'
)

print(f"Original depth: {circuit.depth()}")
print(f"Transpiled depth: {transpiled.depth()}")
```

### Cirq-Specific Features

```python
# Access Cirq's noise models
backend = sqx.get_backend('cirq')

# Add noise model
noise_model = backend.create_noise_model(
    depolarizing_prob=0.01,
    damping_prob=0.02
)

backend.set_noise_model(noise_model)

# Run noisy simulation
circuit = backend.create_circuit(2)
circuit.h(0)
circuit.cx(0, 1)
noisy_result = backend.run(circuit, shots=1000)
```

## 📊 Backend Monitoring

### Performance Monitoring

```python
class BackendMonitor:
    """Monitor backend performance and health."""
    
    def __init__(self):
        self.metrics = {}
    
    def health_check(self, backend_name):
        """Check if backend is healthy."""
        try:
            backend = sqx.get_backend(backend_name)
            
            # Simple test circuit
            circuit = backend.create_circuit(2)
            circuit.h(0)
            circuit.cx(0, 1)
            circuit.measure_all()
            
            # Run test
            start_time = time.time()
            result = backend.run(circuit, shots=100)
            elapsed = time.time() - start_time
            
            # Check results make sense
            counts = result.get_counts()
            total_counts = sum(counts.values())
            
            if total_counts != 100:
                return False, f"Expected 100 shots, got {total_counts}"
            
            if elapsed > 30:  # 30 second timeout
                return False, f"Too slow: {elapsed:.2f}s"
            
            return True, f"Healthy (response time: {elapsed:.3f}s)"
            
        except Exception as e:
            return False, f"Error: {e}"
    
    def monitor_all_backends(self):
        """Monitor all available backends."""
        backends = sqx.list_backends()
        status = {}
        
        for backend_name in backends:
            healthy, message = self.health_check(backend_name)
            status[backend_name] = {
                'healthy': healthy,
                'message': message,
                'timestamp': time.time()
            }
        
        return status

# Example usage
monitor = BackendMonitor()
status_report = monitor.monitor_all_backends()

print("🏥 Backend Health Report:")
for backend, status in status_report.items():
    icon = "✅" if status['healthy'] else "❌"
    print(f"{icon} {backend}: {status['message']}")
```

## 🔀 Backend Migration

### Algorithm Migration

```python
def migrate_algorithm_to_backend(algorithm_func, from_backend, to_backend):
    """Migrate an algorithm from one backend to another."""
    
    print(f"🔄 Migrating from {from_backend} to {to_backend}")
    
    # Test with original backend
    original_result = algorithm_func(from_backend)
    
    # Test with new backend
    new_result = algorithm_func(to_backend)
    
    # Compare results
    def compare_results(result1, result2, tolerance=0.1):
        """Compare quantum measurement results."""
        if not result1 or not result2:
            return False
        
        # Compare state probabilities
        all_states = set(result1.keys()) | set(result2.keys())
        
        for state in all_states:
            prob1 = result1.get(state, 0) / sum(result1.values())
            prob2 = result2.get(state, 0) / sum(result2.values())
            
            if abs(prob1 - prob2) > tolerance:
                return False
        
        return True
    
    is_compatible = compare_results(original_result, new_result)
    
    print(f"📊 Original result: {original_result}")
    print(f"📊 New result: {new_result}")
    print(f"✅ Compatible: {is_compatible}")
    
    return is_compatible

# Example: Migrate Bell state creation
def bell_state_algorithm(backend_name):
    backend = sqx.get_backend(backend_name)
    circuit = backend.create_circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    result = backend.run(circuit, shots=1000)
    return result.get_counts()

# Test migration
compatible = migrate_algorithm_to_backend(
    bell_state_algorithm, 
    'simulator', 
    'pennylane'
)
```

## 🛠️ Custom Backend Development

### Creating a Custom Backend

```python
from superquantx.backends import BaseBackend

class MyCustomBackend(BaseBackend):
    """Example custom backend implementation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "my_custom_backend"
        self.backend_type = "custom"
    
    def create_circuit(self, num_qubits):
        """Create a quantum circuit."""
        # Your circuit implementation
        return MyCustomCircuit(num_qubits)
    
    def run(self, circuit, shots=1000):
        """Execute a quantum circuit."""
        # Your execution logic
        pass
    
    def get_capabilities(self):
        """Return backend capabilities."""
        return BackendCapabilities(
            max_qubits=32,
            has_noise=False,
            supports_gpu=True,
            native_gates=['h', 'x', 'cx'],
            hardware_access=False
        )

# Register custom backend
sqx.register_backend('my_custom', MyCustomBackend)

# Use custom backend
custom_backend = sqx.get_backend('my_custom')
```

## 📚 Best Practices

### Backend Selection Guidelines

1. **For Learning**: Use `simulator` backend - simple and always available
2. **For QML Research**: Use `pennylane` backend - excellent differentiation support
3. **For IBM Hardware**: Use `qiskit` backend - direct access to IBM systems
4. **For Google Hardware**: Use `cirq` backend - optimized for Google devices
5. **For Cloud Computing**: Use `braket` backend - access to multiple hardware providers

### Performance Optimization

```python
# Backend-specific optimizations
def optimize_for_backend(circuit, backend_name):
    """Apply backend-specific optimizations."""
    
    backend = sqx.get_backend(backend_name)
    
    if backend_name == 'pennylane':
        # PennyLane optimizations
        circuit = circuit.optimize_parameters()
        circuit = circuit.merge_rotations()
    
    elif backend_name == 'qiskit':
        # Qiskit optimizations
        circuit = backend.transpile(circuit, optimization_level=3)
    
    elif backend_name == 'cirq':
        # Cirq optimizations
        circuit = circuit.optimize_for_nisq()
    
    return circuit
```

### Error Handling

```python
def robust_backend_execution(circuit, backend_priorities=['pennylane', 'qiskit', 'simulator']):
    """Execute circuit with fallback backends."""
    
    for backend_name in backend_priorities:
        try:
            backend = sqx.get_backend(backend_name)
            result = backend.run(circuit, shots=1000)
            print(f"✅ Successfully executed on {backend_name}")
            return result, backend_name
            
        except Exception as e:
            print(f"❌ {backend_name} failed: {e}")
            continue
    
    raise RuntimeError("All backends failed!")

# Example usage
result, successful_backend = robust_backend_execution(my_circuit)
```

## 🔮 Future Backend Support

SuperQuantX is actively working on support for additional backends:

- **IonQ**: Ion trap quantum computers
- **Rigetti**: Superconducting quantum processors
- **Xanadu**: Photonic quantum computing
- **Oxford Quantum Computing**: TKET optimization
- **Quantinuum**: Trapped ion systems

## 📞 Getting Help

If you need help with backends:

- **[Backend-Specific Guides](../backends/pennylane.md)**: Detailed guides for each backend
- **[Troubleshooting](../help/troubleshooting.md)**: Common backend issues
- **[FAQ](../help/faq.md)**: Frequently asked questions
- **[GitHub Issues](https://github.com/SuperagenticAI/superquantx/issues)**: Report backend bugs

---

!!! tip "Backend Recommendation"
    Start with the `simulator` backend for learning, then move to `pennylane` for quantum machine learning research. Use other backends when you need specific hardware access or features.

!!! warning "Backend Availability"
    Not all backends may be available depending on your installation. Use `sqx.list_backends()` to check which backends are installed and ready to use.