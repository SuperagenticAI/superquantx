# Backends API Reference

SuperQuantX provides a unified interface to multiple quantum computing backends, enabling seamless switching between simulators and real quantum hardware from different providers.

## Base Classes

### Base Backend

::: superquantx.backends.BaseBackend
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

## Simulator Backends

### Python Simulator Backend

::: superquantx.backends.SimulatorBackend
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

## Hardware Backends

### IBM Qiskit Backend

::: superquantx.backends.QiskitBackend
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

### PennyLane Backend

::: superquantx.backends.PennyLaneBackend
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

### Google Cirq Backend

::: superquantx.backends.CirqBackend
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

### Amazon Braket Backend

::: superquantx.backends.BraketBackend
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

### TKET/Quantinuum Backend

::: superquantx.backends.TKETBackend
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

### D-Wave Ocean Backend

::: superquantx.backends.OceanBackend
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

## Utility Functions

### Backend Management

::: superquantx.backends.get_backend
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.backends.list_available_backends
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.backends.check_backend_compatibility
    handler: python
    options:
      docstring_style: google
      show_source: true

## Backend Selection Guide

### By Use Case

**Development & Prototyping**
```python
# Fast local simulation
backend = sqx.get_backend('simulator', max_qubits=10)

# Cross-platform testing
backend = sqx.get_backend('pennylane', device='default.qubit')
```

**Production & Research**
```python
# IBM Quantum hardware
backend = sqx.get_backend('qiskit', device='ibmq_manila', token='your-token')

# Google Quantum AI
backend = sqx.get_backend('cirq', processor_id='rainbow', project_id='your-project')

# AWS Braket cloud
backend = sqx.get_backend('braket', device_arn='arn:aws:braket:us-east-1::device/qpu/ionq/ionQdevice')
```

**Specialized Applications**
```python
# Quantum chemistry (TKET optimization)
backend = sqx.get_backend('tket', device='H1-1E', api_key='quantinuum-key')

# Optimization problems (D-Wave annealing)
backend = sqx.get_backend('ocean', device='advantage', token='dwave-token')
```

### By Performance Requirements

**High Accuracy (Large Shot Counts)**
```python
backends_high_accuracy = {
    'simulator': sqx.get_backend('simulator', shots=100000),
    'ibm_hardware': sqx.get_backend('qiskit', device='ibmq_montreal', shots=8192),
    'aws_sv1': sqx.get_backend('braket', device='sv1', shots=100000)
}
```

**Fast Iteration (Low Latency)**
```python
backends_fast = {
    'local_sim': sqx.get_backend('simulator', shots=1000),
    'pennylane': sqx.get_backend('pennylane', device='lightning.qubit'),
    'braket_local': sqx.get_backend('braket', device='braket_sv')
}
```

**Cost Optimization**
```python
# Free tier resources
free_backends = {
    'simulator': sqx.get_backend('simulator'),
    'ibm_free': sqx.get_backend('qiskit', device='ibmq_qasm_simulator'),
    'aws_sim': sqx.get_backend('braket', device='braket_sv')  # First 1 hour free
}
```

## Integration Examples

### Multi-Backend Algorithm Testing

```python
import superquantx as sqx

def test_algorithm_across_backends(algorithm_func, backends, *args, **kwargs):
    """Test quantum algorithm across multiple backends."""
    
    results = {}
    
    for backend_name, backend in backends.items():
        try:
            print(f"Testing on {backend_name}...")
            
            # Run algorithm
            result = algorithm_func(backend, *args, **kwargs)
            results[backend_name] = result
            
            print(f"  ✓ Success: {result}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[backend_name] = None
    
    return results

# Define test backends
test_backends = {
    'simulator': sqx.get_backend('simulator'),
    'pennylane': sqx.get_backend('pennylane', device='default.qubit'),
}

# Example quantum algorithm
def simple_bell_state(backend):
    circuit = backend.create_circuit(2)
    circuit = backend.add_gate(circuit, 'h', 0)
    circuit = backend.add_gate(circuit, 'cx', [0, 1])
    return backend.execute_circuit(circuit)

# Test across backends
results = test_algorithm_across_backends(simple_bell_state, test_backends)
```

### Backend Fallback Strategy

```python
def get_backend_with_fallback(preferences, **kwargs):
    """Get backend with fallback strategy."""
    
    for backend_name in preferences:
        try:
            backend = sqx.get_backend(backend_name, **kwargs)
            if backend.is_available():
                print(f"Using backend: {backend_name}")
                return backend
        except Exception as e:
            print(f"Backend {backend_name} failed: {e}")
            continue
    
    raise RuntimeError("No backends available")

# Usage
backend = get_backend_with_fallback([
    'qiskit',      # Try IBM first
    'pennylane',   # Fallback to PennyLane  
    'simulator'    # Final fallback
], shots=1024)
```

### Performance Benchmarking

```python
import time
import numpy as np

def benchmark_backends(backends, circuit_sizes=[2, 4, 6, 8]):
    """Benchmark quantum backends across different circuit sizes."""
    
    results = {}
    
    for backend_name, backend in backends.items():
        print(f"\nBenchmarking {backend_name}:")
        results[backend_name] = {}
        
        for n_qubits in circuit_sizes:
            if n_qubits > backend.get_backend_info().get('capabilities', {}).get('max_qubits', 32):
                continue
                
            # Create test circuit
            circuit = backend.create_circuit(n_qubits)
            
            # Add random gates
            for _ in range(n_qubits * 2):
                qubit = np.random.randint(0, n_qubits)
                circuit = backend.add_gate(circuit, 'h', qubit)
                
            for _ in range(n_qubits - 1):
                q1, q2 = np.random.choice(n_qubits, 2, replace=False)
                circuit = backend.add_gate(circuit, 'cx', [q1, q2])
            
            # Benchmark execution
            start_time = time.time()
            try:
                result = backend.execute_circuit(circuit, shots=1000)
                execution_time = time.time() - start_time
                
                results[backend_name][n_qubits] = {
                    'time': execution_time,
                    'success': result.get('success', True),
                    'total_counts': sum(result.get('counts', {}).values())
                }
                
                print(f"  {n_qubits} qubits: {execution_time:.3f}s")
                
            except Exception as e:
                print(f"  {n_qubits} qubits: Failed - {e}")
                results[backend_name][n_qubits] = {'error': str(e)}
    
    return results

# Run benchmark
benchmark_backends({
    'simulator': sqx.get_backend('simulator'),
    'pennylane': sqx.get_backend('pennylane'),
})
```

## Backend Configuration

### Environment Setup

```python
import os
import superquantx as sqx

# Configure backends via environment variables
os.environ['QISKIT_IBMQ_TOKEN'] = 'your-ibm-token'
os.environ['BRAKET_AWS_REGION'] = 'us-east-1'
os.environ['DWAVE_API_TOKEN'] = 'your-dwave-token'

# Test configurations
def validate_backend_config():
    """Validate backend configurations."""
    
    configs = {
        'qiskit': {
            'required_env': ['QISKIT_IBMQ_TOKEN'],
            'test_device': 'ibmq_qasm_simulator'
        },
        'braket': {
            'required_env': ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'],
            'test_device': 'braket_sv'
        },
        'ocean': {
            'required_env': ['DWAVE_API_TOKEN'],
            'test_device': 'simulator'
        }
    }
    
    for backend_name, config in configs.items():
        print(f"Validating {backend_name}:")
        
        # Check environment variables
        missing_env = [env for env in config['required_env'] 
                      if not os.getenv(env)]
        
        if missing_env:
            print(f"  ✗ Missing environment variables: {missing_env}")
            continue
        
        # Test backend creation
        try:
            backend = sqx.get_backend(backend_name, device=config['test_device'])
            if backend.is_available():
                print(f"  ✓ Backend available")
            else:
                print(f"  ✗ Backend not available")
        except Exception as e:
            print(f"  ✗ Configuration error: {e}")

validate_backend_config()
```

### Custom Backend Configuration

```python
# Advanced backend configuration
backend_configs = {
    'qiskit_optimized': {
        'backend_type': 'qiskit',
        'device': 'ibmq_montreal',
        'shots': 4096,
        'optimization_level': 3,
        'error_mitigation': True,
        'noise_model': 'device_noise'
    },
    
    'pennylane_gpu': {
        'backend_type': 'pennylane', 
        'device': 'lightning.gpu',
        'shots': 100000,
        'batch_size': 1024
    },
    
    'braket_cost_optimized': {
        'backend_type': 'braket',
        'device': 'sv1',  # Simulator for cost optimization
        'shots': 1000,
        'max_parallel_tasks': 5
    }
}

def create_configured_backend(config_name):
    """Create backend from configuration."""
    config = backend_configs[config_name]
    backend_type = config.pop('backend_type')
    return sqx.get_backend(backend_type, **config)
```

## Error Handling

### Backend-Specific Error Handling

```python
from superquantx.exceptions import (
    QuantumBackendError, 
    DeviceOfflineError,
    AuthenticationError,
    ResourceExhaustedError
)

def robust_backend_execution(backend, circuit, max_retries=3):
    """Execute circuit with comprehensive error handling."""
    
    for attempt in range(max_retries):
        try:
            result = backend.execute_circuit(circuit)
            return result
            
        except AuthenticationError:
            print("Authentication failed - check API tokens")
            raise  # Don't retry auth errors
            
        except DeviceOfflineError as e:
            print(f"Device offline: {e}")
            if attempt < max_retries - 1:
                print("Waiting 30s before retry...")
                time.sleep(30)
            
        except ResourceExhaustedError as e:
            print(f"Resources exhausted: {e}")
            # Reduce shots and retry
            circuit.shots = max(circuit.shots // 2, 100)
            
        except QuantumBackendError as e:
            print(f"Backend error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise
                
    raise RuntimeError("All execution attempts failed")

# Usage with error handling
try:
    backend = sqx.get_backend('qiskit', device='ibmq_manila')
    result = robust_backend_execution(backend, circuit)
except Exception as e:
    print(f"Execution failed: {e}")
    # Fallback to simulator
    fallback_backend = sqx.get_backend('simulator')
    result = fallback_backend.execute_circuit(circuit)
```

## Best Practices

### Backend Selection Strategy

1. **Development Phase**: Use `simulator` or `pennylane` for rapid iteration
2. **Testing Phase**: Use cloud simulators (`braket_sv`, `qiskit_aer`) for validation
3. **Production Phase**: Use real hardware with appropriate error mitigation
4. **Cost Optimization**: Use free simulators when possible, batch hardware jobs

### Performance Optimization

```python
# Optimize circuit for specific backend
def optimize_for_backend(circuit, backend):
    """Optimize circuit for specific backend capabilities."""
    
    backend_info = backend.get_backend_info()
    
    if 'qiskit' in backend_info.get('backend_name', ''):
        # Use Qiskit-specific optimizations
        return backend.transpile_circuit(circuit, optimization_level=2)
        
    elif 'tket' in backend_info.get('backend_name', ''):
        # Use TKET optimizations
        return backend.optimize_circuit(circuit, optimization_level=3)
        
    elif 'ocean' in backend_info.get('backend_name', ''):
        # Ocean backends use different problem formulations
        return backend.convert_to_qubo(circuit)
        
    else:
        # Generic optimization
        return circuit

# Batch processing for hardware backends  
def batch_execute_circuits(backend, circuits, max_batch_size=10):
    """Execute circuits in batches for efficiency."""
    
    results = []
    
    for i in range(0, len(circuits), max_batch_size):
        batch = circuits[i:i + max_batch_size]
        
        if hasattr(backend, 'execute_batch'):
            # Use native batch execution if available
            batch_results = backend.execute_batch(batch)
        else:
            # Sequential execution fallback
            batch_results = [backend.execute_circuit(c) for c in batch]
        
        results.extend(batch_results)
    
    return results
```

---

For detailed backend integration guides, see:
- [Qiskit Integration](../backends/qiskit.md)
- [PennyLane Integration](../backends/pennylane.md)
- [Cirq Integration](../backends/cirq.md)
- [Braket Integration](../backends/braket.md)
- [TKET Integration](../backends/tket.md)
- [Ocean Integration](../backends/ocean.md)
- [Simulator Backend](../backends/simulator.md)