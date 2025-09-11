# Qiskit Backend Integration

Qiskit is IBM's open-source quantum computing framework that provides access to quantum simulators and real IBM quantum hardware. SuperQuantX integrates seamlessly with Qiskit to provide access to IBM's quantum ecosystem.

## Overview

The Qiskit backend in SuperQuantX offers:

- **IBM Quantum Hardware Access**: Connect to real IBM quantum computers
- **High-Performance Simulators**: Use Qiskit Aer simulators
- **Noise Models**: Simulate realistic quantum device noise
- **Quantum Circuits**: Native Qiskit circuit manipulation
- **Job Management**: Automatic job queue handling

## Installation

```bash
# Install SuperQuantX with Qiskit support
pip install superquantx[qiskit]

# Or install Qiskit directly
pip install qiskit qiskit-aer qiskit-ibm-runtime
```

## Quick Start

### Basic Usage

```python
import superquantx as sqx

# Get Qiskit backend
backend = sqx.get_backend('qiskit')
print(f"Backend: {backend}")

# Create Bell state circuit
circuit = backend.create_circuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

# Execute on simulator
result = backend.run(circuit, shots=1000)
print(f"Bell state results: {result.get_counts()}")
```

### IBM Quantum Hardware

```python
# Connect to IBM Quantum
from qiskit_ibm_runtime import QiskitRuntimeService

# Save your credentials (one-time setup)
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="YOUR_IBM_QUANTUM_TOKEN",
    overwrite=True
)

# Use IBM Quantum backend
backend = sqx.QiskitBackend(
    backend_name='ibmq_qasm_simulator',  # or real hardware like 'ibm_osaka'
    shots=1024
)

# Execute on IBM hardware
result = backend.run(circuit, shots=1024)
```

## Configuration Options

### Backend Selection

```python
# Different Qiskit backends
backends = {
    'qasm_simulator': 'Classical QASM simulator',
    'statevector_simulator': 'Statevector simulator', 
    'unitary_simulator': 'Unitary matrix simulator',
    'ibmq_qasm_simulator': 'IBM cloud simulator',
    'ibm_osaka': 'IBM quantum processor (127 qubits)',
    'ibm_kyoto': 'IBM quantum processor (127 qubits)'
}

# Create backend with specific device
backend = sqx.QiskitBackend(
    backend_name='qasm_simulator',
    shots=2048,
    optimization_level=3
)
```

### Advanced Configuration

```python
# Custom backend configuration
backend = sqx.QiskitBackend(
    backend_name='qasm_simulator',
    shots=1000,
    optimization_level=2,  # 0-3, higher = more optimization
    seed_simulator=42,     # For reproducible results
    max_parallel_threads=4,
    memory=True,          # Enable memory snapshots
    noise_model=None      # Custom noise model
)

# Access underlying Qiskit backend
qiskit_backend = backend.backend
print(f"Backend configuration: {qiskit_backend.configuration()}")
```

## Working with Circuits

### Circuit Creation

```python
def demonstrate_qiskit_circuits():
    """Show various Qiskit circuit features."""
    
    backend = sqx.get_backend('qiskit')
    
    # Create 4-qubit circuit
    circuit = backend.create_circuit(4)
    
    # Single qubit gates
    circuit.h(0)           # Hadamard
    circuit.x(1)           # Pauli-X
    circuit.y(2)           # Pauli-Y  
    circuit.z(3)           # Pauli-Z
    
    # Rotation gates
    import numpy as np
    circuit.rx(np.pi/4, 0) # X rotation
    circuit.ry(np.pi/3, 1) # Y rotation
    circuit.rz(np.pi/6, 2) # Z rotation
    
    # Two-qubit gates
    circuit.cx(0, 1)       # CNOT
    circuit.cz(1, 2)       # Controlled-Z
    circuit.swap(2, 3)     # SWAP
    
    # Multi-controlled gates
    circuit.ccx(0, 1, 2)   # Toffoli (CCX)
    
    # Phase gates
    circuit.s(0)           # S gate (√Z)
    circuit.t(1)           # T gate (√S)
    
    # Measure all qubits
    circuit.measure_all()
    
    # Execute circuit
    result = backend.run(circuit, shots=1000)
    print(f"Complex circuit results: {result.get_counts()}")
    
    # Visualize circuit (if available)
    try:
        print(circuit.draw())
    except:
        print("Circuit visualization not available")

demonstrate_qiskit_circuits()
```

### Parameterized Circuits

```python
def qiskit_parameterized_circuits():
    """Demonstrate parameterized quantum circuits."""
    
    backend = sqx.get_backend('qiskit')
    
    # Create parameterized circuit for VQE
    def create_ansatz(params, n_qubits=4, layers=2):
        circuit = backend.create_circuit(n_qubits)
        
        param_idx = 0
        
        # Initial layer
        for i in range(n_qubits):
            circuit.ry(params[param_idx], i)
            param_idx += 1
        
        # Entangling layers
        for layer in range(layers):
            # CNOT ladder
            for i in range(n_qubits - 1):
                circuit.cx(i, i + 1)
            
            # Parameterized rotations
            for i in range(n_qubits):
                circuit.ry(params[param_idx], i)
                param_idx += 1
                circuit.rz(params[param_idx], i) 
                param_idx += 1
        
        return circuit
    
    # Generate random parameters
    n_params = 4 + 2 * 2 * 4  # Initial + layers * 2 * qubits
    params = np.random.random(n_params) * 2 * np.pi
    
    # Create and run ansatz
    ansatz = create_ansatz(params)
    ansatz.measure_all()
    
    result = backend.run(ansatz, shots=1000)
    print(f"Parameterized circuit results: {result.get_counts()}")
    
    # Parameter sensitivity analysis
    print("\nParameter Sensitivity Analysis:")
    base_result = result.get_counts()
    base_prob_00 = base_result.get('0000', 0) / 1000
    
    # Vary first parameter
    for delta in [-0.1, -0.05, 0.05, 0.1]:
        test_params = params.copy()
        test_params[0] += delta
        
        test_circuit = create_ansatz(test_params)
        test_circuit.measure_all()
        
        test_result = backend.run(test_circuit, shots=1000)
        test_counts = test_result.get_counts()
        test_prob_00 = test_counts.get('0000', 0) / 1000
        
        print(f"  Δθ = {delta:+.2f}: P(0000) = {test_prob_00:.3f} "
              f"(change: {test_prob_00 - base_prob_00:+.3f})")

qiskit_parameterized_circuits()
```

## IBM Quantum Hardware

### Real Hardware Execution

```python
def run_on_ibm_hardware():
    """Execute circuits on real IBM quantum hardware."""
    
    # Note: Requires IBM Quantum account setup
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        
        # Initialize service
        service = QiskitRuntimeService(channel="ibm_quantum")
        
        # List available backends
        available_backends = service.backends()
        print("Available IBM Quantum backends:")
        for backend in available_backends:
            status = backend.status()
            print(f"  {backend.name}: {status.pending_jobs} jobs pending, "
                  f"operational: {status.operational}")
        
        # Select least busy backend
        from qiskit_ibm_runtime.utils import least_busy
        backend_name = least_busy(available_backends).name
        
        print(f"\nUsing backend: {backend_name}")
        
        # Create SuperQuantX backend for IBM hardware
        sqx_backend = sqx.QiskitBackend(
            backend_name=backend_name,
            shots=1024,
            optimization_level=3  # Max optimization for hardware
        )
        
        # Create simple Bell state
        circuit = sqx_backend.create_circuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        
        print("Submitting job to IBM Quantum...")
        result = sqx_backend.run(circuit, shots=1024)
        
        counts = result.get_counts()
        print(f"Hardware results: {counts}")
        
        # Analyze fidelity
        prob_00 = counts.get('00', 0) / 1024
        prob_11 = counts.get('11', 0) / 1024
        bell_fidelity = prob_00 + prob_11
        
        print(f"Bell state fidelity: {bell_fidelity:.3f}")
        
        # Compare with ideal
        if bell_fidelity > 0.8:
            print("✅ Good Bell state fidelity")
        elif bell_fidelity > 0.6:
            print("⚠️ Moderate Bell state fidelity")
        else:
            print("❌ Low Bell state fidelity")
            
    except ImportError:
        print("IBM Quantum access requires: pip install qiskit-ibm-runtime")
    except Exception as e:
        print(f"IBM Quantum access error: {e}")
        print("Make sure to set up IBM Quantum credentials")

# Uncomment to run with proper IBM Quantum setup
# run_on_ibm_hardware()
```

### Device Calibration Data

```python
def analyze_device_properties():
    """Analyze IBM quantum device properties and calibration data."""
    
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        
        service = QiskitRuntimeService(channel="ibm_quantum")
        backend = service.get_backend('ibm_osaka')  # 127-qubit processor
        
        # Get backend properties
        properties = backend.properties()
        configuration = backend.configuration()
        
        print(f"Backend: {backend.name}")
        print(f"Number of qubits: {configuration.n_qubits}")
        print(f"Coupling map: {len(configuration.coupling_map)} connections")
        
        # Analyze qubit properties
        if properties:
            print("\nQubit Error Rates:")
            for i in range(min(10, configuration.n_qubits)):  # First 10 qubits
                # T1 and T2 times
                t1 = properties.t1(i)
                t2 = properties.t2(i)
                
                # Readout error
                readout_error = properties.readout_error(i)
                
                # Gate error (single qubit)
                try:
                    sx_error = properties.gate_error('sx', i)
                    print(f"  Qubit {i}: T1={t1:.1f}μs, T2={t2:.1f}μs, "
                          f"Readout={readout_error:.4f}, SX={sx_error:.4f}")
                except:
                    print(f"  Qubit {i}: T1={t1:.1f}μs, T2={t2:.1f}μs, "
                          f"Readout={readout_error:.4f}")
            
            # Two-qubit gate errors
            print("\nTwo-Qubit Gate Errors:")
            coupling_map = configuration.coupling_map
            for i, (q1, q2) in enumerate(coupling_map[:10]):  # First 10 pairs
                try:
                    cx_error = properties.gate_error('cx', [q1, q2])
                    print(f"  CX({q1},{q2}): {cx_error:.4f}")
                except:
                    continue
        
        # Create noise model for simulation
        from qiskit_aer.noise import NoiseModel
        noise_model = NoiseModel.from_backend(backend)
        
        print(f"\nNoise model created with {len(noise_model.noise_instructions)} "
              f"noise instructions")
        
        # Use noise model in simulation
        sqx_backend = sqx.QiskitBackend(
            backend_name='qasm_simulator',
            noise_model=noise_model,
            shots=1000
        )
        
        # Test with Bell state
        circuit = sqx_backend.create_circuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        
        print("\nTesting with device noise model:")
        noisy_result = sqx_backend.run(circuit, shots=1000)
        noisy_counts = noisy_result.get_counts()
        
        print(f"Noisy simulation: {noisy_counts}")
        
        # Calculate noise impact
        prob_00 = noisy_counts.get('00', 0) / 1000
        prob_11 = noisy_counts.get('11', 0) / 1000
        prob_01 = noisy_counts.get('01', 0) / 1000
        prob_10 = noisy_counts.get('10', 0) / 1000
        
        fidelity = prob_00 + prob_11
        error_rate = prob_01 + prob_10
        
        print(f"Bell state fidelity with noise: {fidelity:.3f}")
        print(f"Error rate: {error_rate:.3f}")
        
    except Exception as e:
        print(f"Device analysis requires IBM Quantum access: {e}")

# Uncomment to run with IBM Quantum access
# analyze_device_properties()
```

## Advanced Features

### Transpilation and Optimization

```python
def demonstrate_transpilation():
    """Show Qiskit circuit transpilation and optimization."""
    
    backend = sqx.get_backend('qiskit')
    
    # Create complex circuit that needs optimization
    circuit = backend.create_circuit(5)
    
    # Add many gates that can be optimized
    for i in range(5):
        circuit.h(i)
        
    for i in range(4):
        circuit.cx(i, i + 1)
    
    # Add more layers
    for _ in range(3):
        for i in range(5):
            circuit.rz(np.pi/4, i)
        for i in range(4):
            circuit.cx(i, i + 1)
    
    circuit.measure_all()
    
    print("Original circuit stats:")
    print(f"  Gates: {len(circuit._circuit.data) if hasattr(circuit, '_circuit') else 'N/A'}")
    
    # Test different optimization levels
    optimization_levels = [0, 1, 2, 3]
    
    for opt_level in optimization_levels:
        opt_backend = sqx.QiskitBackend(
            backend_name='qasm_simulator',
            optimization_level=opt_level,
            shots=1000
        )
        
        # Execute with optimization
        result = opt_backend.run(circuit, shots=1000)
        counts = result.get_counts()
        
        print(f"\nOptimization level {opt_level}:")
        print(f"  Result distribution: {len(counts)} unique outcomes")
        print(f"  Most probable: {max(counts.keys(), key=counts.get)} "
              f"({max(counts.values())/1000:.3f})")

demonstrate_transpilation()
```

### Custom Noise Models

```python
def create_custom_noise_models():
    """Create and use custom noise models."""
    
    try:
        from qiskit_aer.noise import (NoiseModel, depolarizing_error, 
                                    thermal_relaxation_error, ReadoutError)
        
        # Create custom noise model
        noise_model = NoiseModel()
        
        # Single-qubit gate errors
        single_qubit_error = depolarizing_error(0.001, 1)  # 0.1% error rate
        noise_model.add_all_qubit_quantum_error(single_qubit_error, ['u1', 'u2', 'u3'])
        
        # Two-qubit gate errors
        two_qubit_error = depolarizing_error(0.01, 2)  # 1% error rate
        noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx'])
        
        # Thermal relaxation
        t1 = 50000  # T1 time (ns)
        t2 = 70000  # T2 time (ns)
        
        for qubit in range(5):  # Apply to 5 qubits
            thermal_error = thermal_relaxation_error(t1, t2, 35)  # 35ns gate time
            noise_model.add_quantum_error(thermal_error, ['u1', 'u2', 'u3'], [qubit])
        
        # Readout errors
        readout_error = ReadoutError([[0.95, 0.05], [0.1, 0.9]])  # Asymmetric readout
        noise_model.add_all_qubit_readout_error(readout_error)
        
        print("Custom noise model created:")
        print(f"  Quantum errors: {len(noise_model.noise_instructions)}")
        print(f"  Readout errors: {len(noise_model._default_readout_errors)}")
        
        # Use custom noise model
        backend = sqx.QiskitBackend(
            backend_name='qasm_simulator',
            noise_model=noise_model,
            shots=2000
        )
        
        # Test various circuits
        test_circuits = {
            'Single H': lambda: backend.create_circuit(1).h(0).measure_all(),
            'Bell State': lambda: backend.create_circuit(2).h(0).cx(0,1).measure_all(),
            'GHZ State': lambda: backend.create_circuit(3).h(0).cx(0,1).cx(1,2).measure_all()
        }
        
        print("\nNoise impact analysis:")
        
        for name, circuit_func in test_circuits.items():
            # Noiseless simulation
            clean_backend = sqx.QiskitBackend(backend_name='qasm_simulator', shots=2000)
            clean_circuit = circuit_func()
            clean_result = clean_backend.run(clean_circuit, shots=2000)
            clean_counts = clean_result.get_counts()
            
            # Noisy simulation  
            noisy_circuit = circuit_func()
            noisy_result = backend.run(noisy_circuit, shots=2000)
            noisy_counts = noisy_result.get_counts()
            
            # Compare fidelities
            if name == 'Single H':
                # Expected: 50-50 distribution
                clean_fidelity = min(clean_counts.get('0', 0), clean_counts.get('1', 0)) / 1000
                noisy_fidelity = min(noisy_counts.get('0', 0), noisy_counts.get('1', 0)) / 1000
            else:
                # Expected: only |00..0⟩ and |11..1⟩
                all_zeros = '0' * (2 if 'Bell' in name else 3)
                all_ones = '1' * (2 if 'Bell' in name else 3)
                
                clean_fidelity = (clean_counts.get(all_zeros, 0) + 
                                clean_counts.get(all_ones, 0)) / 2000
                noisy_fidelity = (noisy_counts.get(all_zeros, 0) + 
                                noisy_counts.get(all_ones, 0)) / 2000
            
            fidelity_loss = clean_fidelity - noisy_fidelity
            
            print(f"  {name}:")
            print(f"    Clean fidelity: {clean_fidelity:.3f}")
            print(f"    Noisy fidelity: {noisy_fidelity:.3f}")
            print(f"    Fidelity loss: {fidelity_loss:.3f}")
    
    except ImportError:
        print("Noise modeling requires: pip install qiskit-aer")

create_custom_noise_models()
```

### Error Mitigation

```python
def demonstrate_error_mitigation():
    """Show error mitigation techniques with Qiskit."""
    
    try:
        from qiskit_aer.noise import NoiseModel, depolarizing_error
        
        # Create noisy backend
        noise_model = NoiseModel()
        error = depolarizing_error(0.02, 1)  # 2% single-qubit error
        noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])
        
        noisy_backend = sqx.QiskitBackend(
            backend_name='qasm_simulator',
            noise_model=noise_model,
            shots=2000
        )
        
        # Test circuit: |+⟩ state
        def create_plus_state():
            circuit = noisy_backend.create_circuit(1)
            circuit.h(0)
            circuit.measure_all()
            return circuit
        
        # 1. Zero-noise extrapolation
        print("Error Mitigation: Zero-Noise Extrapolation")
        
        noise_factors = [1, 2, 3]  # Amplify noise by these factors
        expectation_values = []
        
        for factor in noise_factors:
            # Amplify noise by repeating gates
            circuit = noisy_backend.create_circuit(1)
            
            # Apply H gate 'factor' times (odd repetitions = H, even = I)
            for _ in range(factor):
                circuit.h(0)
            
            # If even number of H gates, apply one more to get |+⟩
            if factor % 2 == 0:
                circuit.h(0)
            
            circuit.measure_all()
            
            result = noisy_backend.run(circuit, shots=2000)
            counts = result.get_counts()
            
            # Expectation value of |+⟩ state
            prob_0 = counts.get('0', 0) / 2000
            expectation = 2 * prob_0 - 1  # Convert to [-1, 1]
            expectation_values.append(expectation)
            
            print(f"  Noise factor {factor}: <Z> = {expectation:.3f}")
        
        # Extrapolate to zero noise
        # Fit linear model and extrapolate to factor=0
        import numpy as np
        coeffs = np.polyfit(noise_factors, expectation_values, 1)
        zero_noise_value = coeffs[1]  # y-intercept
        
        print(f"  Extrapolated zero-noise: <Z> = {zero_noise_value:.3f}")
        print(f"  Theoretical value: <Z> = 0.000")
        print(f"  Error correction: {abs(zero_noise_value):.3f}")
        
        # 2. Readout error mitigation
        print(f"\nError Mitigation: Readout Correction")
        
        # Calibrate readout errors
        def calibrate_readout():
            # Measure |0⟩ state
            circuit_0 = noisy_backend.create_circuit(1)
            circuit_0.measure_all()
            result_0 = noisy_backend.run(circuit_0, shots=1000)
            counts_0 = result_0.get_counts()
            
            # Measure |1⟩ state
            circuit_1 = noisy_backend.create_circuit(1)
            circuit_1.x(0)  # Prepare |1⟩
            circuit_1.measure_all()
            result_1 = noisy_backend.run(circuit_1, shots=1000)
            counts_1 = result_1.get_counts()
            
            # Build calibration matrix
            p00 = counts_0.get('0', 0) / 1000  # P(measure 0 | prepared 0)
            p01 = counts_0.get('1', 0) / 1000  # P(measure 1 | prepared 0)
            p10 = counts_1.get('0', 0) / 1000  # P(measure 0 | prepared 1) 
            p11 = counts_1.get('1', 0) / 1000  # P(measure 1 | prepared 1)
            
            calibration_matrix = np.array([[p00, p10], [p01, p11]])
            return calibration_matrix
        
        cal_matrix = calibrate_readout()
        print(f"  Calibration matrix:")
        print(f"    [[{cal_matrix[0,0]:.3f}, {cal_matrix[0,1]:.3f}],")
        print(f"     [{cal_matrix[1,0]:.3f}, {cal_matrix[1,1]:.3f}]]")
        
        # Test readout correction
        test_circuit = create_plus_state()
        raw_result = noisy_backend.run(test_circuit, shots=1000)
        raw_counts = raw_result.get_counts()
        
        # Apply correction
        raw_probs = np.array([
            raw_counts.get('0', 0) / 1000,
            raw_counts.get('1', 0) / 1000
        ])
        
        corrected_probs = np.linalg.inv(cal_matrix) @ raw_probs
        
        print(f"  Raw probabilities: [P(0)={raw_probs[0]:.3f}, P(1)={raw_probs[1]:.3f}]")
        print(f"  Corrected probs: [P(0)={corrected_probs[0]:.3f}, P(1)={corrected_probs[1]:.3f}]")
        print(f"  Theoretical: [P(0)=0.500, P(1)=0.500]")
        
        # Calculate improvement
        raw_error = abs(raw_probs[0] - 0.5) + abs(raw_probs[1] - 0.5)
        corrected_error = abs(corrected_probs[0] - 0.5) + abs(corrected_probs[1] - 0.5)
        improvement = (raw_error - corrected_error) / raw_error * 100
        
        print(f"  Error reduction: {improvement:.1f}%")
        
    except ImportError:
        print("Error mitigation requires advanced Qiskit components")

demonstrate_error_mitigation()
```

## Performance Optimization

### Execution Time Analysis

```python
def qiskit_performance_analysis():
    """Analyze Qiskit backend performance characteristics."""
    
    import time
    
    # Test different simulators
    simulators = [
        ('qasm_simulator', 'Basic QASM simulator'),
        ('statevector_simulator', 'Statevector simulator'),
    ]
    
    # Circuit complexity levels
    complexities = [
        (2, 1, "Simple 2-qubit, 1 layer"),
        (4, 2, "Medium 4-qubit, 2 layers"),
        (6, 3, "Complex 6-qubit, 3 layers"),
    ]
    
    results = {}
    
    print("Qiskit Performance Analysis")
    print("=" * 30)
    
    for sim_name, sim_desc in simulators:
        print(f"\nTesting {sim_name} ({sim_desc}):")
        results[sim_name] = {}
        
        try:
            for n_qubits, n_layers, desc in complexities:
                # Create test circuit
                backend = sqx.QiskitBackend(
                    backend_name=sim_name,
                    shots=1000,
                    optimization_level=2
                )
                
                circuit = backend.create_circuit(n_qubits)
                
                # Build parameterized circuit
                params = np.random.random(n_qubits * n_layers * 2) * 2 * np.pi
                param_idx = 0
                
                for layer in range(n_layers):
                    # Rotation layer
                    for qubit in range(n_qubits):
                        circuit.ry(params[param_idx], qubit)
                        param_idx += 1
                        circuit.rz(params[param_idx], qubit)
                        param_idx += 1
                    
                    # Entangling layer
                    for qubit in range(n_qubits - 1):
                        circuit.cx(qubit, qubit + 1)
                
                circuit.measure_all()
                
                # Time execution
                start_time = time.time()
                result = backend.run(circuit, shots=1000)
                execution_time = time.time() - start_time
                
                results[sim_name][(n_qubits, n_layers)] = {
                    'time': execution_time,
                    'counts': len(result.get_counts())
                }
                
                print(f"  {desc}: {execution_time:.3f}s "
                      f"({len(result.get_counts())} unique outcomes)")
                
        except Exception as e:
            print(f"  Error with {sim_name}: {e}")
    
    # Shot count vs accuracy analysis
    print(f"\nShot Count vs Accuracy Analysis:")
    backend = sqx.QiskitBackend(backend_name='qasm_simulator')
    
    # Bell state circuit
    circuit = backend.create_circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    
    shot_counts = [100, 500, 1000, 2000, 5000]
    theoretical_prob = 0.5  # For Bell state |00⟩ probability
    
    for shots in shot_counts:
        start_time = time.time()
        result = backend.run(circuit, shots=shots)
        exec_time = time.time() - start_time
        
        counts = result.get_counts()
        prob_00 = counts.get('00', 0) / shots
        error = abs(prob_00 - theoretical_prob)
        
        print(f"  {shots:4d} shots: P(00)={prob_00:.3f}, "
              f"Error={error:.3f}, Time={exec_time:.3f}s")
    
    # Optimization level impact
    print(f"\nOptimization Level Impact:")
    
    # Complex circuit for optimization testing
    test_backend = sqx.QiskitBackend(backend_name='qasm_simulator')
    complex_circuit = test_backend.create_circuit(4)
    
    # Create circuit that can be optimized
    for i in range(4):
        complex_circuit.h(i)
        complex_circuit.x(i)  # These X gates after H can be optimized
        complex_circuit.h(i)  # H-X-H = Z-H can be optimized
    
    for i in range(3):
        complex_circuit.cx(i, i + 1)
    
    complex_circuit.measure_all()
    
    for opt_level in range(4):
        opt_backend = sqx.QiskitBackend(
            backend_name='qasm_simulator',
            optimization_level=opt_level,
            shots=1000
        )
        
        start_time = time.time()
        result = opt_backend.run(complex_circuit, shots=1000)
        exec_time = time.time() - start_time
        
        print(f"  Opt level {opt_level}: {exec_time:.3f}s")

qiskit_performance_analysis()
```

## Integration Examples

### Quantum Machine Learning with Qiskit

```python
def qiskit_quantum_ml_example():
    """Comprehensive QML example using Qiskit backend."""
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    
    print("Quantum Machine Learning with Qiskit")
    print("=" * 38)
    
    # Generate dataset
    X, y = make_classification(
        n_samples=100, 
        n_features=4, 
        n_classes=2,
        n_redundant=0,
        random_state=42
    )
    
    # Preprocess data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features, 2 classes")
    print(f"Training: {len(X_train)}, Testing: {len(X_test)}")
    
    # Create Quantum SVM using Qiskit backend
    qsvm = sqx.QuantumSVM(
        backend='qiskit',
        feature_map='ZFeatureMap',
        num_features=4,
        shots=1024
    )
    
    # Train quantum SVM
    print("\nTraining Quantum SVM with Qiskit backend...")
    qsvm.fit(X_train, y_train)
    
    # Make predictions
    train_pred = qsvm.predict(X_train)
    test_pred = qsvm.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"Training Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy: {test_acc:.3f}")
    
    # Compare with different Qiskit simulators
    simulators = ['qasm_simulator', 'statevector_simulator']
    
    print(f"\nComparing Qiskit simulators:")
    for sim in simulators:
        try:
            sim_qsvm = sqx.QuantumSVM(
                backend=sqx.QiskitBackend(backend_name=sim),
                feature_map='ZFeatureMap',
                num_features=4
            )
            
            sim_qsvm.fit(X_train, y_train)
            sim_pred = sim_qsvm.predict(X_test)
            sim_acc = accuracy_score(y_test, sim_pred)
            
            print(f"  {sim}: {sim_acc:.3f}")
            
        except Exception as e:
            print(f"  {sim}: Error - {e}")
    
    # Feature map comparison
    print(f"\nFeature map comparison:")
    feature_maps = ['ZFeatureMap', 'ZZFeatureMap', 'PauliFeatureMap']
    
    for fm in feature_maps:
        try:
            fm_qsvm = sqx.QuantumSVM(
                backend='qiskit',
                feature_map=fm,
                num_features=4
            )
            
            fm_qsvm.fit(X_train, y_train)
            fm_pred = fm_qsvm.predict(X_test)
            fm_acc = accuracy_score(y_test, fm_pred)
            
            print(f"  {fm}: {fm_acc:.3f}")
            
        except Exception as e:
            print(f"  {fm}: Error - {e}")

qiskit_quantum_ml_example()
```

## Troubleshooting

### Common Issues

#### 1. IBM Quantum Authentication

```python
# Fix authentication issues
from qiskit_ibm_runtime import QiskitRuntimeService

# Check saved accounts
saved_accounts = QiskitRuntimeService.saved_accounts()
print("Saved accounts:", saved_accounts)

# Re-save credentials if needed
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="YOUR_TOKEN",
    overwrite=True
)
```

#### 2. Backend Availability

```python
# Check backend status
def check_backend_status():
    try:
        service = QiskitRuntimeService()
        backends = service.backends()
        
        for backend in backends:
            status = backend.status()
            print(f"{backend.name}: "
                  f"operational={status.operational}, "
                  f"pending_jobs={status.pending_jobs}")
    except Exception as e:
        print(f"Cannot check backends: {e}")

check_backend_status()
```

#### 3. Memory Issues

```python
# Optimize for large circuits
backend = sqx.QiskitBackend(
    backend_name='qasm_simulator',
    max_parallel_threads=1,  # Reduce memory usage
    shots=1000,
    optimization_level=3     # More optimization
)
```

### Performance Tips

1. **Use appropriate simulators**: `statevector_simulator` for small circuits, `qasm_simulator` for general use
2. **Optimize circuits**: Use `optimization_level=3` for hardware execution
3. **Manage shots**: Balance accuracy vs execution time
4. **Batch jobs**: Submit multiple circuits together when possible
5. **Monitor queue**: Check IBM Quantum queue status before submission

## Best Practices

### Code Organization

```python
# qiskit_helpers.py
class QiskitHelpers:
    @staticmethod
    def create_efficient_backend(n_qubits, use_hardware=False):
        if use_hardware:
            return sqx.QiskitBackend(
                backend_name='least_busy',
                optimization_level=3,
                shots=1024
            )
        else:
            simulator = 'statevector_simulator' if n_qubits <= 10 else 'qasm_simulator'
            return sqx.QiskitBackend(
                backend_name=simulator,
                optimization_level=2,
                shots=1000
            )
    
    @staticmethod
    def submit_with_retry(backend, circuit, max_retries=3):
        for attempt in range(max_retries):
            try:
                return backend.run(circuit)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
```

### Error Handling

```python
def robust_qiskit_execution(circuit, backend_name='qasm_simulator'):
    """Execute circuit with comprehensive error handling."""
    
    try:
        backend = sqx.QiskitBackend(backend_name=backend_name)
        result = backend.run(circuit, shots=1000)
        return result
        
    except Exception as e:
        print(f"Execution failed with {backend_name}: {e}")
        
        # Fallback to basic simulator
        if backend_name != 'qasm_simulator':
            print("Falling back to qasm_simulator...")
            return robust_qiskit_execution(circuit, 'qasm_simulator')
        else:
            raise e
```

## Resources

### Documentation
- **Qiskit Documentation**: https://qiskit.org/documentation/
- **IBM Quantum**: https://quantum-computing.ibm.com/
- **Qiskit Textbook**: https://qiskit.org/textbook/

### Hardware Access
- **IBM Quantum Network**: Apply for access to premium quantum systems
- **Qiskit Runtime**: Optimized quantum program execution
- **IBM Quantum Lab**: Cloud-based development environment

---

*Qiskit's integration with SuperQuantX provides seamless access to IBM's quantum ecosystem, from high-performance simulators to cutting-edge quantum processors. The unified API makes it easy to develop quantum applications while leveraging IBM's quantum expertise.*

!!! tip "Optimization"
    For best performance on IBM hardware, use `optimization_level=3` and minimize circuit depth. Consider using Qiskit Runtime for reduced latency.

!!! warning "Queue Times"
    Real IBM quantum hardware may have significant queue times. Check device status and consider using simulators for development.

!!! info "Hardware Access"
    IBM Quantum Network membership provides access to advanced quantum systems and priority queue access.