# Circuits API Reference

SuperQuantX provides comprehensive quantum circuit construction, manipulation, and execution capabilities. This API reference covers quantum circuits, gates, measurements, and noise modeling.

## Quantum Circuits

### Quantum Circuit

::: superquantx.circuits.QuantumCircuit
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

### Quantum Gate

::: superquantx.circuits.QuantumGate
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

### Classical Register

::: superquantx.circuits.ClassicalRegister
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

## Quantum Gates

### Gate Matrix Library

::: superquantx.gates.GateMatrix
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

### Pauli String Operations

::: superquantx.gates.PauliString
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

### Hamiltonian Operations

::: superquantx.gates.Hamiltonian
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

## Measurements

### Measurement Result

::: superquantx.measurements.MeasurementResult
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

### Quantum Measurement

::: superquantx.measurements.QuantumMeasurement
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

### Result Analyzer

::: superquantx.measurements.ResultAnalyzer
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

## Noise Models

### Noise Channel Base

::: superquantx.noise.NoiseChannel
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

### Specific Noise Channels

::: superquantx.noise.BitFlipChannel
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

::: superquantx.noise.PhaseFlipChannel
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

::: superquantx.noise.DepolarizingChannel
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

::: superquantx.noise.AmplitudeDampingChannel
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

### Noise Model

::: superquantx.noise.NoiseModel
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

## Usage Examples

### Basic Circuit Construction

```python
import superquantx as sqx
import numpy as np

# Create quantum circuit
circuit = sqx.QuantumCircuit(n_qubits=3, n_classical=3)

# Add quantum gates
circuit.h(0)                    # Hadamard gate
circuit.cx(0, 1)               # CNOT gate
circuit.ry(np.pi/4, 2)         # Y-rotation

# Add measurements
circuit.measure(0, 0)          # Measure qubit 0 to classical bit 0
circuit.measure_all()          # Measure all qubits

print(circuit)
```

### Advanced Gate Operations

```python
from superquantx.gates import GateMatrix, PauliString

# Custom rotation gates
theta = np.pi/3
rx_gate = GateMatrix.rx(theta)
ry_gate = GateMatrix.ry(theta) 
rz_gate = GateMatrix.rz(theta)

# Pauli string operations
pauli_str = PauliString('XYZX')  # X⊗Y⊗Z⊗X
print(f"Pauli string: {pauli_str}")
print(f"Matrix shape: {pauli_str.to_matrix().shape}")

# Hamiltonian construction
from superquantx.gates import Hamiltonian, PauliString

# Create Heisenberg model Hamiltonian
hamiltonian = Hamiltonian.heisenberg_model(num_qubits=4, Jx=1.0, Jy=1.0, Jz=1.0)
print(f"Hamiltonian: {hamiltonian}")

# Time evolution
time_evolution_op = hamiltonian.time_evolution(time=0.1)
circuit.unitary(time_evolution_op, range(4))
```

### Measurement Analysis

```python
# Execute circuit and analyze results
backend = sqx.get_backend('simulator')
result = backend.execute_circuit(circuit, shots=1000)

# Create measurement result object
measurement = sqx.MeasurementResult(
    counts=result['counts'],
    shots=result['shots'],
    metadata=result.get('metadata', {})
)

# Analyze measurements
print(f"Most frequent outcome: {measurement.most_frequent}")
print(f"Measurement probabilities: {measurement.probabilities}")

# Marginal analysis
marginal = measurement.marginal_counts([0, 1])  # Marginalize over qubits 0,1
print(f"Marginal counts: {marginal}")

# Statistical analysis
entropy = measurement.measurement_entropy()
print(f"Measurement entropy: {entropy:.4f}")
```

### Expectation Value Calculations

```python
from superquantx.measurements import QuantumMeasurement
from superquantx.gates import PauliString

# Create measurement system
measurement = QuantumMeasurement(backend="simulator")

# Define observable
observable = 'ZZ'  # Z⊗Z measurement

# Calculate expectation value
exp_value = measurement.measure_observable(
    circuit=circuit,
    observable=observable,
    shots=5000
)

print(f"⟨ZZ⟩ = {exp_value:.4f}")

# Multiple observables
observables = ['XX', 'YY', 'ZZ']

exp_values = []
for obs in observables:
    val = measurement.measure_observable(circuit, obs, shots=5000)
    exp_values.append(val)
    print(f"⟨{obs}⟩ = {val:.4f}")
```

### Quantum State Tomography

```python
from superquantx.measurements import QuantumMeasurement

# Create measurement system
measurement = QuantumMeasurement(backend="simulator")

# Perform tomography measurements
tomography_results = measurement.tomography_measurements(
    circuit=circuit,
    qubits=[0, 1],  # Reconstruct 2-qubit state
    shots_per_measurement=1000
)

# Reconstruct density matrix
density_matrix = measurement.reconstruct_state(tomography_results)
print(f"Reconstructed density matrix:\n{density_matrix}")

# Calculate fidelity with target state
bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)  # |Φ⁺⟩
fidelity = measurement.fidelity(density_matrix, bell_state)

print(f"Fidelity with |Φ⁺⟩: {fidelity:.4f}")
print(f"State purity: {np.trace(density_matrix @ density_matrix).real:.4f}")
```

### Noise Modeling

```python
from superquantx.noise import (
    NoiseModel, BitFlipChannel, PhaseFlipChannel, 
    DepolarizingChannel, AmplitudeDampingChannel
)

# Create noise channels
bit_flip = BitFlipChannel(probability=0.01)
phase_flip = PhaseFlipChannel(probability=0.01)
depolarizing = DepolarizingChannel(probability=0.02)
amplitude_damping = AmplitudeDampingChannel(probability=0.05)

# Combine into noise model
noise_model = NoiseModel()

# Add gate-specific noise
noise_model.add_gate_noise('X', bit_flip)
noise_model.add_gate_noise('H', depolarizing)
noise_model.add_gate_noise('CNOT', phase_flip)

# Add readout noise
noise_model.add_readout_noise(
    p0_given_1=0.02,  # Probability of measuring 0 given |1⟩
    p1_given_0=0.03   # Probability of measuring 1 given |0⟩
)

# Apply noise model to backend
noisy_backend = sqx.get_backend('simulator', noise_model=noise_model)

# Execute noisy simulation
noisy_result = noisy_backend.execute_circuit(circuit, shots=1000)
print("Noisy simulation results:", noisy_result['counts'])
```

### Error Mitigation

```python
from superquantx.measurements import ResultAnalyzer

# Zero-noise extrapolation
noise_levels = [1, 2, 3]
measurement_results = []

# Collect measurements at different noise levels
for noise_level in noise_levels:
    # Create noisy backend (implementation dependent)
    result = measurement.measure_circuit(circuit, shots=1000)
    measurement_results.append(result)

# Perform extrapolation
zero_noise_value, fit_info = ResultAnalyzer.error_mitigation_zero_noise_extrapolation(
    noise_levels=noise_levels,
    measurement_results=measurement_results,
    observable='Z'
)

print(f"Zero-noise extrapolated value: {zero_noise_value:.4f}")
print(f"Fit R²: {fit_info.get('r_squared', 'N/A'):.4f}")

# Readout error mitigation
calibration_results = {"0": result, "1": result}  # Calibration measurements
corrected_result = ResultAnalyzer.readout_error_mitigation(
    calibration_results=calibration_results,
    measurement_result=result
)
print(f"Corrected counts: {corrected_result.counts}")
```

### Circuit Optimization

```python
from superquantx.circuits import CircuitOptimizer

# Create optimizer
optimizer = CircuitOptimizer(
    optimization_level=2,
    target_backend=backend
)

# Optimize circuit
optimized_circuit = optimizer.optimize(circuit)

print(f"Original circuit: {circuit.depth()} depth, {circuit.count_ops()} gates")
print(f"Optimized circuit: {optimized_circuit.depth()} depth, {optimized_circuit.count_ops()} gates")

# Specific optimizations
optimized_circuit = optimizer.apply_passes([
    'RemoveRedundantGates',
    'CombineRotations',
    'CancelInverses',
    'CommuteThroughClifford'
])
```

### Parameterized Circuits

```python
from superquantx.circuits import Parameter, ParameterVector

# Create parameters
theta = Parameter('theta')
phi_vec = ParameterVector('phi', 3)

# Build parameterized circuit
param_circuit = sqx.QuantumCircuit(3)

param_circuit.ry(theta, 0)
param_circuit.rx(phi_vec[0], 1)
param_circuit.rz(phi_vec[1], 2)
param_circuit.cx(0, 1)
param_circuit.cry(phi_vec[2], 1, 2)

# Bind parameters
bound_circuit = param_circuit.bind_parameters({
    theta: np.pi/4,
    phi_vec: [np.pi/3, np.pi/6, np.pi/2]
})

# Execute bound circuit
result = backend.execute_circuit(bound_circuit)
```

### Quantum Circuit Libraries

```python
from superquantx.circuits.library import (
    EfficientSU2,
    RealAmplitudes,
    ZZFeatureMap,
    PauliFeatureMap
)

# Variational ansatz circuits
ansatz1 = EfficientSU2(num_qubits=4, reps=3)
ansatz2 = RealAmplitudes(num_qubits=4, reps=2)

# Feature map circuits for quantum machine learning
feature_map1 = ZZFeatureMap(feature_dimension=4, reps=2)
feature_map2 = PauliFeatureMap(feature_dimension=4, reps=1)

# Combine feature map and ansatz
ml_circuit = feature_map1.compose(ansatz1)

# Bind training data
training_data = [0.1, 0.5, -0.3, 0.8]
bound_ml_circuit = ml_circuit.bind_parameters({
    **feature_map1.parameter_dict(training_data),
    **ansatz1.random_parameters()
})
```

### Circuit Simulation and Debugging

```python
from superquantx.circuits import CircuitSimulator

# Create detailed simulator
sim = CircuitSimulator(backend='statevector')

# Step-by-step execution
sim.initialize_circuit(circuit)

for step, gate in enumerate(circuit.data):
    print(f"Step {step}: Applying {gate}")
    
    # Apply gate
    sim.apply_gate(gate)
    
    # Get current state
    current_state = sim.get_statevector()
    print(f"  Statevector norm: {np.linalg.norm(current_state):.6f}")
    
    # Check entanglement
    entanglement = sim.calculate_entanglement()
    print(f"  Entanglement entropy: {entanglement:.4f}")

# Final state analysis
final_state = sim.get_statevector()
final_density_matrix = sim.get_density_matrix()

print("Final state analysis:")
print(f"  Purity: {np.trace(final_density_matrix @ final_density_matrix).real:.4f}")
print(f"  Von Neumann entropy: {sim.von_neumann_entropy():.4f}")
```

## Performance Optimization

### Circuit Compilation

```python
from superquantx.circuits import QuantumCompiler

# Create compiler for specific backend
compiler = QuantumCompiler(target_backend=backend)

# Compile circuit
compiled_circuit = compiler.compile(
    circuit=circuit,
    optimization_level=3,
    scheduling='as_late_as_possible',
    layout='dense'
)

# Analysis compilation results
print(f"Compilation report:")
print(f"  Original gates: {circuit.count_ops()}")
print(f"  Compiled gates: {compiled_circuit.count_ops()}")
print(f"  Gate reduction: {(1 - compiled_circuit.count_ops()/circuit.count_ops())*100:.1f}%")
```

### Batch Circuit Execution

```python
# Execute multiple circuits efficiently
circuits = [create_random_circuit(n_qubits=4) for _ in range(10)]

# Parallel execution
results = backend.execute_circuits_parallel(
    circuits=circuits,
    shots=1000,
    max_workers=4
)

# Batch analysis
success_rate = sum(1 for r in results if r['success']) / len(results)
avg_execution_time = np.mean([r.get('execution_time', 0) for r in results])

print(f"Batch execution: {success_rate*100:.1f}% success rate")
print(f"Average execution time: {avg_execution_time:.3f}s")
```

## Best Practices

### Circuit Construction

1. **Use Native Gates**: Prefer gates supported by target backend
2. **Minimize Depth**: Reduce circuit depth for better fidelity
3. **Optimize Layout**: Consider qubit connectivity constraints
4. **Parameter Management**: Use symbolic parameters for flexibility

### Measurement Strategy

1. **Statistical Significance**: Use sufficient shots for reliable statistics
2. **Marginal Analysis**: Focus on relevant qubit subsets
3. **Error Bars**: Always include measurement uncertainties
4. **Baseline Comparison**: Compare with theoretical expectations

### Noise Handling

1. **Realistic Models**: Use device-specific noise parameters
2. **Error Mitigation**: Apply appropriate mitigation techniques
3. **Validation**: Compare noisy vs. noiseless simulations
4. **Calibration**: Regularly update noise models from hardware data

---

For more detailed examples and tutorials, see:
- [Circuit Construction Tutorial](../tutorials/circuits.md)
- [Noise Modeling Guide](../user-guide/noise.md)
- [Backend Integration](../backends/)