# Utilities API Reference

SuperQuantX provides comprehensive utility modules for optimization, visualization, benchmarking, data handling, and command-line operations. This API reference covers all utility functions and classes.

## Optimization Utilities

### Circuit Optimization

::: superquantx.utils.optimization.optimize_circuit
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.utils.optimization.optimize_parameters
    handler: python
    options:
      docstring_style: google
      show_source: true

### Optimizers

::: superquantx.utils.optimization.gradient_descent
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.utils.optimization.adam_optimizer
    handler: python
    options:
      docstring_style: google
      show_source: true

## Visualization Utilities

### Result Visualization

::: superquantx.utils.visualization.visualize_results
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.utils.visualization.plot_optimization_history
    handler: python
    options:
      docstring_style: google
      show_source: true

### Quantum State Visualization

::: superquantx.utils.visualization.plot_circuit
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.utils.visualization.plot_quantum_state
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.utils.visualization.plot_bloch_sphere
    handler: python
    options:
      docstring_style: google
      show_source: true

## Benchmarking Utilities

### Algorithm Benchmarking

::: superquantx.utils.benchmarking.benchmark_algorithm
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.utils.benchmarking.benchmark_backend
    handler: python
    options:
      docstring_style: google
      show_source: true

### Performance Analysis

::: superquantx.utils.benchmarking.performance_metrics
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.utils.benchmarking.compare_algorithms
    handler: python
    options:
      docstring_style: google
      show_source: true

## Feature Mapping Utilities

### Quantum Feature Maps

::: superquantx.utils.feature_mapping.QuantumFeatureMap
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

::: superquantx.utils.feature_mapping.create_feature_map
    handler: python
    options:
      docstring_style: google
      show_source: true

### Specific Feature Maps

::: superquantx.utils.feature_mapping.pauli_feature_map
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.utils.feature_mapping.zz_feature_map
    handler: python
    options:
      docstring_style: google
      show_source: true

## Quantum Utilities

### Quantum Information Measures

::: superquantx.utils.quantum_utils.fidelity
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.utils.quantum_utils.trace_distance
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.utils.quantum_utils.quantum_mutual_information
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.utils.quantum_utils.entanglement_measure
    handler: python
    options:
      docstring_style: google
      show_source: true

## Classical Utilities

### Machine Learning Utilities

::: superquantx.utils.classical_utils.cross_validation
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.utils.classical_utils.hyperparameter_search
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.utils.classical_utils.model_selection
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.utils.classical_utils.data_splitting
    handler: python
    options:
      docstring_style: google
      show_source: true

## Datasets

### Quantum-Adapted Classical Datasets

::: superquantx.datasets.load_iris_quantum
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.datasets.load_wine_quantum
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.datasets.load_digits_quantum
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.datasets.load_breast_cancer_quantum
    handler: python
    options:
      docstring_style: google
      show_source: true

### Synthetic Data Generators

::: superquantx.datasets.generate_classification_data
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.datasets.generate_regression_data
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.datasets.generate_clustering_data
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.datasets.generate_portfolio_data
    handler: python
    options:
      docstring_style: google
      show_source: true

### Molecular Datasets

::: superquantx.datasets.load_molecule
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.datasets.load_h2_molecule
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.datasets.load_lih_molecule
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.datasets.load_beh2_molecule
    handler: python
    options:
      docstring_style: google
      show_source: true

### Data Preprocessing

::: superquantx.datasets.QuantumFeatureEncoder
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

::: superquantx.datasets.AmplitudeEncoder
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

::: superquantx.datasets.AngleEncoder
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

::: superquantx.datasets.normalize_quantum_data
    handler: python
    options:
      docstring_style: google
      show_source: true

## Command Line Interface

### Main CLI Application

::: superquantx.cli.main
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.cli.create_app
    handler: python
    options:
      docstring_style: google
      show_source: true

### CLI Commands

::: superquantx.cli.run_algorithm
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.cli.list_algorithms
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.cli.list_backends
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.cli.benchmark
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.cli.configure
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.cli.info
    handler: python
    options:
      docstring_style: google
      show_source: true

## Usage Examples

### Optimization Workflow

```python
import superquantx as sqx
import numpy as np

# Create parameterized circuit
backend = sqx.get_backend('simulator')
circuit = backend.create_circuit(4)

# Add parameterized gates
params = sqx.Parameter('theta', shape=(8,))
for i in range(4):
    circuit = backend.add_gate(circuit, 'ry', i, [params[i]])
    
for i in range(3):
    circuit = backend.add_gate(circuit, 'cx', [i, i+1])
    
for i in range(4):
    circuit = backend.add_gate(circuit, 'ry', i, [params[i+4]])

# Define cost function
def cost_function(parameters):
    bound_circuit = circuit.bind_parameters({params: parameters})
    result = backend.execute_circuit(bound_circuit)
    # Calculate some cost based on measurement results
    counts = result['counts']
    return -sum(int(bitstring, 2) * count for bitstring, count in counts.items())

# Optimize parameters
from superquantx.utils import optimize_parameters, adam_optimizer

optimal_params = optimize_parameters(
    cost_function=cost_function,
    initial_params=np.random.random(8) * 2 * np.pi,
    optimizer=adam_optimizer(learning_rate=0.01),
    max_iterations=100
)

print(f"Optimal parameters: {optimal_params}")
```

### Visualization Example

```python
from superquantx.utils import visualize_results, plot_optimization_history

# Execute circuit with optimal parameters
final_circuit = circuit.bind_parameters({params: optimal_params})
result = backend.execute_circuit(final_circuit, shots=1000)

# Visualize measurement results
fig = visualize_results(
    result['counts'],
    title='Optimized Circuit Results',
    plot_type='histogram'
)
fig.show()

# Plot optimization history
history = {
    'iteration': list(range(100)),
    'cost': [cost_function(p) for p in optimization_history],
    'gradient_norm': [np.linalg.norm(g) for g in gradient_history]
}

plot_optimization_history(history, metrics=['cost', 'gradient_norm'])
```

### Benchmarking Example

```python
from superquantx.utils import benchmark_algorithm, compare_algorithms

# Define test problem
def test_classification_problem():
    X, y = sqx.datasets.load_iris_quantum()
    return X[:100], y[:100]  # Use subset for faster benchmarking

# Benchmark single algorithm
qsvm_metrics = benchmark_algorithm(
    algorithm_class=sqx.QuantumSVM,
    problem_generator=test_classification_problem,
    backend='simulator',
    n_trials=5,
    metrics=['accuracy', 'training_time', 'inference_time']
)

print("QSVM Benchmarks:")
for metric, value in qsvm_metrics.items():
    print(f"  {metric}: {value:.4f} ± {qsvm_metrics[f'{metric}_std']:.4f}")

# Compare multiple algorithms
algorithms = {
    'QSVM': sqx.QuantumSVM,
    'QNN': sqx.QuantumNN,
    'Hybrid': sqx.HybridClassifier
}

comparison = compare_algorithms(
    algorithms=algorithms,
    problem_generator=test_classification_problem,
    backend='simulator',
    n_trials=3
)

# Print comparison table
print("\nAlgorithm Comparison:")
print("Algorithm | Accuracy | Training Time | Inference Time")
print("-" * 50)
for algo_name, metrics in comparison.items():
    print(f"{algo_name:9} | {metrics['accuracy']:.3f}    | {metrics['training_time']:.3f}s        | {metrics['inference_time']:.3f}s")
```

### Feature Mapping Example

```python
from superquantx.utils import QuantumFeatureMap, zz_feature_map

# Create custom feature map
feature_map = QuantumFeatureMap(
    feature_dimension=4,
    reps=2,
    entanglement='linear',
    rotation_gates=['ry', 'rz']
)

# Sample data
X = np.random.random((10, 4))

# Encode data into quantum circuits
encoded_circuits = []
for x in X:
    circuit = feature_map.encode(x, backend=backend)
    encoded_circuits.append(circuit)

# Pre-built ZZ feature map
zz_map = zz_feature_map(feature_dimension=4, reps=2)
zz_circuit = zz_map.encode(X[0], backend=backend)

# Calculate quantum kernel matrix
def quantum_kernel(x1, x2, feature_map, backend):
    """Calculate quantum kernel between two data points."""
    circuit1 = feature_map.encode(x1, backend)
    circuit2 = feature_map.encode(x2, backend)
    
    # Create kernel circuit: |0⟩ -> U†(x2) U(x1) |0⟩
    kernel_circuit = circuit1.compose(circuit2.inverse())
    
    # Measure overlap
    result = backend.execute_circuit(kernel_circuit)
    prob_zero = result['counts'].get('0' * kernel_circuit.n_qubits, 0) / sum(result['counts'].values())
    
    return prob_zero

# Compute kernel matrix
kernel_matrix = np.zeros((len(X), len(X)))
for i in range(len(X)):
    for j in range(i, len(X)):
        kernel_val = quantum_kernel(X[i], X[j], feature_map, backend)
        kernel_matrix[i, j] = kernel_val
        kernel_matrix[j, i] = kernel_val

print(f"Quantum kernel matrix shape: {kernel_matrix.shape}")
```

### Dataset Usage Example

```python
# Load quantum-adapted datasets
X_iris, y_iris = sqx.datasets.load_iris_quantum()
X_wine, y_wine = sqx.datasets.load_wine_quantum()

print(f"Iris dataset: {X_iris.shape} features, {len(set(y_iris))} classes")
print(f"Wine dataset: {X_wine.shape} features, {len(set(y_wine))} classes")

# Generate synthetic data
X_synthetic, y_synthetic = sqx.datasets.generate_classification_data(
    n_samples=200,
    n_features=4,
    n_classes=3,
    n_informative=3,
    n_clusters_per_class=1,
    random_state=42
)

# Portfolio data for financial applications
portfolio_data = sqx.datasets.generate_portfolio_data(
    n_assets=10,
    n_time_periods=100,
    correlation_structure='block',
    volatility_regime='changing'
)

print(f"Portfolio returns shape: {portfolio_data['returns'].shape}")
print(f"Risk factors: {list(portfolio_data['risk_factors'].keys())}")

# Molecular datasets for quantum chemistry
h2_data = sqx.datasets.load_h2_molecule(bond_length=0.735)
print(f"H2 molecule: {h2_data['n_qubits']} qubits, {h2_data['n_orbitals']} orbitals")
print(f"Ground state energy: {h2_data['ground_energy']:.6f} Ha")
```

### Data Preprocessing Example

```python
from superquantx.datasets import QuantumFeatureEncoder, AmplitudeEncoder, AngleEncoder

# Amplitude encoding
amplitude_encoder = AmplitudeEncoder()
X_normalized = amplitude_encoder.fit_transform(X_iris)

print(f"Original range: [{X_iris.min():.3f}, {X_iris.max():.3f}]")
print(f"Encoded range: [{X_normalized.min():.3f}, {X_normalized.max():.3f}]")

# Angle encoding
angle_encoder = AngleEncoder(encoding_type='linear')
X_angles = angle_encoder.fit_transform(X_iris)

print(f"Angle encoding shape: {X_angles.shape}")
print(f"Angle range: [{X_angles.min():.3f}, {X_angles.max():.3f}]")

# Quantum feature encoding with dimensionality reduction
quantum_encoder = QuantumFeatureEncoder(
    target_dimension=8,  # Reduce to 8 features for quantum circuit
    encoding_method='pca',
    normalization='standard'
)

X_quantum = quantum_encoder.fit_transform(X_iris)
print(f"Quantum encoding: {X_iris.shape} -> {X_quantum.shape}")
```

### Command Line Usage

```bash
# List available algorithms
superquantx list-algorithms

# List available backends  
superquantx list-backends

# Run algorithm from command line
superquantx run-algorithm QSVM \
    --data iris \
    --backend simulator \
    --feature-map ZZFeatureMap \
    --shots 1000 \
    --output results.json

# Benchmark algorithms
superquantx benchmark \
    --algorithms QSVM,QNN,HybridClassifier \
    --dataset wine \
    --backend simulator \
    --trials 5 \
    --output benchmark_results.csv

# Configure SuperQuantX
superquantx configure \
    --backend-preference "pennylane,qiskit,simulator" \
    --default-shots 1024 \
    --optimization-level 2

# Get system information
superquantx info --backends --versions --capabilities
```

### Quantum Information Analysis

```python
from superquantx.utils import fidelity, trace_distance, entanglement_measure

# Create two quantum states
backend = sqx.get_backend('simulator')

# Bell state
bell_circuit = backend.create_circuit(2)
bell_circuit = backend.add_gate(bell_circuit, 'h', 0)
bell_circuit = backend.add_gate(bell_circuit, 'cx', [0, 1])
bell_state = backend.get_statevector(bell_circuit)

# Random state  
random_circuit = backend.create_circuit(2)
random_circuit = backend.add_gate(random_circuit, 'ry', 0, [np.pi/3])
random_circuit = backend.add_gate(random_circuit, 'rz', 1, [np.pi/4])
random_state = backend.get_statevector(random_circuit)

# Calculate quantum information measures
state_fidelity = fidelity(bell_state, random_state)
trace_dist = trace_distance(bell_state, random_state)
entanglement = entanglement_measure(bell_state)

print(f"Fidelity: {state_fidelity:.4f}")
print(f"Trace distance: {trace_dist:.4f}")
print(f"Entanglement (Bell state): {entanglement:.4f}")
```

## Best Practices

### Optimization Guidelines

1. **Start Simple**: Begin with basic optimizers before advanced methods
2. **Monitor Convergence**: Track optimization metrics throughout training
3. **Parameter Initialization**: Use informed initial parameter guesses
4. **Early Stopping**: Implement convergence criteria to avoid overfitting

### Visualization Standards

1. **Consistent Styling**: Use consistent color schemes and layouts
2. **Clear Labels**: Always include axis labels and titles
3. **Error Bars**: Show confidence intervals when appropriate
4. **Interactive Plots**: Use interactive visualizations for complex data

### Benchmarking Protocol

1. **Multiple Trials**: Run multiple independent trials for statistical significance
2. **Controlled Environment**: Fix random seeds for reproducible results
3. **Baseline Comparison**: Always compare against classical baselines
4. **Resource Tracking**: Monitor computational resources (time, memory, shots)

### Data Handling

1. **Preprocessing**: Always preprocess data appropriately for quantum algorithms
2. **Validation**: Use proper train/validation/test splits
3. **Feature Scaling**: Normalize features to appropriate ranges
4. **Dimensionality**: Consider quantum-appropriate feature dimensions

---

For additional examples and advanced usage patterns, see:
- [Optimization Tutorial](../tutorials/optimization.md)
- [Visualization Guide](../user-guide/visualization.md)
- [Benchmarking Best Practices](../user-guide/benchmarking.md)
- [Data Preprocessing Guide](../user-guide/data-preprocessing.md)