# Algorithms API Reference

SuperQuantX provides a comprehensive collection of quantum algorithms for machine learning, optimization, cryptography, and simulation. All algorithms support multiple quantum backends and provide unified interfaces for easy integration.

## Base Classes

### Base Algorithm

::: superquantx.algorithms.BaseQuantumAlgorithm
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

### Quantum Result

::: superquantx.algorithms.QuantumResult
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

## Machine Learning Algorithms

### Quantum Support Vector Machine

::: superquantx.algorithms.QuantumSVM
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

### Quantum Neural Network

::: superquantx.algorithms.QuantumNN
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

::: superquantx.algorithms.QuantumNeuralNetwork
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

### Hybrid Classifier

::: superquantx.algorithms.HybridClassifier
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

### Quantum Principal Component Analysis

::: superquantx.algorithms.QuantumPCA
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

### Quantum K-Means

::: superquantx.algorithms.QuantumKMeans
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

## Optimization Algorithms

### Variational Quantum Eigensolver

::: superquantx.algorithms.VQE
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

::: superquantx.algorithms.create_vqe_for_molecule
    handler: python
    options:
      docstring_style: google
      show_source: true

### Quantum Approximate Optimization Algorithm

::: superquantx.algorithms.QAOA
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

## Quantum Agents

### Base Quantum Agent

::: superquantx.algorithms.quantum_agents.QuantumAgent
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

### Specialized Agents

::: superquantx.algorithms.quantum_agents.QuantumPortfolioAgent
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

::: superquantx.algorithms.quantum_agents.QuantumOptimizationAgent
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

::: superquantx.algorithms.quantum_agents.QuantumClassificationAgent
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

## Examples and Usage Patterns

### Machine Learning Example

```python
import numpy as np
from sklearn.datasets import make_classification
import superquantx as sqx

# Generate sample data
X, y = make_classification(
    n_samples=100, 
    n_features=4, 
    n_classes=2, 
    random_state=42
)

# Split data
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# Create and train Quantum SVM
qsvm = sqx.QuantumSVM(
    backend='simulator',
    feature_map='ZFeatureMap',
    num_features=4
)

qsvm.fit(X_train, y_train)
predictions = qsvm.predict(X_test)
accuracy = qsvm.score(X_test, y_test)

print(f"Quantum SVM Accuracy: {accuracy:.3f}")
```

### VQE Molecule Example

```python
import superquantx as sqx

# Create VQE for H2 molecule
vqe = sqx.create_vqe_for_molecule(
    molecule='H2',
    bond_length=0.735,  # Angstroms
    backend='simulator'
)

# Find ground state
ground_energy = vqe.find_ground_state()
print(f"H2 Ground State Energy: {ground_energy:.6f} Ha")

# Get optimized parameters
optimal_params = vqe.get_optimal_parameters()
print(f"Optimal parameters: {optimal_params}")
```

### QAOA Optimization Example

```python
import superquantx as sqx

# Define Max-Cut problem
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]

qaoa = sqx.QAOA(
    problem_type='max_cut',
    graph_edges=edges,
    layers=3,
    backend='simulator'
)

# Solve optimization problem
solution = qaoa.solve()
optimal_value = qaoa.get_optimal_value()

print(f"Max-Cut Solution: {solution}")
print(f"Cut Value: {optimal_value}")
```

### Quantum Agent Example

```python
import superquantx as sqx

# Create quantum portfolio optimization agent
portfolio_agent = sqx.QuantumPortfolioAgent(
    backend='simulator',
    risk_tolerance=0.1,
    optimization_method='qaoa'
)

# Sample portfolio data
assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
returns = [0.12, 0.15, 0.08, 0.20]
risks = [0.05, 0.10, 0.03, 0.15]

# Optimize portfolio
optimal_weights = portfolio_agent.optimize_portfolio(
    assets=assets,
    expected_returns=returns,
    risk_estimates=risks,
    budget=10000
)

print("Optimal Portfolio:")
for asset, weight in zip(assets, optimal_weights):
    print(f"{asset}: {weight:.2%}")
```

## Best Practices

### Algorithm Selection

1. **For Classification**: Use `QuantumSVM` for small datasets, `QuantumNN` for complex patterns
2. **For Optimization**: Use `VQE` for chemistry problems, `QAOA` for combinatorial optimization
3. **For Autonomous Tasks**: Use specialized `QuantumAgent` implementations

### Backend Optimization

```python
# Choose appropriate backend for algorithm
algorithms_backends = {
    'QuantumSVM': 'pennylane',      # Best autodiff support
    'VQE': 'qiskit',                # Good hardware access
    'QAOA': 'cirq',                 # Flexible circuit construction
    'QuantumAgents': 'simulator',    # Fast prototyping
}

# Use backend-specific optimizations
qsvm = sqx.QuantumSVM(
    backend='pennylane',
    feature_map='ZZFeatureMap',     # More expressive for PennyLane
    optimization_level=2
)
```

### Performance Monitoring

```python
import time
import superquantx as sqx

# Benchmark different algorithms
def benchmark_algorithm(algorithm_class, X, y, **kwargs):
    start_time = time.time()
    
    algorithm = algorithm_class(**kwargs)
    algorithm.fit(X, y)
    predictions = algorithm.predict(X)
    
    execution_time = time.time() - start_time
    accuracy = algorithm.score(X, y)
    
    return {
        'accuracy': accuracy,
        'time': execution_time,
        'algorithm': algorithm_class.__name__
    }

# Compare algorithms
algorithms = [sqx.QuantumSVM, sqx.QuantumNN, sqx.HybridClassifier]
results = []

for algo in algorithms:
    result = benchmark_algorithm(algo, X_train, y_train, backend='simulator')
    results.append(result)
    print(f"{result['algorithm']}: {result['accuracy']:.3f} accuracy in {result['time']:.2f}s")
```

---

For complete algorithm implementations and advanced usage patterns, see:
- [Backend Integration Guide](../backends/)
- [Tutorial Examples](../tutorials/)
- [User Guide](../user-guide/)