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

All snippets below run on SuperQuantX 0.3.0 with the built-in simulator.

### Quantum SVM

```python
import numpy as np
from sklearn.datasets import make_classification
import superquantx as sqx

X, y = make_classification(
    n_samples=60, n_features=4, n_informative=3, n_redundant=0, random_state=0
)
X_train, X_test, y_train, y_test = X[:45], X[45:], y[:45], y[45:]

qsvm = sqx.QuantumSVM(backend="simulator", feature_map="ZFeatureMap")
qsvm.fit(X_train, y_train)
print(float(np.mean(qsvm.predict(X_test) == y_test)))
```

### VQE for a Molecule

```python
import superquantx as sqx

vqe = sqx.create_vqe_for_molecule(
    molecule_name="H2",
    bond_distance=0.735,
    backend="simulator",
    maxiter=40,
)
energy = vqe.find_ground_state()
print(f"H2 ground-state energy: {energy:.6f} Ha")
```

### QAOA MaxCut

```python
import numpy as np
import superquantx as sqx

adjacency = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [0, 1, 1, 0],
], dtype=float)

qaoa = sqx.QAOA(backend="simulator", p=1, maxiter=30, n_restarts=1)
result = qaoa.solve(adjacency)
print(result["bitstring"], result["cut_value"], result["classical_optimum"])
```

### Quantum Portfolio Agent

```python
import superquantx as sqx

returns, cov, _, meta = sqx.generate_portfolio_data(
    n_assets=4, n_scenarios=40, random_state=0
)
agent = sqx.QuantumPortfolioAgent(
    backend="simulator", n_select=2, maxiter=40, n_restarts=1
)
agent.fit(returns)
result = agent.solve()
print(result.result["selected_assets"], result.result["matches_classical_optimum"])
```

### Comparing Algorithms

```python
import time
import numpy as np
from sklearn.datasets import make_classification
import superquantx as sqx

X, y = make_classification(
    n_samples=40, n_features=2, n_informative=2, n_redundant=0, random_state=0
)

def time_fit(ctor, **kwargs):
    model = ctor(**kwargs)
    t0 = time.time()
    model.fit(X, y)
    return time.time() - t0, float(np.mean(model.predict(X) == y))

for name, ctor, kwargs in [
    ("QuantumSVM", sqx.QuantumSVM, {"backend": "simulator", "feature_map": "ZFeatureMap"}),
    ("QuantumNN", sqx.QuantumNN, {"backend": "simulator", "n_layers": 1, "max_epochs": 8, "seed": 0}),
]:
    secs, acc = time_fit(ctor, **kwargs)
    print(f"{name}: acc={acc:.3f} in {secs:.2f}s")
```

---

For complete algorithm implementations and advanced usage patterns, see:

- [Backend Integration Guide](../backends/simulator.md)
- [Tutorial Examples](../tutorials/basic-quantum.md)
- [User Guide](../user-guide/overview.md)
