# What is SuperQuantX?

SuperQuantX is an **experimental quantum AI research toolkit** that provides a unified API for quantum-agentic systems research. It bridges the gap between multiple quantum computing frameworks and makes quantum machine learning accessible to researchers, developers, and students.

## 🔬 The Vision

### Unifying Quantum Ecosystems

The quantum computing landscape is fragmented across different frameworks, each with unique strengths:

- **PennyLane**: Excellent for quantum machine learning and differentiable programming
- **Qiskit**: IBM's robust framework with extensive hardware access
- **Cirq**: Google's framework optimized for NISQ devices
- **Amazon Braket**: Cloud-based quantum computing with diverse hardware
- **TKET**: Cambridge Quantum Computing's compiler-focused approach
- **D-Wave Ocean**: Specialized for quantum annealing

**SuperQuantX solves this fragmentation** by providing one consistent API that works across all these frameworks.

### The Problem SuperQuantX Solves

```python
# Without SuperQuantX: Different APIs for each framework
import pennylane as qml
import qiskit
import cirq

# PennyLane approach
dev_pl = qml.device('default.qubit', wires=2)
@qml.qnode(dev_pl)
def pennylane_circuit():
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliZ(0))

# Qiskit approach
circuit_qiskit = qiskit.QuantumCircuit(2)
circuit_qiskit.h(0)
# ... different measurement setup

# Cirq approach
circuit_cirq = cirq.Circuit()
qubit = cirq.GridQubit(0, 0)
circuit_cirq.append(cirq.H(qubit))
# ... different execution model
```

```python
# With SuperQuantX: One API for everything
import superquantx as sqx

# Same code works with any gate-model backend
for backend_name in ['pennylane', 'qiskit', 'cirq']:
    backend = sqx.get_backend(backend_name)
    circuit = backend.create_circuit(2)
    circuit = backend.add_gate(circuit, 'H', 0)        # same gate names everywhere
    circuit = backend.add_gate(circuit, 'CNOT', [0, 1])
    circuit = backend.add_measurement(circuit)
    result = backend.execute_circuit(circuit, shots=1000)
    print(f"{backend_name}: {result['counts']}")
```

## 🧠 Core Concepts

### 1. Unified Backend Abstraction

SuperQuantX provides a common interface over different quantum computing frameworks:

```python
import superquantx as sqx

# All gate-model backends follow the same interface
backend = sqx.get_backend('pennylane')  # or 'qiskit', 'cirq', 'braket', 'tket', 'simulator'

# Create circuit
circuit = backend.create_circuit(3)

# Add gates (same names across all backends)
circuit = backend.add_gate(circuit, 'H', 0)           # Hadamard
circuit = backend.add_gate(circuit, 'CNOT', [0, 1])   # CNOT
circuit = backend.add_gate(circuit, 'RY', 2, [0.5])   # Y rotation

# Execute circuit
circuit = backend.add_measurement(circuit)
result = backend.execute_circuit(circuit, shots=1000)
counts = result['counts']
```

### 2. Quantum-Agentic Systems

SuperQuantX is designed for research in **quantum-agentic systems** - AI agents that leverage quantum computation:

```python
import numpy as np
import superquantx as sqx


# Example: a quantum neural network as the policy of a simple agent
class QuantumAgent:
    def __init__(self, backend='pennylane'):
        self.policy = sqx.QuantumNN(backend=backend, n_layers=2, max_epochs=10, seed=0)

    def learn(self, states, good_actions):
        self.policy.fit(states, good_actions)
        return self

    def decide_action(self, state):
        return int(self.policy.predict(np.atleast_2d(state))[0])


rng = np.random.default_rng(0)
states = rng.uniform(-1, 1, size=(30, 2))
actions = (states[:, 0] > 0).astype(int)        # move right when x > 0
agent = QuantumAgent().learn(states, actions)
print(agent.decide_action([0.8, 0.1]))
```

### 3. Research-First Design

Every feature in SuperQuantX is designed with research needs in mind:

- **Reproducibility**: Built-in seed management and deterministic execution
- **Experimentation**: Easy backend comparison and parameter sweeps
- **Flexibility**: Low-level circuit access combined with high-level algorithms
- **Educational**: Comprehensive documentation and examples

## 🚀 Key Features

### Unified Quantum Framework API

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=40, noise=0.1, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Switch backends without changing your code
results = {}
for backend_name in ['simulator', 'pennylane', 'qiskit', 'cirq']:
    qsvm = sqx.QuantumSVM(backend=backend_name, feature_map='ZFeatureMap')
    qsvm.fit(X_train, y_train)
    results[backend_name] = float(np.mean(qsvm.predict(X_test) == y_test))

print(f"Backend comparison: {results}")
```

### Built-in Quantum Machine Learning

Ready-to-use quantum ML algorithms:

```python
from superquantx.datasets import load_iris_quantum

X_train, X_test, y_train, y_test, _ = load_iris_quantum(n_features=2, encoding='angle')

# Quantum Support Vector Machine
qsvm = sqx.QuantumSVM(backend='pennylane', feature_map='ZZFeatureMap')
qsvm.fit(X_train, y_train)
predictions = qsvm.predict(X_test)

# Variational Quantum Eigensolver
H = sqx.Hamiltonian.from_dict({'ZZ': 1.0, 'XI': 0.5, 'IX': 0.5})
vqe = sqx.VQE(hamiltonian=H, backend='qiskit', seed=0)
ground_energy = vqe.find_ground_state()

# Quantum Neural Network
qnn = sqx.QuantumNN(backend='simulator', n_layers=2, learning_rate=0.1, max_epochs=10, seed=0)
qnn.fit(X_train[y_train < 2], y_train[y_train < 2])
print(f"VQE energy {ground_energy:.4f} (exact {H.ground_state_energy():.4f})")
```

### Advanced Circuit Building

Flexible circuit construction with optimization:

```python
# Build a layered circuit with the shared gate names
backend = sqx.get_backend('qiskit')
circuit = backend.create_circuit(4)

for q, angle in enumerate([0.1, 0.2, 0.3, 0.4]):      # RY layer
    circuit = backend.add_gate(circuit, 'RY', q, [angle])
for a, b in [(0, 1), (2, 3), (1, 2)]:                  # entangling layer
    circuit = backend.add_gate(circuit, 'CNOT', [a, b])

# Or use a library ansatz
ansatz = backend.create_ansatz('RealAmplitudes', 4, np.zeros(backend.ansatz_param_count('RealAmplitudes', 4, reps=1)), reps=1)

# Analysis with the native framework (here a qiskit.QuantumCircuit)
print(f"Depth: {circuit.depth()}, Gates: {circuit.size()}")
```

### Comprehensive Visualization

Built-in tools for understanding quantum systems:

```python
# Circuit visualization: the backend-independent description draws as text
circuit = sqx.QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()
print(circuit.draw())

# Result visualization: plot counts with matplotlib
import matplotlib.pyplot as plt

backend = sqx.get_backend('simulator')
native = backend.create_circuit(2)
native.h(0)
native.cx(0, 1)
native.measure_all()
counts = backend.run(native, shots=1000).get_counts()
plt.bar(list(counts), list(counts.values()))
plt.title("Bell state counts")
```

Bloch-sphere plots and `result.plot_*` helpers are not available in 0.3.0.

## 🔍 Architecture Overview

### Layered Design

SuperQuantX follows a layered architecture:

```
┌─────────────────────────────────────────────────┐
│                User Interface                    │
├─────────────────────────────────────────────────┤
│  Algorithms  │  Circuits  │  Visualization      │
├─────────────────────────────────────────────────┤
│              Backend Abstraction                │  
├─────────────────────────────────────────────────┤
│ PennyLane │ Qiskit │ Cirq │ Braket │ TKET       │
├─────────────────────────────────────────────────┤
│        Hardware/Simulators/Cloud Services       │
└─────────────────────────────────────────────────┘
```

#### Layer Details

1. **User Interface**: High-level API for researchers and developers
2. **Algorithm Layer**: Pre-built quantum ML and optimization algorithms
3. **Circuit Layer**: Quantum circuit construction and manipulation tools
4. **Backend Abstraction**: Unified interface over different frameworks
5. **Framework Layer**: Individual quantum computing frameworks
6. **Hardware Layer**: Quantum computers, simulators, and cloud services

### Backend Management

```python
# Backend discovery and capabilities
available = sqx.list_available_backends()
print("Available:", [name for name, info in available.items() if info['available']])

backend = sqx.get_backend('qiskit')
print(backend.capabilities)          # capability flags for this backend
print(backend.get_version_info())    # installed framework versions
```

## 🎯 Use Cases

### 1. Quantum Machine Learning Research

```python
# Compare ML algorithms across quantum backends
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=40, noise=0.1, random_state=0)
models = {
    'QuantumSVM': lambda b: sqx.QuantumSVM(backend=b, feature_map='ZFeatureMap'),
    'QuantumNN': lambda b: sqx.QuantumNN(backend=b, n_layers=2, max_epochs=10, seed=0),
}

results_matrix = {}
for algo_name, make in models.items():
    results_matrix[algo_name] = {}
    for backend_name in ['simulator', 'pennylane']:
        model = make(backend_name).fit(X, y)
        results_matrix[algo_name][backend_name] = float(np.mean(model.predict(X) == y))

print(results_matrix)
```

### 2. Quantum Algorithm Development

```python
import numpy as np


# Implement and test new quantum algorithms
class MyQuantumAlgorithm(sqx.BaseQuantumAlgorithm):
    """Learns a single RY angle so that P(1) matches a target probability."""

    def fit(self, X, y=None, **kwargs):
        target = float(np.mean(y))
        angles = np.linspace(0, np.pi, 50)
        self.theta_ = min(angles, key=lambda t: abs(self._p1(t) - target))
        self.is_fitted = True
        return self

    def predict(self, X, **kwargs):
        return np.full(len(X), self._p1(self.theta_))

    def _p1(self, theta):
        circuit = self.backend.create_circuit(1)
        self.backend.add_gate(circuit, 'RY', 0, [theta])
        self.backend.add_measurement(circuit)
        counts = self.backend.execute_circuit(circuit, shots=500)['counts']
        return counts.get('1', 0) / 500


algo = MyQuantumAlgorithm(backend='simulator', seed=0).fit(np.zeros((8, 1)), [1, 1, 0, 0, 0, 0, 0, 0])
print(f"P(1) learned: {algo.predict([[0]])[0]:.2f}  (target 0.25)")
```

### 3. Educational Quantum Computing

```python
# Interactive quantum education
class QuantumLesson:
    def __init__(self, title):
        self.title = title
        self.backend = sqx.get_backend('simulator')

    def demonstrate_superposition(self):
        circuit = self.backend.create_circuit(1)
        circuit.h(0)  # create superposition
        circuit.measure_all()

        counts = self.backend.run(circuit, shots=1000).get_counts()
        print(f"{self.title}: {counts}  (about half 0, half 1)")
        return counts


QuantumLesson("Superposition").demonstrate_superposition()
```

### 4. Hardware Benchmarking

!!! note "Requires credentials"
    Hardware backends need provider accounts (IBM Quantum, AWS, Quantinuum). The loop below runs on the local backends; add a hardware backend name once its credentials are set up.

```python
import time


def ghz_circuit(backend, n):
    circuit = backend.create_circuit(n)
    backend.add_gate(circuit, 'H', 0)
    for q in range(n - 1):
        backend.add_gate(circuit, 'CNOT', [q, q + 1])
    backend.add_measurement(circuit)
    return circuit


def benchmark_backends(names=('simulator', 'qiskit', 'cirq', 'pennylane'), n=3, shots=1000):
    """Run the same GHZ circuit everywhere: time taken and fraction of ideal outcomes."""
    results = {}
    for name in names:
        backend = sqx.get_backend(name)
        start = time.time()
        counts = backend.execute_circuit(ghz_circuit(backend, n), shots=shots)['counts']
        ideal = counts.get('0' * n, 0) + counts.get('1' * n, 0)
        results[name] = {'seconds': round(time.time() - start, 3), 'ideal_fraction': ideal / shots}
    return results


print(benchmark_backends())
```

## 🛠️ Development Philosophy

### Research-First Approach

1. **Reproducibility**: Every experiment can be exactly reproduced
2. **Flexibility**: Low-level control when needed, high-level convenience by default  
3. **Interoperability**: Seamless switching between quantum frameworks
4. **Education**: Clear documentation and examples for learning

### Code Example: Research Workflow

```python
import numpy as np
import superquantx as sqx

# Reproducible comparison: fixed dataset splits (random_state) and seeds
datasets = {
    'iris': sqx.datasets.load_iris_quantum,
    'wine': sqx.datasets.load_wine_quantum,
}
feature_maps = ['ZFeatureMap', 'ZZFeatureMap']

results = []
for dataset_name, loader in datasets.items():
    X_train, X_test, y_train, y_test, metadata = loader(
        n_features=4, encoding='angle', random_state=42
    )
    for feature_map in feature_maps:
        qsvm = sqx.QuantumSVM(backend='simulator', feature_map=feature_map, seed=42)
        qsvm.fit(X_train, y_train)
        results.append({
            'dataset': dataset_name,
            'feature_map': feature_map,
            'accuracy': float(np.mean(qsvm.predict(X_test) == y_test)),
        })

for row in results:
    print(row)
best = max(results, key=lambda r: r['accuracy'])
print("Best configuration:", best)
```

There is no `Experiment` tracker, `set_seed` or `cross_val_score` in 0.3.0; pass `seed=` to each algorithm and use your own logging (or `sqx.compare_to_classical` for cross-validated comparisons).

## 🌟 What Makes SuperQuantX Different?

### Compared to Individual Frameworks

| Feature | Individual Frameworks | SuperQuantX |
|---------|----------------------|-------------|
| **API Consistency** | Different for each | Unified across all |
| **Backend Switching** | Requires code rewrite | One line change |
| **Algorithm Library** | Framework-specific | Works with any backend |
| **Learning Curve** | Steep for each framework | Learn once, use everywhere |
| **Research Tools** | Basic | Advanced experiment tracking |

### Compared to Other Unification Attempts

- **More Comprehensive**: Covers more frameworks and features
- **Research-Focused**: Built specifically for quantum ML research
- **Actively Maintained**: Regular updates and community support
- **Production-Ready**: Well-tested and documented
- **Educational**: Extensive learning materials and examples

## 🔮 Future Roadmap

### Short Term (Next 6 months)
- Additional backend integrations (IonQ, Rigetti)
- Enhanced quantum ML algorithm library
- Improved visualization tools
- Performance optimizations

### Medium Term (6-12 months)
- Quantum error correction integration
- Advanced quantum neural network architectures
- Cloud deployment templates
- Quantum advantage benchmarking suite

### Long Term (1+ years)
- Quantum-classical hybrid computing
- Automated quantum circuit discovery
- Quantum reinforcement learning frameworks
- Industry-specific quantum applications

## 🤝 Community and Ecosystem

### Open Source Community

SuperQuantX thrives on community contributions:

- **Researchers**: Share algorithms and results
- **Developers**: Contribute backends and tools
- **Educators**: Create tutorials and examples
- **Students**: Learn and provide feedback

### Integration Ecosystem

SuperQuantX integrates with:

- **Machine Learning**: scikit-learn, TensorFlow, PyTorch
- **Scientific Computing**: NumPy, SciPy, matplotlib
- **Data Science**: pandas, Jupyter notebooks
- **Cloud Platforms**: AWS, Google Cloud, Azure
- **Development Tools**: Git, Docker, CI/CD pipelines

## 📚 Learning Resources

### Getting Started
- **[Quick Start Guide](../getting-started/quickstart.md)**: 5-minute introduction
- **[Installation Guide](../getting-started/installation.md)**: Detailed setup
- **[First Program](../getting-started/first-program.md)**: Build your first quantum app

### Deep Dive
- **[Quantum Basics Tutorial](../tutorials/basic-quantum.md)**: Learn quantum computing
- **[Quantum ML Tutorial](../tutorials/quantum-ml.md)**: Apply ML to quantum systems
- **[Backend Guides](../backends/pennylane.md)**: Framework-specific details

### Reference
- **[API Documentation](../api/core.md)**: Complete function reference
- **[Algorithm Library](algorithms.md)**: Built-in algorithms
- **[Backend Reference](backends.md)**: Supported frameworks

## 🎯 Ready to Start?

Choose your path based on your background:

### New to Quantum Computing
Start with our [Basic Quantum Tutorial](../tutorials/basic-quantum.md) to learn quantum fundamentals, then move to the [Quick Start Guide](../getting-started/quickstart.md).

### Familiar with Quantum Computing
Jump straight to the [Quick Start Guide](../getting-started/quickstart.md) and explore [Backend Integration](backends.md).

### Quantum ML Researcher
Check out the [Quantum ML Tutorial](../tutorials/quantum-ml.md) and [Algorithm Library](algorithms.md).

### Framework Developer
Read the [Architecture Guide](../development/architecture.md) and [Contributing Guidelines](../development/contributing.md).

---

!!! info "Research Software Notice"
    SuperQuantX is experimental research software designed for academic and research purposes. While we strive for reliability and performance, it is **not intended for production use** in critical systems.

!!! tip "Best Learning Approach"
    The best way to understand SuperQuantX is through hands-on experimentation. Start with simple examples and gradually explore more advanced features as you become comfortable with the platform.