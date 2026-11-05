# Quantum Algorithms

SuperQuantX provides quantum algorithms for machine learning and optimisation, each with a classical check. This guide shows how to use them, how to build textbook algorithms from gates, and how to write your own algorithm class.

Every example on this page runs on SuperQuantX 0.3.0 with the built-in `'simulator'` backend, except the annealing example, which needs the `ocean` extra (`uv add "superquantx[ocean]==0.3.0"`).

## 🎯 What ships in 0.3.0

| Category | Classes and functions |
|---|---|
| Machine learning | `QuantumSVM`, `QuantumNN`, `QuantumKMeans`, `QuantumPCA` (`HybridClassifier` has a known issue, see below) |
| Optimisation | `VQE`, `QAOA`, `create_vqe_for_molecule`, `OceanBackend.solve_qubo` / `solve_ising` / `solve_max_cut` |
| Physics | `Hamiltonian`, `PauliString` (exact ground state, time evolution, expectation values) |
| Evaluation | `compare_to_classical`, `benchmark_algorithm` |
| Agents | see the [Agents API](../api/agents.md) |

!!! note "Not available as library classes"
    There are no `QuantumFourierTransform`, `QuantumPhaseEstimation`, `QuantumTeleportation`, `GroversAlgorithm`, `QuantumWalk`, `QuantumGAN`, `VariationalQuantumClassifier`, `VariationalQuantumSimulator`, `BB84Protocol`, `ShorsAlgorithm`, `MaxCutProblem` or `QuantumAnnealer` classes in 0.3.0. The section below builds several textbook algorithms directly from gates instead.

## 🔬 Textbook algorithms built from gates

The simulator circuit has gate methods (`h`, `x`, `y`, `z`, `rx`, `ry`, `rz`, `cx`, `cz`) and `apply_gate(matrix, qubits)` for any other unitary. `backend.statevector_big_endian(circuit)` returns the state with qubit 0 as the most significant bit.

### Quantum Fourier Transform (QFT)

```python
import numpy as np
import superquantx as sqx

backend = sqx.get_backend('simulator')


def cphase(theta):
    """Controlled phase gate as a 4x4 matrix."""
    return np.diag([1, 1, 1, np.exp(1j * theta)])


SWAP = np.eye(4)[[0, 2, 1, 3]]


def qft(circuit, n):
    for j in range(n):
        circuit.h(j)
        for k in range(j + 1, n):
            circuit.apply_gate(cphase(np.pi / 2 ** (k - j)), [k, j])
    for j in range(n // 2):
        circuit.apply_gate(SWAP, [j, n - 1 - j])
    return circuit


n = 3
circuit = backend.create_circuit(n)
circuit.x(2)          # prepare |001> = basis state 1
qft(circuit, n)

state = backend.statevector_big_endian(circuit)
expected = np.exp(2j * np.pi * 1 * np.arange(2 ** n) / 2 ** n) / np.sqrt(2 ** n)
print("QFT output phases (units of 2pi):", np.round(np.angle(state) / (2 * np.pi) % 1, 3))
print("Matches the DFT of |1>:", np.allclose(state, expected))
```

### Quantum Phase Estimation (QPE)

Estimate the phase of the T gate, whose eigenvalue on |1> is exp(2 pi i / 8). Three counting qubits recover the phase 1/8 exactly. This reuses `cphase` and `SWAP` from the QFT example.

```python
def inverse_qft(circuit, qubits):
    n = len(qubits)
    for j in range(n // 2):
        circuit.apply_gate(SWAP, [qubits[j], qubits[n - 1 - j]])
    for j in reversed(range(n)):
        for k in reversed(range(j + 1, n)):
            circuit.apply_gate(cphase(-np.pi / 2 ** (k - j)), [qubits[k], qubits[j]])
        circuit.h(qubits[j])


phase = 1 / 8          # T gate
counting = [0, 1, 2]
target = 3

circuit = backend.create_circuit(4)
circuit.x(target)      # eigenstate |1>
for q in counting:
    circuit.h(q)
for i, q in enumerate(counting):
    power = 2 ** (len(counting) - 1 - i)    # qubit 0 is the most significant bit
    circuit.apply_gate(cphase(2 * np.pi * phase * power), [q, target])
inverse_qft(circuit, counting)

probs = np.abs(backend.statevector_big_endian(circuit)) ** 2
best = format(int(np.argmax(probs)), '04b')
estimate = int(best[:3], 2) / 2 ** 3
print(f"Most likely outcome {best}, estimated phase {estimate} (true {phase})")
```

### Quantum teleportation

This version uses the deferred-measurement form: the classically controlled corrections become controlled gates, so the whole protocol is one unitary circuit and fidelity can be checked from the statevector.

```python
theta, phi = 1.1, 0.4

# The state to teleport, prepared on its own for reference
ref = backend.create_circuit(1)
ref.ry(theta, 0)
ref.rz(phi, 0)
psi = backend.statevector_big_endian(ref)

circuit = backend.create_circuit(3)
circuit.ry(theta, 0)
circuit.rz(phi, 0)       # qubit 0 holds |psi>
circuit.h(1)
circuit.cx(1, 2)         # Bell pair on qubits 1 and 2
circuit.cx(0, 1)
circuit.h(0)             # Bell measurement basis
circuit.cx(1, 2)         # X correction
circuit.cz(0, 2)         # Z correction

state = backend.statevector_big_endian(circuit).reshape(2, 2, 2)
rho_bob = np.einsum('abi,abj->ij', state, state.conj())   # trace out qubits 0 and 1
fidelity = np.real(psi.conj() @ rho_bob @ psi)
print(f"Teleportation fidelity: {fidelity:.6f}")
```

### Grover's search

Two-qubit Grover search for the marked item |11>. One iteration finds it with certainty.

```python
circuit = backend.create_circuit(2)
circuit.h(0); circuit.h(1)          # uniform superposition
circuit.cz(0, 1)                    # oracle: flip the phase of |11>
circuit.h(0); circuit.h(1)          # diffusion operator
circuit.x(0); circuit.x(1)
circuit.cz(0, 1)
circuit.x(0); circuit.x(1)
circuit.h(0); circuit.h(1)
circuit.measure_all()

counts = backend.run(circuit, shots=1000).get_counts()
print("Grover counts:", counts)
```

## 🧠 Quantum Machine Learning

### Quantum Support Vector Machine (QSVM)

`QuantumSVM` computes a quantum kernel and trains a classical SVM on it. Supported feature maps are `'ZZFeatureMap'`, `'ZFeatureMap'`, `'PauliFeatureMap'` and `'AngleEncoding'`; `feature_map_reps` sets the depth.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import superquantx as sqx

X, y = make_classification(n_samples=80, n_features=4, n_classes=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

for feature_map in ['ZFeatureMap', 'ZZFeatureMap', 'PauliFeatureMap']:
    qsvm = sqx.QuantumSVM(backend='simulator', feature_map=feature_map, feature_map_reps=2)
    qsvm.fit(X_train, y_train)
    print(f"QSVM with {feature_map}: {qsvm.score(X_test, y_test):.3f}")

classical = SVC(kernel='rbf').fit(X_train, y_train)
print(f"Classical RBF SVM: {classical.score(X_test, y_test):.3f}")
```

A single train/test split is noisy. For a fair comparison use `compare_to_classical` (see [Benchmarking](benchmarking.md)), which tunes the classical baselines and reports a p-value.

### Quantum Neural Network (QNN)

`QuantumNN` is a trainable variational circuit with angle encoding. It covers the "variational quantum classifier" use case.

```python
qnn = sqx.QuantumNN(
    backend='simulator',
    n_layers=2,
    encoding='angle',
    entanglement='linear',
    learning_rate=0.05,
    max_epochs=20,
    task_type='classification',
    seed=0,
)
qnn.fit(X_train, y_train)
print(f"QNN test accuracy: {qnn.score(X_test, y_test):.3f}")
history = qnn.get_training_history()
print("Training history keys:", list(history))
```

### Hybrid classifier

!!! bug "Known issue in 0.3.0"
    `HybridClassifier` is exported but does not work reliably in 0.3.0: the default `'ensemble'` mode and the `'voting'` and `'stacking'` modes raise a NumPy shape error in `fit`, and `'sequential'` trains but gives poor accuracy. Until this is fixed, compare quantum and classical models with `compare_to_classical` (below) instead.

!!! note "Not available in 0.3.0"
    Quantum GANs and Keras-style layer APIs (`sqx.layers`, `model.compile`) are not implemented.

## ⚡ Optimization Algorithms

### Variational Quantum Eigensolver (VQE)

```python
import superquantx as sqx

# Two-qubit reduced H2 Hamiltonian (STO-3G, 0.74 Angstrom)
h2 = sqx.Hamiltonian.from_dict({
    'II': -1.0523732,
    'ZI': 0.39793742,
    'IZ': -0.39793742,
    'ZZ': -0.01128010,
    'XX': 0.18093119,
})

vqe = sqx.VQE(hamiltonian=h2, ansatz='RealAmplitudes', optimizer='COBYLA', backend='simulator', seed=0)
ground_energy = vqe.find_ground_state()
exact_energy = h2.ground_state_energy()

print(f"VQE energy:   {ground_energy:.6f} Ha")
print(f"Exact energy: {exact_energy:.6f} Ha")
print(f"Error:        {abs(ground_energy - exact_energy):.2e} Ha")
```

`create_vqe_for_molecule('H2')` builds the same problem from the bundled molecular data. A `'UCCSD'` ansatz is not available; use `'RealAmplitudes'` or a callable.

### Quantum Approximate Optimization Algorithm (QAOA)

`QAOA.solve(adjacency)` runs weighted MaxCut and checks the answer by brute force.

```python
import numpy as np

edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
adjacency = np.zeros((4, 4))
for i, j in edges:
    adjacency[i, j] = adjacency[j, i] = 1.0

qaoa = sqx.QAOA(backend='simulator', p=3, optimizer='COBYLA', seed=0)
result = qaoa.solve(adjacency)

print("Best partition:", result['bitstring'])
print("Cut value:", result['cut_value'], "of optimum", result['classical_optimum'])
print("Approximation ratio:", round(result['approximation_ratio'], 3))
```

### Quantum annealing (D-Wave Ocean)

Annealing problems go through the `ocean` backend. Without D-Wave credentials it uses the local simulated annealer.

```python
ocean = sqx.get_backend('ocean')

# Minimise -x0 - x1 - x2 + 2 x0 x1 + 2 x0 x2 + 2 x1 x2 (pick exactly one variable)
Q = {(0, 0): -1, (1, 1): -1, (2, 2): -1, (0, 1): 2, (0, 2): 2, (1, 2): 2}
result = ocean.solve_qubo(Q, num_reads=100)

best = int(np.argmin(result['energies']))
print("Best solution:", {k: int(v) for k, v in result['samples'][best].items()})
print("Best energy:", float(result['energies'][best]))
```

## 🧮 Simulating quantum systems

`Hamiltonian` gives exact reference values for small systems: the matrix, ground-state energy, expectation values and time evolution.

```python
ising = sqx.Hamiltonian.ising_model(4, J=1.0, h=0.5)
print(f"Ising ground-state energy: {ising.ground_state_energy():.6f}")

# Time evolution of |0000> under the Ising Hamiltonian: the energy is conserved,
# but the magnetisation of qubit 0 oscillates because of the transverse field
z0 = sqx.Hamiltonian.from_dict({'ZIII': 1.0})
psi0 = np.zeros(2 ** 4, dtype=complex)
psi0[0] = 1.0
for t in [0.0, 0.5, 1.0, 2.0]:
    psi_t = ising.time_evolution(t) @ psi0
    energy = np.real(ising.expectation_value(psi_t))
    magnetisation = np.real(z0.expectation_value(psi_t))
    print(f"t={t}: <H> = {energy:.4f}, <Z0> = {magnetisation:.4f}")

# The same Hamiltonian solved variationally
vqe = sqx.VQE(hamiltonian=ising, backend='simulator', reps=2, seed=0)
print(f"VQE: {vqe.find_ground_state():.6f}")
```

!!! note "Not available in 0.3.0"
    Excited-state solvers, quantum walks, BB84 key distribution and Shor's algorithm are not implemented.

## 🔧 Custom Algorithm Development

Subclass `BaseQuantumAlgorithm` and implement `fit` and `predict`. The base class resolves `backend` to a backend object and stores `shots` and `seed`.

```python
import numpy as np
from scipy.optimize import minimize
from superquantx.algorithms import BaseQuantumAlgorithm


class GroundStateProbability(BaseQuantumAlgorithm):
    """Train RY angles so that measuring gives |00...0> as often as possible."""

    def __init__(self, n_qubits=3, backend='simulator', **kwargs):
        super().__init__(backend=backend, **kwargs)
        self.n_qubits = n_qubits

    def circuit(self, params):
        circuit = self.backend.create_circuit(self.n_qubits)
        for i, param in enumerate(params):
            circuit.ry(param, i)
        for i in range(self.n_qubits - 1):
            circuit.cx(i, i + 1)
        return circuit

    def cost(self, params):
        probs = np.abs(self.backend.statevector_big_endian(self.circuit(params))) ** 2
        return -probs[0]

    def fit(self, X=None, y=None, **kwargs):
        x0 = np.random.default_rng(self.seed).uniform(0, 2 * np.pi, self.n_qubits)
        res = minimize(self.cost, x0, method='COBYLA')
        self.optimal_params_, self.optimal_cost_ = res.x, res.fun
        self.is_fitted = True
        return self

    def predict(self, X=None, **kwargs):
        circuit = self.circuit(self.optimal_params_)
        circuit.measure_all()
        return self.backend.run(circuit, shots=self.shots).get_counts()


algo = GroundStateProbability(n_qubits=3, backend='simulator', seed=0).fit()
print(f"P(|000>) = {-algo.optimal_cost_:.4f}")
print("Sampled counts:", algo.predict())
```

### Combining algorithms

Algorithms are plain objects, so composition is ordinary Python. Here QAOA answers a MaxCut problem and VQE checks the energy of the corresponding Ising Hamiltonian.

```python
adjacency = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
cut = sqx.QAOA(backend='simulator', p=2, seed=0).solve(adjacency)

# MaxCut as an Ising energy: sum over edges of (Z_i Z_j - 1) / 2
terms = {}
for i in range(3):
    for j in range(i + 1, 3):
        if adjacency[i, j]:
            label = ''.join('Z' if q in (i, j) else 'I' for q in range(3))
            terms[label] = terms.get(label, 0) + 0.5 * adjacency[i, j]
            terms['III'] = terms.get('III', 0) - 0.5 * adjacency[i, j]
H = sqx.Hamiltonian.from_dict(terms)

print("QAOA cut:", cut['cut_value'], "| -ground energy:", -H.ground_state_energy())
```

## 📊 Algorithm Performance Analysis

Use `compare_to_classical` to compare a quantum classifier against tuned classical baselines on the same cross-validation folds. See [Benchmarking](benchmarking.md) for the full guide.

```python
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=60, noise=0.15, random_state=0)
comparison = sqx.compare_to_classical(X, y, sqx.QuantumSVM(backend='simulator'), cv=3, seed=0)
print(comparison.table())
print(comparison.verdict)
```

## 📚 Algorithm Selection Guide

```python
def recommend_algorithm(problem_type):
    """Map a problem type to the 0.3.0 class to start with."""
    return {
        'classification': 'QuantumSVM (kernel) or QuantumNN (variational)',
        'clustering': 'QuantumKMeans',
        'dimensionality_reduction': 'QuantumPCA',
        'combinatorial_optimization': 'QAOA, or OceanBackend.solve_qubo on annealers',
        'ground_state': 'VQE',
        'portfolio': 'QuantumPortfolioAgent',
    }.get(problem_type, 'not covered in 0.3.0')


available = [name for name, info in sqx.list_available_backends().items() if info['available']]
print("Available backends:", available)
for problem in ['classification', 'combinatorial_optimization', 'ground_state', 'search']:
    print(f"{problem}: {recommend_algorithm(problem)}")
```

## 📞 Getting Help

- **[Tutorials](../tutorials/quantum-ml.md)**: step-by-step algorithm tutorials
- **[API Reference](../api/algorithms.md)**: class and method reference
- **[Showcase](../showcase.md)**: a full benchmark run with its output
- **[GitHub Issues](https://github.com/SuperagenticAI/superquantx/issues)**: report bugs

!!! warning "NISQ limitations"
    Current quantum hardware is noisy and small. Every result on this page comes from noiseless simulation; expect lower accuracy on hardware.
