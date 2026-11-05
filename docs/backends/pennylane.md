# PennyLane Backend Integration

PennyLane is a cross-platform quantum machine learning library focused on
variational circuits and differentiable programming. SuperQuantX wraps it
behind the same gate-model API used by the other backends.

## Overview

The PennyLane backend is a good fit when you want:

- Variational circuits and hybrid quantum-classical loops
- Local simulators (`default.qubit`, `lightning.qubit`)
- The same portable circuit API as Qiskit, Cirq, Braket and TKET

## Installation

```bash
uv add "superquantx[pennylane]==0.3.0"

# Optional: faster local simulator
uv add pennylane-lightning
```

## Quick Start

### Basic Usage

```python
import superquantx as sqx

# Local PennyLane simulator (default.qubit)
backend = sqx.get_backend("pennylane")

circuit = backend.create_circuit(2)
circuit = backend.add_gate(circuit, "H", 0)
circuit = backend.add_gate(circuit, "CNOT", [0, 1])
circuit = backend.add_measurement(circuit)

result = backend.execute_circuit(circuit, shots=1000)
print(result["counts"])  # about 500 each of '00' and '11'
```

PennyLane circuits are `PennyLaneCircuit` objects. They do **not** expose
methods such as `circuit.h(0)`. Always build them with `backend.add_gate` /
`backend.add_measurement`, or with the higher-level helpers below.

## Configuration

### Device Selection

```python
import superquantx as sqx

# Devices PennyLaneBackend understands out of the box
devices = {
    "default.qubit": "Pure-Python PennyLane simulator (default)",
    "lightning.qubit": "C++ simulator (needs pennylane-lightning)",
}

backend = sqx.PennyLaneBackend(device="lightning.qubit", wires=8, shots=1024)
print(backend.get_device_info())
print(backend.device)  # underlying PennyLane device name
```

Plugins such as `qiskit.aer` or IBM hardware devices are not configured
through `PennyLaneBackend` in 0.3.0. Use the Qiskit backend for IBM access,
or build a native PennyLane device yourself and drive it outside SuperQuantX.

### Constructor Arguments

`PennyLaneBackend(device='default.qubit', wires=16, shots=1024, **kwargs)`:

- `device`: PennyLane device name
- `wires`: maximum number of qubits the device is created with
- `shots`: default shot count for `execute_circuit`

## Working with Circuits

### Portable Gate API

```python
import numpy as np
import superquantx as sqx

backend = sqx.get_backend("pennylane")
circuit = backend.create_circuit(3)

backend.add_gate(circuit, "H", 0)
backend.add_gate(circuit, "RY", 1, [np.pi / 4])
backend.add_gate(circuit, "CNOT", [0, 1])
backend.add_gate(circuit, "CNOT", [1, 2])
backend.add_measurement(circuit)

counts = backend.execute_circuit(circuit, shots=1000)["counts"]
print(counts)
print(backend.counts_big_endian(counts))  # qubit 0 first
```

Supported gate names match the other gate-model backends (`H`, `X`, `Y`, `Z`,
`RX`, `RY`, `RZ`, `CNOT`/`CX`, `CZ`, `SWAP`, `TOFFOLI`, …).

### Feature Maps and Kernels

```python
import numpy as np
import superquantx as sqx

backend = sqx.get_backend("pennylane")
X = np.array([[0.1, 0.2], [0.4, 0.8], [0.7, 0.3], [0.9, 0.6]])

# create_feature_map returns a FeatureMapSpec, not a callable
spec = backend.create_feature_map(n_features=2, feature_map="ZZFeatureMap", reps=2)
encoded = backend.build_feature_map_circuit(X[0], spec)
print(type(encoded).__name__)

# Kernel matrix used by QuantumSVM
K = backend.compute_kernel_matrix(X, X, feature_map="ZZFeatureMap", shots=200)
print(K.shape, np.round(np.diag(K), 3))
```

Supported feature maps: `ZZFeatureMap`, `ZFeatureMap`, `PauliFeatureMap`,
`AngleEncoding`.

### Ansätze

```python
import numpy as np
import superquantx as sqx

backend = sqx.get_backend("pennylane")
n_qubits, reps = 3, 1
n_params = backend.ansatz_param_count("RealAmplitudes", n_qubits, reps)
params = np.zeros(n_params)

# create_ansatz returns a ready-to-run circuit, not a factory
ansatz = backend.create_ansatz("RealAmplitudes", n_qubits, params, reps=reps)
backend.add_measurement(ansatz)
print(backend.execute_circuit(ansatz, shots=500)["counts"])

su2_params = np.zeros(backend.ansatz_param_count("EfficientSU2", n_qubits, reps))
su2 = backend.create_ansatz("EfficientSU2", n_qubits, su2_params, reps=reps)
print(type(su2).__name__)
```

## Quantum Machine Learning

### Quantum Neural Network

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import superquantx as sqx

X, y = make_moons(n_samples=60, noise=0.1, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

qnn = sqx.QuantumNN(
    backend="pennylane",
    n_layers=2,
    max_epochs=15,
    seed=0,
)
qnn.fit(X_train, y_train)
acc = float(np.mean(qnn.predict(X_test) == y_test))
print(f"QuantumNN accuracy: {acc:.3f}")
```

There is no `VariationalQuantumClassifier` class in 0.3.0. Use
`QuantumNN` (alias `QuantumNeuralNetwork`) for variational classifiers.

### Quantum SVM

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import superquantx as sqx

X, y = make_moons(n_samples=40, noise=0.1, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

qsvm = sqx.QuantumSVM(backend="pennylane", feature_map="ZFeatureMap")
qsvm.fit(X_train, y_train)
acc = float(np.mean(qsvm.predict(X_test) == y_test))
print(f"QuantumSVM accuracy: {acc:.3f}")
```

## Comparison Across Backends

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import superquantx as sqx

X, y = make_moons(n_samples=40, noise=0.1, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

results = {}
for name in ["simulator", "pennylane"]:
    model = sqx.QuantumSVM(backend=name, feature_map="ZFeatureMap")
    model.fit(X_train, y_train)
    results[name] = float(np.mean(model.predict(X_test) == y_test))
print(results)
```

## Tips

- Prefer `backend.add_gate` over native PennyLane syntax when you want the
  example to run unchanged on other SuperQuantX backends.
- `lightning.qubit` is usually faster than `default.qubit` for more than a
  few qubits.
- Hardware access through PennyLane plugins is outside the 0.3.0
  `PennyLaneBackend` surface; use the matching SuperQuantX backend instead.
