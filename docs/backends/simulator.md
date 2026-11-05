# Python Simulator Backend

The Python simulator is a pure-Python statevector backend that ships with
SuperQuantX. It needs no extra dependencies and is the default choice for
tests, tutorials and local experiments.

## Overview

- Pure Python, no Qiskit / PennyLane / Cirq required
- Direct statevector access
- Fluent gate methods on the circuit object (`circuit.h(0)`, `circuit.cx(0, 1)`)
  **and** the portable `backend.add_gate` API
- Feature maps, kernels and variational Ansätze shared with the other backends

It favours clarity over speed. For larger circuits prefer PennyLane Lightning,
Qiskit Aer or Cirq.

## Installation

```bash
uv add "superquantx==0.3.0"
```

### Verify Installation

```python
import superquantx as sqx

backend = sqx.get_backend("simulator")
print(backend.is_available())
print(backend.max_qubits)
print(backend.get_backend_info()["backend_name"])
```

## Quick Start

### Portable API

```python
import superquantx as sqx

backend = sqx.get_backend("simulator", max_qubits=5, shots=1024)

circuit = backend.create_circuit(2)
circuit = backend.add_gate(circuit, "H", 0)
circuit = backend.add_gate(circuit, "CX", [0, 1])
circuit = backend.add_measurement(circuit)

result = backend.execute_circuit(circuit, shots=1000)
print(result["counts"])
```

### Fluent Simulator API

On the simulator only, the circuit object is a `QuantumState` with gate
methods and `backend.run` returns a `QuantumResult`:

```python
import superquantx as sqx

backend = sqx.get_backend("simulator")
circuit = backend.create_circuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

result = backend.run(circuit, shots=1000)
print(result.get_counts())
```

Other backends do not implement these fluent methods. Prefer `add_gate` /
`execute_circuit` when you want portable examples.

### Parameterized Circuit and Statevector

```python
import numpy as np
import superquantx as sqx

backend = sqx.get_backend("simulator")
circuit = backend.create_circuit(3)
backend.add_gate(circuit, "RY", 0, [np.pi / 4])
backend.add_gate(circuit, "RX", 1, [np.pi / 3])
backend.add_gate(circuit, "RZ", 2, [np.pi / 2])
backend.add_gate(circuit, "CX", [0, 1])
backend.add_gate(circuit, "CX", [1, 2])

statevector = backend.get_statevector(circuit)
print(statevector.shape, float(np.linalg.norm(statevector)))
```

## Configuration

```python
import superquantx as sqx

backend = sqx.SimulatorBackend(max_qubits=10, shots=1024)
print(backend.capabilities)
```

## Feature Maps and Kernels

`create_feature_map` returns a `FeatureMapSpec`. Build a circuit from it with
`build_feature_map_circuit`; do not call the spec like a function.

```python
import numpy as np
import superquantx as sqx

backend = sqx.get_backend("simulator")
X = np.array([[0.1, 0.2, 0.3], [0.4, 0.8, 0.2], [0.7, 0.3, 0.9]])

spec = backend.create_feature_map(n_features=3, feature_map="ZZFeatureMap", reps=2)
encoded = backend.build_feature_map_circuit(X[0], spec)
sv = backend.get_statevector(encoded)
print(spec, float(np.linalg.norm(sv)))

K = backend.compute_kernel_matrix(X, X, feature_map="ZZFeatureMap", shots=200)
print(K.shape)
```

Supported maps: `ZZFeatureMap`, `ZFeatureMap`, `PauliFeatureMap`, `AngleEncoding`.

## Variational Ansätze

`create_ansatz` returns a circuit, not a factory.

```python
import numpy as np
import superquantx as sqx

backend = sqx.get_backend("simulator")
n_qubits, reps = 4, 1
n_params = backend.ansatz_param_count("RealAmplitudes", n_qubits, reps)
params = np.random.default_rng(0).random(n_params)

ansatz = backend.create_ansatz("RealAmplitudes", n_qubits, params, reps=reps)
backend.add_measurement(ansatz)
print(backend.execute_circuit(ansatz, shots=500)["counts"])

su2_n = backend.ansatz_param_count("EfficientSU2", n_qubits, reps)
su2 = backend.create_ansatz(
    "EfficientSU2", n_qubits, np.zeros(su2_n), reps=reps
)
print(type(su2).__name__)
```

## Expectation Values

```python
import numpy as np
import superquantx as sqx

backend = sqx.get_backend("simulator")
circuit = backend.create_circuit(2)
backend.add_gate(circuit, "H", 0)
backend.add_gate(circuit, "CX", [0, 1])

# Pauli dict: qubit 0 is the leftmost character of the label
zz = backend.compute_expectation(circuit, {"ZZ": 1.0})
print(zz)
```

## Quantum ML on the Simulator

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import superquantx as sqx

X, y = make_moons(n_samples=40, noise=0.1, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

qsvm = sqx.QuantumSVM(backend="simulator", feature_map="ZFeatureMap")
qsvm.fit(X_train, y_train)
print(float(np.mean(qsvm.predict(X_test) == y_test)))
```

## Tips

- Use the fluent API for teaching; use `add_gate` for portable docs and tests.
- `backend.run(...).get_counts()` is simulator-only. Everywhere else read
  `result["counts"]` from `execute_circuit`.
- Cap `max_qubits` when you want to fail fast on oversized circuits.
