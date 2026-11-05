# Utilities API Reference

Helpers for feature maps, benchmarking and entanglement measures.

## Feature Maps

```python
import numpy as np
import superquantx as sqx

backend = sqx.get_backend("simulator")
X = np.array([[0.1, 0.2], [0.4, 0.8], [0.7, 0.3]])

spec = backend.create_feature_map(n_features=2, feature_map="ZZFeatureMap", reps=1)
circuit = backend.build_feature_map_circuit(X[0], spec)
print(spec.name, backend.get_statevector(circuit).shape)

K = backend.compute_kernel_matrix(X, X, feature_map="ZFeatureMap", shots=100)
print(np.round(K, 3))
```

Concrete feature-map classes also live under `superquantx.utils.feature_mapping`
(`ZFeatureMap`, `ZZFeatureMap`, …). Prefer the backend helpers above for portable
code.

## Benchmarking

`sqx.benchmark_algorithm` exists in 0.3.0 but assumes `algorithm.backend` is a
dict with a `name` key. Real algorithm instances store a backend object, so the
helper raises `AttributeError` today. Time a fit yourself instead:

```python
import time
import numpy as np
from sklearn.datasets import make_classification
import superquantx as sqx

X, y = make_classification(
    n_samples=30, n_features=2, n_informative=2, n_redundant=0, random_state=0
)
model = sqx.QuantumSVM(backend="simulator", feature_map="ZFeatureMap")
t0 = time.time()
model.fit(X, y)
acc = float(np.mean(model.predict(X) == y))
print(f"accuracy={acc:.3f} in {time.time() - t0:.2f}s")
```

## Entanglement Measure

```python
import numpy as np
from superquantx.utils import entanglement_measure

# Bell state |00>+|11> as a 4-amplitude statevector
bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
value = entanglement_measure(bell, subsystem_dims=(2, 2), measure="negativity")
print(value)
```

## Dataset Helpers

```python
import superquantx as sqx

X_train, X_test, y_train, y_test, meta = sqx.load_iris_quantum(
    n_features=2, encoding="angle"
)
print(X_train.shape, y_train.shape, meta.get("n_features"))
```

## See Also

- [Algorithms API](algorithms.md)
- [Circuits API](circuits.md)
