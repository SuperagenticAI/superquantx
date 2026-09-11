# Design note: QuantumSVM `feature_map="auto"` (Qmes-style spike)

Status: spike on branch `spike/qmes-feature-map-auto`.
Inspired by [Qmes](https://github.com/tungduy1704/Qmes) ([arXiv:2609.04652](https://arxiv.org/abs/2609.04652)).

## Why

QuantumSVM performance depends strongly on the encoding circuit. Exhaustive
kernel evaluation of every candidate is expensive. Qmes shows that classical
dataset complexity features can rank encoding circuits without quantum
evaluation at inference time. This spike adds a practical `feature_map="auto"`
path that recommends a name from SuperQuantX's existing feature-map registry.

## What we adopted

1. **Registry** (`utils/feature_map_registry.py`)
   Canonical names: `ZZFeatureMap`, `PauliFeatureMap`, `AmplitudeMap`,
   `AngleEncoding`, `ZFeatureMap`. Manual string aliases still work.

2. **Qmes mapping** (`QMES_TO_SUPERQUANTX`)

   | Qmes pool | SuperQuantX registry | Rationale |
   |---|---|---|
   | unit | AmplitudeMap | Amplitude / square-root style embedding |
   | SRx | AngleEncoding | Separable single-qubit rotations |
   | RY | AngleEncoding | Separable RY angle encoding |
   | HERx | PauliFeatureMap | Hardware-efficient rotations + CX |
   | RY_CX | PauliFeatureMap | Angle + linear CX |
   | ZFM | ZFeatureMap | Separable Z rotations |
   | HD | ZZFeatureMap | High-dim / entangling analogue |

3. **Selector** (`utils/feature_map_auto.py`)
   - `mode="auto"`: try Qmes if installed, else heuristic.
   - `mode="qmes"`: require optional `[qmes]` extra.
   - `mode="heuristic"`: local meta-features only (no problexity, no quantum).
   - Heuristic uses sample/feature ratio, mean |correlation|, class imbalance,
     and a cheap logistic-regression separability proxy.

4. **QuantumSVM**
   - `feature_map="auto"` or `None` resolves at `fit()` time.
   - Stores `selected_feature_map_` and `feature_map_recommendation_`.
   - Existing manual strings/objects are unchanged.

5. **Backends**
   Simulator / PennyLane / Qiskit / Cirq `create_feature_map` recognize
   `AngleEncoding` and `ZFeatureMap` (AmplitudeMap already present).

## What we deferred

- Full Qmes circuit pool implementations (unit, SRx, HERx, RY_CX, HD gate
  schedules, Qsun backend, fixed 4-qubit PCA pipeline).
- Shipping or re-training a SuperQuantX-native pairwise OvO recommender on
  our own meta-dataset and backends.
- Problexity's full 22-d / 12-d complexity vectors in the default heuristic.
- Noise-aware or hardware-aware recommendation.
- Regression-task auto selection wired into a QuantumSVR (QSVM remains
  classification-focused).
- Changing the default constructor value away from `ZZFeatureMap` (auto is
  opt-in).

## Circuit-pool gaps

Qmes evaluates seven Qsun encodings with a shared 4-qubit PCA + MinMax
preprocess. SuperQuantX backends implement a smaller, differently shaped set.
Closest-name mapping is approximate:

- Exact Qmes unit / HD gate lists are not reproduced.
- `AmplitudeMap` remains a partial/placeholder on several backends.
- Entanglement patterns (linear CX vs ISWAP brickwork) differ.
- Recommendation quality of the heuristic is unvalidated against Qmes regret.

## How to try

```python
from sklearn.datasets import make_classification
from superquantx.algorithms import QuantumSVM

X, y = make_classification(n_samples=40, n_features=4, n_informative=3, random_state=0)
qsvm = QuantumSVM(backend="simulator", feature_map="auto", shots=100)
qsvm.fit(X, y)
print(qsvm.selected_feature_map_)
print(qsvm.feature_map_recommendation_)
```

Optional Qmes path:

```bash
pip install "superquantx[qmes]"
# or: pip install "git+https://github.com/tungduy1704/Qmes.git"
```

```python
qsvm = QuantumSVM(
    backend="simulator",
    feature_map="auto",
    auto_feature_map_mode="qmes",
)
```

## Follow-ups

1. Expand the registry with first-class Qmes-like encodings (or adapters).
2. Build a SuperQuantX meta-dataset on simulator kernels and train a pairwise
   recommender (or fine-tune Qmes with registered SQX circuits).
3. Expose `recommend_feature_map` in the CLI / docs tutorials.
4. Add regression auto-selection once a Quantum kernel regressor lands.
5. Benchmark heuristic vs Qmes vs fixed `ZZFeatureMap` on shared datasets.
