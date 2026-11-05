# Design note: QuantumSVM `feature_map="auto"` (Qmes-style spike)

Status: experimental feature included in SuperQuantX 0.3.0.
Inspired by [Qmes](https://github.com/tungduy1704/Qmes) ([arXiv:2609.04652](https://arxiv.org/abs/2609.04652)).

## Paper scope and limits (read before demoing)

The Qmes results in arXiv:2609.04652 hold only within this scope:

- **7 circuits**: a fixed pool (unit, SRx, RY, HERx, RY_CX, ZFM, HD),
  implemented in the bundled Qsun simulator, not in SuperQuantX backends.
- **4 qubits**: every dataset is reduced by PCA to at most 4 components and
  rescaled before kernel evaluation, so rankings describe 4-qubit circuits.
- **Noiseless simulation**: all circuit scores come from noiseless
  state-vector simulation. Hardware noise and noise models are not modelled,
  so a recommendation says nothing about performance on real QPUs.
- Supervised tabular data only (labels required), meta-features computed on
  at most 600 subsampled points, fixed non-trainable kernels.
- Reported gains (2.2x lower mean regret for classification, 4.2x for
  regression versus a non-adaptive baseline) are leave-one-out results on the
  authors' 105 classification and 86 regression benchmark datasets.

SuperQuantX maps the Qmes circuit names onto its own, differently shaped
feature maps (see below), so the paper's regret numbers do not transfer
directly to SuperQuantX kernels. Treat `feature_map="auto"` as a heuristic
starting point, not a validated optimum.

## Installing Qmes (optional)

Qmes is an optional runtime import, not a package extra. It is not published
on PyPI, and PyPI (and hatchling) reject direct git references in extras, so
SuperQuantX does not declare it as a dependency. Install it manually if you
want the pre-trained recommender:

```bash
pip install git+https://github.com/tungduy1704/Qmes
```

Without Qmes, `feature_map="auto"` (mode `auto`) uses the built-in classical
heuristic. Tests run without Qmes installed.

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
   - `mode="qmes"`: require the optional Qmes package (runtime import only).
   - `mode="heuristic"`: local meta-features only (no problexity, no quantum).
   - Heuristic uses sample/feature ratio, mean |correlation|, class imbalance,
     and a cheap logistic-regression separability proxy.

4. **QuantumSVM**
   - `feature_map="auto"` or `None` resolves at `fit()` time.
   - Stores `selected_feature_map_` and `feature_map_recommendation_`.
   - Existing manual strings/objects are unchanged.

5. **Backends**
   Simulator / PennyLane / Qiskit / Cirq `create_feature_map` recognize
   `AngleEncoding` and `ZFeatureMap` through the shared feature-map implementation.

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
- `AmplitudeMap` is a registry candidate but is not implemented by the shared backends.
  Automatic selection chooses the next implemented map from the ranking and records
  `unavailable_feature_map` in the recommendation details. Explicit requests raise an error.
- Entanglement patterns (linear CX vs ISWAP brickwork) differ.
- Recommendation quality of the heuristic is unvalidated against Qmes regret.

## How to try

```python
from sklearn.datasets import make_classification
from superquantx.algorithms import QuantumSVM

X, y = make_classification(n_samples=40, n_features=4, n_informative=3, n_redundant=0, n_repeated=0, random_state=0)
qsvm = QuantumSVM(backend="simulator", feature_map="auto", shots=100)
qsvm.fit(X, y)
print(qsvm.selected_feature_map_)
print(qsvm.feature_map_recommendation_)
```

Optional Qmes path:

```bash
pip install git+https://github.com/tungduy1704/Qmes
```

!!! note "Optional Qmes dependency"
    Needs the Qmes package from GitHub; not run in the docs checker.

<!-- docs-check: skip -->
```python
from superquantx.algorithms import QuantumSVM

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
