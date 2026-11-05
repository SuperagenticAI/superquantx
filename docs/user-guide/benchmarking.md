# Benchmarking Against Classical Baselines

A quantum model result means little on its own. `compare_to_classical` puts
a quantum classifier next to tuned classical models on identical
cross-validation folds and tells you whether the difference is real.

```python
from sklearn.datasets import make_moons
import superquantx as sqx

X, y = make_moons(n_samples=100, noise=0.2, random_state=0)
result = sqx.compare_to_classical(X, y, sqx.QuantumSVM(backend="pennylane"), cv=5)
print(result.table())
print(result.verdict)
```

Real output (from [the showcase](../showcase.md)):

```text
model                                      accuracy  wall time  grid  QPU circuits
----------------------------------------------------------------------------------
QuantumSVM (PennyLane), tuned       0.930 +/- 0.051      3.51s    25       119,000
QuantumSVM (PennyLane), untuned     0.670 +/- 0.125      0.64s     1             -
SVC (RBF), tuned C and gamma        0.950 +/- 0.055      0.89s    25             -
Logistic regression, tuned C        0.860 +/- 0.097      0.32s     4             -
----------------------------------------------------------------------------------
100 samples, 2 features, 5-fold stratified CV (identical folds for every model), seed=0
quantum minus best classical: -0.020; no significant difference from SVC (RBF), tuned C and gamma (p=0.54)
Protocol: arXiv:2610.42029. Quantum kernels are classically simulated.
```

## Protocol

The function follows the evaluation protocol of
[Bhalgami (2026), "Empirical Evaluation of Quantum Kernel Methods for
Classification: A Comparative Study with Classical Kernels",
arXiv:2610.42029](https://arxiv.org/abs/2610.42029):

- **Same folds for every model.** `StratifiedKFold(cv, shuffle=True, random_state=seed)`.
- **Equal tuning budget.** The quantum kernel bandwidth (input scaling) and
  `C` are tuned on a 5 x 5 grid, the same size as the RBF SVM's `C` x `gamma`
  grid, with nested inner cross-validation. Logistic regression tunes `C`.
- **Per-fold preprocessing.** Scaling is refit inside each fold, so no test
  information leaks into training.
- **Corrected significance test.** The p-value uses the Nadeau-Bengio
  corrected resampled t-test, which accounts for overlapping training sets
  in cross-validation. A plain paired t-test overstates significance.
- **Hardware cost.** The "QPU circuits" column counts the circuits a real
  device would need for the tuned kernel: `n(n-1)/2` training-kernel entries
  plus `n_test * n_train` test entries, summed over folds and grid points.
- **Untuned reference.** The untuned row (bandwidth 1, `C` = 1) shows how much
  of the quantum score comes from tuning rather than from the circuit.

## How to read the result

The paper and related work (Shaydulin and Wild 2022; arXiv:2503.05602) find
that once the bandwidth is tuned, fidelity quantum kernels behave much like
an RBF kernel, and tuned quantum kernels are statistically indistinguishable
from a tuned RBF SVM on standard datasets. The showcase reproduces exactly
that: bandwidth tuning lifts the quantum kernel from 0.67 to 0.93, and the
remaining gap to the RBF SVM is not significant (p = 0.54).

So a "no significant difference" verdict is the expected outcome on
small classical datasets. A claim of quantum benefit needs data where the
quantum model wins this test, at a hardware cost you can afford.

## API

`compare_to_classical(X, y, classifier, *, cv=5, seed=0, inner_cv=3,
bandwidth_grid=..., C_grid=..., rbf_gamma_grid=..., logreg_C_grid=...,
include_untuned=True)` returns a `ComparisonResult` with:

- `scores`: one `ModelScore` per model (`accuracy_mean`, `accuracy_std`,
  `fold_accuracies`, `wall_time_s`, `grid_size`, `best_params`,
  `hardware_circuits`);
- `quantum`, `best_classical`: shortcuts to those rows;
- `p_value_vs_best_classical`, `delta_vs_best_classical`, `verdict`;
- `table()` and `to_dict()` for reporting.

From the command line:

```bash
superquantx benchmark --dataset moons --backend pennylane
```
