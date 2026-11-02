"""Honest classical baselines for quantum classifiers.

``compare_to_classical`` runs a quantum classifier, a tuned RBF SVM and a tuned
logistic regression on exactly the same stratified cross-validation folds, with
preprocessing fitted inside each fold, and reports mean and standard deviation of
the test accuracy together with wall time.

The protocol follows the controls recommended by Bhalgami (2026), "Empirical
Evaluation of Quantum Kernel Methods for Classification: A Comparative Study with
Classical Kernels", arXiv:2610.42029:

* the quantum kernel bandwidth is tuned on a grid of the same size as the RBF
  ``gamma`` grid, so neither side is handicapped;
* the scaler is fitted on the training fold only and the quantum kernel is
  recomputed inside every fold;
* the hardware cost of the fidelity kernel is reported in circuit evaluations
  (``n_train * (n_train - 1) / 2 + n_test * n_train`` per fold and grid point),
  because simulator wall time says nothing about QPU cost.

That paper finds tuned quantum kernels statistically indistinguishable from a
tuned RBF SVM on four standard datasets. Expect the same here: on classical data
the honest default is a tie or a classical win. All quantum results in this
package are classically simulated.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

__all__ = [
    "ModelScore",
    "ComparisonResult",
    "compare_to_classical",
    "corrected_paired_ttest",
]

ARXIV_REFERENCE = "arXiv:2610.42029"

DEFAULT_C_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
DEFAULT_RBF_GAMMA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_BANDWIDTH_GRID = (0.05, 0.1, 0.25, 0.5, 1.0)
DEFAULT_LOGREG_C_GRID = (0.01, 0.1, 1.0, 10.0)


@dataclass
class ModelScore:
    """Cross-validated score for one model."""

    name: str
    accuracy_mean: float
    accuracy_std: float
    fold_accuracies: list[float]
    wall_time_s: float
    grid_size: int
    best_params: list[dict[str, Any]] = field(default_factory=list)
    hardware_circuits: int | None = None
    is_quantum: bool | None = False  # None marks the untuned quantum reference row


@dataclass
class ComparisonResult:
    """Result of :func:`compare_to_classical`."""

    scores: list[ModelScore]
    n_samples: int
    n_features: int
    cv: int
    seed: int
    p_value_vs_best_classical: float | None = None
    delta_vs_best_classical: float | None = None
    reference: str = ARXIV_REFERENCE

    @property
    def quantum(self) -> ModelScore:
        return next(s for s in self.scores if s.is_quantum is True)

    @property
    def best_classical(self) -> ModelScore:
        classical = [s for s in self.scores if s.is_quantum is False]
        return max(classical, key=lambda s: s.accuracy_mean)

    @property
    def verdict(self) -> str:
        """Plain-language verdict at the 5% level (corrected resampled t-test)."""
        if self.p_value_vs_best_classical is None or np.isnan(self.p_value_vs_best_classical):
            return "not enough folds for a significance test"
        if self.p_value_vs_best_classical >= 0.05:
            return (f"no significant difference from {self.best_classical.name} "
                    f"(p={self.p_value_vs_best_classical:.2f})")
        better = "better" if self.delta_vs_best_classical > 0 else "worse"
        return (f"quantum is significantly {better} than {self.best_classical.name} "
                f"(p={self.p_value_vs_best_classical:.3f})")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict
        return d

    def table(self) -> str:
        """Fixed-width text table, one row per model."""
        header = f"{'model':<34} {'accuracy':>16} {'wall time':>10} {'grid':>5} {'QPU circuits':>13}"
        lines = [header, "-" * len(header)]
        for s in self.scores:
            acc = f"{s.accuracy_mean:.3f} +/- {s.accuracy_std:.3f}"
            circ = f"{s.hardware_circuits:,}" if s.hardware_circuits is not None else "-"
            lines.append(f"{s.name:<34} {acc:>16} {s.wall_time_s:>9.2f}s {s.grid_size:>5} {circ:>13}")
        lines.append("-" * len(header))
        lines.append(
            f"{self.n_samples} samples, {self.n_features} features, {self.cv}-fold stratified CV "
            f"(identical folds for every model), seed={self.seed}"
        )
        if self.delta_vs_best_classical is not None:
            lines.append(
                f"quantum minus best classical: {self.delta_vs_best_classical:+.3f}; {self.verdict}"
            )
        lines.append(f"Protocol: {self.reference}. Quantum kernels are classically simulated.")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.table()


def corrected_paired_ttest(a: np.ndarray, b: np.ndarray, n_train: int, n_test: int) -> float:
    """Two-sided p-value of the Nadeau and Bengio corrected resampled t-test.

    Plain paired t-tests over CV folds are over-confident because the training
    sets overlap; the correction inflates the variance by ``n_test / n_train``.
    """
    from scipy import stats

    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    k = len(d)
    if k < 2:
        return float("nan")
    var = np.var(d, ddof=1)
    if var == 0:
        return 1.0 if np.mean(d) == 0 else 0.0
    t = np.mean(d) / np.sqrt((1.0 / k + n_test / n_train) * var)
    return float(2 * stats.t.sf(abs(t), df=k - 1))


def _is_kernel_classifier(clf: Any) -> bool:
    backend = getattr(clf, "backend", None)
    return (
        backend is not None
        and hasattr(backend, "compute_kernel_matrix")
        and hasattr(clf, "feature_map")
        and getattr(clf, "quantum_kernel", None) is None
    )


def _quantum_kernel_fold(clf, X_tr, y_tr, X_te, bandwidths, C_grid, inner_cv, seed):
    """Tune bandwidth and C on the training fold with a precomputed quantum kernel."""
    scaler = StandardScaler().fit(X_tr)
    Xa, Xb = scaler.transform(X_tr), scaler.transform(X_te)
    n_tr = len(Xa)
    spec = clf.backend.create_feature_map(
        n_features=X_tr.shape[1],
        feature_map=clf.feature_map,
        reps=getattr(clf, "feature_map_reps", 2),
    )
    inner = StratifiedKFold(inner_cv, shuffle=True, random_state=seed)
    best = (-np.inf, None, None, None)
    for c in bandwidths:
        Z = np.vstack([Xa, Xb]) * c
        K = clf.backend.compute_kernel_matrix(Z, Z, feature_map=spec)
        K_tr, K_te = K[:n_tr, :n_tr], K[n_tr:, :n_tr]
        gs = GridSearchCV(SVC(kernel="precomputed"), {"C": list(C_grid)}, cv=inner)
        gs.fit(K_tr, y_tr)
        if gs.best_score_ > best[0]:
            best = (gs.best_score_, c, gs.best_estimator_, K_te)
    _, c, model, K_te = best
    circuits = len(bandwidths) * (n_tr * (n_tr - 1) // 2 + len(Xb) * n_tr)
    return model.predict(K_te), {"bandwidth": c, "C": model.C}, circuits


def compare_to_classical(
    X: np.ndarray,
    y: np.ndarray,
    classifier: Any,
    *,
    cv: int = 5,
    seed: int = 0,
    inner_cv: int = 3,
    bandwidth_grid: tuple[float, ...] | None = DEFAULT_BANDWIDTH_GRID,
    C_grid: tuple[float, ...] = DEFAULT_C_GRID,
    rbf_gamma_grid: tuple[float, ...] = DEFAULT_RBF_GAMMA_GRID,
    logreg_C_grid: tuple[float, ...] = DEFAULT_LOGREG_C_GRID,
    quantum_name: str | None = None,
    include_untuned: bool = True,
) -> ComparisonResult:
    """Compare a quantum classifier against tuned classical baselines.

    Every model sees the same ``StratifiedKFold(cv, shuffle=True, random_state=seed)``
    splits. Baselines:

    * ``SVC(kernel="rbf")`` with ``GridSearchCV`` over ``C`` and ``gamma``;
    * ``LogisticRegression`` with ``GridSearchCV`` over ``C``.

    For a quantum kernel classifier (for example :class:`~superquantx.QuantumSVM`)
    the bandwidth and ``C`` are tuned on a grid of the same size as the RBF grid,
    using a precomputed fidelity kernel recomputed inside each fold. Pass
    ``bandwidth_grid=None`` to keep the classifier's own settings and just call
    ``fit``/``predict`` per fold (this is also what happens for any other
    estimator with ``fit`` and ``predict``). Note that skipping the bandwidth
    tuning handicaps the quantum model (arXiv:2610.42029, Sec. V.2).

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        y: Labels.
        classifier: Quantum classifier instance, refit on every fold.
        cv: Number of outer folds.
        seed: Seed for the fold split.
        inner_cv: Folds for inner hyperparameter search.
        bandwidth_grid: Quantum bandwidths tried (features are multiplied by it).
        C_grid: SVM regularisation grid (quantum and RBF).
        rbf_gamma_grid: RBF ``gamma`` grid.
        logreg_C_grid: Logistic regression ``C`` grid.
        quantum_name: Row label for the quantum model.
        include_untuned: For quantum kernel classifiers, also report the same
            kernel with bandwidth 1 and C 1, to show what tuning is worth.

    Returns:
        ComparisonResult with per-model accuracy mean/std, wall time and a
        ``table()`` for printing.

    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    outer = StratifiedKFold(cv, shuffle=True, random_state=seed)
    folds = list(outer.split(X, y))

    tuned_kernel = bandwidth_grid is not None and _is_kernel_classifier(classifier)
    backend_name = getattr(getattr(classifier, "backend", None), "__class__", type(None)).__name__
    if quantum_name is None:
        quantum_name = f"{type(classifier).__name__} ({backend_name.replace('Backend', '')})"
        if tuned_kernel:
            quantum_name += ", tuned"

    def run(name, fit_predict, grid_size, is_quantum=False):
        accs, params, circuits = [], [], 0
        start = time.perf_counter()
        for tr, te in folds:
            pred, p, circ = fit_predict(X[tr], y[tr], X[te])
            accs.append(float(np.mean(pred == y[te])))
            params.append(p)
            circuits += circ or 0
        wall = time.perf_counter() - start
        return ModelScore(
            name=name,
            accuracy_mean=float(np.mean(accs)),
            accuracy_std=float(np.std(accs)),
            fold_accuracies=accs,
            wall_time_s=wall,
            grid_size=grid_size,
            best_params=params,
            hardware_circuits=circuits if is_quantum and circuits else None,
            is_quantum=is_quantum,
        )

    def quantum_fp(X_tr, y_tr, X_te):
        if tuned_kernel:
            return _quantum_kernel_fold(
                classifier, X_tr, y_tr, X_te, bandwidth_grid, C_grid, inner_cv, seed
            )
        classifier.fit(X_tr, y_tr)
        n_tr = len(X_tr)
        circ = n_tr * (n_tr - 1) // 2 + len(X_te) * n_tr if _is_kernel_classifier(classifier) else None
        return np.asarray(classifier.predict(X_te)), {}, circ

    inner = StratifiedKFold(inner_cv, shuffle=True, random_state=seed)

    def rbf_fp(X_tr, y_tr, X_te):
        gs = GridSearchCV(
            make_pipeline(StandardScaler(), SVC(kernel="rbf")),
            {"svc__C": list(C_grid), "svc__gamma": list(rbf_gamma_grid)},
            cv=inner,
        ).fit(X_tr, y_tr)
        return gs.predict(X_te), gs.best_params_, None

    def logreg_fp(X_tr, y_tr, X_te):
        gs = GridSearchCV(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            {"logisticregression__C": list(logreg_C_grid)},
            cv=inner,
        ).fit(X_tr, y_tr)
        return gs.predict(X_te), gs.best_params_, None

    def untuned_fp(X_tr, y_tr, X_te):
        return _quantum_kernel_fold(classifier, X_tr, y_tr, X_te, (1.0,), (1.0,), inner_cv, seed)

    q_grid = len(bandwidth_grid) * len(C_grid) if tuned_kernel else 1
    scores = [run(quantum_name, quantum_fp, q_grid, is_quantum=True)]
    if tuned_kernel and include_untuned:
        untuned = run(quantum_name.replace(", tuned", ", untuned"), untuned_fp, 1)
        untuned.hardware_circuits = None
        scores.append(untuned)
    scores += [
        run("SVC (RBF), tuned C and gamma", rbf_fp, len(C_grid) * len(rbf_gamma_grid)),
        run("Logistic regression, tuned C", logreg_fp, len(logreg_C_grid)),
    ]
    # The untuned quantum row is a reference, not a classical baseline.
    for sc in scores[1:]:
        if "untuned" in sc.name and sc.name.startswith(type(classifier).__name__):
            sc.is_quantum = None

    result = ComparisonResult(scores=scores, n_samples=len(X), n_features=X.shape[1], cv=cv, seed=seed)
    q, best = result.quantum, result.best_classical
    n_test = len(X) // cv
    result.delta_vs_best_classical = q.accuracy_mean - best.accuracy_mean
    result.p_value_vs_best_classical = corrected_paired_ttest(
        q.fold_accuracies, best.fold_accuracies, len(X) - n_test, n_test
    )
    return result
