"""Qmes-inspired automatic feature-map selection for QuantumSVM.

Selection is entirely classical at inference time (no quantum evaluation).

Strategy:
1. If the optional ``Qmes`` package is importable, call its pre-trained
   recommender and map the top circuit onto SuperQuantX's registry.
2. Otherwise (or if Qmes fails), run a lightweight meta-feature heuristic
   that picks among SuperQuantX registry names only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .feature_map_registry import (
    DEFAULT_FEATURE_MAP,
    QMES_TO_SUPERQUANTX,
    map_qmes_circuit,
    normalize_feature_map_name,
)


logger = logging.getLogger(__name__)

AutoMode = Literal["auto", "qmes", "heuristic"]


@dataclass
class FeatureMapRecommendation:
    """Result of automatic feature-map selection."""

    feature_map: str
    source: str
    ranking: list[str] = field(default_factory=list)
    qmes_ranking: list[str] = field(default_factory=list)
    meta_features: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_map": self.feature_map,
            "source": self.source,
            "ranking": list(self.ranking),
            "qmes_ranking": list(self.qmes_ranking),
            "meta_features": dict(self.meta_features),
            "details": dict(self.details),
        }


def qmes_available() -> bool:
    """Return True if the optional Qmes package can be imported."""
    try:
        import Qmes  # noqa: F401

        return True
    except ImportError:
        return False


def extract_lightweight_meta_features(
    X: np.ndarray,
    y: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute cheap classical descriptors used by the heuristic fallback.

    These are intentionally simpler than Qmes/problexity measures so the
    spike has no hard dependency on problexity when Qmes is absent.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    n_samples, n_features = X.shape
    feats: dict[str, float] = {
        "n_samples": float(n_samples),
        "n_features": float(n_features),
        "samples_per_feature": float(n_samples) / max(n_features, 1),
        "feature_std_mean": float(np.mean(np.std(X, axis=0))) if n_samples > 1 else 0.0,
        "feature_corr_abs_mean": _mean_abs_corr(X),
    }

    if y is not None:
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        feats["n_classes"] = float(len(classes))
        probs = counts.astype(float) / max(n_samples, 1)
        # Normalized entropy imbalance in [0, 1]; 0 = balanced binary.
        if len(classes) > 1:
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            feats["class_imbalance"] = float(
                1.0 - entropy / np.log(len(classes))
            )
        else:
            feats["class_imbalance"] = 0.0
        feats["linear_sep_proxy"] = _linear_separability_proxy(X, y)
    else:
        feats["n_classes"] = 0.0
        feats["class_imbalance"] = 0.0
        feats["linear_sep_proxy"] = 0.5

    return feats


def _mean_abs_corr(X: np.ndarray) -> float:
    if X.shape[0] < 3 or X.shape[1] < 2:
        return 0.0
    std = np.std(X, axis=0)
    keep = std > 1e-12
    if np.count_nonzero(keep) < 2:
        return 0.0
    corr = np.corrcoef(X[:, keep], rowvar=False)
    if corr.ndim == 0:
        return 0.0
    mask = ~np.eye(corr.shape[0], dtype=bool)
    vals = np.abs(corr[mask])
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else 0.0


def _linear_separability_proxy(X: np.ndarray, y: np.ndarray) -> float:
    """Train a tiny linear model; higher score => more linearly separable."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import LabelEncoder, StandardScaler
    except ImportError:
        return 0.5

    y_enc = LabelEncoder().fit_transform(y)
    if len(np.unique(y_enc)) < 2 or X.shape[0] < 6:
        return 0.5

    Xs = StandardScaler().fit_transform(X)
    # Cap size for speed in the selector path.
    if Xs.shape[0] > 200:
        rng = np.random.default_rng(0)
        idx = rng.choice(Xs.shape[0], size=200, replace=False)
        Xs, y_enc = Xs[idx], y_enc[idx]

    clf = LogisticRegression(max_iter=200, solver="lbfgs")
    try:
        n_splits = 3 if Xs.shape[0] >= 15 else 2
        scores = cross_val_score(clf, Xs, y_enc, cv=n_splits, scoring="accuracy")
        return float(np.mean(scores))
    except Exception:
        clf.fit(Xs, y_enc)
        return float(clf.score(Xs, y_enc))


def recommend_feature_map_heuristic(
    X: np.ndarray,
    y: np.ndarray | None = None,
) -> FeatureMapRecommendation:
    """Select a registry feature map from lightweight meta-features only."""
    meta = extract_lightweight_meta_features(X, y)
    n_features = meta["n_features"]
    spp = meta["samples_per_feature"]
    lin = meta["linear_sep_proxy"]
    corr = meta["feature_corr_abs_mean"]
    imbalance = meta["class_imbalance"]

    # Prefer compact amplitude embedding when dimensionality is high and data
    # is scarce relative to features (inspired by Qmes unit for dense tabular).
    if n_features >= 16 and spp < 8.0:
        choice = "AmplitudeMap"
        reason = "high_dim_scarce_samples"
    # Strong feature correlation + weak linear separation favors entangling ZZ.
    elif corr >= 0.35 and lin < 0.75:
        choice = "ZZFeatureMap"
        reason = "correlated_nonlinear"
    # Mild nonlinearity / multiclass: Pauli-style hardware-efficient map.
    elif lin < 0.85 or imbalance > 0.25 or meta["n_classes"] > 2:
        choice = "PauliFeatureMap"
        reason = "moderate_complexity"
    # Nearly linear / separable: separable Z rotations.
    elif lin >= 0.92 and corr < 0.2:
        choice = "ZFeatureMap"
        reason = "highly_linear"
    else:
        choice = "AngleEncoding"
        reason = "default_angle"

    # Stable preference order for reporting (selected first).
    preference = [
        choice,
        "ZZFeatureMap",
        "PauliFeatureMap",
        "AngleEncoding",
        "ZFeatureMap",
        "AmplitudeMap",
    ]
    ranking: list[str] = []
    for name in preference:
        if name not in ranking:
            ranking.append(name)

    return FeatureMapRecommendation(
        feature_map=choice,
        source="heuristic",
        ranking=ranking,
        meta_features=meta,
        details={"reason": reason},
    )


def recommend_feature_map_qmes(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: Literal["classification", "regression"] = "classification",
    top_k: int = 3,
) -> FeatureMapRecommendation:
    """Use Qmes recommend() and map onto SuperQuantX registry names."""
    from Qmes import get_extractor, load_default_recommender, recommend

    result = recommend(
        X,
        y,
        extractor=get_extractor(task),
        recommender=load_default_recommender(task),
        top_k=top_k,
    )
    qmes_ranking = list(result.get("ranking") or result.get("top_k") or [])
    if not qmes_ranking:
        raise RuntimeError("Qmes recommend() returned an empty ranking")

    mapped: list[str] = []
    for qname in qmes_ranking:
        sqx = map_qmes_circuit(qname)
        if sqx not in mapped:
            mapped.append(sqx)

    top = mapped[0]
    return FeatureMapRecommendation(
        feature_map=top,
        source="qmes",
        ranking=mapped,
        qmes_ranking=qmes_ranking,
        details={
            "top_k_qmes": list(result.get("top_k", [])),
            "votes": dict(result.get("votes", {})),
            "mapping": dict(QMES_TO_SUPERQUANTX),
        },
    )


def recommend_feature_map(
    X: np.ndarray,
    y: np.ndarray | None = None,
    *,
    mode: AutoMode = "auto",
    task: Literal["classification", "regression"] = "classification",
) -> FeatureMapRecommendation:
    """Recommend a SuperQuantX feature-map name for the given dataset.

    Args:
        X: Feature matrix.
        y: Labels/targets. Required for Qmes; optional for heuristic.
        mode: ``auto`` tries Qmes then falls back; ``qmes`` requires Qmes;
            ``heuristic`` forces the local meta-feature rule.
        task: Qmes task type (classification or regression).

    Returns:
        FeatureMapRecommendation with a registry feature_map string.
    """
    X = np.asarray(X)
    if mode == "heuristic":
        return recommend_feature_map_heuristic(X, y)

    if mode in ("auto", "qmes"):
        if y is None and mode == "qmes":
            raise ValueError("Qmes mode requires y labels/targets")
        if y is not None and qmes_available():
            try:
                return recommend_feature_map_qmes(X, y, task=task)
            except Exception as exc:
                if mode == "qmes":
                    raise
                logger.warning(
                    "Qmes recommendation failed (%s); using heuristic fallback",
                    exc,
                )
        elif mode == "qmes":
            raise ImportError(
                "Qmes is not installed. Install the optional extra: "
                'pip install "superquantx[qmes]" '
                "(or pip install git+https://github.com/tungduy1704/Qmes.git)"
            )

    return recommend_feature_map_heuristic(X, y)


def resolve_feature_map(
    feature_map: str | None,
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    *,
    auto_mode: AutoMode = "auto",
    task: Literal["classification", "regression"] = "classification",
) -> FeatureMapRecommendation:
    """Resolve a user-supplied feature_map argument.

    ``None`` or ``"auto"`` triggers recommendation. Any other string is
    normalized against the registry without looking at the data.
    """
    if feature_map is None or (
        isinstance(feature_map, str) and feature_map.strip().lower() == "auto"
    ):
        if X is None:
            return FeatureMapRecommendation(
                feature_map=DEFAULT_FEATURE_MAP,
                source="default",
                ranking=[DEFAULT_FEATURE_MAP],
                details={"reason": "auto_without_data"},
            )
        return recommend_feature_map(X, y, mode=auto_mode, task=task)

    if not isinstance(feature_map, str):
        # Caller passed a custom object; leave it alone.
        return FeatureMapRecommendation(
            feature_map=feature_map,  # type: ignore[arg-type]
            source="custom_object",
            ranking=[],
            details={},
        )

    name = normalize_feature_map_name(feature_map)
    return FeatureMapRecommendation(
        feature_map=name,
        source="manual",
        ranking=[name],
    )
