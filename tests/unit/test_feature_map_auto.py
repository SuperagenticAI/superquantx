"""Unit tests for Qmes-inspired feature_map='auto' selection (no hardware)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from superquantx.utils.feature_map_auto import (
    FeatureMapRecommendation,
    extract_lightweight_meta_features,
    qmes_available,
    recommend_feature_map,
    recommend_feature_map_heuristic,
    resolve_feature_map,
)
from superquantx.utils.feature_map_registry import (
    FEATURE_MAP_REGISTRY,
    QMES_TO_SUPERQUANTX,
    list_feature_maps,
    map_qmes_circuit,
    normalize_feature_map_name,
)


class TestFeatureMapRegistry:
    def test_registry_contains_expected_names(self):
        names = list_feature_maps()
        assert "ZZFeatureMap" in names
        assert "PauliFeatureMap" in names
        assert "AmplitudeMap" in names
        assert "AngleEncoding" in names
        assert "ZFeatureMap" in names
        assert set(names) == set(FEATURE_MAP_REGISTRY)

    def test_qmes_mapping_covers_full_pool(self):
        expected = {"unit", "SRx", "RY", "HERx", "RY_CX", "ZFM", "HD"}
        assert set(QMES_TO_SUPERQUANTX) == expected
        for qname, sqx in QMES_TO_SUPERQUANTX.items():
            assert sqx in FEATURE_MAP_REGISTRY
            assert map_qmes_circuit(qname) == sqx

    def test_normalize_aliases(self):
        assert normalize_feature_map_name("zz") == "ZZFeatureMap"
        assert normalize_feature_map_name("AngleEncoding") == "AngleEncoding"
        assert normalize_feature_map_name("zfm") == "ZFeatureMap"
        with pytest.raises(ValueError):
            normalize_feature_map_name("not-a-real-map")


class TestHeuristicSelector:
    def test_meta_features_shape(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 4))
        y = np.array([0] * 15 + [1] * 15)
        meta = extract_lightweight_meta_features(X, y)
        assert meta["n_samples"] == 30.0
        assert meta["n_features"] == 4.0
        assert 0.0 <= meta["class_imbalance"] <= 1.0
        assert 0.0 <= meta["linear_sep_proxy"] <= 1.0

    def test_heuristic_returns_registry_name(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(40, 3))
        y = (X[:, 0] > 0).astype(int)
        rec = recommend_feature_map_heuristic(X, y)
        assert isinstance(rec, FeatureMapRecommendation)
        assert rec.source == "heuristic"
        assert rec.feature_map in FEATURE_MAP_REGISTRY
        assert rec.ranking[0] == rec.feature_map

    def test_high_dim_scarce_prefers_amplitude(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(20, 20))
        y = rng.integers(0, 2, size=20)
        rec = recommend_feature_map_heuristic(X, y)
        assert rec.feature_map == "AmplitudeMap"
        assert rec.details["reason"] == "high_dim_scarce_samples"

    def test_mode_heuristic_forces_local_path(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(25, 4))
        y = rng.integers(0, 2, size=25)
        rec = recommend_feature_map(X, y, mode="heuristic")
        assert rec.source == "heuristic"


class TestResolveFeatureMap:
    def test_manual_string(self):
        rec = resolve_feature_map("PauliFeatureMap")
        assert rec.source == "manual"
        assert rec.feature_map == "PauliFeatureMap"

    def test_auto_without_data_defaults(self):
        rec = resolve_feature_map("auto")
        assert rec.feature_map == "ZZFeatureMap"
        assert rec.source == "default"

    def test_auto_with_data_uses_selector(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(20, 4))
        y = rng.integers(0, 2, size=20)
        rec = resolve_feature_map("auto", X=X, y=y, auto_mode="heuristic")
        assert rec.feature_map in FEATURE_MAP_REGISTRY
        assert rec.source == "heuristic"

    def test_none_is_auto(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(16, 3))
        y = rng.integers(0, 2, size=16)
        rec = resolve_feature_map(None, X=X, y=y, auto_mode="heuristic")
        assert rec.source == "heuristic"


class TestQmesOptionalPath:
    def test_qmes_available_is_bool(self):
        assert isinstance(qmes_available(), bool)

    def test_qmes_mode_without_install_raises(self):
        rng = np.random.default_rng(6)
        X = rng.normal(size=(12, 3))
        y = rng.integers(0, 2, size=12)
        with patch(
            "superquantx.utils.feature_map_auto.qmes_available",
            return_value=False,
        ):
            with pytest.raises(ImportError, match="Qmes is not installed"):
                recommend_feature_map(X, y, mode="qmes")

    def test_qmes_recommendation_is_mapped(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(20, 4))
        y = rng.integers(0, 2, size=20)

        fake_result = {
            "top_k": ["HERx", "RY", "unit"],
            "ranking": ["HERx", "RY", "unit", "SRx", "RY_CX", "HD", "ZFM"],
            "votes": {"HERx": 5, "RY": 4},
        }

        mock_qmes = MagicMock()
        mock_qmes.get_extractor.return_value = MagicMock()
        mock_qmes.load_default_recommender.return_value = MagicMock()
        mock_qmes.recommend.return_value = fake_result

        with patch.dict("sys.modules", {"Qmes": mock_qmes}):
            from superquantx.utils import feature_map_auto as fma

            rec = fma.recommend_feature_map_qmes(X, y)
            assert rec.source == "qmes"
            assert rec.feature_map == "PauliFeatureMap"  # HERx mapped
            assert rec.qmes_ranking[0] == "HERx"
            assert "AngleEncoding" in rec.ranking  # RY mapped
            assert "AmplitudeMap" in rec.ranking  # unit mapped

    def test_auto_falls_back_when_qmes_errors(self):
        rng = np.random.default_rng(8)
        X = rng.normal(size=(18, 4))
        y = rng.integers(0, 2, size=18)

        with patch(
            "superquantx.utils.feature_map_auto.qmes_available",
            return_value=True,
        ), patch(
            "superquantx.utils.feature_map_auto.recommend_feature_map_qmes",
            side_effect=RuntimeError("boom"),
        ):
            rec = recommend_feature_map(X, y, mode="auto")
            assert rec.source == "heuristic"


class TestQuantumSVMAutoWiring:
    def test_qsvm_auto_sets_selected_feature_map(self):
        from superquantx.algorithms.quantum_svm import QuantumSVM

        rng = np.random.default_rng(9)
        X = rng.normal(size=(20, 3))
        y = rng.integers(0, 2, size=20)

        qsvm = QuantumSVM(
            backend="simulator",
            feature_map="auto",
            auto_feature_map_mode="heuristic",
            shots=32,
        )

        # Avoid expensive kernel path: stub backend kernel + SVM fit pieces
        qsvm.backend.compute_kernel_matrix = MagicMock(
            return_value=np.eye(X.shape[0])
        )
        # predict during fit uses kernel against training data
        def _kernel(X1, X2=None, feature_map=None, shots=None):
            n1 = len(X1)
            n2 = len(X2) if X2 is not None else n1
            return np.eye(max(n1, n2))[:n1, :n2]

        qsvm.backend.compute_kernel_matrix = _kernel
        qsvm.fit(X, y)

        assert qsvm.selected_feature_map_ in FEATURE_MAP_REGISTRY
        assert qsvm.feature_map_recommendation_ is not None
        assert qsvm.feature_map_recommendation_["source"] == "heuristic"
        assert qsvm.is_fitted

    def test_qsvm_manual_unchanged(self):
        from superquantx.algorithms.quantum_svm import QuantumSVM

        rng = np.random.default_rng(10)
        X = rng.normal(size=(16, 3))
        y = rng.integers(0, 2, size=16)

        qsvm = QuantumSVM(
            backend="simulator",
            feature_map="ZZFeatureMap",
            shots=32,
        )

        def _kernel(X1, X2=None, feature_map=None, shots=None):
            n1 = len(X1)
            n2 = len(X2) if X2 is not None else n1
            return np.eye(max(n1, n2))[:n1, :n2]

        qsvm.backend.compute_kernel_matrix = _kernel
        qsvm.fit(X, y)
        assert qsvm.selected_feature_map_ == "ZZFeatureMap"
        assert qsvm.feature_map_recommendation_["source"] == "manual"
