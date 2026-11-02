"""Tests for superquantx.benchmarks.compare_to_classical."""

import numpy as np
import pytest
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier

import superquantx as sqx
from superquantx.benchmarks import (
    ComparisonResult,
    compare_to_classical,
    corrected_paired_ttest,
)


@pytest.fixture(scope="module")
def moons():
    return make_moons(n_samples=40, noise=0.2, random_state=0)


@pytest.fixture(scope="module")
def result(moons):
    X, y = moons
    return compare_to_classical(
        X, y, sqx.QuantumSVM(backend="simulator"), cv=4, seed=1,
        bandwidth_grid=(0.1, 1.0), C_grid=(1.0, 100.0), rbf_gamma_grid=(0.1, 1.0),
    )


def test_exported_from_package():
    assert sqx.compare_to_classical is compare_to_classical


def test_result_structure(result):
    assert isinstance(result, ComparisonResult)
    names = [s.name for s in result.scores]
    assert names[0].startswith("QuantumSVM")
    assert any("untuned" in n for n in names)
    assert any("RBF" in n for n in names)
    assert any("Logistic" in n for n in names)
    for s in result.scores:
        assert len(s.fold_accuracies) == 4
        assert 0.0 <= s.accuracy_mean <= 1.0
        assert s.accuracy_std >= 0.0
        assert s.wall_time_s > 0.0


def test_equal_tuning_budget(result):
    q = result.quantum
    rbf = next(s for s in result.scores if "RBF" in s.name)
    assert q.grid_size == rbf.grid_size == 4
    assert all(set(p) == {"bandwidth", "C"} for p in q.best_params)
    assert q.hardware_circuits and q.hardware_circuits > 0


def test_best_classical_excludes_quantum_rows(result):
    assert result.best_classical.is_quantum is False
    assert result.delta_vs_best_classical == pytest.approx(
        result.quantum.accuracy_mean - result.best_classical.accuracy_mean
    )
    assert 0.0 <= result.p_value_vs_best_classical <= 1.0


def test_table_and_dict(result):
    table = result.table()
    assert "arXiv:2610.42029" in table
    assert "identical folds" in table
    assert str(result) == table
    d = result.to_dict()
    assert d["cv"] == 4 and "verdict" in d


def test_same_folds_are_deterministic(moons):
    X, y = moons
    kw = {"cv": 3, "seed": 5}
    a = compare_to_classical(X, y, KNeighborsClassifier(), **kw)
    b = compare_to_classical(X, y, KNeighborsClassifier(), **kw)
    assert [s.fold_accuracies for s in a.scores] == [s.fold_accuracies for s in b.scores]


def test_generic_estimator_path(moons):
    X, y = moons
    r = compare_to_classical(X, y, KNeighborsClassifier(), cv=3, quantum_name="kNN")
    assert r.quantum.name == "kNN"
    assert r.quantum.hardware_circuits is None
    assert len(r.scores) == 3


def test_corrected_ttest():
    assert corrected_paired_ttest([0.9, 0.9], [0.9, 0.9], 80, 20) == 1.0
    p = corrected_paired_ttest([0.9, 0.8, 0.85, 0.95], [0.7, 0.6, 0.65, 0.7], 75, 25)
    assert 0.0 < p < 0.05
    assert np.isnan(corrected_paired_ttest([0.9], [0.8], 80, 20))
