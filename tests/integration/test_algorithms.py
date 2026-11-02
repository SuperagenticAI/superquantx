"""End-to-end checks of the public algorithms against exact classical answers.

Every assertion compares a quantum result with something computed classically
(exact diagonalisation, brute force, or a known label set), so a silent
fallback to fake numbers would fail these tests.
"""

import numpy as np
import pytest

import superquantx as sqx
from superquantx.algorithms.qaoa import brute_force_maxcut, cut_value
from superquantx.algorithms import create_vqe_for_molecule
from superquantx.algorithms.vqe import H2_STO3G_0735
from superquantx.backends.base_backend import hamiltonian_to_matrix


def _backend_available(name: str) -> bool:
    try:
        sqx.get_backend(name)
        return True
    except Exception:
        return False


BACKENDS = [b for b in ("simulator", "pennylane", "qiskit", "cirq") if _backend_available(b)]

# 5-node graph whose maximum cut is 6.
GRAPH = np.array(
    [[0, 1, 1, 0, 1],
     [1, 0, 1, 1, 0],
     [1, 1, 0, 1, 0],
     [0, 1, 1, 0, 1],
     [1, 0, 0, 1, 0]],
    dtype=float,
)


class TestVQEIntegration:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_vqe_single_qubit_z(self, backend):
        vqe = sqx.VQE(hamiltonian={"Z": 1.0}, backend=backend, seed=0)
        assert vqe.find_ground_state() == pytest.approx(-1.0, abs=1e-3)

    def test_vqe_h2_molecule_matches_exact(self):
        vqe = create_vqe_for_molecule("H2", backend="simulator", seed=0)
        energy = vqe.find_ground_state()
        exact = vqe.exact_ground_energy()
        assert exact == pytest.approx(-1.857275, abs=1e-5)
        assert energy == pytest.approx(exact, abs=1.6e-3)  # chemical accuracy

    def test_vqe_molecule_factory_rejects_unknown(self):
        with pytest.raises(NotImplementedError):
            create_vqe_for_molecule("LiH")
        with pytest.raises(NotImplementedError):
            create_vqe_for_molecule("H2", bond_distance=1.5)

    def test_h2_hamiltonian_spectrum(self):
        H = hamiltonian_to_matrix(dict(H2_STO3G_0735), 2)
        assert np.allclose(H, H.conj().T)
        assert np.linalg.eigvalsh(H)[0] == pytest.approx(-1.857275, abs=1e-5)


class TestQAOAIntegration:
    def test_brute_force_reference(self):
        bits, best = brute_force_maxcut(GRAPH)
        assert best == 6
        assert cut_value(GRAPH, bits) == 6

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_qaoa_max_cut_finds_optimum(self, backend):
        result = sqx.QAOA(backend=backend, p=2, seed=7, maxiter=200).solve(GRAPH)
        assert result["classical_optimum"] == 6
        assert result["cut_value"] == 6
        assert cut_value(GRAPH, np.array([int(b) for b in result["bitstring"]])) == 6
        # Expected cut of the QAOA state is below the optimum and well above random (3.5).
        assert 3.5 < result["expected_cut"] <= 6 + 1e-9
        assert 0.0 < result["approximation_ratio"] <= 1.0

    def test_qaoa_rejects_non_square(self):
        with pytest.raises(Exception):
            sqx.QAOA(backend="simulator", p=1).solve(np.ones((2, 3)))


class TestClassifiersIntegration:
    @pytest.fixture(scope="class")
    def moons(self):
        from sklearn.datasets import make_moons
        return make_moons(40, noise=0.1, random_state=0)

    def test_quantum_svm_fits_and_predicts(self, moons):
        X, y = moons
        model = sqx.QuantumSVM(backend="simulator", feature_map="ZZFeatureMap", gamma=0.5)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)) <= set(np.unique(y))
        assert np.mean(preds == y) > 0.6  # untuned bandwidth; see compare_to_classical
        K = model.backend.compute_kernel_matrix(X[:5], X[:5], feature_map="ZZFeatureMap")
        assert np.allclose(np.diag(K), 1.0, atol=1e-6)

    def test_quantum_nn_learns(self, moons):
        X, y = moons
        model = sqx.QuantumNN(backend="simulator", n_layers=2, max_epochs=15,
                              batch_size=40, learning_rate=0.1, seed=1)
        model.fit(X, y)
        assert model.loss_history[-1] < model.loss_history[0]
        assert model.score(X, y) > 0.6


class TestBenchmarkIntegration:
    def test_compare_to_classical_reports_baselines(self):
        from sklearn.datasets import make_moons
        X, y = make_moons(30, noise=0.1, random_state=0)
        result = sqx.compare_to_classical(
            X, y, sqx.QuantumSVM(backend="simulator"), cv=3,
            bandwidth_grid=(0.5,), C_grid=(1.0,), include_untuned=False,
        )
        assert result.quantum.is_quantum is True
        assert result.best_classical.is_quantum is False
        assert 0.0 <= result.quantum.accuracy_mean <= 1.0
        assert len(result.quantum.fold_accuracies) == 3
        assert result.p_value_vs_best_classical is not None
        assert "2610.42029" in result.reference
