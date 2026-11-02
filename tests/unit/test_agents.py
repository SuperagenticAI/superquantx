"""Agents must return real computations with classical checks, never fabricated numbers."""

import numpy as np
import pytest
from sklearn.datasets import make_moons

import superquantx as sqx
from superquantx.algorithms.quantum_agents import portfolio_qubo_values


@pytest.fixture(scope="module")
def returns():
    rng = np.random.default_rng(3)
    mu = np.array([0.010, 0.008, 0.012, 0.006])
    A = rng.normal(size=(4, 4)) * 0.01
    return rng.multivariate_normal(mu, A @ A.T + np.eye(4) * 1e-4, size=200)


def test_portfolio_qubo_penalises_wrong_cardinality(returns):
    v = portfolio_qubo_values(returns.mean(0), np.cov(returns.T), k=2)
    feasible = [i for i in range(16) if bin(i).count("1") == 2]
    infeasible = [i for i in range(16) if bin(i).count("1") != 2]
    assert v[feasible].min() < v[infeasible].min()


def test_portfolio_agent_matches_brute_force(returns):
    res = sqx.QuantumPortfolioAgent(backend="simulator", n_select=2, seed=0).solve(returns)
    assert res.error is None
    r = res.result
    assert r["feasible"]
    assert r["matches_classical_optimum"]
    assert r["objective"] == pytest.approx(r["classical_objective"])
    assert np.isclose(r["weights"].sum(), 1.0)
    assert np.isclose(r["markowitz_reference"]["weights"].sum(), 1.0)
    assert "quantum_advantage" not in r


def test_portfolio_agent_is_deterministic(returns):
    a = sqx.QuantumPortfolioAgent(backend="simulator", n_select=2, seed=1).solve(returns).result
    b = sqx.QuantumPortfolioAgent(backend="simulator", n_select=2, seed=1).solve(returns).result
    assert a["selected_assets"] == b["selected_assets"]
    assert a["probability_of_optimum"] == pytest.approx(b["probability_of_optimum"])


def test_optimization_agent_maxcut():
    W = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]], float)
    res = sqx.QuantumOptimizationAgent(backend="simulator", seed=1).solve(W)
    assert res.error is None
    assert res.result["classical_optimum"] == 4.0
    assert res.result["cut_value"] <= res.result["classical_optimum"]
    assert 0 < res.result["approximation_ratio"] <= 1.0


def test_optimization_agent_ground_state():
    res = sqx.QuantumOptimizationAgent(
        backend="simulator", problem_type="continuous", seed=0
    ).solve({"ZZ": 1.0, "XI": 0.5})
    assert res.result["ground_energy"] == pytest.approx(res.result["exact_energy"], abs=1e-4)


def test_classification_agent_reports_cv_against_classical():
    X, y = make_moons(n_samples=40, noise=0.2, random_state=0)
    res = sqx.QuantumClassificationAgent(backend="simulator", cv=3, seed=0).solve(
        (X, y), bandwidth_grid=(0.1, 1.0), C_grid=(1.0, 100.0), rbf_gamma_grid=(0.1, 1.0)
    )
    r = res.result
    assert 0.0 <= r["cv_accuracy"] <= 1.0
    assert "RBF" in r["best_classical"] or "Logistic" in r["best_classical"]
    assert "verdict" in r


def test_research_agent_is_honest():
    agent = sqx.QuantumResearchAgent(backend="simulator", seed=0)
    with pytest.raises(NotImplementedError):
        agent.investigate("Find a room-temperature superconductor")
    r = agent.simulate_ground_state().result
    assert r["exact_energy"] == pytest.approx(-1.857275, abs=1e-5)
    assert r["error"] < 1e-4


def test_trading_agent_has_no_fabricated_numbers(returns):
    agent = sqx.QuantumTradingAgent(backend="simulator", seed=0)
    with pytest.raises(NotImplementedError):
        agent.predict(returns)
    with pytest.raises(NotImplementedError):
        sqx.QuantumTradingAgent(backend="simulator", strategy="momentum").deploy(returns)
    r = agent.deploy(returns).result
    assert "quantum_advantage" not in r
    assert "classical_objective" in r
