"""Quantum agents demo: every quantum answer next to an exact classical check.

Run:
    uv run python examples/quantum_agents_demo.py

What it does (all circuits are classically simulated):
1. QuantumPortfolioAgent picks 3 of 6 assets with CVaR-VQE and is checked
   against brute force over all selections and a classical Markowitz solution.
2. QuantumClassificationAgent trains a quantum-kernel SVM and benchmarks it
   against a tuned RBF SVM and logistic regression on identical CV folds.

Set SQX_BACKEND to choose the backend (default: pennylane, falls back to the
built-in simulator if PennyLane is not installed). The script exits non-zero
if any step fails.
"""

import os
import sys
import time
import warnings

import numpy as np
from sklearn.datasets import make_moons

import superquantx as sqx

warnings.filterwarnings("ignore")


def pick_backend() -> str:
    name = os.environ.get("SQX_BACKEND", "pennylane")
    try:
        sqx.get_backend(name)
        return name
    except Exception as exc:  # backend extra not installed
        print(f"Backend '{name}' unavailable ({exc}); using 'simulator'")
        return "simulator"


def portfolio_demo(backend: str) -> bool:
    print("\n1. QuantumPortfolioAgent: choose 3 of 6 assets")
    print("-" * 60)
    rng = np.random.default_rng(3)
    mu = np.array([0.010, 0.008, 0.012, 0.006, 0.009, 0.011])
    A = rng.normal(size=(6, 6)) * 0.01
    returns = rng.multivariate_normal(mu, A @ A.T + np.eye(6) * 1e-4, size=250)

    agent = sqx.QuantumPortfolioAgent(backend=backend, n_select=3, seed=0)
    res = agent.solve(returns)
    if res.error:
        print(f"FAILED: {res.error}")
        return False
    r = res.result
    mk = r["markowitz_reference"]
    print(f"   quantum (CVaR-VQE) selection : {r['selected_assets']}  objective {r['objective']:.6f}")
    print(f"   brute-force optimum          : {r['classical_selected_assets']}  objective {r['classical_objective']:.6f}")
    print(f"   matches exact optimum        : {r['matches_classical_optimum']}"
          f"  (P(optimum) in final state = {r['probability_of_optimum']:.2f})")
    print(f"   quantum portfolio            : return {r['expected_return']:.4f}, risk {r['risk']:.4f}")
    print(f"   classical Markowitz weights  : return {mk['expected_return']:.4f}, risk {mk['risk']:.4f}"
          f"  {np.round(mk['weights'], 2)}")
    print(f"   time {res.execution_time:.1f}s on {backend}")
    return True


def classification_demo(backend: str) -> bool:
    print("\n2. QuantumClassificationAgent: make_moons, 5-fold CV")
    print("-" * 60)
    X, y = make_moons(n_samples=100, noise=0.2, random_state=0)
    agent = sqx.QuantumClassificationAgent(backend=backend, seed=0)
    res = agent.solve((X, y))
    if res.error:
        print(f"FAILED: {res.error}")
        return False
    print(res.result["benchmark"].table())
    print(f"   time {res.execution_time:.1f}s on {backend}")
    return True


def main() -> int:
    print(f"SuperQuantX {sqx.__version__}: quantum agents with classical baselines")
    backend = pick_backend()
    start = time.time()
    ok = portfolio_demo(backend) and classification_demo(backend)
    print("\n" + ("All agent demos completed" if ok else "Agent demo FAILED")
          + f" in {time.time() - start:.1f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
