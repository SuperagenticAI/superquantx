# Quantum Agents API Reference

Quantum agents bundle one or more SuperQuantX algorithms behind a task-level interface. Every agent reports a classical baseline next to the quantum answer, computed in the same call, so you can see how the quantum result compares.

All examples on this page run on SuperQuantX 0.3.0 with the built-in simulator.

## Overview

| Agent | What it does | Classical check |
|---|---|---|
| [QuantumPortfolioAgent](#quantumportfolioagent) | Picks `k` of `n` assets with CVaR-VQE on a mean-variance QUBO | Brute-force optimum and a Markowitz solution |
| [QuantumOptimizationAgent](#quantumoptimizationagent) | Weighted MaxCut with QAOA, or a ground state with VQE | Brute force or exact diagonalisation |
| [QuantumClassificationAgent](#quantumclassificationagent) | Trains quantum classifiers, optionally as an ensemble | Tuned RBF SVM and logistic regression on the same folds |
| [QuantumTradingAgent](#quantumtradingagent) | Experimental: portfolio selection only | Same as the portfolio agent |
| [QuantumResearchAgent](#quantumresearchagent) | Experimental: VQE ground-state simulation only | Exact diagonalisation |

Agents do not report a "quantum advantage" figure. Features that are not implemented raise `NotImplementedError` instead of returning placeholder output.

## Base class

### QuantumAgent

```python
from superquantx.algorithms.quantum_agents import QuantumAgent


class MyQuantumAgent(QuantumAgent):
    def _initialize_agent(self):
        # Create the algorithms this agent uses and store them in self.algorithms
        pass

    def solve(self, problem_instance, **kwargs):
        # Return a QuantumResult, for example via self._result(result_dict, start_time)
        pass

    # QuantumAgent is a BaseQuantumAlgorithm, so fit and predict are required too
    def fit(self, X, y=None, **kwargs):
        self.is_fitted = True
        return self

    def predict(self, X, **kwargs):
        return X


agent = MyQuantumAgent(backend='simulator')
print(agent.get_agent_info()['agent_type'])
```

**Constructor:** `QuantumAgent(backend, agent_config=None, shots=1024, **kwargs)`

- `backend` (str or backend): backend name such as `'simulator'` or a backend instance
- `agent_config` (dict, optional): configuration stored on the agent
- `shots` (int): measurement shots
- `**kwargs`: base algorithm options such as `seed`

**Methods:**

- `_initialize_agent()`: abstract; create the agent's algorithms
- `solve(problem_instance, **kwargs)`: abstract; returns a `QuantumResult`
- `get_agent_info()`: agent type, configuration, algorithms and metrics

**Attributes:** `algorithms`, `results_history`, `performance_metrics`, `agent_config`.

`QuantumResult` has the fields `result` (dict), `metadata` (dict), `execution_time` (seconds), `backend_info`, `error` and `intermediate_results`.

## QuantumPortfolioAgent

Selects `n_select` of `n` assets with equal weights. The selection problem is encoded as a diagonal Hamiltonian; a RealAmplitudes circuit is trained to minimise the CVaR of the energy, and the best sampled selection is returned. The brute-force optimum of the same QUBO and a long-only Markowitz portfolio are computed in the same call. Practical up to about 10 assets.

**Constructor:** `QuantumPortfolioAgent(backend, n_select=None, risk_aversion=1.0, reps=2, alpha=0.25, n_restarts=5, maxiter=300, **kwargs)`

- `n_select`: number of assets to hold (default: half the assets)
- `risk_aversion`: weight of the variance term
- `reps`: ansatz repetitions
- `alpha`: CVaR fraction (1.0 is ordinary VQE)
- `n_restarts`, `maxiter`: optimizer restarts and iterations per restart
- `seed` (keyword): makes the run reproducible

**Methods:**

- `fit(returns)`: estimate mean returns and covariance from a (periods x assets) matrix
- `solve(returns=None)`: run the selection; refits if returns are given
- `predict(returns=None)`: equal weights of the selected assets

```python
import numpy as np
from superquantx import QuantumPortfolioAgent

rng = np.random.default_rng(42)
returns = rng.multivariate_normal(
    mean=[0.08, 0.12, 0.10, 0.15],
    cov=[[0.16, 0.05, 0.03, 0.02],
         [0.05, 0.25, 0.04, 0.03],
         [0.03, 0.04, 0.20, 0.05],
         [0.02, 0.03, 0.05, 0.30]],
    size=252,
) / 100

agent = QuantumPortfolioAgent(backend='simulator', n_select=2, risk_aversion=2.0, seed=0)
result = agent.solve(returns)
r = result.result

print("Quantum selection:     ", r['selected_assets'], r['weights'])
print("Brute-force selection: ", r['classical_selected_assets'])
print("Matches optimum:       ", r['matches_classical_optimum'])
print(f"P(optimum) after VQE:   {r['probability_of_optimum']:.3f}")
print(f"Expected return {r['expected_return']:.4f}, risk {r['risk']:.4f}")
print("Markowitz weights:     ", np.round(r['markowitz_reference']['weights'], 3))
```

Keys in `result.result`: `selected_assets`, `weights`, `objective`, `feasible`, `classical_selected_assets`, `classical_objective`, `matches_classical_optimum`, `objective_gap`, `probability_of_optimum`, `optimum_sampled`, `markowitz_reference`, `expected_return`, `risk`, `sharpe_ratio`, plus `cvar_rescaled` and `shots`.

!!! note "Not available in 0.3.0"
    Black-Litterman or factor risk models, Sharpe or risk-parity objectives, per-asset weight limits and sector constraints are not implemented. The agent selects an equal-weight subset; use `markowitz_reference` for continuous weights.

## QuantumOptimizationAgent

**Constructor:** `QuantumOptimizationAgent(backend, problem_type='combinatorial', p=2, algorithms=None, maxiter=200, n_restarts=3, reps=2, **kwargs)`

- `problem_type='combinatorial'`: `solve(adjacency)` runs weighted MaxCut with QAOA and compares against brute force
- `problem_type='continuous'`: `solve(hamiltonian)` runs VQE and compares against exact diagonalisation
- `p`: QAOA layers; `reps`: VQE ansatz repetitions

### MaxCut with QAOA

```python
import numpy as np
from superquantx import QuantumOptimizationAgent

# 4-node graph: edges (0,1), (1,2), (2,3), (3,0) with weight 1 and (0,2) with weight 2
adjacency = np.array([
    [0, 1, 2, 1],
    [1, 0, 1, 0],
    [2, 1, 0, 1],
    [1, 0, 1, 0],
], dtype=float)

agent = QuantumOptimizationAgent(backend='simulator', problem_type='combinatorial', p=2, seed=0)
result = agent.solve(adjacency)

print("Partition:", result.result['bitstring'])
print("Cut value:", result.result['cut_value'], "optimum:", result.result['classical_optimum'])
print("Approximation ratio:", round(result.result['approximation_ratio'], 3))
print("Algorithm:", result.metadata['algorithm_used'])
```

### Ground state with VQE

```python
from superquantx import Hamiltonian, QuantumOptimizationAgent

# Two-qubit transverse-field Ising model
H = Hamiltonian.ising_model(2, J=1.0, h=0.5)

agent = QuantumOptimizationAgent(backend='simulator', problem_type='continuous', seed=0)
result = agent.solve(H)
print(f"VQE energy {result.result['ground_energy']:.6f}, exact {result.result['exact_energy']:.6f}")
```

`predict(problem)` returns the MaxCut partition as an array of 0/1, or the VQE energy.

## QuantumClassificationAgent

**Constructor:** `QuantumClassificationAgent(backend, algorithms=None, ensemble_method='voting', cv=5, **kwargs)`

- `algorithms`: any of `'quantum_svm'` (default), `'quantum_nn'`, `'hybrid'` (`'hybrid'` uses `HybridClassifier`, which fails in `fit` in 0.3.0; see the [algorithms guide](../user-guide/algorithms.md#hybrid-classifier))
- `ensemble_method`: `'voting'`, `'weighted'` (weights votes by training accuracy) or `'best'`
- `cv`: folds used by `solve`

**Methods:**

- `fit(X, y)`: train every algorithm and record its training accuracy in `performance_metrics`
- `predict(X)`: predictions from one algorithm, or the ensemble vote
- `solve((X, y))`: cross-validated benchmark of the first algorithm against classical baselines with `compare_to_classical`, then fit on all the data

### Ensemble

```python
from sklearn.datasets import make_classification
from superquantx import QuantumClassificationAgent

X, y = make_classification(n_samples=60, n_features=4, n_informative=3, n_redundant=0, random_state=42)
X_train, X_test = X[:45], X[45:]
y_train, y_test = y[:45], y[45:]

agent = QuantumClassificationAgent(
    backend='simulator',
    algorithms=['quantum_svm', 'quantum_nn'],
    ensemble_method='weighted',
    seed=0,
)
agent.algorithms['quantum_nn'].max_epochs = 20  # keep the example quick

agent.fit(X_train, y_train)
predictions = agent.predict(X_test)
print(f"Ensemble test accuracy: {(predictions == y_test).mean():.3f}")
print("Training accuracy per algorithm:", agent.performance_metrics)
```

### Benchmark against classical models

```python
from sklearn.datasets import make_classification
from superquantx import QuantumClassificationAgent

X, y = make_classification(n_samples=60, n_features=4, n_informative=3, n_redundant=0, random_state=0)

agent = QuantumClassificationAgent(backend='simulator', cv=3, seed=0)
result = agent.solve((X, y))
r = result.result

print(f"QuantumSVM CV accuracy: {r['cv_accuracy']:.3f} +/- {r['cv_accuracy_std']:.3f}")
print(f"Best classical ({r['best_classical']}): {r['best_classical_cv_accuracy']:.3f}")
print(f"p-value: {r['p_value']:.3f}")
print("Verdict:", r['verdict'])
```

!!! note "Not available in 0.3.0"
    There is no `'stacking'` ensemble and no `auto_tune` option.

## QuantumTradingAgent

!!! warning "Experimental"
    Only `strategy='quantum_portfolio'` is implemented. `deploy(returns)` runs the portfolio agent and returns its result. Other strategies (for example momentum or mean reversion) and `predict()` raise `NotImplementedError`. There are no trading signals or backtests.

**Constructor:** `QuantumTradingAgent(backend='simulator', strategy='quantum_portfolio', risk_tolerance=0.5, markets=None, **kwargs)`. Higher `risk_tolerance` puts less weight on variance.

```python
import numpy as np
from superquantx import QuantumTradingAgent

rng = np.random.default_rng(1)
returns = rng.normal(0.001, 0.02, size=(250, 4))  # 250 periods, 4 assets

agent = QuantumTradingAgent(strategy='quantum_portfolio', risk_tolerance=0.4, seed=0)
result = agent.deploy(returns)
print("Selected assets:", result.result['selected_assets'])
print("Matches brute-force optimum:", result.result['matches_classical_optimum'])
print(f"Sharpe ratio of selection: {result.result['sharpe_ratio']:.3f}")
```

## QuantumResearchAgent

!!! warning "Experimental"
    Only ground-state simulation is implemented. `investigate()` (automated research planning, hypothesis generation, experiment design) raises `NotImplementedError`.

**Constructor:** `QuantumResearchAgent(backend='simulator', domain='materials_science', **kwargs)`

`simulate_ground_state(hamiltonian=None)` runs VQE and reports the exact energy. With no argument it uses H2 in STO-3G at 0.735 Angstrom.

```python
from superquantx import QuantumResearchAgent

agent = QuantumResearchAgent(seed=0)
result = agent.simulate_ground_state()
print(result.result['hamiltonian'])
print(f"VQE {result.result['ground_energy']:.6f} Ha, exact {result.result['exact_energy']:.6f} Ha")
```

## Combining agents

Agents are ordinary Python objects, so you can chain them yourself:

```python
import numpy as np
from superquantx import QuantumOptimizationAgent, QuantumPortfolioAgent

rng = np.random.default_rng(7)
returns = rng.normal(0.001, 0.02, size=(250, 4))

portfolio = QuantumPortfolioAgent(backend='simulator', n_select=2, seed=0).solve(returns)
held = portfolio.result['selected_assets']

# Split the held assets' correlation graph with MaxCut
corr = np.abs(np.corrcoef(returns[:, held].T))
np.fill_diagonal(corr, 0)
split = QuantumOptimizationAgent(backend='simulator', seed=0).solve(corr)

print("Held:", held, "split:", split.result['bitstring'])
for name, res in [('portfolio', portfolio), ('maxcut', split)]:
    print(f"{name}: {res.execution_time:.2f}s, error={res.error}")
```

## Error handling

```python
import numpy as np
from superquantx import QuantumResearchAgent, QuantumTradingAgent

try:
    QuantumTradingAgent(strategy='momentum').deploy(np.zeros((10, 3)))
except NotImplementedError as e:
    print("Not implemented:", e)

try:
    QuantumResearchAgent().investigate("Find a room-temperature superconductor")
except NotImplementedError as e:
    print("Not implemented:", e)
```

Invalid arguments raise `ValueError`, for example an unknown algorithm name in `QuantumClassificationAgent` or a `problem_type` other than `'combinatorial'` or `'continuous'`.

!!! tip "Choosing an agent"
    Start on the `'simulator'` backend. Every agent accepts any SuperQuantX gate-model backend name, for example `'qiskit'` or `'pennylane'`, once the matching extra is installed.
