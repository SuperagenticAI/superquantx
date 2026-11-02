"""Quantum agents: task-level wrappers around the quantum algorithms.

An agent bundles one or more quantum algorithms for a problem domain and,
crucially, always reports a classical baseline next to the quantum answer.
No agent in this module reports a "quantum advantage": every number it returns
is either computed by a (classically simulated) quantum circuit or by an exact
classical method run in the same call, and the two are labelled as such.

Agents:

* :class:`QuantumPortfolioAgent`: pick ``k`` of ``n`` assets by minimising a
  mean-variance QUBO with VQE; checked against brute force and a classical
  Markowitz solution.
* :class:`QuantumClassificationAgent`: trains quantum classifiers and
  benchmarks them against tuned classical models with
  :func:`superquantx.benchmarks.compare_to_classical`.
* :class:`QuantumOptimizationAgent`: weighted MaxCut with QAOA (checked by
  brute force) or ground-state search with VQE (checked by exact
  diagonalisation).
* :class:`QuantumTradingAgent` and :class:`QuantumResearchAgent`: experimental.
  Only the parts backed by real computation are implemented; everything else
  raises ``NotImplementedError``.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score

from .base_algorithm import BaseQuantumAlgorithm, QuantumResult
from .hybrid_classifier import HybridClassifier
from .qaoa import QAOA
from .quantum_nn import QuantumNN
from .quantum_svm import QuantumSVM
from .vqe import H2_STO3G_0735, VQE

logger = logging.getLogger(__name__)


class QuantumAgent(BaseQuantumAlgorithm, ABC):
    """Base class for quantum agents.

    Args:
        backend: Quantum backend for circuit execution
        agent_config: Configuration dictionary for the agent
        shots: Number of measurement shots
        **kwargs: Additional parameters (for example ``seed``)

    """

    def __init__(
        self,
        backend: str | Any,
        agent_config: dict[str, Any] | None = None,
        shots: int = 1024,
        **kwargs
    ) -> None:
        super().__init__(backend=backend, shots=shots, **kwargs)
        self.agent_config = agent_config or {}
        self.algorithms: dict[str, Any] = {}
        self.results_history: list[Any] = []
        self.performance_metrics: dict[str, Any] = {}
        self._initialize_agent()
        logger.info("Initialized %s", self.__class__.__name__)

    @abstractmethod
    def _initialize_agent(self) -> None:
        """Initialize agent-specific algorithms and configurations."""

    @abstractmethod
    def solve(self, problem_instance: Any, **kwargs) -> QuantumResult:
        """Solve a problem using the quantum agent."""

    def _result(self, result: dict[str, Any], start: float, **metadata) -> QuantumResult:
        return QuantumResult(
            result=result,
            metadata=metadata,
            execution_time=time.time() - start,
            backend_info=self.get_circuit_info(),
        )

    def get_agent_info(self) -> dict[str, Any]:
        """Get information about the agent and its algorithms."""
        return {
            'agent_type': self.__class__.__name__,
            'algorithms': list(self.algorithms.keys()),
            'config': self.agent_config,
            'backend': type(self.backend).__name__,
            'performance_metrics': self.performance_metrics,
        }


# ---------------------------------------------------------------------------
# Portfolio selection
# ---------------------------------------------------------------------------

def portfolio_qubo_values(mu: np.ndarray, sigma: np.ndarray, k: int,
                          risk_aversion: float = 1.0, penalty: float | None = None) -> np.ndarray:
    """Objective of every asset selection, indexed by big-endian bitstring.

    For selection ``x`` (``x_i = 1`` means asset i is held, equal weight ``1/k``):

        f(x) = risk_aversion * x^T Sigma x / k^2 - mu^T x / k + penalty * (sum(x) - k)^2

    The penalty enforces holding exactly ``k`` assets. By default it is 1.5 times
    the spread of the unpenalised objective, which is the smallest safe choice
    and keeps the energy landscape well scaled for VQE.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = len(mu)
    idx = np.arange(2 ** n)
    X = ((idx[:, None] >> (n - 1 - np.arange(n))) & 1).astype(float)
    risk = np.einsum('bi,ij,bj->b', X, sigma, X) / k ** 2
    ret = X @ mu / k
    f0 = risk_aversion * risk - ret
    if penalty is None:
        penalty = 1.5 * max(float(f0.max() - f0.min()), 1e-12)
    return f0 + penalty * (X.sum(axis=1) - k) ** 2


class QuantumPortfolioAgent(QuantumAgent):
    """Pick ``k`` of ``n`` assets with CVaR-VQE, checked against exact classical answers.

    The selection problem is encoded as a diagonal Ising Hamiltonian whose
    ground state is the best equal-weight ``k``-asset portfolio. A
    RealAmplitudes circuit is trained on the chosen backend to minimise the
    CVaR of the energy (the mean of the best ``alpha`` fraction of outcomes,
    Barkoutsos et al., Quantum 4, 256 (2020)), which works much better than the
    plain expectation for combinatorial problems. ``shots`` samples are drawn
    from the final state and the best sampled selection is returned. Two classical references are always
    computed in the same call: the brute-force optimum of the same QUBO and a
    long-only Markowitz solution with continuous weights. Practical up to
    about 10 assets (``2^n`` statevector).

    Args:
        backend: Quantum backend
        n_select: Number of assets to hold (default: half the assets)
        risk_aversion: Weight of the variance term
        reps: Ansatz repetitions
        alpha: CVaR fraction (1.0 is ordinary VQE)
        n_restarts: Optimizer restarts from different seeds
        maxiter: Optimizer iterations per restart
        **kwargs: ``seed`` and other base parameters

    Example:
        >>> agent = QuantumPortfolioAgent(backend='simulator', n_select=2, seed=0)
        >>> res = agent.solve(returns)            # returns: (periods x assets)
        >>> res.result['selected_assets'], res.result['matches_classical_optimum']

    """

    def __init__(
        self,
        backend: str | Any,
        n_select: int | None = None,
        risk_aversion: float = 1.0,
        reps: int = 2,
        alpha: float = 0.25,
        n_restarts: int = 5,
        maxiter: int = 300,
        **kwargs
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.n_select = n_select
        self.risk_aversion = risk_aversion
        self.reps = reps
        self.n_restarts = n_restarts
        self.maxiter = maxiter
        super().__init__(backend=backend, agent_config={
            'n_select': n_select, 'risk_aversion': risk_aversion, 'reps': reps,
        }, **kwargs)
        self.expected_returns = None
        self.covariance_matrix = None
        self.optimal_weights = None

    def _initialize_agent(self) -> None:
        self.algorithms = {}

    def fit(self, X: np.ndarray, y: np.ndarray | None = None, **kwargs) -> QuantumPortfolioAgent:
        """Estimate mean returns and covariance from a (periods x assets) matrix."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] < 2:
            raise ValueError("Pass historical returns as a (periods x assets) matrix")
        self.expected_returns = X.mean(axis=0)
        self.covariance_matrix = np.cov(X.T)
        self.is_fitted = True
        return self

    def _markowitz(self) -> dict[str, Any]:
        """Classical reference: long-only, fully invested mean-variance weights."""
        mu, sigma, n = self.expected_returns, self.covariance_matrix, len(self.expected_returns)

        def objective(w):
            return self.risk_aversion * w @ sigma @ w - mu @ w

        res = minimize(objective, np.full(n, 1.0 / n), method='SLSQP',
                       bounds=[(0.0, 1.0)] * n,
                       constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}])
        w = np.clip(res.x, 0, None)
        w /= w.sum()
        return {'weights': w, **self._metrics(w)}

    def _metrics(self, w: np.ndarray) -> dict[str, float]:
        ret = float(w @ self.expected_returns)
        risk = float(np.sqrt(w @ self.covariance_matrix @ w))
        return {'expected_return': ret, 'risk': risk,
                'sharpe_ratio': ret / risk if risk > 0 else float('nan')}

    def solve(self, problem_instance: np.ndarray | None = None, **kwargs) -> QuantumResult:
        """Select assets with VQE and report the classical checks.

        Args:
            problem_instance: Optional returns matrix; refits the agent if given.

        Returns:
            QuantumResult whose ``result`` has the quantum selection and weights,
            its objective, the brute-force optimum, whether they match, the
            probability VQE puts on the optimal selection, and a classical
            Markowitz reference.

        """
        start = time.time()
        if problem_instance is not None:
            self.fit(problem_instance)
        if not self.is_fitted:
            raise ValueError("Call fit(returns) or pass returns to solve()")

        n = len(self.expected_returns)
        k = self.n_select or max(1, n // 2)
        values = portfolio_qubo_values(self.expected_returns, self.covariance_matrix, k,
                                       self.risk_aversion)
        # Rescale so feasible selections span [0, 1]: same ground state, but the
        # optimizer sees the differences that matter instead of the penalty wall.
        feasible = np.array([bin(i).count('1') == k for i in range(2 ** n)])
        spread = max(values[feasible].max() - values[feasible].min(), 1e-12)
        H = np.diag((values - values.min()) / spread)

        order = np.argsort(H.diagonal())
        h_sorted = H.diagonal()[order]
        n_params = self.backend.ansatz_param_count('RealAmplitudes', n, reps=self.reps)

        def probabilities(params):
            circuit = self.backend.create_ansatz('RealAmplitudes', n, params, reps=self.reps)
            return np.abs(self.backend.statevector_big_endian(circuit)) ** 2

        def cvar(params):
            # Mean energy of the best alpha fraction of the distribution (exact, from probabilities).
            p_sorted = probabilities(params)[order]
            cum = np.cumsum(p_sorted)
            take = np.clip(self.alpha - (cum - p_sorted), 0.0, p_sorted)
            return float(np.sum(take * h_sorted) / self.alpha)

        base_seed = 0 if self.seed is None else self.seed
        best = None
        for r in range(max(1, self.n_restarts)):
            x0 = np.random.default_rng(base_seed + r).uniform(-np.pi, np.pi, n_params)
            res = minimize(cvar, x0, method='COBYLA', options={'maxiter': self.maxiter})
            if best is None or res.fun < best.fun:
                best = res
        probs = probabilities(best.x)
        probs = probs / probs.sum()
        samples = np.random.default_rng(base_seed).choice(len(probs), size=self.shots, p=probs)
        q_index = int(samples[np.argmin(values[samples])])
        c_index = int(np.argmin(values))
        self.optimal_params_ = best.x

        def bits(i):
            return np.array([(i >> (n - 1 - j)) & 1 for j in range(n)])

        q_bits, c_bits = bits(q_index), bits(c_index)
        weights = q_bits / max(1, q_bits.sum())
        self.optimal_weights = weights

        result = {
            'selected_assets': [int(i) for i in np.flatnonzero(q_bits)],
            'weights': weights,
            'objective': float(values[q_index]),
            'cvar_rescaled': float(best.fun),
            'shots': self.shots,
            'feasible': bool(q_bits.sum() == k),
            'classical_selected_assets': [int(i) for i in np.flatnonzero(c_bits)],
            'classical_objective': float(values[c_index]),
            'matches_classical_optimum': bool(q_index == c_index),
            'objective_gap': float(values[q_index] - values[c_index]),
            'probability_of_optimum': float(probs[c_index]),
            'optimum_sampled': bool(np.any(samples == c_index)),
            'markowitz_reference': self._markowitz(),
            **self._metrics(weights),
        }
        self.performance_metrics = {k_: result[k_] for k_ in
                                    ('matches_classical_optimum', 'probability_of_optimum')}
        return self._result(result, start, n_assets=n, n_select=k,
                            method='CVaR-VQE on a mean-variance QUBO; brute force check',
                            n_evaluations=int(best.nfev))

    def predict(self, X: np.ndarray | None = None, **kwargs) -> np.ndarray:
        """Return the equal weights of the VQE selection."""
        if self.optimal_weights is None:
            self.solve(X)
        return self.optimal_weights


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class QuantumClassificationAgent(QuantumAgent):
    """Train quantum classifiers and benchmark them against tuned classical models.

    ``solve((X, y))`` runs :func:`superquantx.benchmarks.compare_to_classical`
    on the first algorithm (QuantumSVM by default) so that the reported
    accuracy is a cross-validated test accuracy next to a tuned RBF SVM and a
    logistic regression on identical folds, then refits on all the data.

    Args:
        backend: Quantum backend
        algorithms: Any of 'quantum_svm', 'quantum_nn', 'hybrid'
        ensemble_method: 'voting', 'weighted' or 'best' (used by predict when
            several algorithms are trained)
        cv: Folds for the benchmark in ``solve``
        **kwargs: Base parameters

    """

    def __init__(
        self,
        backend: str | Any,
        algorithms: list[str] | None = None,
        ensemble_method: str = 'voting',
        cv: int = 5,
        **kwargs
    ) -> None:
        self.available_algorithms = algorithms or ['quantum_svm']
        self.ensemble_method = ensemble_method
        self.cv = cv
        super().__init__(backend=backend, agent_config={
            'algorithms': self.available_algorithms, 'ensemble_method': ensemble_method,
        }, **kwargs)

    def _initialize_agent(self) -> None:
        builders = {
            'quantum_svm': lambda: QuantumSVM(backend=self.backend, shots=self.shots),
            'quantum_nn': lambda: QuantumNN(backend=self.backend, shots=self.shots,
                                            task_type='classification', seed=self.seed),
            'hybrid': lambda: HybridClassifier(backend=self.backend, shots=self.shots),
        }
        for name in self.available_algorithms:
            if name not in builders:
                raise ValueError(f"Unknown algorithm '{name}'. Use {sorted(builders)}")
            self.algorithms[name] = builders[name]()

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> QuantumClassificationAgent:
        """Fit every algorithm. Errors propagate; nothing is silently scored as 0."""
        for name, algorithm in self.algorithms.items():
            algorithm.fit(X, y)
            self.performance_metrics[name] = {
                'train_accuracy': float(accuracy_score(y, algorithm.predict(X)))
            }
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict with one algorithm, or combine several."""
        if not self.is_fitted:
            raise ValueError("Agent must be fitted before prediction")
        preds = {name: np.asarray(a.predict(X)) for name, a in self.algorithms.items()}
        if len(preds) == 1:
            return next(iter(preds.values()))
        if self.ensemble_method == 'best':
            best = max(self.performance_metrics,
                       key=lambda n: self.performance_metrics[n]['train_accuracy'])
            return preds[best]
        weights = {n: (self.performance_metrics[n]['train_accuracy']
                       if self.ensemble_method == 'weighted' else 1.0) for n in preds}
        out = []
        for i in range(len(X)):
            votes: dict[Any, float] = {}
            for n, p in preds.items():
                votes[p[i]] = votes.get(p[i], 0.0) + weights[n]
            out.append(max(votes.items(), key=lambda kv: kv[1])[0])
        return np.array(out)

    def solve(self, problem_instance: tuple[np.ndarray, np.ndarray], **kwargs) -> QuantumResult:
        """Benchmark against classical baselines with CV, then fit on all data.

        Keyword arguments are passed to ``compare_to_classical``.
        """
        from ..benchmarks import compare_to_classical

        start = time.time()
        X, y = problem_instance
        name, algorithm = next(iter(self.algorithms.items()))
        kwargs.setdefault('cv', self.cv)
        kwargs.setdefault('seed', 0 if self.seed is None else self.seed)
        benchmark = compare_to_classical(X, y, algorithm, **kwargs)
        self.fit(X, y)
        result = {
            'algorithm': name,
            'cv_accuracy': benchmark.quantum.accuracy_mean,
            'cv_accuracy_std': benchmark.quantum.accuracy_std,
            'best_classical': benchmark.best_classical.name,
            'best_classical_cv_accuracy': benchmark.best_classical.accuracy_mean,
            'delta_vs_best_classical': benchmark.delta_vs_best_classical,
            'p_value': benchmark.p_value_vs_best_classical,
            'verdict': benchmark.verdict,
            'benchmark': benchmark,
            'train_accuracy': self.performance_metrics[name]['train_accuracy'],
        }
        return self._result(result, start, n_samples=len(X), n_features=X.shape[1],
                            n_classes=len(np.unique(y)), algorithms_used=list(self.algorithms))


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

class QuantumOptimizationAgent(QuantumAgent):
    """Solve small optimisation problems with QAOA or VQE plus an exact classical check.

    * ``problem_type='combinatorial'``: ``solve(adjacency)`` runs weighted
      MaxCut with QAOA and compares against brute force.
    * ``problem_type='continuous'``: ``solve(hamiltonian)`` runs VQE and
      compares against exact diagonalisation.

    Args:
        backend: Quantum backend
        problem_type: 'combinatorial' or 'continuous'
        p: QAOA layers
        **kwargs: Base parameters and extra QAOA/VQE options (``maxiter``,
            ``n_restarts`` for QAOA, ``reps`` for VQE)

    """

    def __init__(
        self,
        backend: str | Any,
        problem_type: str = 'combinatorial',
        p: int = 2,
        algorithms: list[str] | None = None,
        maxiter: int = 200,
        n_restarts: int = 3,
        reps: int = 2,
        **kwargs
    ) -> None:
        if problem_type not in ('combinatorial', 'continuous'):
            raise ValueError("problem_type must be 'combinatorial' or 'continuous'")
        self.problem_type = problem_type
        self.p = p
        self.maxiter = maxiter
        self.n_restarts = n_restarts
        self.reps = reps
        self.available_algorithms = algorithms or (
            ['qaoa'] if problem_type == 'combinatorial' else ['vqe'])
        super().__init__(backend=backend, agent_config={
            'problem_type': problem_type, 'p': p, 'algorithms': self.available_algorithms,
        }, **kwargs)
        self.problem_instance = None

    def _initialize_agent(self) -> None:
        if 'qaoa' in self.available_algorithms:
            self.algorithms['qaoa'] = QAOA(backend=self.backend, p=self.p, shots=self.shots,
                                           maxiter=self.maxiter, n_restarts=self.n_restarts,
                                           seed=self.seed)
        if 'vqe' in self.available_algorithms:
            self.algorithms['vqe'] = VQE(hamiltonian=None, backend=self.backend, shots=self.shots,
                                         maxiter=self.maxiter, reps=self.reps, seed=self.seed)

    def fit(self, X: Any, y: np.ndarray | None = None, **kwargs) -> QuantumOptimizationAgent:
        """Store the problem: an adjacency matrix or a Hamiltonian."""
        self.problem_instance = X
        self.is_fitted = True
        return self

    def solve(self, problem_instance: Any = None, **kwargs) -> QuantumResult:
        """Solve and return the quantum answer with its exact classical check."""
        start = time.time()
        if problem_instance is not None:
            self.fit(problem_instance)
        if self.problem_instance is None:
            raise ValueError("Pass an adjacency matrix (combinatorial) or Hamiltonian (continuous)")

        if self.problem_type == 'combinatorial':
            out = self.algorithms['qaoa'].solve(np.asarray(self.problem_instance, dtype=float))
            out['matches_classical_optimum'] = bool(
                np.isclose(out['cut_value'], out['classical_optimum']))
            return self._result(out, start, problem_type='maxcut', algorithm_used='QAOA',
                                classical_check='brute force')

        vqe = self.algorithms['vqe']
        vqe.fit(self.problem_instance)
        opt = vqe._run_optimization()
        out = {
            'ground_energy': opt['ground_energy'],
            'exact_energy': opt['exact_energy'],
            'error': abs(opt['ground_energy'] - opt['exact_energy']),
            'optimal_params': opt['optimal_params'],
            'n_evaluations': opt['n_iterations'],
        }
        return self._result(out, start, problem_type='ground_state', algorithm_used='VQE',
                            classical_check='exact diagonalisation')

    def predict(self, X: Any = None, **kwargs) -> np.ndarray:
        """MaxCut partition (combinatorial) or energy (continuous)."""
        res = self.solve(X)
        if self.problem_type == 'combinatorial':
            return np.array([int(b) for b in res.result['bitstring']])
        return np.array([res.result['ground_energy']])


# ---------------------------------------------------------------------------
# Experimental agents
# ---------------------------------------------------------------------------

class QuantumTradingAgent(QuantumAgent):
    """Experimental. Only portfolio selection is implemented.

    ``deploy(returns)`` delegates to :class:`QuantumPortfolioAgent` and returns
    its result, including the brute-force and Markowitz baselines. There are no
    trading signals, backtests or "quantum advantage" numbers: strategies other
    than ``'quantum_portfolio'`` raise ``NotImplementedError``.
    """

    def __init__(
        self,
        backend: str | Any = 'simulator',
        strategy: str = 'quantum_portfolio',
        risk_tolerance: float = 0.5,
        markets: list[str] | None = None,
        **kwargs
    ) -> None:
        self.strategy = strategy
        self.risk_tolerance = risk_tolerance
        self.markets = markets or ['stocks']
        kwargs.pop('quantum_advantage_threshold', None)
        super().__init__(backend=backend, agent_config={'strategy': strategy}, **kwargs)

    def _initialize_agent(self) -> None:
        # Higher risk tolerance means less weight on variance.
        self.algorithms['portfolio'] = QuantumPortfolioAgent(
            backend=self.backend, risk_aversion=2.0 * (1.0 - self.risk_tolerance) + 0.1,
            seed=self.seed)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None, **kwargs) -> QuantumTradingAgent:
        self.algorithms['portfolio'].fit(X)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError("Trading signal prediction is not implemented")

    def solve(self, problem_instance: Any, **kwargs) -> QuantumResult:
        return self.deploy(problem_instance, **kwargs)

    def deploy(self, market_data: np.ndarray | None = None, **kwargs) -> QuantumResult:
        """Run quantum portfolio selection on a (periods x assets) returns matrix."""
        if self.strategy != 'quantum_portfolio':
            raise NotImplementedError(
                f"Strategy '{self.strategy}' is not implemented; use 'quantum_portfolio'")
        if market_data is None and not self.algorithms['portfolio'].is_fitted:
            raise ValueError("deploy() needs a (periods x assets) returns matrix")
        return self.algorithms['portfolio'].solve(market_data, **kwargs)


class QuantumResearchAgent(QuantumAgent):
    """Experimental. Only ground-state simulation is implemented.

    ``simulate_ground_state(hamiltonian)`` runs VQE and reports the exact
    energy next to it (default: H2 in STO-3G at 0.735 Angstrom). Automated
    hypothesis generation, literature synthesis and experiment design are not
    implemented and raise ``NotImplementedError`` rather than returning
    template text.
    """

    def __init__(
        self,
        backend: str | Any = 'simulator',
        domain: str = 'materials_science',
        **kwargs
    ) -> None:
        self.domain = domain
        for flag in ('hypothesis_generation', 'experiment_design', 'literature_synthesis'):
            kwargs.pop(flag, None)
        super().__init__(backend=backend, agent_config={'domain': domain}, **kwargs)

    def _initialize_agent(self) -> None:
        self.algorithms['vqe'] = VQE(hamiltonian=dict(H2_STO3G_0735), backend=self.backend,
                                     seed=self.seed)

    def fit(self, X: Any = None, y: np.ndarray | None = None, **kwargs) -> QuantumResearchAgent:
        if X is not None:
            self.algorithms['vqe'].fit(X)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError("QuantumResearchAgent has no predictive model")

    def solve(self, problem_instance: Any = None, **kwargs) -> QuantumResult:
        return self.simulate_ground_state(problem_instance)

    def simulate_ground_state(self, hamiltonian: Any = None) -> QuantumResult:
        """VQE ground-state energy with the exact value for comparison."""
        start = time.time()
        vqe = self.algorithms['vqe']
        vqe.fit(hamiltonian)
        opt = vqe._run_optimization()
        return self._result({
            'ground_energy': opt['ground_energy'],
            'exact_energy': opt['exact_energy'],
            'error': abs(opt['ground_energy'] - opt['exact_energy']),
            'hamiltonian': 'H2 STO-3G 0.735 A' if hamiltonian is None else 'user supplied',
        }, start, agent_type='QuantumResearchAgent')

    def investigate(self, research_question: str, constraints: dict[str, Any] | None = None,
                    **kwargs) -> QuantumResult:
        raise NotImplementedError(
            "Automated research planning and hypothesis generation are not implemented. "
            "Use simulate_ground_state() for a real VQE run with an exact baseline."
        )


__all__ = [
    'QuantumAgent',
    'QuantumPortfolioAgent',
    'QuantumClassificationAgent',
    'QuantumOptimizationAgent',
    'QuantumTradingAgent',
    'QuantumResearchAgent',
    'portfolio_qubo_values',
]

