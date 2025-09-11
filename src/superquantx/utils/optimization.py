"""Optimization utilities for quantum machine learning.

This module provides optimization functions for quantum circuits and parameters,
including classical optimizers commonly used in quantum machine learning.
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np


class Optimizer(ABC):
    """Base class for optimizers."""

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.history = []

    @abstractmethod
    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Perform one optimization step."""
        pass

    def reset(self):
        """Reset optimizer state."""
        self.history = []


class GradientDescentOptimizer(Optimizer):
    """Simple gradient descent optimizer."""

    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)

    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Perform gradient descent step."""
        new_params = params - self.learning_rate * gradients
        return new_params


class AdamOptimizer(Optimizer):
    """Adam optimizer for quantum parameter optimization."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Perform Adam optimization step."""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients

        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)

        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Update parameters
        new_params = params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return new_params

    def reset(self):
        """Reset Adam optimizer state."""
        super().reset()
        self.m = None
        self.v = None
        self.t = 0


def optimize_circuit(
    cost_function: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    gradient_function: Callable[[np.ndarray], np.ndarray] | None = None,
    optimizer: str = 'adam',
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    learning_rate: float = 0.01,
    verbose: bool = False
) -> dict[str, Any]:
    """Optimize quantum circuit parameters.

    Args:
        cost_function: Function to minimize f(params) -> cost
        initial_params: Initial parameter values
        gradient_function: Function to compute gradients (optional)
        optimizer: Optimizer type ('adam', 'sgd', 'lbfgs')
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        learning_rate: Learning rate for gradient-based optimizers
        verbose: Whether to print progress

    Returns:
        Dictionary with optimization results

    """
    start_time = time.time()

    # Initialize optimizer
    if optimizer == 'adam':
        opt = AdamOptimizer(learning_rate)
    elif optimizer == 'sgd':
        opt = GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    params = initial_params.copy()
    costs = []

    # If no gradient function provided, use finite differences
    if gradient_function is None:
        def gradient_function(p):
            return finite_difference_gradient(cost_function, p)

    for iteration in range(max_iterations):
        # Compute cost and gradient
        cost = cost_function(params)
        gradients = gradient_function(params)

        costs.append(cost)

        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}: Cost = {cost:.6f}")

        # Check convergence
        if iteration > 0 and abs(costs[-2] - cost) < tolerance:
            if verbose:
                print(f"Converged at iteration {iteration}")
            break

        # Update parameters
        params = opt.step(params, gradients)

    optimization_time = time.time() - start_time

    return {
        'optimal_params': params,
        'optimal_cost': costs[-1],
        'cost_history': costs,
        'n_iterations': len(costs),
        'converged': len(costs) < max_iterations,
        'optimization_time': optimization_time,
        'optimizer': optimizer
    }


def optimize_parameters(
    objective_function: Callable,
    bounds: list[tuple[float, float]],
    method: str = 'scipy',
    max_evaluations: int = 1000,
    random_state: int | None = None
) -> dict[str, Any]:
    """Optimize parameters using various methods.

    Args:
        objective_function: Function to minimize
        bounds: Parameter bounds as list of (min, max) tuples
        method: Optimization method ('scipy', 'random_search', 'grid_search')
        max_evaluations: Maximum function evaluations
        random_state: Random seed

    Returns:
        Optimization results dictionary

    """
    if method == 'scipy':
        return _scipy_optimize(objective_function, bounds, max_evaluations)
    elif method == 'random_search':
        return _random_search_optimize(objective_function, bounds, max_evaluations, random_state)
    elif method == 'grid_search':
        return _grid_search_optimize(objective_function, bounds, max_evaluations)
    else:
        raise ValueError(f"Unknown optimization method: {method}")


def gradient_descent(
    cost_function: Callable[[np.ndarray], float],
    gradient_function: Callable[[np.ndarray], np.ndarray],
    initial_params: np.ndarray,
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
    tolerance: float = 1e-6
) -> tuple[np.ndarray, list[float]]:
    """Perform gradient descent optimization.

    Args:
        cost_function: Cost function to minimize
        gradient_function: Function returning gradients
        initial_params: Initial parameter values
        learning_rate: Learning rate
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        Tuple of (optimal_params, cost_history)

    """
    params = initial_params.copy()
    cost_history = []

    for i in range(max_iterations):
        cost = cost_function(params)
        cost_history.append(cost)

        if i > 0 and abs(cost_history[-2] - cost) < tolerance:
            break

        gradients = gradient_function(params)
        params = params - learning_rate * gradients

    return params, cost_history


def adam_optimizer(
    cost_function: Callable[[np.ndarray], float],
    gradient_function: Callable[[np.ndarray], np.ndarray],
    initial_params: np.ndarray,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    max_iterations: int = 1000,
    tolerance: float = 1e-6
) -> tuple[np.ndarray, list[float]]:
    """Perform Adam optimization.

    Args:
        cost_function: Cost function to minimize
        gradient_function: Function returning gradients
        initial_params: Initial parameter values
        learning_rate: Learning rate
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for second moment
        epsilon: Small constant for numerical stability
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        Tuple of (optimal_params, cost_history)

    """
    optimizer = AdamOptimizer(learning_rate, beta1, beta2, epsilon)
    params = initial_params.copy()
    cost_history = []

    for i in range(max_iterations):
        cost = cost_function(params)
        cost_history.append(cost)

        if i > 0 and abs(cost_history[-2] - cost) < tolerance:
            break

        gradients = gradient_function(params)
        params = optimizer.step(params, gradients)

    return params, cost_history


def finite_difference_gradient(
    function: Callable[[np.ndarray], float],
    params: np.ndarray,
    epsilon: float = 1e-6
) -> np.ndarray:
    """Compute gradient using finite differences.

    Args:
        function: Function to differentiate
        params: Parameters at which to compute gradient
        epsilon: Finite difference step size

    Returns:
        Gradient vector

    """
    gradients = np.zeros_like(params)

    for i in range(len(params)):
        # Forward difference
        params_plus = params.copy()
        params_plus[i] += epsilon

        params_minus = params.copy()
        params_minus[i] -= epsilon

        gradients[i] = (function(params_plus) - function(params_minus)) / (2 * epsilon)

    return gradients


def _scipy_optimize(
    objective_function: Callable,
    bounds: list[tuple[float, float]],
    max_evaluations: int
) -> dict[str, Any]:
    """Optimize using scipy methods."""
    try:
        from scipy.optimize import minimize

        # Initial guess (center of bounds)
        x0 = [(b[0] + b[1]) / 2 for b in bounds]

        result = minimize(
            objective_function,
            x0,
            bounds=bounds,
            options={'maxiter': max_evaluations}
        )

        return {
            'optimal_params': result.x,
            'optimal_value': result.fun,
            'n_evaluations': result.nfev,
            'success': result.success,
            'method': 'scipy'
        }
    except ImportError:
        raise ImportError("scipy is required for scipy optimization")


def _random_search_optimize(
    objective_function: Callable,
    bounds: list[tuple[float, float]],
    max_evaluations: int,
    random_state: int | None
) -> dict[str, Any]:
    """Random search optimization."""
    np.random.seed(random_state)

    best_params = None
    best_value = float('inf')

    for _ in range(max_evaluations):
        # Generate random parameters within bounds
        params = [np.random.uniform(b[0], b[1]) for b in bounds]
        value = objective_function(params)

        if value < best_value:
            best_value = value
            best_params = params

    return {
        'optimal_params': np.array(best_params),
        'optimal_value': best_value,
        'n_evaluations': max_evaluations,
        'success': True,
        'method': 'random_search'
    }


def _grid_search_optimize(
    objective_function: Callable,
    bounds: list[tuple[float, float]],
    max_evaluations: int
) -> dict[str, Any]:
    """Grid search optimization."""
    n_params = len(bounds)
    n_points_per_dim = int(max_evaluations ** (1 / n_params))

    # Create grid points
    grids = []
    for b in bounds:
        grids.append(np.linspace(b[0], b[1], n_points_per_dim))

    best_params = None
    best_value = float('inf')
    n_evaluations = 0

    # Evaluate all grid points
    for params in np.ndindex(*[len(g) for g in grids]):
        if n_evaluations >= max_evaluations:
            break

        param_values = [grids[i][params[i]] for i in range(n_params)]
        value = objective_function(param_values)
        n_evaluations += 1

        if value < best_value:
            best_value = value
            best_params = param_values

    return {
        'optimal_params': np.array(best_params),
        'optimal_value': best_value,
        'n_evaluations': n_evaluations,
        'success': True,
        'method': 'grid_search'
    }
