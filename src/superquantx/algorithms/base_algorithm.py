"""Base classes for quantum machine learning algorithms.

This module provides abstract base classes that define the common interface
for all quantum algorithms in SuperQuantX.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)

@dataclass
class QuantumResult:
    """Container for quantum algorithm results.

    This class provides a standardized way to return results from quantum
    algorithms, including the main result, metadata, and performance metrics.

    Attributes:
        result: The main algorithm result
        metadata: Additional information about the computation
        execution_time: Time taken to execute the algorithm (seconds)
        backend_info: Information about the backend used
        error: Error information if computation failed
        intermediate_results: Optional intermediate results for debugging

    """

    result: Any
    metadata: dict[str, Any]
    execution_time: float
    backend_info: dict[str, Any]
    error: str | None = None
    intermediate_results: dict[str, Any] | None = None

    def __post_init__(self):
        """Validate and process result after initialization."""
        if self.metadata is None:
            self.metadata = {}
        if self.backend_info is None:
            self.backend_info = {}

class BaseQuantumAlgorithm(ABC):
    """Abstract base class for all quantum machine learning algorithms.

    This class defines the common interface that all quantum algorithms must
    implement, providing consistency across different algorithm types and backends.

    Args:
        backend: Quantum backend to use for computation
        shots: Number of measurement shots (default: 1024)
        seed: Random seed for reproducibility
        optimization_level: Circuit optimization level (0-3)
        **kwargs: Additional algorithm-specific parameters

    """

    def __init__(
        self,
        backend: str | Any,
        shots: int = 1024,
        seed: int | None = None,
        optimization_level: int = 1,
        **kwargs
    ) -> None:
        """Initialize the quantum algorithm."""
        self.backend = self._initialize_backend(backend)
        self.shots = shots
        self.seed = seed
        self.optimization_level = optimization_level

        # Algorithm state
        self.is_fitted = False
        self.training_history = []
        self.best_params = None
        self.best_score = None

        # Store additional parameters
        self.algorithm_params = kwargs

        # Performance tracking
        self.execution_times = []
        self.backend_stats = {}

        logger.info(f"Initialized {self.__class__.__name__} with backend {type(self.backend).__name__}")

    def _initialize_backend(self, backend: str | Any) -> Any:
        """Initialize the quantum backend."""
        if isinstance(backend, str):
            # Import here to avoid circular imports
            from ..backends import get_backend
            return get_backend(backend)
        return backend

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None, **kwargs) -> 'BaseQuantumAlgorithm':
        """Train the quantum algorithm.

        Args:
            X: Training data features
            y: Training data labels (for supervised learning)
            **kwargs: Additional training parameters

        Returns:
            Self for method chaining

        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Make predictions using the trained algorithm.

        Args:
            X: Input data for prediction
            **kwargs: Additional prediction parameters

        Returns:
            Predictions array

        """
        pass

    def score(self, X: np.ndarray, y: np.ndarray, **kwargs) -> float:
        """Compute the algorithm's score on the given test data.

        Args:
            X: Test data features
            y: True test data labels
            **kwargs: Additional scoring parameters

        Returns:
            Algorithm score (higher is better)

        """
        predictions = self.predict(X, **kwargs)
        return self._compute_score(predictions, y)

    def _compute_score(self, predictions: np.ndarray, y_true: np.ndarray) -> float:
        """Compute accuracy score by default."""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_true, predictions)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get algorithm parameters.

        Args:
            deep: Whether to return deep copy of parameters

        Returns:
            Parameter dictionary

        """
        params = {
            'backend': self.backend,
            'shots': self.shots,
            'seed': self.seed,
            'optimization_level': self.optimization_level,
        }
        params.update(self.algorithm_params)
        return params

    def set_params(self, **params) -> 'BaseQuantumAlgorithm':
        """Set algorithm parameters.

        Args:
            **params: Parameters to set

        Returns:
            Self for method chaining

        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.algorithm_params[key] = value
        return self

    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk.

        Args:
            filepath: Path where to save the model

        """
        import pickle

        if not self.is_fitted:
            logger.warning("Model is not fitted yet. Saving unfitted model.")

        model_data = {
            'class': self.__class__.__name__,
            'params': self.get_params(),
            'is_fitted': self.is_fitted,
            'training_history': self.training_history,
            'best_params': self.best_params,
            'best_score': self.best_score,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'BaseQuantumAlgorithm':
        """Load a trained model from disk.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded algorithm instance

        """
        import pickle

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # Create new instance with saved parameters
        instance = cls(**model_data['params'])
        instance.is_fitted = model_data['is_fitted']
        instance.training_history = model_data['training_history']
        instance.best_params = model_data['best_params']
        instance.best_score = model_data['best_score']

        logger.info(f"Model loaded from {filepath}")
        return instance

    def benchmark(self, X: np.ndarray, y: np.ndarray | None = None, runs: int = 5) -> dict[str, Any]:
        """Benchmark algorithm performance.

        Args:
            X: Test data
            y: Test labels (optional)
            runs: Number of benchmark runs

        Returns:
            Benchmark results dictionary

        """
        execution_times = []
        scores = []

        for i in range(runs):
            start_time = time.time()

            if y is not None:
                score = self.score(X, y)
                scores.append(score)
            else:
                self.predict(X)

            execution_time = time.time() - start_time
            execution_times.append(execution_time)

        results = {
            'execution_times': execution_times,
            'mean_execution_time': np.mean(execution_times),
            'std_execution_time': np.std(execution_times),
            'min_execution_time': np.min(execution_times),
            'max_execution_time': np.max(execution_times),
        }

        if scores:
            results.update({
                'scores': scores,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
            })

        return results

    def get_circuit_info(self) -> dict[str, Any]:
        """Get information about the quantum circuit.

        Returns:
            Circuit information dictionary

        """
        return {
            'backend': type(self.backend).__name__,
            'shots': self.shots,
            'optimization_level': self.optimization_level,
        }

    def reset(self) -> None:
        """Reset algorithm to untrained state."""
        self.is_fitted = False
        self.training_history = []
        self.best_params = None
        self.best_score = None
        self.execution_times = []
        self.backend_stats = {}

        logger.info(f"Reset {self.__class__.__name__} to untrained state")

    def __repr__(self) -> str:
        """String representation of the algorithm."""
        params = self.get_params()
        param_str = ", ".join([f"{k}={v}" for k, v in list(params.items())[:3]])
        return f"{self.__class__.__name__}({param_str}, ...)"

    def __str__(self) -> str:
        """Human-readable string representation."""
        status = "fitted" if self.is_fitted else "unfitted"
        return f"{self.__class__.__name__} ({status})"

class SupervisedQuantumAlgorithm(BaseQuantumAlgorithm):
    """Base class for supervised quantum learning algorithms."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classes_ = None
        self.n_features_ = None
        self.n_classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'SupervisedQuantumAlgorithm':
        """Fit supervised algorithm."""
        self._validate_data(X, y)
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self.n_classes_ = len(self.classes_)
        return self

    def _validate_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate input data."""
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        if len(X.shape) != 2:
            raise ValueError("X must be 2D array")

class UnsupervisedQuantumAlgorithm(BaseQuantumAlgorithm):
    """Base class for unsupervised quantum learning algorithms."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_features_ = None
        self.n_samples_ = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None, **kwargs) -> 'UnsupervisedQuantumAlgorithm':
        """Fit unsupervised algorithm."""
        self._validate_data(X)
        self.n_features_ = X.shape[1]
        self.n_samples_ = X.shape[0]
        return self

    def _validate_data(self, X: np.ndarray) -> None:
        """Validate input data."""
        if len(X.shape) != 2:
            raise ValueError("X must be 2D array")

class OptimizationQuantumAlgorithm(BaseQuantumAlgorithm):
    """Base class for quantum optimization algorithms."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimal_params_ = None
        self.optimal_value_ = None
        self.optimization_history_ = []

    def optimize(self, objective_function, initial_params: np.ndarray | None = None, **kwargs) -> QuantumResult:
        """Optimize objective function.

        Args:
            objective_function: Function to optimize
            initial_params: Initial parameter values
            **kwargs: Additional optimization parameters

        Returns:
            Optimization result

        """
        start_time = time.time()

        try:
            result = self._run_optimization(objective_function, initial_params, **kwargs)

            return QuantumResult(
                result=result,
                metadata={
                    'converged': True,
                    'n_iterations': len(self.optimization_history_),
                    'final_cost': self.optimal_value_,
                },
                execution_time=time.time() - start_time,
                backend_info=self.get_circuit_info(),
            )

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return QuantumResult(
                result=None,
                metadata={'converged': False},
                execution_time=time.time() - start_time,
                backend_info=self.get_circuit_info(),
                error=str(e),
            )

    @abstractmethod
    def _run_optimization(self, objective_function, initial_params, **kwargs):
        """Run the optimization algorithm."""
        pass
