"""Quantum Support Vector Machine (QSVM) implementation.

This module provides a quantum support vector machine algorithm that can be used
for binary and multiclass classification tasks using quantum kernels.
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .base_algorithm import SupervisedQuantumAlgorithm


logger = logging.getLogger(__name__)

class QuantumSVM(SupervisedQuantumAlgorithm):
    """Quantum Support Vector Machine for classification.

    This implementation uses quantum feature maps to transform data into
    a high-dimensional Hilbert space where linear separation is possible.
    The quantum kernel is computed using quantum circuits.

    The algorithm works by:
    1. Encoding classical data into quantum states using feature maps
    2. Computing quantum kernels between data points
    3. Training a classical SVM using the quantum kernel matrix

    Args:
        backend: Quantum backend for circuit execution
        feature_map: Type of quantum feature map ('ZZFeatureMap', 'PauliFeatureMap', etc.)
        feature_map_reps: Number of repetitions in the feature map
        C: Regularization parameter for SVM
        gamma: Kernel coefficient (for RBF-like quantum kernels)
        quantum_kernel: Custom quantum kernel function
        shots: Number of measurement shots
        **kwargs: Additional parameters

    Example:
        >>> qsvm = QuantumSVM(backend='pennylane', feature_map='ZZFeatureMap')
        >>> qsvm.fit(X_train, y_train)
        >>> predictions = qsvm.predict(X_test)
        >>> accuracy = qsvm.score(X_test, y_test)

    """

    def __init__(
        self,
        backend: str | Any,
        feature_map: str = 'ZZFeatureMap',
        feature_map_reps: int = 2,
        C: float = 1.0,
        gamma: float | None = None,
        quantum_kernel: Callable | None = None,
        shots: int = 1024,
        normalize_data: bool = True,
        **kwargs
    ) -> None:
        super().__init__(backend=backend, shots=shots, **kwargs)

        self.feature_map = feature_map
        self.feature_map_reps = feature_map_reps
        self.C = C
        self.gamma = gamma
        self.quantum_kernel = quantum_kernel
        self.normalize_data = normalize_data

        # Classical components
        self.svm = None
        self.scaler = StandardScaler() if normalize_data else None

        # Quantum components
        self.kernel_matrix_ = None
        self.feature_map_circuit_ = None

        # Training data storage (needed for kernel computation)
        self.X_train_ = None

        logger.info(f"Initialized QuantumSVM with feature_map={feature_map}, reps={feature_map_reps}")

    def _create_feature_map(self, n_features: int) -> Any:
        """Create quantum feature map circuit."""
        try:
            if hasattr(self.backend, 'create_feature_map'):
                return self.backend.create_feature_map(
                    n_features=n_features,
                    feature_map=self.feature_map,
                    reps=self.feature_map_reps
                )
            else:
                # Fallback implementation
                return self._default_feature_map(n_features)
        except Exception as e:
            logger.error(f"Failed to create feature map: {e}")
            return self._default_feature_map(n_features)

    def _default_feature_map(self, n_features: int) -> Any:
        """Create default feature map when backend doesn't provide one."""
        # This would be implemented based on the backend
        logger.warning("Using default feature map - results may vary")
        return None

    def _compute_quantum_kernel(self, X1: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray:
        """Compute quantum kernel matrix between data points.

        Args:
            X1: First set of data points
            X2: Second set of data points (if None, use X1)

        Returns:
            Quantum kernel matrix

        """
        if X2 is None:
            X2 = X1

        n1, n2 = len(X1), len(X2)
        kernel_matrix = np.zeros((n1, n2))

        if self.quantum_kernel:
            # Use custom quantum kernel
            for i in range(n1):
                for j in range(n2):
                    kernel_matrix[i, j] = self.quantum_kernel(X1[i], X2[j])
        else:
            # Use backend's kernel computation
            try:
                kernel_matrix = self.backend.compute_kernel_matrix(
                    X1, X2,
                    feature_map=self.feature_map_circuit_,
                    shots=self.shots
                )
            except Exception as e:
                logger.error(f"Quantum kernel computation failed: {e}")
                # Fallback to classical RBF kernel
                from sklearn.metrics.pairwise import rbf_kernel
                gamma = self.gamma if self.gamma else 1.0 / X1.shape[1]
                kernel_matrix = rbf_kernel(X1, X2, gamma=gamma)
                logger.warning("Using classical RBF kernel as fallback")

        return kernel_matrix

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'QuantumSVM':
        """Train the quantum SVM.

        Args:
            X: Training data features
            y: Training data labels
            **kwargs: Additional training parameters

        Returns:
            Self for method chaining

        """
        logger.info(f"Training QuantumSVM on {X.shape[0]} samples with {X.shape[1]} features")

        # Validate and preprocess data
        super().fit(X, y, **kwargs)

        if self.normalize_data:
            X = self.scaler.fit_transform(X)

        self.X_train_ = X.copy()

        # Create quantum feature map
        self.feature_map_circuit_ = self._create_feature_map(X.shape[1])

        # Compute quantum kernel matrix
        logger.info("Computing quantum kernel matrix...")
        self.kernel_matrix_ = self._compute_quantum_kernel(X)

        # Train classical SVM with quantum kernel
        self.svm = SVC(
            kernel='precomputed',
            C=self.C,
        )

        self.svm.fit(self.kernel_matrix_, y)
        self.is_fitted = True

        # Compute training accuracy
        train_predictions = self.predict(X)
        train_accuracy = accuracy_score(y, train_predictions)

        self.training_history.append({
            'train_accuracy': train_accuracy,
            'n_support_vectors': self.svm.n_support_,
            'kernel_matrix_shape': self.kernel_matrix_.shape,
        })

        logger.info(f"Training completed. Accuracy: {train_accuracy:.3f}, "
                   f"Support vectors: {sum(self.svm.n_support_)}")

        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Make predictions using the trained quantum SVM.

        Args:
            X: Input data for prediction
            **kwargs: Additional prediction parameters

        Returns:
            Predicted labels

        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if self.normalize_data:
            X = self.scaler.transform(X)

        # Compute kernel matrix between test data and training data
        test_kernel = self._compute_quantum_kernel(X, self.X_train_)

        # Make predictions using the trained SVM
        predictions = self.svm.predict(test_kernel)

        return predictions

    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Input data for prediction
            **kwargs: Additional parameters

        Returns:
            Predicted class probabilities

        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if self.normalize_data:
            X = self.scaler.transform(X)

        test_kernel = self._compute_quantum_kernel(X, self.X_train_)

        # Need to recreate SVM with probability=True for probabilities
        if not hasattr(self.svm, 'predict_proba'):
            logger.warning("Probability prediction not available, returning decision scores")
            return self.decision_function(X)

        return self.svm.predict_proba(test_kernel)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function values.

        Args:
            X: Input data

        Returns:
            Decision function values

        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing decision function")

        if self.normalize_data:
            X = self.scaler.transform(X)

        test_kernel = self._compute_quantum_kernel(X, self.X_train_)
        return self.svm.decision_function(test_kernel)

    def get_support_vectors(self) -> np.ndarray:
        """Get support vectors from the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get support vectors")

        return self.X_train_[self.svm.support_]

    def get_quantum_kernel_matrix(self, X: np.ndarray | None = None) -> np.ndarray:
        """Get the quantum kernel matrix.

        Args:
            X: Data to compute kernel matrix for (default: training data)

        Returns:
            Quantum kernel matrix

        """
        if X is None:
            if self.kernel_matrix_ is None:
                raise ValueError("No kernel matrix available")
            return self.kernel_matrix_
        else:
            if self.normalize_data:
                X = self.scaler.transform(X)
            return self._compute_quantum_kernel(X)

    def analyze_kernel(self) -> dict[str, Any]:
        """Analyze properties of the quantum kernel.

        Returns:
            Dictionary with kernel analysis results

        """
        if self.kernel_matrix_ is None:
            raise ValueError("Model must be fitted to analyze kernel")

        K = self.kernel_matrix_

        # Compute kernel properties
        eigenvalues = np.linalg.eigvals(K)

        analysis = {
            'kernel_shape': K.shape,
            'kernel_rank': np.linalg.matrix_rank(K),
            'condition_number': np.linalg.cond(K),
            'trace': np.trace(K),
            'frobenius_norm': np.linalg.norm(K, 'fro'),
            'eigenvalue_stats': {
                'mean': np.mean(eigenvalues),
                'std': np.std(eigenvalues),
                'min': np.min(eigenvalues),
                'max': np.max(eigenvalues),
            },
            'is_positive_definite': np.all(eigenvalues > 0),
        }

        return analysis

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get algorithm parameters."""
        params = super().get_params(deep)
        params.update({
            'feature_map': self.feature_map,
            'feature_map_reps': self.feature_map_reps,
            'C': self.C,
            'gamma': self.gamma,
            'normalize_data': self.normalize_data,
        })
        return params

    def set_params(self, **params) -> 'QuantumSVM':
        """Set algorithm parameters."""
        if self.is_fitted and any(key in params for key in
                                 ['feature_map', 'feature_map_reps', 'C', 'gamma']):
            logger.warning("Changing core parameters requires refitting the model")
            self.is_fitted = False

        return super().set_params(**params)
