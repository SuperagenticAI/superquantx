"""Quantum Principal Component Analysis (QPCA) implementation.

This module provides quantum algorithms for principal component analysis,
including quantum matrix diagonalization and dimensionality reduction.
"""

import logging
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .base_algorithm import UnsupervisedQuantumAlgorithm


logger = logging.getLogger(__name__)

class QuantumPCA(UnsupervisedQuantumAlgorithm):
    """Quantum Principal Component Analysis for dimensionality reduction.

    This implementation uses quantum algorithms to perform PCA, potentially
    offering exponential speedup for certain types of data matrices.

    The algorithm can use different quantum approaches:
    - Quantum Matrix Inversion: For density matrix diagonalization
    - Variational Quantum Eigensolver: For finding principal eigenvectors
    - Quantum Phase Estimation: For eigenvalue extraction
    - Quantum Singular Value Decomposition: Direct SVD approach

    Args:
        backend: Quantum backend for circuit execution
        n_components: Number of principal components to extract
        method: Quantum method ('vqe', 'phase_estimation', 'matrix_inversion', 'qsvd')
        encoding: Data encoding method ('amplitude', 'dense', 'sparse')
        max_iterations: Maximum iterations for variational methods
        tolerance: Convergence tolerance
        shots: Number of measurement shots
        classical_fallback: Use classical PCA if quantum fails
        **kwargs: Additional parameters

    Example:
        >>> qpca = QuantumPCA(backend='pennylane', n_components=3, method='vqe')
        >>> qpca.fit(X_train)
        >>> X_reduced = qpca.transform(X_test)
        >>> X_reconstructed = qpca.inverse_transform(X_reduced)

    """

    def __init__(
        self,
        backend: str | Any,
        n_components: int = 2,
        method: str = 'vqe',
        encoding: str = 'amplitude',
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        shots: int = 1024,
        classical_fallback: bool = True,
        normalize_data: bool = True,
        **kwargs
    ) -> None:
        super().__init__(backend=backend, shots=shots, **kwargs)

        self.n_components = n_components
        self.method = method
        self.encoding = encoding
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.classical_fallback = classical_fallback
        self.normalize_data = normalize_data

        # PCA components
        self.components_ = None
        self.eigenvalues_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

        # Quantum-specific attributes
        self.density_matrix_ = None
        self.quantum_state_ = None
        self.n_qubits = None

        # Classical components for fallback/comparison
        self.scaler = StandardScaler() if normalize_data else None
        self.classical_pca = PCA(n_components=n_components)

        # Method-specific parameters
        self.vqe_params = None
        self.convergence_history = []

        logger.info(f"Initialized QuantumPCA with method={method}, n_components={n_components}")

    def _prepare_data_matrix(self, X: np.ndarray) -> np.ndarray:
        """Prepare data matrix for quantum processing."""
        if self.normalize_data:
            X = self.scaler.fit_transform(X)

        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        return X_centered

    def _create_density_matrix(self, X: np.ndarray) -> np.ndarray:
        """Create density matrix from data."""
        # Compute covariance matrix
        n_samples = X.shape[0]
        cov_matrix = (X.T @ X) / (n_samples - 1)

        # Convert to density matrix (normalized)
        trace = np.trace(cov_matrix)
        if trace > 0:
            density_matrix = cov_matrix / trace
        else:
            density_matrix = cov_matrix

        self.density_matrix_ = density_matrix
        return density_matrix

    def _determine_qubits(self, n_features: int) -> int:
        """Determine number of qubits needed."""
        if self.encoding == 'amplitude':
            return int(np.ceil(np.log2(n_features)))
        elif self.encoding == 'dense':
            return n_features
        else:
            return int(np.ceil(np.log2(n_features)))

    def _encode_density_matrix(self, density_matrix: np.ndarray) -> Any:
        """Encode density matrix into quantum state."""
        try:
            if hasattr(self.backend, 'encode_density_matrix'):
                return self.backend.encode_density_matrix(
                    density_matrix=density_matrix,
                    encoding=self.encoding,
                    n_qubits=self.n_qubits
                )
            else:
                return self._fallback_encoding(density_matrix)
        except Exception as e:
            logger.error(f"Failed to encode density matrix: {e}")
            return self._fallback_encoding(density_matrix)

    def _fallback_encoding(self, density_matrix: np.ndarray) -> Any:
        """Fallback encoding implementation."""
        logger.warning("Using fallback density matrix encoding")
        return None

    def _quantum_eigensolver_vqe(self, density_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Use VQE to find principal eigenvectors and eigenvalues."""
        logger.info("Running VQE for quantum PCA")

        try:
            if hasattr(self.backend, 'run_pca_vqe'):
                result = self.backend.run_pca_vqe(
                    density_matrix=density_matrix,
                    n_components=self.n_components,
                    max_iterations=self.max_iterations,
                    tolerance=self.tolerance,
                    shots=self.shots
                )

                eigenvalues = result.get('eigenvalues', np.zeros(self.n_components))
                eigenvectors = result.get('eigenvectors', np.eye(density_matrix.shape[0], self.n_components))
                self.convergence_history = result.get('convergence_history', [])

                return eigenvalues, eigenvectors
            else:
                return self._fallback_vqe_eigensolver(density_matrix)

        except Exception as e:
            logger.error(f"VQE eigensolver failed: {e}")
            return self._fallback_vqe_eigensolver(density_matrix)

    def _fallback_vqe_eigensolver(self, density_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fallback VQE implementation using classical eigensolver."""
        logger.warning("Using classical fallback for VQE eigensolver")
        eigenvals, eigenvecs = np.linalg.eigh(density_matrix)
        # Sort by eigenvalue magnitude (descending)
        idx = np.argsort(eigenvals)[::-1]
        return eigenvals[idx][:self.n_components], eigenvecs[:, idx][:, :self.n_components]

    def _quantum_phase_estimation(self, density_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Use quantum phase estimation for eigenvalue extraction."""
        logger.info("Running quantum phase estimation for PCA")

        try:
            if hasattr(self.backend, 'run_phase_estimation_pca'):
                result = self.backend.run_phase_estimation_pca(
                    density_matrix=density_matrix,
                    n_components=self.n_components,
                    precision_bits=8,
                    shots=self.shots
                )

                eigenvalues = result.get('eigenvalues', np.zeros(self.n_components))
                eigenvectors = result.get('eigenvectors', np.eye(density_matrix.shape[0], self.n_components))

                return eigenvalues, eigenvectors
            else:
                return self._fallback_phase_estimation(density_matrix)

        except Exception as e:
            logger.error(f"Quantum phase estimation failed: {e}")
            return self._fallback_phase_estimation(density_matrix)

    def _fallback_phase_estimation(self, density_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fallback phase estimation using classical methods."""
        logger.warning("Using classical fallback for phase estimation")
        return self._fallback_vqe_eigensolver(density_matrix)

    def _quantum_svd(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Quantum singular value decomposition approach."""
        logger.info("Running quantum SVD for PCA")

        try:
            if hasattr(self.backend, 'run_quantum_svd'):
                result = self.backend.run_quantum_svd(
                    data_matrix=X,
                    n_components=self.n_components,
                    shots=self.shots
                )

                singular_values = result.get('singular_values', np.zeros(self.n_components))
                components = result.get('components', np.eye(X.shape[1], self.n_components))

                # Convert singular values to eigenvalues
                eigenvalues = singular_values ** 2 / (X.shape[0] - 1)

                return eigenvalues, components.T
            else:
                return self._fallback_svd(X)

        except Exception as e:
            logger.error(f"Quantum SVD failed: {e}")
            return self._fallback_svd(X)

    def _fallback_svd(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fallback SVD using classical methods."""
        logger.warning("Using classical fallback for SVD")
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        eigenvalues = (s ** 2) / (X.shape[0] - 1)
        return eigenvalues[:self.n_components], Vt[:self.n_components].T

    def _quantum_matrix_inversion(self, density_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Quantum matrix inversion approach."""
        logger.info("Running quantum matrix inversion for PCA")

        try:
            if hasattr(self.backend, 'quantum_matrix_inversion_pca'):
                result = self.backend.quantum_matrix_inversion_pca(
                    density_matrix=density_matrix,
                    n_components=self.n_components,
                    condition_threshold=1e-6,
                    shots=self.shots
                )

                eigenvalues = result.get('eigenvalues', np.zeros(self.n_components))
                eigenvectors = result.get('eigenvectors', np.eye(density_matrix.shape[0], self.n_components))

                return eigenvalues, eigenvectors
            else:
                return self._fallback_matrix_inversion(density_matrix)

        except Exception as e:
            logger.error(f"Quantum matrix inversion failed: {e}")
            return self._fallback_matrix_inversion(density_matrix)

    def _fallback_matrix_inversion(self, density_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fallback matrix inversion using classical methods."""
        logger.warning("Using classical fallback for matrix inversion")
        return self._fallback_vqe_eigensolver(density_matrix)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None, **kwargs) -> 'QuantumPCA':
        """Fit quantum PCA to the data.

        Args:
            X: Training data
            y: Ignored (unsupervised learning)
            **kwargs: Additional fitting parameters

        Returns:
            Self for method chaining

        """
        logger.info(f"Fitting QuantumPCA to data of shape {X.shape}")

        # Validate and preprocess data
        super().fit(X, y, **kwargs)

        # Prepare data
        X_processed = self._prepare_data_matrix(X)

        # Determine quantum circuit size
        self.n_qubits = self._determine_qubits(X.shape[1])

        # Choose quantum method
        if self.method == 'vqe':
            density_matrix = self._create_density_matrix(X_processed)
            eigenvalues, eigenvectors = self._quantum_eigensolver_vqe(density_matrix)
        elif self.method == 'phase_estimation':
            density_matrix = self._create_density_matrix(X_processed)
            eigenvalues, eigenvectors = self._quantum_phase_estimation(density_matrix)
        elif self.method == 'matrix_inversion':
            density_matrix = self._create_density_matrix(X_processed)
            eigenvalues, eigenvectors = self._quantum_matrix_inversion(density_matrix)
        elif self.method == 'qsvd':
            eigenvalues, eigenvectors = self._quantum_svd(X_processed)
        else:
            raise ValueError(f"Unknown quantum PCA method: {self.method}")

        # Store results
        self.eigenvalues_ = eigenvalues
        self.components_ = eigenvectors.T  # Store as rows
        self.explained_variance_ = eigenvalues

        # Calculate explained variance ratio
        total_variance = np.sum(eigenvalues) if np.sum(eigenvalues) > 0 else 1.0
        self.explained_variance_ratio_ = eigenvalues / total_variance

        # Fit classical PCA for comparison/fallback
        if self.classical_fallback:
            try:
                self.classical_pca.fit(X_processed)
            except Exception as e:
                logger.warning(f"Classical PCA fitting failed: {e}")

        self.is_fitted = True

        logger.info(f"Quantum PCA completed. Explained variance ratio: {self.explained_variance_ratio_}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to lower dimensional space.

        Args:
            X: Data to transform

        Returns:
            Transformed data

        """
        if not self.is_fitted:
            raise ValueError("QuantumPCA must be fitted before transform")

        # Preprocess data
        if self.normalize_data:
            X = self.scaler.transform(X)

        # Center data
        X_centered = X - self.mean_

        # Project onto principal components
        X_transformed = X_centered @ self.components_.T

        return X_transformed

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Reconstruct data from lower dimensional representation.

        Args:
            X_transformed: Transformed data

        Returns:
            Reconstructed data

        """
        if not self.is_fitted:
            raise ValueError("QuantumPCA must be fitted before inverse_transform")

        # Reconstruct in original space
        X_reconstructed = X_transformed @ self.components_

        # Add back the mean
        X_reconstructed += self.mean_

        # Inverse scaling if applied
        if self.normalize_data:
            X_reconstructed = self.scaler.inverse_transform(X_reconstructed)

        return X_reconstructed

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit PCA and transform data in one step."""
        return self.fit(X, y).transform(X)

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Transform data (alias for transform method)."""
        return self.transform(X)

    def get_quantum_advantage_metrics(self) -> dict[str, Any]:
        """Analyze potential quantum advantage."""
        if not self.is_fitted:
            raise ValueError("Must fit model first")

        n_features = self.components_.shape[1]

        metrics = {
            'data_dimension': n_features,
            'reduced_dimension': self.n_components,
            'compression_ratio': n_features / self.n_components,
            'quantum_circuit_qubits': self.n_qubits,
            'quantum_vs_classical_qubits': self.n_qubits / int(np.ceil(np.log2(n_features))),
        }

        # Potential speedup estimates (theoretical)
        classical_complexity = n_features ** 3  # O(d^3) for eigendecomposition
        quantum_complexity = self.n_qubits ** 2 * np.log(n_features)  # Estimated quantum complexity

        metrics.update({
            'classical_complexity_estimate': classical_complexity,
            'quantum_complexity_estimate': quantum_complexity,
            'theoretical_speedup': classical_complexity / quantum_complexity if quantum_complexity > 0 else 1,
        })

        return metrics

    def compare_with_classical(self, X: np.ndarray) -> dict[str, Any]:
        """Compare quantum PCA results with classical PCA."""
        if not self.is_fitted or not hasattr(self.classical_pca, 'components_'):
            raise ValueError("Both quantum and classical PCA must be fitted")

        # Transform data with both methods
        X_quantum = self.transform(X)
        X_classical = self.classical_pca.transform(X - self.mean_)

        # Compute reconstruction errors
        X_quantum_reconstructed = self.inverse_transform(X_quantum)
        X_classical_reconstructed = self.classical_pca.inverse_transform(X_classical) + self.mean_

        quantum_error = np.mean((X - X_quantum_reconstructed) ** 2)
        classical_error = np.mean((X - X_classical_reconstructed) ** 2)

        # Compare explained variance
        quantum_var_ratio = np.sum(self.explained_variance_ratio_)
        classical_var_ratio = np.sum(self.classical_pca.explained_variance_ratio_)

        # Component similarity (using absolute cosine similarity)
        component_similarities = []
        min_components = min(self.n_components, len(self.classical_pca.components_))

        for i in range(min_components):
            # Cosine similarity between components
            cos_sim = np.abs(np.dot(self.components_[i], self.classical_pca.components_[i]))
            cos_sim /= (np.linalg.norm(self.components_[i]) * np.linalg.norm(self.classical_pca.components_[i]))
            component_similarities.append(cos_sim)

        return {
            'quantum_reconstruction_error': quantum_error,
            'classical_reconstruction_error': classical_error,
            'error_ratio': quantum_error / classical_error if classical_error > 0 else float('inf'),
            'quantum_variance_explained': quantum_var_ratio,
            'classical_variance_explained': classical_var_ratio,
            'variance_explained_ratio': quantum_var_ratio / classical_var_ratio if classical_var_ratio > 0 else float('inf'),
            'component_similarities': component_similarities,
            'mean_component_similarity': np.mean(component_similarities) if component_similarities else 0,
        }

    def analyze_convergence(self) -> dict[str, Any]:
        """Analyze convergence properties of the quantum algorithm."""
        if not self.convergence_history:
            return {'message': 'No convergence history available'}

        convergence_data = np.array(self.convergence_history)

        return {
            'total_iterations': len(convergence_data),
            'final_cost': convergence_data[-1],
            'initial_cost': convergence_data[0],
            'cost_reduction': convergence_data[0] - convergence_data[-1],
            'converged': abs(convergence_data[-1] - convergence_data[-2]) < self.tolerance if len(convergence_data) > 1 else False,
            'convergence_rate': np.mean(np.diff(convergence_data)) if len(convergence_data) > 1 else 0,
        }

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get quantum PCA parameters."""
        params = super().get_params(deep)
        params.update({
            'n_components': self.n_components,
            'method': self.method,
            'encoding': self.encoding,
            'max_iterations': self.max_iterations,
            'tolerance': self.tolerance,
            'classical_fallback': self.classical_fallback,
            'normalize_data': self.normalize_data,
        })
        return params

    def set_params(self, **params) -> 'QuantumPCA':
        """Set quantum PCA parameters."""
        if self.is_fitted and any(key in params for key in
                                 ['n_components', 'method', 'encoding']):
            logger.warning("Changing core parameters requires refitting the model")
            self.is_fitted = False

        return super().set_params(**params)
