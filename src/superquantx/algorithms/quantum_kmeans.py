"""Quantum K-Means clustering implementation.

This module provides quantum algorithms for K-means clustering, including
quantum distance calculations and quantum amplitude estimation approaches.
"""

import logging
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .base_algorithm import UnsupervisedQuantumAlgorithm


logger = logging.getLogger(__name__)

class QuantumKMeans(UnsupervisedQuantumAlgorithm):
    """Quantum K-Means clustering algorithm.

    This implementation uses quantum algorithms to perform K-means clustering,
    potentially offering speedup for high-dimensional data through quantum
    distance calculations and amplitude estimation.

    The algorithm can use different quantum approaches:
    - Quantum Distance Calculation: Use quantum circuits to compute distances
    - Quantum Amplitude Estimation: For probabilistic distance measurements
    - Variational Quantum Clustering: Use VQC for cluster optimization
    - Quantum Annealing: For global cluster optimization

    Args:
        backend: Quantum backend for circuit execution
        n_clusters: Number of clusters (k)
        method: Quantum method ('distance', 'amplitude', 'variational', 'annealing')
        distance_metric: Distance metric ('euclidean', 'manhattan', 'quantum')
        encoding: Data encoding method ('amplitude', 'angle', 'dense')
        max_iterations: Maximum iterations for clustering
        tolerance: Convergence tolerance
        init_method: Centroid initialization ('random', 'k-means++', 'quantum')
        shots: Number of measurement shots
        classical_fallback: Use classical K-means if quantum fails
        **kwargs: Additional parameters

    Example:
        >>> qkmeans = QuantumKMeans(backend='pennylane', n_clusters=3, method='distance')
        >>> qkmeans.fit(X_train)
        >>> labels = qkmeans.predict(X_test)
        >>> centroids = qkmeans.cluster_centers_

    """

    def __init__(
        self,
        backend: str | Any,
        n_clusters: int = 3,
        method: str = 'distance',
        distance_metric: str = 'euclidean',
        encoding: str = 'amplitude',
        max_iterations: int = 300,
        tolerance: float = 1e-4,
        init_method: str = 'k-means++',
        shots: int = 1024,
        classical_fallback: bool = True,
        normalize_data: bool = True,
        random_state: int | None = None,
        **kwargs
    ) -> None:
        super().__init__(backend=backend, shots=shots, **kwargs)

        self.n_clusters = n_clusters
        self.method = method
        self.distance_metric = distance_metric
        self.encoding = encoding
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.init_method = init_method
        self.classical_fallback = classical_fallback
        self.normalize_data = normalize_data
        self.random_state = random_state

        # Clustering results
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

        # Quantum-specific attributes
        self.n_qubits = None
        self.quantum_distances_ = None
        self.convergence_history = []

        # Classical components
        self.scaler = StandardScaler() if normalize_data else None
        self.classical_kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iterations,
            tol=tolerance,
            random_state=random_state
        )

        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)

        logger.info(f"Initialized QuantumKMeans with method={method}, n_clusters={n_clusters}")

    def _determine_qubits(self, n_features: int) -> int:
        """Determine number of qubits needed for encoding."""
        if self.encoding == 'amplitude':
            return max(1, int(np.ceil(np.log2(n_features))))
        elif self.encoding == 'angle':
            return n_features
        elif self.encoding == 'dense':
            return n_features
        else:
            return max(1, int(np.ceil(np.log2(n_features))))

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initialize cluster centroids."""
        n_samples, n_features = X.shape

        if self.init_method == 'random':
            # Random initialization
            centroids = np.random.uniform(
                X.min(axis=0), X.max(axis=0),
                size=(self.n_clusters, n_features)
            )
        elif self.init_method == 'k-means++':
            # K-means++ initialization
            centroids = self._kmeans_plus_plus_init(X)
        elif self.init_method == 'quantum':
            # Quantum-inspired initialization
            centroids = self._quantum_init(X)
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")

        return centroids

    def _kmeans_plus_plus_init(self, X: np.ndarray) -> np.ndarray:
        """K-means++ initialization algorithm."""
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))

        # Choose first centroid randomly
        centroids[0] = X[np.random.randint(n_samples)]

        for i in range(1, self.n_clusters):
            # Compute distances to nearest centroid
            distances = np.array([
                min([np.linalg.norm(x - c) ** 2 for c in centroids[:i]])
                for x in X
            ])

            # Choose next centroid with probability proportional to distance^2
            probabilities = distances / distances.sum()
            cumulative_probs = probabilities.cumsum()
            r = np.random.random()

            for j, p in enumerate(cumulative_probs):
                if r < p:
                    centroids[i] = X[j]
                    break

        return centroids

    def _quantum_init(self, X: np.ndarray) -> np.ndarray:
        """Quantum-inspired centroid initialization."""
        try:
            if hasattr(self.backend, 'quantum_centroid_init'):
                return self.backend.quantum_centroid_init(
                    X, self.n_clusters, encoding=self.encoding, shots=self.shots
                )
            else:
                logger.warning("Quantum initialization not available, using k-means++")
                return self._kmeans_plus_plus_init(X)
        except Exception as e:
            logger.error(f"Quantum initialization failed: {e}")
            return self._kmeans_plus_plus_init(X)

    def _encode_data_point(self, x: np.ndarray) -> Any:
        """Encode data point into quantum state."""
        try:
            if hasattr(self.backend, 'encode_data_point'):
                return self.backend.encode_data_point(
                    x, encoding=self.encoding, n_qubits=self.n_qubits
                )
            else:
                return self._fallback_encoding(x)
        except Exception as e:
            logger.error(f"Data encoding failed: {e}")
            return self._fallback_encoding(x)

    def _fallback_encoding(self, x: np.ndarray) -> Any:
        """Fallback data encoding."""
        logger.warning("Using fallback data encoding")
        return None

    def _quantum_distance(self, x: np.ndarray, centroid: np.ndarray) -> float:
        """Compute quantum distance between point and centroid."""
        try:
            if hasattr(self.backend, 'compute_quantum_distance'):
                return self.backend.compute_quantum_distance(
                    x, centroid,
                    metric=self.distance_metric,
                    encoding=self.encoding,
                    n_qubits=self.n_qubits,
                    shots=self.shots
                )
            else:
                return self._fallback_distance(x, centroid)
        except Exception as e:
            logger.error(f"Quantum distance computation failed: {e}")
            return self._fallback_distance(x, centroid)

    def _fallback_distance(self, x: np.ndarray, centroid: np.ndarray) -> float:
        """Fallback classical distance computation."""
        if self.distance_metric == 'euclidean':
            return np.linalg.norm(x - centroid)
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x - centroid))
        else:
            return np.linalg.norm(x - centroid)  # Default to euclidean

    def _compute_distances_batch(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Compute distances between all points and all centroids."""
        n_samples, n_features = X.shape
        distances = np.zeros((n_samples, self.n_clusters))

        if self.method == 'distance':
            # Use quantum distance calculations
            for i in range(n_samples):
                for j in range(self.n_clusters):
                    distances[i, j] = self._quantum_distance(X[i], centroids[j])

        elif self.method == 'amplitude':
            # Use quantum amplitude estimation
            distances = self._amplitude_estimation_distances(X, centroids)

        elif self.method == 'variational':
            # Use variational quantum circuits
            distances = self._variational_distances(X, centroids)

        else:
            # Fallback to classical distances
            for i in range(n_samples):
                for j in range(self.n_clusters):
                    distances[i, j] = self._fallback_distance(X[i], centroids[j])

        return distances

    def _amplitude_estimation_distances(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Use quantum amplitude estimation for distance computation."""
        try:
            if hasattr(self.backend, 'amplitude_estimation_distances'):
                return self.backend.amplitude_estimation_distances(
                    X, centroids,
                    encoding=self.encoding,
                    shots=self.shots
                )
            else:
                logger.warning("Amplitude estimation not available, using classical")
                return self._classical_distances(X, centroids)
        except Exception as e:
            logger.error(f"Amplitude estimation failed: {e}")
            return self._classical_distances(X, centroids)

    def _variational_distances(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Use variational quantum circuits for distance computation."""
        try:
            if hasattr(self.backend, 'variational_distances'):
                return self.backend.variational_distances(
                    X, centroids,
                    encoding=self.encoding,
                    shots=self.shots
                )
            else:
                logger.warning("Variational distances not available, using classical")
                return self._classical_distances(X, centroids)
        except Exception as e:
            logger.error(f"Variational distance computation failed: {e}")
            return self._classical_distances(X, centroids)

    def _classical_distances(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Classical distance computation."""
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))

        for i in range(n_samples):
            for j in range(self.n_clusters):
                distances[i, j] = self._fallback_distance(X[i], centroids[j])

        return distances

    def _assign_clusters(self, distances: np.ndarray) -> np.ndarray:
        """Assign points to clusters based on distances."""
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Update cluster centroids."""
        centroids = np.zeros((self.n_clusters, X.shape[1]))

        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # Handle empty clusters - reinitialize randomly
                centroids[k] = X[np.random.randint(len(X))]

        return centroids

    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """Compute within-cluster sum of squares (inertia)."""
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                distances = np.sum((cluster_points - centroids[k]) ** 2, axis=1)
                inertia += np.sum(distances)
        return inertia

    def _check_convergence(self, old_centroids: np.ndarray, new_centroids: np.ndarray) -> bool:
        """Check if centroids have converged."""
        centroid_shift = np.max(np.linalg.norm(new_centroids - old_centroids, axis=1))
        return centroid_shift < self.tolerance

    def fit(self, X: np.ndarray, y: np.ndarray | None = None, **kwargs) -> 'QuantumKMeans':
        """Fit quantum K-means to the data.

        Args:
            X: Training data
            y: Ignored (unsupervised learning)
            **kwargs: Additional fitting parameters

        Returns:
            Self for method chaining

        """
        logger.info(f"Fitting QuantumKMeans to data of shape {X.shape}")

        # Validate and preprocess data
        super().fit(X, y, **kwargs)

        # Normalize data if specified
        if self.normalize_data:
            X = self.scaler.fit_transform(X)

        # Determine quantum circuit requirements
        self.n_qubits = self._determine_qubits(X.shape[1])

        # Initialize centroids
        centroids = self._initialize_centroids(X)

        # Reset convergence history
        self.convergence_history = []

        # Main K-means iteration loop
        for iteration in range(self.max_iterations):
            # Store old centroids for convergence check
            old_centroids = centroids.copy()

            # Compute distances and assign clusters
            distances = self._compute_distances_batch(X, centroids)
            labels = self._assign_clusters(distances)

            # Update centroids
            centroids = self._update_centroids(X, labels)

            # Compute inertia
            inertia = self._compute_inertia(X, labels, centroids)
            self.convergence_history.append(inertia)

            # Check convergence
            if self._check_convergence(old_centroids, centroids):
                logger.info(f"Converged after {iteration + 1} iterations")
                break

            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: Inertia = {inertia:.6f}")

        # Store final results
        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_iter_ = iteration + 1

        # Fit classical K-means for comparison
        if self.classical_fallback:
            try:
                self.classical_kmeans.fit(X)
            except Exception as e:
                logger.warning(f"Classical K-means fitting failed: {e}")

        self.is_fitted = True

        logger.info(f"Quantum K-means completed. Final inertia: {self.inertia_:.6f}")

        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict cluster labels for new data.

        Args:
            X: Data to cluster
            **kwargs: Additional parameters

        Returns:
            Cluster labels

        """
        if not self.is_fitted:
            raise ValueError("QuantumKMeans must be fitted before prediction")

        # Normalize data if specified
        if self.normalize_data:
            X = self.scaler.transform(X)

        # Compute distances to centroids
        distances = self._compute_distances_batch(X, self.cluster_centers_)

        # Assign to nearest cluster
        return self._assign_clusters(distances)

    def fit_predict(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit K-means and return cluster labels."""
        return self.fit(X, y).labels_

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to cluster-distance space."""
        if not self.is_fitted:
            raise ValueError("QuantumKMeans must be fitted before transform")

        # Normalize data if specified
        if self.normalize_data:
            X = self.scaler.transform(X)

        # Return distances to all centroids
        return self._compute_distances_batch(X, self.cluster_centers_)

    def get_quantum_advantage_metrics(self) -> dict[str, Any]:
        """Analyze potential quantum advantage."""
        if not self.is_fitted:
            raise ValueError("Must fit model first")

        n_features = self.cluster_centers_.shape[1]
        n_samples = self.n_samples_

        metrics = {
            'data_dimension': n_features,
            'n_samples': n_samples,
            'n_clusters': self.n_clusters,
            'quantum_circuit_qubits': self.n_qubits,
            'encoding_efficiency': n_features / self.n_qubits if self.n_qubits > 0 else 1,
        }

        # Complexity estimates
        classical_complexity = n_samples * self.n_clusters * n_features * self.n_iter_
        quantum_complexity = n_samples * self.n_clusters * self.n_qubits * self.shots * self.n_iter_

        metrics.update({
            'classical_complexity_estimate': classical_complexity,
            'quantum_complexity_estimate': quantum_complexity,
            'theoretical_speedup': classical_complexity / quantum_complexity if quantum_complexity > 0 else 1,
        })

        return metrics

    def compare_with_classical(self, X: np.ndarray, y_true: np.ndarray | None = None) -> dict[str, Any]:
        """Compare quantum K-means results with classical K-means."""
        if not self.is_fitted or not hasattr(self.classical_kmeans, 'cluster_centers_'):
            raise ValueError("Both quantum and classical K-means must be fitted")

        # Get predictions from both methods
        quantum_labels = self.predict(X)
        classical_labels = self.classical_kmeans.predict(X if not self.normalize_data
                                                        else self.scaler.transform(X))

        comparison = {
            'quantum_inertia': self.inertia_,
            'classical_inertia': self.classical_kmeans.inertia_,
            'inertia_ratio': self.inertia_ / self.classical_kmeans.inertia_ if self.classical_kmeans.inertia_ > 0 else float('inf'),
            'quantum_iterations': self.n_iter_,
            'classical_iterations': self.classical_kmeans.n_iter_,
        }

        # Compute silhouette scores
        try:
            if len(np.unique(quantum_labels)) > 1:
                quantum_silhouette = silhouette_score(X, quantum_labels)
                comparison['quantum_silhouette'] = quantum_silhouette

            if len(np.unique(classical_labels)) > 1:
                classical_silhouette = silhouette_score(X, classical_labels)
                comparison['classical_silhouette'] = classical_silhouette

            if 'quantum_silhouette' in comparison and 'classical_silhouette' in comparison:
                comparison['silhouette_ratio'] = quantum_silhouette / classical_silhouette

        except Exception as e:
            logger.warning(f"Silhouette score computation failed: {e}")

        # Compare with ground truth if available
        if y_true is not None:
            try:
                quantum_ari = adjusted_rand_score(y_true, quantum_labels)
                classical_ari = adjusted_rand_score(y_true, classical_labels)

                comparison.update({
                    'quantum_adjusted_rand_score': quantum_ari,
                    'classical_adjusted_rand_score': classical_ari,
                    'ari_ratio': quantum_ari / classical_ari if classical_ari != 0 else float('inf'),
                })

            except Exception as e:
                logger.warning(f"Adjusted rand score computation failed: {e}")

        # Compare centroid similarities
        centroid_distances = []
        for i in range(min(len(self.cluster_centers_), len(self.classical_kmeans.cluster_centers_))):
            dist = np.linalg.norm(self.cluster_centers_[i] - self.classical_kmeans.cluster_centers_[i])
            centroid_distances.append(dist)

        if centroid_distances:
            comparison.update({
                'centroid_distances': centroid_distances,
                'mean_centroid_distance': np.mean(centroid_distances),
                'max_centroid_distance': np.max(centroid_distances),
            })

        return comparison

    def analyze_convergence(self) -> dict[str, Any]:
        """Analyze convergence properties."""
        if not self.convergence_history:
            return {'message': 'No convergence history available'}

        inertias = np.array(self.convergence_history)

        return {
            'total_iterations': len(inertias),
            'final_inertia': inertias[-1],
            'initial_inertia': inertias[0],
            'inertia_reduction': inertias[0] - inertias[-1] if len(inertias) > 0 else 0,
            'convergence_rate': np.mean(np.diff(inertias)) if len(inertias) > 1 else 0,
            'converged': len(inertias) < self.max_iterations,
            'inertia_variance': np.var(inertias[-10:]) if len(inertias) >= 10 else np.var(inertias),
        }

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get quantum K-means parameters."""
        params = super().get_params(deep)
        params.update({
            'n_clusters': self.n_clusters,
            'method': self.method,
            'distance_metric': self.distance_metric,
            'encoding': self.encoding,
            'max_iterations': self.max_iterations,
            'tolerance': self.tolerance,
            'init_method': self.init_method,
            'classical_fallback': self.classical_fallback,
            'normalize_data': self.normalize_data,
            'random_state': self.random_state,
        })
        return params

    def set_params(self, **params) -> 'QuantumKMeans':
        """Set quantum K-means parameters."""
        if self.is_fitted and any(key in params for key in
                                 ['n_clusters', 'method', 'distance_metric', 'encoding']):
            logger.warning("Changing core parameters requires refitting the model")
            self.is_fitted = False

        return super().set_params(**params)
