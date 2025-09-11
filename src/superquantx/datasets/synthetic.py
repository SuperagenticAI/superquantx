"""Synthetic data generators for quantum machine learning.

This module provides functions to generate synthetic datasets specifically
designed for testing quantum algorithms and benchmarking performance.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .preprocessing import normalize_quantum_data


def generate_classification_data(
    n_samples: int = 200,
    n_features: int = 4,
    n_classes: int = 2,
    n_redundant: int = 0,
    n_informative: Optional[int] = None,
    class_sep: float = 1.0,
    encoding: str = 'amplitude',
    normalize: bool = True,
    test_size: float = 0.2,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate synthetic classification data for quantum machine learning.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features (should be power of 2 for quantum efficiency)
        n_classes: Number of classes
        n_redundant: Number of redundant features
        n_informative: Number of informative features (default: n_features)
        class_sep: Class separation factor
        encoding: Type of quantum encoding
        normalize: Whether to normalize features
        test_size: Proportion for test split
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, metadata)

    """
    if n_informative is None:
        n_informative = n_features

    # Generate synthetic data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        class_sep=class_sep,
        random_state=random_state
    )

    # Normalization for quantum encoding
    if normalize:
        if encoding == 'amplitude':
            X = normalize_quantum_data(X, method='l2')
        elif encoding == 'angle':
            scaler = MinMaxScaler(feature_range=(0, 2*np.pi))
            X = scaler.fit_transform(X)
        else:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

    # Train-test split
    n_train = int(n_samples * (1 - test_size))
    indices = np.random.RandomState(random_state).permutation(n_samples)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    metadata = {
        'dataset_type': 'synthetic_classification',
        'n_samples': n_samples,
        'n_features': n_features,
        'n_classes': n_classes,
        'n_informative': n_informative,
        'n_redundant': n_redundant,
        'class_sep': class_sep,
        'encoding': encoding,
        'normalized': normalize
    }

    return X_train, X_test, y_train, y_test, metadata


def generate_regression_data(
    n_samples: int = 200,
    n_features: int = 4,
    n_informative: Optional[int] = None,
    noise: float = 0.1,
    encoding: str = 'amplitude',
    normalize: bool = True,
    test_size: float = 0.2,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate synthetic regression data for quantum machine learning.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        n_informative: Number of informative features
        noise: Noise level in target
        encoding: Type of quantum encoding
        normalize: Whether to normalize features
        test_size: Proportion for test split
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, metadata)

    """
    if n_informative is None:
        n_informative = n_features

    # Generate synthetic regression data
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state
    )

    # Normalization
    if normalize:
        if encoding == 'amplitude':
            X = normalize_quantum_data(X, method='l2')
        elif encoding == 'angle':
            scaler = MinMaxScaler(feature_range=(0, 2*np.pi))
            X = scaler.fit_transform(X)
        else:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Normalize targets
        y = (y - np.mean(y)) / np.std(y)

    # Train-test split
    n_train = int(n_samples * (1 - test_size))
    indices = np.random.RandomState(random_state).permutation(n_samples)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    metadata = {
        'dataset_type': 'synthetic_regression',
        'n_samples': n_samples,
        'n_features': n_features,
        'n_informative': n_informative,
        'noise': noise,
        'encoding': encoding,
        'normalized': normalize
    }

    return X_train, X_test, y_train, y_test, metadata


def generate_clustering_data(
    n_samples: int = 200,
    n_features: int = 4,
    n_clusters: int = 3,
    cluster_std: float = 1.0,
    center_box: Tuple[float, float] = (-10., 10.),
    encoding: str = 'amplitude',
    normalize: bool = True,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate synthetic clustering data for quantum machine learning.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        n_clusters: Number of clusters
        cluster_std: Standard deviation of clusters
        center_box: Bounding box for cluster centers
        encoding: Type of quantum encoding
        normalize: Whether to normalize features
        random_state: Random seed
        
    Returns:
        Tuple of (X, y_true, metadata)

    """
    # Generate clustering data
    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        center_box=center_box,
        random_state=random_state
    )

    # Normalization
    if normalize:
        if encoding == 'amplitude':
            X = normalize_quantum_data(X, method='l2')
        elif encoding == 'angle':
            scaler = MinMaxScaler(feature_range=(0, 2*np.pi))
            X = scaler.fit_transform(X)
        else:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

    metadata = {
        'dataset_type': 'synthetic_clustering',
        'n_samples': n_samples,
        'n_features': n_features,
        'n_clusters': n_clusters,
        'cluster_std': cluster_std,
        'encoding': encoding,
        'normalized': normalize
    }

    return X, y_true, metadata


def generate_portfolio_data(
    n_assets: int = 8,
    n_scenarios: int = 100,
    risk_level: float = 0.2,
    correlation: float = 0.3,
    normalize: bool = True,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate synthetic portfolio optimization data for quantum finance algorithms.
    
    Args:
        n_assets: Number of assets in portfolio
        n_scenarios: Number of return scenarios
        risk_level: Overall risk level (volatility)
        correlation: Average correlation between assets
        normalize: Whether to normalize returns
        random_state: Random seed
        
    Returns:
        Tuple of (returns, covariance_matrix, expected_returns, metadata)

    """
    np.random.seed(random_state)

    # Generate expected returns
    expected_returns = np.random.uniform(0.05, 0.20, n_assets)

    # Generate correlation matrix
    correlations = np.full((n_assets, n_assets), correlation)
    np.fill_diagonal(correlations, 1.0)

    # Add some randomness to correlations
    noise = np.random.uniform(-0.1, 0.1, (n_assets, n_assets))
    correlations += (noise + noise.T) / 2
    np.fill_diagonal(correlations, 1.0)

    # Ensure positive definite
    correlations = np.maximum(correlations, -0.99)
    correlations = np.minimum(correlations, 0.99)

    # Generate volatilities
    volatilities = np.random.uniform(
        risk_level * 0.5,
        risk_level * 1.5,
        n_assets
    )

    # Create covariance matrix
    covariance_matrix = np.outer(volatilities, volatilities) * correlations

    # Generate return scenarios
    returns = np.random.multivariate_normal(
        expected_returns,
        covariance_matrix,
        n_scenarios
    )

    if normalize:
        returns = normalize_quantum_data(returns, method='l2')

    metadata = {
        'dataset_type': 'portfolio_optimization',
        'n_assets': n_assets,
        'n_scenarios': n_scenarios,
        'risk_level': risk_level,
        'avg_correlation': correlation,
        'normalized': normalize
    }

    return returns, covariance_matrix, expected_returns, metadata
