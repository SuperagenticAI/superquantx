"""Quantum-adapted classical datasets for machine learning.

This module provides classical datasets adapted for quantum machine learning,
with proper preprocessing and encoding for quantum circuits.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .preprocessing import normalize_quantum_data


def load_iris_quantum(
    n_features: Optional[int] = None,
    encoding: str = 'amplitude',
    normalize: bool = True,
    test_size: float = 0.2,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load and preprocess the Iris dataset for quantum machine learning.
    
    Args:
        n_features: Number of features to keep (default: all 4)
        encoding: Type of quantum encoding ('amplitude', 'angle', 'basis')
        normalize: Whether to normalize features
        test_size: Proportion of dataset for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, metadata)

    """
    # Load dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Feature selection
    if n_features is not None and n_features < X.shape[1]:
        # Select features with highest variance
        feature_vars = np.var(X, axis=0)
        selected_features = np.argsort(feature_vars)[-n_features:]
        X = X[:, selected_features]
        feature_names = [iris.feature_names[i] for i in selected_features]
    else:
        feature_names = iris.feature_names
        n_features = X.shape[1]

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

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Metadata
    metadata = {
        'dataset_name': 'iris',
        'n_samples': len(X),
        'n_features': n_features,
        'n_classes': len(np.unique(y)),
        'class_names': iris.target_names.tolist(),
        'feature_names': feature_names,
        'encoding': encoding,
        'normalized': normalize
    }

    return X_train, X_test, y_train, y_test, metadata


def load_wine_quantum(
    n_features: Optional[int] = 8,
    encoding: str = 'amplitude',
    normalize: bool = True,
    test_size: float = 0.2,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load and preprocess the Wine dataset for quantum machine learning.
    
    Args:
        n_features: Number of top features to keep (default: 8)
        encoding: Type of quantum encoding
        normalize: Whether to normalize features
        test_size: Proportion of dataset for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, metadata)

    """
    # Load dataset
    wine = datasets.load_wine()
    X, y = wine.data, wine.target

    # Feature selection based on variance
    if n_features is not None and n_features < X.shape[1]:
        feature_vars = np.var(X, axis=0)
        selected_features = np.argsort(feature_vars)[-n_features:]
        X = X[:, selected_features]
        feature_names = [wine.feature_names[i] for i in selected_features]
    else:
        feature_names = wine.feature_names
        n_features = X.shape[1]

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

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    metadata = {
        'dataset_name': 'wine',
        'n_samples': len(X),
        'n_features': n_features,
        'n_classes': len(np.unique(y)),
        'class_names': wine.target_names.tolist(),
        'feature_names': feature_names,
        'encoding': encoding,
        'normalized': normalize
    }

    return X_train, X_test, y_train, y_test, metadata


def load_digits_quantum(
    n_classes: int = 10,
    n_pixels: Optional[int] = 32,
    encoding: str = 'amplitude',
    normalize: bool = True,
    test_size: float = 0.2,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load and preprocess the Digits dataset for quantum machine learning.
    
    Args:
        n_classes: Number of digit classes to include (2-10)
        n_pixels: Number of pixels to keep (reduces from 64)
        encoding: Type of quantum encoding
        normalize: Whether to normalize features
        test_size: Proportion of dataset for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, metadata)

    """
    # Load dataset
    digits = datasets.load_digits()
    X, y = digits.data, digits.target

    # Class filtering
    if n_classes < 10:
        mask = y < n_classes
        X, y = X[mask], y[mask]

    # Feature selection (pixel reduction)
    if n_pixels is not None and n_pixels < X.shape[1]:
        # Select pixels with highest variance
        pixel_vars = np.var(X, axis=0)
        selected_pixels = np.argsort(pixel_vars)[-n_pixels:]
        X = X[:, selected_pixels]
    else:
        n_pixels = X.shape[1]

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

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    metadata = {
        'dataset_name': 'digits',
        'n_samples': len(X),
        'n_features': n_pixels,
        'n_classes': n_classes,
        'original_shape': (8, 8),
        'encoding': encoding,
        'normalized': normalize
    }

    return X_train, X_test, y_train, y_test, metadata


def load_breast_cancer_quantum(
    n_features: Optional[int] = 16,
    encoding: str = 'amplitude',
    normalize: bool = True,
    test_size: float = 0.2,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load and preprocess the Breast Cancer dataset for quantum machine learning.
    
    Args:
        n_features: Number of top features to keep
        encoding: Type of quantum encoding
        normalize: Whether to normalize features
        test_size: Proportion of dataset for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, metadata)

    """
    # Load dataset
    cancer = datasets.load_breast_cancer()
    X, y = cancer.data, cancer.target

    # Feature selection based on correlation with target
    if n_features is not None and n_features < X.shape[1]:
        correlations = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
        selected_features = np.argsort(correlations)[-n_features:]
        X = X[:, selected_features]
        feature_names = [cancer.feature_names[i] for i in selected_features]
    else:
        feature_names = cancer.feature_names
        n_features = X.shape[1]

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

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    metadata = {
        'dataset_name': 'breast_cancer',
        'n_samples': len(X),
        'n_features': n_features,
        'n_classes': 2,
        'class_names': cancer.target_names.tolist(),
        'feature_names': feature_names,
        'encoding': encoding,
        'normalized': normalize
    }

    return X_train, X_test, y_train, y_test, metadata
