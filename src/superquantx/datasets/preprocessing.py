"""Data preprocessing utilities for quantum machine learning.

This module provides preprocessing functions and classes to prepare
classical data for quantum machine learning algorithms.
"""

from typing import Literal, Optional

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class QuantumFeatureEncoder:
    """Base class for quantum feature encoding strategies.
    
    Quantum feature encoding maps classical data to quantum states,
    which is crucial for quantum machine learning algorithms.
    """

    def __init__(self, encoding_type: str = 'amplitude'):
        self.encoding_type = encoding_type
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> 'QuantumFeatureEncoder':
        """Fit the encoder to training data."""
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using the fitted encoder."""
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit encoder and transform data in one step."""
        return self.fit(X).transform(X)


class AmplitudeEncoder(QuantumFeatureEncoder):
    """Amplitude encoding for quantum machine learning.
    
    Encodes classical data as amplitudes of quantum states.
    Each data point becomes a quantum state |ψ⟩ = Σᵢ xᵢ|i⟩.
    
    The data is normalized so that ||x||₂ = 1 for proper quantum state encoding.
    """

    def __init__(self, normalize_samples: bool = True, normalize_features: bool = False):
        super().__init__('amplitude')
        self.normalize_samples = normalize_samples
        self.normalize_features = normalize_features
        self.feature_scaler = None

    def fit(self, X: np.ndarray) -> 'AmplitudeEncoder':
        """Fit the amplitude encoder.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            
        Returns:
            Self for method chaining

        """
        if self.normalize_features:
            self.feature_scaler = StandardScaler()
            self.feature_scaler.fit(X)

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using amplitude encoding.
        
        Args:
            X: Data to transform of shape (n_samples, n_features)
            
        Returns:
            Transformed data with proper normalization

        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transform")

        X_encoded = X.copy()

        # Feature normalization
        if self.normalize_features and self.feature_scaler is not None:
            X_encoded = self.feature_scaler.transform(X_encoded)

        # Sample normalization (L2 norm = 1 for each sample)
        if self.normalize_samples:
            norms = np.linalg.norm(X_encoded, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            X_encoded = X_encoded / norms

        return X_encoded


class AngleEncoder(QuantumFeatureEncoder):
    """Angle encoding for quantum machine learning.
    
    Encodes classical features as rotation angles in quantum circuits.
    Each feature xᵢ becomes a rotation angle, typically in [0, 2π].
    """

    def __init__(self, angle_range: tuple = (0, 2 * np.pi)):
        super().__init__('angle')
        self.angle_range = angle_range
        self.scaler = None

    def fit(self, X: np.ndarray) -> 'AngleEncoder':
        """Fit the angle encoder.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            
        Returns:
            Self for method chaining

        """
        self.scaler = MinMaxScaler(feature_range=self.angle_range)
        self.scaler.fit(X)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using angle encoding.
        
        Args:
            X: Data to transform of shape (n_samples, n_features)
            
        Returns:
            Data scaled to angle range

        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transform")

        return self.scaler.transform(X)


class BasisEncoder(QuantumFeatureEncoder):
    """Basis encoding for quantum machine learning.
    
    Encodes classical data directly in the computational basis.
    Each data point corresponds to a basis state |x⟩.
    """

    def __init__(self, n_qubits: Optional[int] = None):
        super().__init__('basis')
        self.n_qubits = n_qubits

    def fit(self, X: np.ndarray) -> 'BasisEncoder':
        """Fit the basis encoder."""
        if self.n_qubits is None:
            # Determine number of qubits needed
            max_value = np.max(X)
            self.n_qubits = int(np.ceil(np.log2(max_value + 1)))

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data for basis encoding."""
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transform")

        # Convert to integers for basis states
        X_int = np.round(X).astype(int)

        # Clip to valid range
        max_state = 2 ** self.n_qubits - 1
        X_int = np.clip(X_int, 0, max_state)

        return X_int


def normalize_quantum_data(
    X: np.ndarray,
    method: Literal['l1', 'l2', 'max'] = 'l2',
    axis: int = 1
) -> np.ndarray:
    """Normalize data for quantum machine learning.
    
    Args:
        X: Data to normalize of shape (n_samples, n_features)
        method: Normalization method ('l1', 'l2', or 'max')
        axis: Axis along which to normalize (1 for samples, 0 for features)
        
    Returns:
        Normalized data

    """
    if method == 'l1':
        norms = np.sum(np.abs(X), axis=axis, keepdims=True)
    elif method == 'l2':
        norms = np.sqrt(np.sum(X ** 2, axis=axis, keepdims=True))
    elif method == 'max':
        norms = np.max(np.abs(X), axis=axis, keepdims=True)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Avoid division by zero
    norms[norms == 0] = 1

    return X / norms


def pad_to_power_of_two(X: np.ndarray, pad_value: float = 0.0) -> np.ndarray:
    """Pad feature dimension to next power of two.
    
    Many quantum algorithms work more efficiently when the number
    of features is a power of two.
    
    Args:
        X: Input data of shape (n_samples, n_features)
        pad_value: Value to use for padding
        
    Returns:
        Padded data with power-of-two feature dimension

    """
    n_features = X.shape[1]
    n_qubits = int(np.ceil(np.log2(n_features)))
    n_padded = 2 ** n_qubits

    if n_padded == n_features:
        return X

    # Create padding
    n_samples = X.shape[0]
    padding_shape = (n_samples, n_padded - n_features)
    padding = np.full(padding_shape, pad_value)

    return np.hstack([X, padding])


def quantum_feature_map(
    X: np.ndarray,
    feature_map_type: str = 'ZZFeatureMap',
    reps: int = 1,
    parameter_prefix: str = 'x'
) -> dict:
    """Generate quantum feature map parameters.
    
    Args:
        X: Input data
        feature_map_type: Type of feature map
        reps: Number of repetitions
        parameter_prefix: Prefix for parameter names
        
    Returns:
        Dictionary with feature map configuration

    """
    n_features = X.shape[1]

    feature_map_config = {
        'type': feature_map_type,
        'n_qubits': n_features,
        'n_features': n_features,
        'reps': reps,
        'parameter_prefix': parameter_prefix,
        'data_shape': X.shape
    }

    if feature_map_type == 'ZZFeatureMap':
        feature_map_config.update({
            'entanglement': 'linear',
            'alpha': 2.0
        })
    elif feature_map_type == 'ZFeatureMap':
        feature_map_config.update({
            'rotation': 'Y'
        })
    elif feature_map_type == 'PauliFeatureMap':
        feature_map_config.update({
            'paulis': ['Z', 'ZZ']
        })

    return feature_map_config
