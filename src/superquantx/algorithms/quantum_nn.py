"""Quantum Neural Network implementation.

This module provides quantum neural network architectures for machine learning
tasks using parameterized quantum circuits as trainable layers.
"""

import logging
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .base_algorithm import SupervisedQuantumAlgorithm


logger = logging.getLogger(__name__)

class QuantumNN(SupervisedQuantumAlgorithm):
    """Quantum Neural Network for classification and regression.

    This implementation uses parameterized quantum circuits as neural network
    layers, with classical optimization to train the quantum parameters.

    The network can be configured with different architectures:
    - Pure quantum: Only quantum layers
    - Hybrid: Combination of quantum and classical layers
    - Variational: Variational quantum circuits with measurement

    Args:
        backend: Quantum backend for circuit execution
        n_layers: Number of quantum layers
        architecture: Network architecture ('pure', 'hybrid', 'variational')
        encoding: Data encoding method ('amplitude', 'angle', 'basis')
        entanglement: Entanglement pattern ('linear', 'circular', 'full')
        measurement: Measurement strategy ('expectation', 'sampling', 'statevector')
        optimizer: Classical optimizer for training
        learning_rate: Learning rate for training
        batch_size: Training batch size
        shots: Number of measurement shots
        **kwargs: Additional parameters

    Example:
        >>> qnn = QuantumNN(backend='pennylane', n_layers=3, architecture='hybrid')
        >>> qnn.fit(X_train, y_train)
        >>> predictions = qnn.predict(X_test)
        >>> accuracy = qnn.score(X_test, y_test)

    """

    def __init__(
        self,
        backend: str | Any,
        n_layers: int = 3,
        architecture: str = 'hybrid',
        encoding: str = 'angle',
        entanglement: str = 'linear',
        measurement: str = 'expectation',
        optimizer: str = 'adam',
        learning_rate: float = 0.01,
        batch_size: int = 32,
        max_epochs: int = 100,
        shots: int = 1024,
        task_type: str = 'classification',
        **kwargs
    ) -> None:
        super().__init__(backend=backend, shots=shots, **kwargs)

        self.n_layers = n_layers
        self.architecture = architecture
        self.encoding = encoding
        self.entanglement = entanglement
        self.measurement = measurement
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.task_type = task_type

        # Network components
        self.quantum_layers = []
        self.classical_layers = []
        self.n_qubits = None
        self.n_params = None

        # Training components
        self.weights = None
        self.encoder = LabelEncoder() if task_type == 'classification' else None
        self.scaler = StandardScaler()
        self.optimizer = None

        # Training history
        self.loss_history = []
        self.accuracy_history = []

        logger.info(f"Initialized QuantumNN with {n_layers} layers, architecture={architecture}")

    def _determine_qubits(self, n_features: int) -> int:
        """Determine number of qubits needed for encoding."""
        if self.encoding == 'amplitude':
            return max(1, int(np.ceil(np.log2(n_features))))
        elif self.encoding == 'angle':
            return n_features
        elif self.encoding == 'basis':
            return int(np.ceil(np.log2(n_features)))
        else:
            return n_features

    def _create_encoding_layer(self, x: np.ndarray) -> Any:
        """Create data encoding quantum layer."""
        try:
            if hasattr(self.backend, 'create_encoding_layer'):
                return self.backend.create_encoding_layer(
                    data=x,
                    encoding=self.encoding,
                    n_qubits=self.n_qubits
                )
            else:
                return self._fallback_encoding(x)
        except Exception as e:
            logger.error(f"Failed to create encoding layer: {e}")
            return self._fallback_encoding(x)

    def _fallback_encoding(self, x: np.ndarray) -> Any:
        """Fallback data encoding implementation."""
        logger.warning("Using fallback data encoding")
        return None

    def _create_variational_layer(self, params: np.ndarray, layer_idx: int) -> Any:
        """Create parameterized variational quantum layer."""
        try:
            if hasattr(self.backend, 'create_variational_layer'):
                return self.backend.create_variational_layer(
                    params=params,
                    layer_idx=layer_idx,
                    entanglement=self.entanglement,
                    n_qubits=self.n_qubits
                )
            else:
                return self._fallback_variational_layer(params, layer_idx)
        except Exception as e:
            logger.error(f"Failed to create variational layer {layer_idx}: {e}")
            return self._fallback_variational_layer(params, layer_idx)

    def _fallback_variational_layer(self, params: np.ndarray, layer_idx: int) -> Any:
        """Fallback variational layer implementation."""
        logger.warning(f"Using fallback variational layer {layer_idx}")
        return None

    def _create_measurement_layer(self) -> Any:
        """Create measurement layer."""
        try:
            if hasattr(self.backend, 'create_measurement_layer'):
                return self.backend.create_measurement_layer(
                    measurement=self.measurement,
                    n_qubits=self.n_qubits
                )
            else:
                return self._fallback_measurement()
        except Exception as e:
            logger.error(f"Failed to create measurement layer: {e}")
            return self._fallback_measurement()

    def _fallback_measurement(self) -> Any:
        """Fallback measurement implementation."""
        logger.warning("Using fallback measurement layer")
        return None

    def _build_network(self) -> None:
        """Build the complete quantum neural network."""
        logger.info(f"Building {self.architecture} quantum neural network")

        # Calculate number of parameters needed
        params_per_layer = self._get_params_per_layer()
        self.n_params = self.n_layers * params_per_layer

        # Initialize weights
        self.weights = np.random.uniform(-np.pi, np.pi, self.n_params)

        # Build network layers based on architecture
        if self.architecture == 'pure':
            self._build_pure_quantum_network()
        elif self.architecture == 'hybrid':
            self._build_hybrid_network()
        elif self.architecture == 'variational':
            self._build_variational_network()
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

    def _get_params_per_layer(self) -> int:
        """Get number of parameters per quantum layer."""
        if hasattr(self.backend, 'get_layer_param_count'):
            return self.backend.get_layer_param_count(
                n_qubits=self.n_qubits,
                entanglement=self.entanglement
            )
        else:
            # Default parameter count estimate
            return 2 * self.n_qubits  # RY and RZ rotations per qubit

    def _build_pure_quantum_network(self) -> None:
        """Build pure quantum network (only quantum layers)."""
        self.quantum_layers = []
        for i in range(self.n_layers):
            layer = {
                'type': 'variational',
                'params': slice(i * self.n_qubits * 2, (i + 1) * self.n_qubits * 2),
                'layer_idx': i
            }
            self.quantum_layers.append(layer)

    def _build_hybrid_network(self) -> None:
        """Build hybrid quantum-classical network."""
        self._build_pure_quantum_network()

        # Add classical layers for hybrid processing
        # Always create classical layers for hybrid networks to handle output size
        self.classical_layers = [
            {'type': 'dense', 'units': self.n_classes_ or 1, 'activation': 'softmax' if self.task_type == 'classification' else 'linear'}
        ]

    def _build_variational_network(self) -> None:
        """Build variational quantum circuit network."""
        self._build_pure_quantum_network()

    def _forward_pass(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Perform forward pass through the quantum neural network.

        Args:
            x: Input data
            weights: Network weights

        Returns:
            Network output

        """
        try:
            if hasattr(self.backend, 'execute_qnn'):
                result = self.backend.execute_qnn(
                    input_data=x,
                    weights=weights,
                    quantum_layers=self.quantum_layers,
                    classical_layers=self.classical_layers,
                    encoding=self.encoding,
                    measurement=self.measurement,
                    shots=self.shots
                )
                return result
            else:
                return self._fallback_forward_pass(x, weights)
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            return self._fallback_forward_pass(x, weights)

    def _fallback_forward_pass(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Fallback forward pass implementation."""
        logger.warning("Using fallback forward pass")
        batch_size = x.shape[0]
        output_size = self.n_classes_ if self.task_type == 'classification' else 1
        return np.random.random((batch_size, output_size))

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute loss function."""
        if self.task_type == 'classification':
            # Cross-entropy loss
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid log(0)
            if len(y_true.shape) == 1:
                # Convert to one-hot if needed
                y_true_oh = np.zeros((len(y_true), self.n_classes_))
                y_true_oh[np.arange(len(y_true)), y_true.astype(int)] = 1
                y_true = y_true_oh
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        else:
            # Mean squared error for regression
            return np.mean((y_true - y_pred.flatten()) ** 2)

    def _compute_gradients(self, x: np.ndarray, y_true: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Compute gradients using parameter-shift rule."""
        gradients = np.zeros_like(weights)
        shift = np.pi / 2

        for i in range(len(weights)):
            # Forward shift
            weights_plus = weights.copy()
            weights_plus[i] += shift
            y_pred_plus = self._forward_pass(x, weights_plus)
            loss_plus = self._compute_loss(y_true, y_pred_plus)

            # Backward shift
            weights_minus = weights.copy()
            weights_minus[i] -= shift
            y_pred_minus = self._forward_pass(x, weights_minus)
            loss_minus = self._compute_loss(y_true, y_pred_minus)

            # Gradient via parameter-shift rule
            gradients[i] = 0.5 * (loss_plus - loss_minus)

        return gradients

    def _update_weights(self, gradients: np.ndarray) -> None:
        """Update weights using optimizer."""
        if self.optimizer_name == 'sgd':
            self.weights -= self.learning_rate * gradients
        elif self.optimizer_name == 'adam':
            # Simplified Adam optimizer
            if not hasattr(self, 'adam_m'):
                self.adam_m = np.zeros_like(self.weights)
                self.adam_v = np.zeros_like(self.weights)
                self.adam_t = 0

            self.adam_t += 1
            beta1, beta2 = 0.9, 0.999

            self.adam_m = beta1 * self.adam_m + (1 - beta1) * gradients
            self.adam_v = beta2 * self.adam_v + (1 - beta2) * gradients**2

            m_hat = self.adam_m / (1 - beta1**self.adam_t)
            v_hat = self.adam_v / (1 - beta2**self.adam_t)

            self.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        else:
            # Default: simple gradient descent
            self.weights -= self.learning_rate * gradients

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'QuantumNN':
        """Train the quantum neural network.

        Args:
            X: Training data features
            y: Training data labels
            **kwargs: Additional training parameters

        Returns:
            Self for method chaining

        """
        logger.info(f"Training QuantumNN on {X.shape[0]} samples with {X.shape[1]} features")

        # Validate and preprocess data
        super().fit(X, y, **kwargs)

        # Set number of classes for classification BEFORE building network
        if self.task_type == 'classification':
            unique_classes = np.unique(y)
            self.n_classes_ = len(unique_classes)
            logger.info(f"Detected {self.n_classes_} classes for classification: {unique_classes}")

        # Scale features
        X = self.scaler.fit_transform(X)

        # Encode labels for classification after setting n_classes_
        if self.task_type == 'classification' and self.encoder:
            y = self.encoder.fit_transform(y)

        # Determine network architecture
        self.n_qubits = self._determine_qubits(X.shape[1])
        self._build_network()

        # Reset training history
        self.loss_history = []
        self.accuracy_history = []

        logger.info(f"Training network with {self.n_qubits} qubits and {self.n_params} parameters")

        # Training loop
        for epoch in range(self.max_epochs):
            epoch_losses = []
            epoch_accuracies = []

            # Mini-batch training
            for i in range(0, len(X), self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]

                # Forward pass
                y_pred = self._forward_pass(X_batch, self.weights)

                # Compute loss
                loss = self._compute_loss(y_batch, y_pred)
                epoch_losses.append(loss)

                # Compute accuracy for classification
                if self.task_type == 'classification':
                    y_pred_labels = np.argmax(y_pred, axis=1)
                    accuracy = accuracy_score(y_batch, y_pred_labels)
                    epoch_accuracies.append(accuracy)

                # Compute gradients and update weights
                gradients = self._compute_gradients(X_batch, y_batch, self.weights)
                self._update_weights(gradients)

            # Record epoch statistics
            epoch_loss = np.mean(epoch_losses)
            self.loss_history.append(epoch_loss)

            if self.task_type == 'classification' and epoch_accuracies:
                epoch_accuracy = np.mean(epoch_accuracies)
                self.accuracy_history.append(epoch_accuracy)

                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.4f}")
            else:
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")

            # Early stopping check
            if len(self.loss_history) > 10 and self._check_early_stopping():
                logger.info(f"Early stopping at epoch {epoch}")
                break

        self.is_fitted = True

        # Final training statistics
        final_loss = self.loss_history[-1]
        logger.info(f"Training completed. Final loss: {final_loss:.4f}")

        return self

    def _check_early_stopping(self, patience: int = 10, min_delta: float = 1e-4) -> bool:
        """Check if training should stop early."""
        if len(self.loss_history) < patience + 1:
            return False

        recent_losses = self.loss_history[-patience-1:]
        best_loss = min(recent_losses[:-1])
        current_loss = recent_losses[-1]

        return (best_loss - current_loss) < min_delta

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Make predictions using the trained quantum neural network.

        Args:
            X: Input data for prediction
            **kwargs: Additional prediction parameters

        Returns:
            Predicted labels or values

        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Scale features
        X = self.scaler.transform(X)

        # Forward pass
        y_pred = self._forward_pass(X, self.weights)

        if self.task_type == 'classification':
            # Return class labels
            predictions = np.argmax(y_pred, axis=1)
            if self.encoder:
                predictions = self.encoder.inverse_transform(predictions)
            return predictions
        else:
            # Return continuous values for regression
            return y_pred.flatten()

    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Input data for prediction
            **kwargs: Additional parameters

        Returns:
            Predicted class probabilities

        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification tasks")

        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Scale features
        X = self.scaler.transform(X)

        # Forward pass returns probabilities for classification
        return self._forward_pass(X, self.weights)

    def get_circuit_depth(self) -> int:
        """Get the depth of the quantum circuit."""
        if hasattr(self.backend, 'get_circuit_depth'):
            return self.backend.get_circuit_depth(self.quantum_layers)
        else:
            return self.n_layers * 2  # Estimate

    def get_training_history(self) -> dict[str, list[float]]:
        """Get training history."""
        history = {'loss': self.loss_history}
        if self.accuracy_history:
            history['accuracy'] = self.accuracy_history
        return history

    def analyze_expressivity(self) -> dict[str, Any]:
        """Analyze the expressivity of the quantum neural network."""
        analysis = {
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'n_parameters': self.n_params,
            'circuit_depth': self.get_circuit_depth(),
            'entanglement_pattern': self.entanglement,
            'encoding_method': self.encoding,
        }

        # Estimate expressivity metrics
        analysis.update({
            'parameter_space_dimension': self.n_params,
            'hilbert_space_dimension': 2**self.n_qubits,
            'expressivity_ratio': self.n_params / (2**self.n_qubits),
        })

        return analysis

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get quantum neural network parameters."""
        params = super().get_params(deep)
        params.update({
            'n_layers': self.n_layers,
            'architecture': self.architecture,
            'encoding': self.encoding,
            'entanglement': self.entanglement,
            'measurement': self.measurement,
            'optimizer': self.optimizer_name,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'task_type': self.task_type,
        })
        return params

    def set_params(self, **params) -> 'QuantumNN':
        """Set quantum neural network parameters."""
        if self.is_fitted and any(key in params for key in
                                 ['n_layers', 'architecture', 'encoding', 'entanglement']):
            logger.warning("Changing core parameters requires refitting the model")
            self.is_fitted = False

        return super().set_params(**params)

# Alias for backwards compatibility
QuantumNeuralNetwork = QuantumNN
