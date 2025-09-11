"""Hybrid Classical-Quantum Classifier implementation.

This module provides hybrid classifiers that combine classical and quantum
machine learning components for enhanced performance and flexibility.
"""

import logging
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from .base_algorithm import SupervisedQuantumAlgorithm
from .quantum_nn import QuantumNN
from .quantum_svm import QuantumSVM


logger = logging.getLogger(__name__)

class HybridClassifier(SupervisedQuantumAlgorithm):
    """Hybrid Classical-Quantum Classifier.

    This classifier combines classical and quantum machine learning algorithms
    to leverage the strengths of both approaches. It can operate in different modes:
    - Ensemble: Combines predictions from multiple quantum and classical models
    - Sequential: Uses quantum features as input to classical models
    - Voting: Majority voting among quantum and classical predictions
    - Stacking: Uses meta-learner to combine quantum and classical predictions

    Args:
        backend: Quantum backend for quantum components
        hybrid_mode: Mode of operation ('ensemble', 'sequential', 'voting', 'stacking')
        quantum_algorithms: List of quantum algorithms to include
        classical_algorithms: List of classical algorithms to include
        quantum_weight: Weight for quantum predictions (0-1)
        feature_selection: Whether to use quantum feature selection
        meta_learner: Meta-learning algorithm for stacking mode
        shots: Number of measurement shots
        **kwargs: Additional parameters

    Example:
        >>> hybrid = HybridClassifier(
        ...     backend='pennylane',
        ...     hybrid_mode='ensemble',
        ...     quantum_algorithms=['quantum_svm', 'quantum_nn'],
        ...     classical_algorithms=['random_forest', 'svm']
        ... )
        >>> hybrid.fit(X_train, y_train)
        >>> predictions = hybrid.predict(X_test)

    """

    def __init__(
        self,
        backend: str | Any,
        hybrid_mode: str = 'ensemble',
        quantum_algorithms: list[str] | None = None,
        classical_algorithms: list[str] | None = None,
        quantum_weight: float = 0.5,
        feature_selection: bool = False,
        meta_learner: str = 'logistic_regression',
        shots: int = 1024,
        normalize_data: bool = True,
        **kwargs
    ) -> None:
        super().__init__(backend=backend, shots=shots, **kwargs)

        self.hybrid_mode = hybrid_mode
        self.quantum_algorithms = quantum_algorithms or ['quantum_svm']
        self.classical_algorithms = classical_algorithms or ['random_forest']
        self.quantum_weight = quantum_weight
        self.feature_selection = feature_selection
        self.meta_learner_name = meta_learner
        self.normalize_data = normalize_data

        # Initialize models
        self.quantum_models = {}
        self.classical_models = {}
        self.meta_learner = None
        self.feature_selector = None

        # Data preprocessing
        self.scaler = StandardScaler() if normalize_data else None
        self.label_encoder = LabelEncoder()

        # Model performance tracking
        self.quantum_scores = {}
        self.classical_scores = {}
        self.hybrid_score = None
        self.feature_importance_ = None

        self._initialize_models()

        logger.info(f"Initialized HybridClassifier with mode={hybrid_mode}")
        logger.info(f"Quantum algorithms: {self.quantum_algorithms}")
        logger.info(f"Classical algorithms: {self.classical_algorithms}")

    def _initialize_models(self) -> None:
        """Initialize quantum and classical models."""
        # Initialize quantum models
        for algo in self.quantum_algorithms:
            if algo == 'quantum_svm':
                self.quantum_models[algo] = QuantumSVM(
                    backend=self.backend,
                    shots=self.shots
                )
            elif algo == 'quantum_nn':
                self.quantum_models[algo] = QuantumNN(
                    backend=self.backend,
                    shots=self.shots,
                    task_type='classification'
                )
            else:
                logger.warning(f"Unknown quantum algorithm: {algo}")

        # Initialize classical models
        for algo in self.classical_algorithms:
            if algo == 'random_forest':
                self.classical_models[algo] = RandomForestClassifier(
                    n_estimators=100, random_state=42
                )
            elif algo == 'gradient_boosting':
                self.classical_models[algo] = GradientBoostingClassifier(
                    random_state=42
                )
            elif algo == 'logistic_regression':
                self.classical_models[algo] = LogisticRegression(
                    random_state=42, max_iter=1000
                )
            elif algo == 'svm':
                self.classical_models[algo] = SVC(
                    probability=True, random_state=42
                )
            else:
                logger.warning(f"Unknown classical algorithm: {algo}")

        # Initialize meta-learner for stacking
        if self.hybrid_mode == 'stacking':
            if self.meta_learner_name == 'logistic_regression':
                self.meta_learner = LogisticRegression(random_state=42)
            elif self.meta_learner_name == 'random_forest':
                self.meta_learner = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                self.meta_learner = LogisticRegression(random_state=42)

        # Initialize feature selector
        if self.feature_selection:
            from sklearn.feature_selection import SelectKBest, f_classif
            self.feature_selector = SelectKBest(score_func=f_classif, k='all')

    def _apply_feature_selection(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Apply quantum-inspired feature selection."""
        if not self.feature_selection or self.feature_selector is None:
            return X

        if y is not None:
            # Fit and transform
            X_selected = self.feature_selector.fit_transform(X, y)

            # Store feature importance
            if hasattr(self.feature_selector, 'scores_'):
                self.feature_importance_ = self.feature_selector.scores_
        else:
            # Transform only
            X_selected = self.feature_selector.transform(X)

        logger.info(f"Feature selection: {X.shape[1]} -> {X_selected.shape[1]} features")
        return X_selected

    def _train_quantum_models(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Train quantum models and return their scores."""
        scores = {}

        for name, model in self.quantum_models.items():
            try:
                logger.info(f"Training quantum model: {name}")
                model.fit(X, y)

                # Evaluate model
                predictions = model.predict(X)
                score = accuracy_score(y, predictions)
                scores[name] = score

                logger.info(f"{name} training accuracy: {score:.4f}")

            except Exception as e:
                logger.error(f"Failed to train quantum model {name}: {e}")
                scores[name] = 0.0

        return scores

    def _train_classical_models(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Train classical models and return their scores."""
        scores = {}

        for name, model in self.classical_models.items():
            try:
                logger.info(f"Training classical model: {name}")
                model.fit(X, y)

                # Evaluate model
                predictions = model.predict(X)
                score = accuracy_score(y, predictions)
                scores[name] = score

                logger.info(f"{name} training accuracy: {score:.4f}")

            except Exception as e:
                logger.error(f"Failed to train classical model {name}: {e}")
                scores[name] = 0.0

        return scores

    def _get_base_predictions(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get predictions from all base models."""
        quantum_preds = []
        classical_preds = []

        # Get quantum predictions
        for name, model in self.quantum_models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X)
                    quantum_preds.append(pred_proba)
                else:
                    pred = model.predict(X)
                    # Convert to one-hot for consistency
                    pred_proba = np.zeros((len(pred), self.n_classes_))
                    pred_proba[np.arange(len(pred)), pred] = 1.0
                    quantum_preds.append(pred_proba)
            except Exception as e:
                logger.error(f"Failed to get predictions from quantum model {name}: {e}")
                # Add dummy predictions
                dummy_pred = np.ones((X.shape[0], self.n_classes_)) / self.n_classes_
                quantum_preds.append(dummy_pred)

        # Get classical predictions
        for name, model in self.classical_models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X)
                    classical_preds.append(pred_proba)
                else:
                    pred = model.predict(X)
                    # Convert to one-hot for consistency
                    pred_proba = np.zeros((len(pred), self.n_classes_))
                    pred_proba[np.arange(len(pred)), pred] = 1.0
                    classical_preds.append(pred_proba)
            except Exception as e:
                logger.error(f"Failed to get predictions from classical model {name}: {e}")
                # Add dummy predictions
                dummy_pred = np.ones((X.shape[0], self.n_classes_)) / self.n_classes_
                classical_preds.append(dummy_pred)

        quantum_predictions = np.array(quantum_preds) if quantum_preds else np.array([])
        classical_predictions = np.array(classical_preds) if classical_preds else np.array([])

        return quantum_predictions, classical_predictions

    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction mode."""
        quantum_preds, classical_preds = self._get_base_predictions(X)

        # Combine predictions with weighted average
        combined_pred = np.zeros((X.shape[0], self.n_classes_))

        if len(quantum_preds) > 0:
            quantum_avg = np.mean(quantum_preds, axis=0)
            combined_pred += self.quantum_weight * quantum_avg

        if len(classical_preds) > 0:
            classical_avg = np.mean(classical_preds, axis=0)
            combined_pred += (1 - self.quantum_weight) * classical_avg

        return np.argmax(combined_pred, axis=1)

    def _voting_predict(self, X: np.ndarray) -> np.ndarray:
        """Voting prediction mode."""
        quantum_preds, classical_preds = self._get_base_predictions(X)

        all_predictions = []

        # Get hard predictions
        if len(quantum_preds) > 0:
            for pred_proba in quantum_preds:
                all_predictions.append(np.argmax(pred_proba, axis=1))

        if len(classical_preds) > 0:
            for pred_proba in classical_preds:
                all_predictions.append(np.argmax(pred_proba, axis=1))

        if not all_predictions:
            return np.zeros(X.shape[0], dtype=int)

        # Majority voting
        all_predictions = np.array(all_predictions).T
        final_predictions = []

        for sample_preds in all_predictions:
            unique, counts = np.unique(sample_preds, return_counts=True)
            final_predictions.append(unique[np.argmax(counts)])

        return np.array(final_predictions)

    def _sequential_predict(self, X: np.ndarray) -> np.ndarray:
        """Sequential prediction mode (quantum features -> classical models)."""
        # Use quantum models to extract features
        quantum_features = []

        for name, model in self.quantum_models.items():
            try:
                if hasattr(model, 'transform'):
                    features = model.transform(X)
                elif hasattr(model, 'decision_function'):
                    features = model.decision_function(X)
                    if len(features.shape) == 1:
                        features = features.reshape(-1, 1)
                else:
                    # Use prediction probabilities as features
                    features = model.predict_proba(X)

                quantum_features.append(features)

            except Exception as e:
                logger.error(f"Failed to extract features from {name}: {e}")

        if not quantum_features:
            logger.warning("No quantum features extracted, using original features")
            quantum_feature_matrix = X
        else:
            quantum_feature_matrix = np.concatenate(quantum_features, axis=1)

        # Use the best classical model for final prediction
        best_classical = max(self.classical_scores.items(), key=lambda x: x[1])[0]
        model = self.classical_models[best_classical]

        return model.predict(quantum_feature_matrix)

    def _stacking_predict(self, X: np.ndarray) -> np.ndarray:
        """Stacking prediction mode."""
        quantum_preds, classical_preds = self._get_base_predictions(X)

        # Combine all predictions as meta-features
        meta_features = []

        if len(quantum_preds) > 0:
            for pred_proba in quantum_preds:
                meta_features.append(pred_proba)

        if len(classical_preds) > 0:
            for pred_proba in classical_preds:
                meta_features.append(pred_proba)

        if not meta_features:
            return np.zeros(X.shape[0], dtype=int)

        # Concatenate meta-features
        meta_X = np.concatenate(meta_features, axis=1)

        # Use meta-learner for final prediction
        return self.meta_learner.predict(meta_X)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'HybridClassifier':
        """Train the hybrid classifier.

        Args:
            X: Training data features
            y: Training data labels
            **kwargs: Additional training parameters

        Returns:
            Self for method chaining

        """
        logger.info(f"Training HybridClassifier on {X.shape[0]} samples with {X.shape[1]} features")

        # Validate and preprocess data
        super().fit(X, y, **kwargs)

        # Normalize features
        if self.normalize_data:
            X = self.scaler.fit_transform(X)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Apply feature selection
        X_selected = self._apply_feature_selection(X, y_encoded)

        # Train quantum models
        self.quantum_scores = self._train_quantum_models(X_selected, y_encoded)

        # Train classical models
        self.classical_scores = self._train_classical_models(X_selected, y_encoded)

        # Train meta-learner for stacking mode
        if self.hybrid_mode == 'stacking' and self.meta_learner is not None:
            logger.info("Training meta-learner for stacking")

            # Get base model predictions for meta-training
            quantum_preds, classical_preds = self._get_base_predictions(X_selected)

            meta_features = []
            if len(quantum_preds) > 0:
                for pred_proba in quantum_preds:
                    meta_features.append(pred_proba)
            if len(classical_preds) > 0:
                for pred_proba in classical_preds:
                    meta_features.append(pred_proba)

            if meta_features:
                meta_X = np.concatenate(meta_features, axis=1)
                self.meta_learner.fit(meta_X, y_encoded)

        # Train sequential model if needed
        if self.hybrid_mode == 'sequential':
            # Retrain classical models with quantum features
            quantum_features = []

            for name, model in self.quantum_models.items():
                try:
                    if hasattr(model, 'transform'):
                        features = model.transform(X_selected)
                    elif hasattr(model, 'decision_function'):
                        features = model.decision_function(X_selected)
                        if len(features.shape) == 1:
                            features = features.reshape(-1, 1)
                    else:
                        features = model.predict_proba(X_selected)

                    quantum_features.append(features)

                except Exception as e:
                    logger.error(f"Failed to extract features from {name}: {e}")

            if quantum_features:
                quantum_feature_matrix = np.concatenate(quantum_features, axis=1)

                # Retrain classical models with quantum features
                for name, model in self.classical_models.items():
                    try:
                        model.fit(quantum_feature_matrix, y_encoded)
                    except Exception as e:
                        logger.error(f"Failed to retrain {name} with quantum features: {e}")

        self.is_fitted = True

        # Compute hybrid performance
        predictions = self.predict(X)
        self.hybrid_score = accuracy_score(y, predictions)

        logger.info(f"Hybrid classifier training completed. Accuracy: {self.hybrid_score:.4f}")

        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Make predictions using the hybrid classifier.

        Args:
            X: Input data for prediction
            **kwargs: Additional prediction parameters

        Returns:
            Predicted labels

        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Normalize features
        if self.normalize_data:
            X = self.scaler.transform(X)

        # Apply feature selection
        X_selected = self._apply_feature_selection(X)

        # Make predictions based on hybrid mode
        if self.hybrid_mode == 'ensemble':
            predictions = self._ensemble_predict(X_selected)
        elif self.hybrid_mode == 'voting':
            predictions = self._voting_predict(X_selected)
        elif self.hybrid_mode == 'sequential':
            predictions = self._sequential_predict(X_selected)
        elif self.hybrid_mode == 'stacking':
            predictions = self._stacking_predict(X_selected)
        else:
            raise ValueError(f"Unknown hybrid mode: {self.hybrid_mode}")

        # Decode labels
        return self.label_encoder.inverse_transform(predictions)

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

        # Normalize features
        if self.normalize_data:
            X = self.scaler.transform(X)

        # Apply feature selection
        X_selected = self._apply_feature_selection(X)

        # Get base predictions
        quantum_preds, classical_preds = self._get_base_predictions(X_selected)

        if self.hybrid_mode == 'ensemble':
            # Weighted average of probabilities
            combined_pred = np.zeros((X.shape[0], self.n_classes_))

            if len(quantum_preds) > 0:
                quantum_avg = np.mean(quantum_preds, axis=0)
                combined_pred += self.quantum_weight * quantum_avg

            if len(classical_preds) > 0:
                classical_avg = np.mean(classical_preds, axis=0)
                combined_pred += (1 - self.quantum_weight) * classical_avg

            return combined_pred

        elif self.hybrid_mode == 'stacking' and self.meta_learner is not None:
            # Use meta-learner probabilities
            meta_features = []

            if len(quantum_preds) > 0:
                for pred_proba in quantum_preds:
                    meta_features.append(pred_proba)
            if len(classical_preds) > 0:
                for pred_proba in classical_preds:
                    meta_features.append(pred_proba)

            if meta_features:
                meta_X = np.concatenate(meta_features, axis=1)
                if hasattr(self.meta_learner, 'predict_proba'):
                    return self.meta_learner.predict_proba(meta_X)

        # Fallback: convert predictions to probabilities
        predictions = self.predict(X)
        pred_encoded = self.label_encoder.transform(predictions)
        prob_matrix = np.zeros((len(predictions), self.n_classes_))
        prob_matrix[np.arange(len(predictions)), pred_encoded] = 1.0

        return prob_matrix

    def get_model_performance(self) -> dict[str, Any]:
        """Get detailed performance metrics for all models."""
        performance = {
            'quantum_scores': self.quantum_scores.copy(),
            'classical_scores': self.classical_scores.copy(),
            'hybrid_score': self.hybrid_score,
            'hybrid_mode': self.hybrid_mode,
        }

        # Add quantum advantage metrics
        if self.quantum_scores and self.classical_scores:
            best_quantum = max(self.quantum_scores.values()) if self.quantum_scores else 0
            best_classical = max(self.classical_scores.values()) if self.classical_scores else 0

            performance.update({
                'best_quantum_score': best_quantum,
                'best_classical_score': best_classical,
                'quantum_advantage': best_quantum - best_classical,
                'hybrid_vs_best_quantum': self.hybrid_score - best_quantum if self.hybrid_score else 0,
                'hybrid_vs_best_classical': self.hybrid_score - best_classical if self.hybrid_score else 0,
            })

        return performance

    def get_feature_importance(self) -> np.ndarray | None:
        """Get feature importance from feature selection."""
        return self.feature_importance_

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict[str, Any]:
        """Perform cross-validation on the hybrid classifier."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before cross-validation")

        try:
            scores = cross_val_score(self, X, y, cv=cv, scoring='accuracy')

            return {
                'cv_scores': scores.tolist(),
                'cv_mean': np.mean(scores),
                'cv_std': np.std(scores),
                'cv_min': np.min(scores),
                'cv_max': np.max(scores),
            }

        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return {'error': str(e)}

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get hybrid classifier parameters."""
        params = super().get_params(deep)
        params.update({
            'hybrid_mode': self.hybrid_mode,
            'quantum_algorithms': self.quantum_algorithms,
            'classical_algorithms': self.classical_algorithms,
            'quantum_weight': self.quantum_weight,
            'feature_selection': self.feature_selection,
            'meta_learner': self.meta_learner_name,
            'normalize_data': self.normalize_data,
        })
        return params

    def set_params(self, **params) -> 'HybridClassifier':
        """Set hybrid classifier parameters."""
        if self.is_fitted and any(key in params for key in
                                 ['hybrid_mode', 'quantum_algorithms', 'classical_algorithms']):
            logger.warning("Changing core parameters requires refitting the model")
            self.is_fitted = False

        return super().set_params(**params)
