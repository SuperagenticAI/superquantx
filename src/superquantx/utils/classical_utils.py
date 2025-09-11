"""Classical machine learning utilities for quantum algorithms.

This module provides classical ML utilities that complement quantum algorithms,
including cross-validation, hyperparameter search, and model selection.
"""

import itertools
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


@dataclass
class CrossValidationResult:
    """Results from cross-validation."""

    scores: list[float]
    mean_score: float
    std_score: float
    fold_times: list[float]
    mean_time: float
    best_params: dict[str, Any] | None = None


def cross_validation(
    algorithm: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    scoring: str = 'accuracy',
    stratify: bool = True,
    random_state: int | None = 42,
    verbose: bool = False
) -> CrossValidationResult:
    """Perform k-fold cross-validation on quantum algorithm.

    Args:
        algorithm: Quantum algorithm instance
        X: Feature matrix
        y: Target vector
        cv_folds: Number of CV folds
        scoring: Scoring metric ('accuracy', 'mse', 'mae')
        stratify: Whether to use stratified CV for classification
        random_state: Random seed
        verbose: Whether to print progress

    Returns:
        CrossValidationResult with scores and timing info

    """
    len(X)

    # Choose cross-validation strategy
    if stratify and _is_classification_task(y):
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    scores = []
    fold_times = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        if verbose:
            print(f"Fold {fold_idx + 1}/{cv_folds}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        start_time = time.time()

        # Train algorithm
        algorithm.fit(X_train, y_train)

        # Make predictions
        y_pred = algorithm.predict(X_val)

        fold_time = time.time() - start_time
        fold_times.append(fold_time)

        # Calculate score
        if scoring == 'accuracy':
            score = accuracy_score(y_val, y_pred)
        elif scoring == 'mse':
            score = -mean_squared_error(y_val, y_pred)  # Negative for "higher is better"
        elif scoring == 'mae':
            score = -np.mean(np.abs(y_val - y_pred))
        else:
            raise ValueError(f"Unknown scoring metric: {scoring}")

        scores.append(score)

        if verbose:
            print(f"  Score: {score:.4f}, Time: {fold_time:.2f}s")

    return CrossValidationResult(
        scores=scores,
        mean_score=np.mean(scores),
        std_score=np.std(scores),
        fold_times=fold_times,
        mean_time=np.mean(fold_times)
    )


def hyperparameter_search(
    algorithm_class: type,
    param_grid: dict[str, list[Any]],
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 3,
    scoring: str = 'accuracy',
    n_jobs: int = 1,
    random_state: int | None = 42,
    verbose: bool = False
) -> dict[str, Any]:
    """Perform grid search for hyperparameter optimization.

    Args:
        algorithm_class: Quantum algorithm class
        param_grid: Dictionary of parameter names and values to try
        X: Feature matrix
        y: Target vector
        cv_folds: Number of CV folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs (not implemented)
        random_state: Random seed
        verbose: Whether to print progress

    Returns:
        Dictionary with best parameters and results

    """
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))

    best_score = float('-inf')
    best_params = None
    all_results = []

    if verbose:
        print(f"Testing {len(param_combinations)} parameter combinations...")

    for i, param_values in enumerate(param_combinations):
        # Create parameter dictionary
        params = dict(zip(param_names, param_values, strict=False))

        if verbose:
            print(f"Combination {i+1}/{len(param_combinations)}: {params}")

        try:
            # Create algorithm instance with these parameters
            algorithm = algorithm_class(**params)

            # Perform cross-validation
            cv_result = cross_validation(
                algorithm, X, y, cv_folds=cv_folds,
                scoring=scoring, random_state=random_state,
                verbose=False
            )

            mean_score = cv_result.mean_score

            # Track results
            result = {
                'params': params,
                'mean_score': mean_score,
                'std_score': cv_result.std_score,
                'mean_time': cv_result.mean_time
            }
            all_results.append(result)

            # Update best score
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

            if verbose:
                print(f"  Score: {mean_score:.4f} ± {cv_result.std_score:.4f}")

        except Exception as e:
            if verbose:
                print(f"  Failed: {str(e)}")

            result = {
                'params': params,
                'mean_score': None,
                'std_score': None,
                'mean_time': None,
                'error': str(e)
            }
            all_results.append(result)

    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': all_results,
        'n_combinations': len(param_combinations)
    }


def model_selection(
    algorithms: list[tuple[str, type, dict[str, Any]]],
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    scoring: str = 'accuracy',
    test_size: float = 0.2,
    random_state: int | None = 42,
    verbose: bool = False
) -> dict[str, Any]:
    """Compare multiple algorithms and select the best one.

    Args:
        algorithms: List of (name, class, params) tuples
        X: Feature matrix
        y: Target vector
        cv_folds: Number of CV folds
        scoring: Scoring metric
        test_size: Proportion for test set
        random_state: Random seed
        verbose: Whether to print progress

    Returns:
        Dictionary with model selection results

    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if _is_classification_task(y) else None
    )

    results = {}
    best_algorithm = None
    best_score = float('-inf')

    for name, algorithm_class, params in algorithms:
        if verbose:
            print(f"Evaluating {name}...")

        try:
            # Create algorithm instance
            algorithm = algorithm_class(**params)

            # Cross-validation on training set
            cv_result = cross_validation(
                algorithm, X_train, y_train, cv_folds=cv_folds,
                scoring=scoring, random_state=random_state,
                verbose=False
            )

            # Final evaluation on test set
            algorithm.fit(X_train, y_train)
            y_pred = algorithm.predict(X_test)

            if scoring == 'accuracy':
                test_score = accuracy_score(y_test, y_pred)
            elif scoring == 'mse':
                test_score = -mean_squared_error(y_test, y_pred)
            elif scoring == 'mae':
                test_score = -np.mean(np.abs(y_test - y_pred))
            else:
                test_score = 0  # Fallback

            results[name] = {
                'cv_mean': cv_result.mean_score,
                'cv_std': cv_result.std_score,
                'test_score': test_score,
                'mean_time': cv_result.mean_time,
                'params': params
            }

            # Track best algorithm
            if cv_result.mean_score > best_score:
                best_score = cv_result.mean_score
                best_algorithm = name

            if verbose:
                print(f"  CV: {cv_result.mean_score:.4f} ± {cv_result.std_score:.4f}")
                print(f"  Test: {test_score:.4f}")

        except Exception as e:
            if verbose:
                print(f"  Failed: {str(e)}")

            results[name] = {
                'cv_mean': None,
                'cv_std': None,
                'test_score': None,
                'mean_time': None,
                'params': params,
                'error': str(e)
            }

    return {
        'results': results,
        'best_algorithm': best_algorithm,
        'best_score': best_score
    }


def data_splitting(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    stratify: bool = True,
    random_state: int | None = 42
) -> tuple[np.ndarray, ...]:
    """Split data into train, validation, and test sets.

    Args:
        X: Feature matrix
        y: Target vector
        train_size: Proportion for training
        val_size: Proportion for validation
        test_size: Proportion for testing
        stratify: Whether to stratify splits for classification
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)

    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("Split sizes must sum to 1.0")

    stratify_target = y if (stratify and _is_classification_task(y)) else None

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=stratify_target
    )

    # Second split: separate train and validation
    relative_val_size = val_size / (train_size + val_size)
    stratify_temp = y_temp if (stratify and _is_classification_task(y)) else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size,
        random_state=random_state, stratify=stratify_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def learning_curve_analysis(
    algorithm: Any,
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: list[float] | None = None,
    cv_folds: int = 5,
    scoring: str = 'accuracy',
    random_state: int | None = 42,
    verbose: bool = False
) -> dict[str, Any]:
    """Analyze learning curve by varying training set size.

    Args:
        algorithm: Quantum algorithm instance
        X: Feature matrix
        y: Target vector
        train_sizes: Fractions of training data to use
        cv_folds: Number of CV folds
        scoring: Scoring metric
        random_state: Random seed
        verbose: Whether to print progress

    Returns:
        Learning curve analysis results

    """
    if train_sizes is None:
        train_sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    results = {
        'train_sizes': [],
        'train_scores': [],
        'val_scores': [],
        'training_times': []
    }

    for size in train_sizes:
        if verbose:
            print(f"Training size: {size:.1%}")

        # Subsample training data
        n_samples = int(len(X) * size)
        indices = np.random.RandomState(random_state).choice(
            len(X), n_samples, replace=False
        )
        X_subset = X[indices]
        y_subset = y[indices]

        # Cross-validation
        cv_result = cross_validation(
            algorithm, X_subset, y_subset, cv_folds=cv_folds,
            scoring=scoring, random_state=random_state,
            verbose=False
        )

        results['train_sizes'].append(n_samples)
        results['val_scores'].append(cv_result.mean_score)
        results['training_times'].append(cv_result.mean_time)

        # Training score (fit on full subset, score on same data)
        algorithm.fit(X_subset, y_subset)
        y_pred = algorithm.predict(X_subset)

        if scoring == 'accuracy':
            train_score = accuracy_score(y_subset, y_pred)
        elif scoring == 'mse':
            train_score = -mean_squared_error(y_subset, y_pred)
        elif scoring == 'mae':
            train_score = -np.mean(np.abs(y_subset - y_pred))
        else:
            train_score = 0

        results['train_scores'].append(train_score)

        if verbose:
            print(f"  Train: {train_score:.4f}, Val: {cv_result.mean_score:.4f}")

    return results


def feature_importance_analysis(
    algorithm: Any,
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'permutation',
    n_repeats: int = 10,
    scoring: str = 'accuracy',
    random_state: int | None = 42
) -> np.ndarray:
    """Analyze feature importance using permutation or other methods.

    Args:
        algorithm: Fitted quantum algorithm
        X: Feature matrix
        y: Target vector
        method: Method for importance ('permutation', 'ablation')
        n_repeats: Number of permutation repeats
        scoring: Scoring metric
        random_state: Random seed

    Returns:
        Feature importance scores

    """
    if method == 'permutation':
        return _permutation_importance(
            algorithm, X, y, n_repeats, scoring, random_state
        )
    elif method == 'ablation':
        return _ablation_importance(algorithm, X, y, scoring)
    else:
        raise ValueError(f"Unknown importance method: {method}")


def _permutation_importance(
    algorithm: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int,
    scoring: str,
    random_state: int | None
) -> np.ndarray:
    """Calculate permutation feature importance."""
    # Baseline score
    y_pred = algorithm.predict(X)
    if scoring == 'accuracy':
        baseline_score = accuracy_score(y, y_pred)
    elif scoring == 'mse':
        baseline_score = mean_squared_error(y, y_pred)
    else:
        baseline_score = 0

    n_features = X.shape[1]
    importance_scores = np.zeros(n_features)

    rng = np.random.RandomState(random_state)

    for feature_idx in range(n_features):
        feature_scores = []

        for _ in range(n_repeats):
            # Permute feature
            X_permuted = X.copy()
            X_permuted[:, feature_idx] = rng.permutation(X_permuted[:, feature_idx])

            # Score with permuted feature
            y_pred_perm = algorithm.predict(X_permuted)
            if scoring == 'accuracy':
                perm_score = accuracy_score(y, y_pred_perm)
                # For accuracy, importance = decrease in accuracy
                feature_importance = baseline_score - perm_score
            elif scoring == 'mse':
                perm_score = mean_squared_error(y, y_pred_perm)
                # For MSE, importance = increase in MSE
                feature_importance = perm_score - baseline_score
            else:
                feature_importance = 0

            feature_scores.append(feature_importance)

        importance_scores[feature_idx] = np.mean(feature_scores)

    return importance_scores


def _ablation_importance(
    algorithm: Any,
    X: np.ndarray,
    y: np.ndarray,
    scoring: str
) -> np.ndarray:
    """Calculate ablation feature importance."""
    n_features = X.shape[1]
    importance_scores = np.zeros(n_features)

    # Baseline score with all features
    y_pred = algorithm.predict(X)
    if scoring == 'accuracy':
        baseline_score = accuracy_score(y, y_pred)
    elif scoring == 'mse':
        baseline_score = mean_squared_error(y, y_pred)
    else:
        baseline_score = 0

    for feature_idx in range(n_features):
        # Remove feature (set to zero)
        X_ablated = X.copy()
        X_ablated[:, feature_idx] = 0

        # Score without feature
        y_pred_ablated = algorithm.predict(X_ablated)
        if scoring == 'accuracy':
            ablated_score = accuracy_score(y, y_pred_ablated)
            importance = baseline_score - ablated_score
        elif scoring == 'mse':
            ablated_score = mean_squared_error(y, y_pred_ablated)
            importance = ablated_score - baseline_score
        else:
            importance = 0

        importance_scores[feature_idx] = importance

    return importance_scores


def _is_classification_task(y: np.ndarray) -> bool:
    """Check if task is classification based on target values."""
    unique_vals = np.unique(y)

    # Classification if:
    # 1. Integer values
    # 2. Small number of unique values relative to sample size
    # 3. Values look like class labels (0, 1, 2, ...)

    is_integer = np.all(y == y.astype(int))
    n_unique = len(unique_vals)
    n_samples = len(y)

    if is_integer and n_unique <= min(20, n_samples * 0.1):
        return True

    return False
