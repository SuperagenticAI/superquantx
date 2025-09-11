"""Benchmarking utilities for quantum machine learning.

This module provides functions to benchmark quantum algorithms and backends,
compare performance, and generate comprehensive performance reports.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    algorithm_name: str
    backend_name: str
    dataset_name: str
    execution_time: float
    memory_usage: float | None
    accuracy: float | None
    loss: float | None
    n_parameters: int | None
    n_qubits: int | None
    n_iterations: int | None
    success: bool
    error_message: str | None
    metadata: dict[str, Any]


def benchmark_algorithm(
    algorithm: Any,
    datasets: list[tuple[str, Any]],
    metrics: list[str] | None = None,
    n_runs: int = 1,
    verbose: bool = True
) -> list[BenchmarkResult]:
    """Benchmark quantum algorithm performance across multiple datasets.

    Args:
        algorithm: Quantum algorithm instance
        datasets: List of (name, dataset) tuples
        metrics: List of metrics to compute
        n_runs: Number of runs for averaging
        verbose: Whether to print progress

    Returns:
        List of benchmark results

    """
    if metrics is None:
        metrics = ['accuracy', 'execution_time', 'memory_usage']

    results = []

    for dataset_name, dataset in datasets:
        if verbose:
            print(f"Benchmarking {algorithm.__class__.__name__} on {dataset_name}...")

        dataset_results = []

        for run in range(n_runs):
            if verbose and n_runs > 1:
                print(f"  Run {run + 1}/{n_runs}")

            result = _run_single_benchmark(
                algorithm, dataset_name, dataset, metrics
            )
            dataset_results.append(result)

        # Average results if multiple runs
        if n_runs > 1:
            averaged_result = _average_benchmark_results(dataset_results)
            results.append(averaged_result)
        else:
            results.extend(dataset_results)

    return results


def benchmark_backend(
    backends: list[Any],
    test_circuit: Callable,
    n_qubits_range: list[int] = None,
    n_shots: int = 1024,
    verbose: bool = True
) -> dict[str, list[BenchmarkResult]]:
    """Benchmark different quantum backends.

    Args:
        backends: List of backend instances
        test_circuit: Function that creates test circuit
        n_qubits_range: Range of qubit numbers to test
        n_shots: Number of shots for each measurement
        verbose: Whether to print progress

    Returns:
        Dictionary mapping backend names to benchmark results

    """
    if n_qubits_range is None:
        n_qubits_range = [2, 4, 6, 8]
    results = {}

    for backend in backends:
        backend_name = getattr(backend, 'name', backend.__class__.__name__)
        if verbose:
            print(f"Benchmarking backend: {backend_name}")

        backend_results = []

        for n_qubits in n_qubits_range:
            if verbose:
                print(f"  Testing {n_qubits} qubits...")

            try:
                start_time = time.time()
                start_memory = _get_memory_usage()

                # Create and run circuit
                circuit = test_circuit(n_qubits)
                result = backend.run(circuit, shots=n_shots)

                execution_time = time.time() - start_time
                memory_usage = _get_memory_usage() - start_memory if start_memory else None

                benchmark_result = BenchmarkResult(
                    algorithm_name="test_circuit",
                    backend_name=backend_name,
                    dataset_name=f"{n_qubits}_qubits",
                    execution_time=execution_time,
                    memory_usage=memory_usage,
                    accuracy=None,
                    loss=None,
                    n_parameters=None,
                    n_qubits=n_qubits,
                    n_iterations=None,
                    success=True,
                    error_message=None,
                    metadata={
                        'n_shots': n_shots,
                        'result_counts': getattr(result, 'counts', None)
                    }
                )

            except Exception as e:
                benchmark_result = BenchmarkResult(
                    algorithm_name="test_circuit",
                    backend_name=backend_name,
                    dataset_name=f"{n_qubits}_qubits",
                    execution_time=0,
                    memory_usage=None,
                    accuracy=None,
                    loss=None,
                    n_parameters=None,
                    n_qubits=n_qubits,
                    n_iterations=None,
                    success=False,
                    error_message=str(e),
                    metadata={'n_shots': n_shots}
                )

            backend_results.append(benchmark_result)

        results[backend_name] = backend_results

    return results


def performance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = 'classification'
) -> dict[str, float]:
    """Compute performance metrics for predictions.

    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values
        task_type: Type of task ('classification' or 'regression')

    Returns:
        Dictionary of computed metrics

    """
    metrics = {}

    if task_type == 'classification':
        # Accuracy
        metrics['accuracy'] = np.mean(y_true == y_pred)

        # Precision, Recall, F1 for binary classification
        if len(np.unique(y_true)) == 2:
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1_score'] = f1

        # Confusion matrix elements
        unique_labels = np.unique(y_true)
        confusion_matrix = np.zeros((len(unique_labels), len(unique_labels)))

        for i, true_label in enumerate(unique_labels):
            for j, pred_label in enumerate(unique_labels):
                confusion_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))

        metrics['confusion_matrix'] = confusion_matrix.tolist()

    elif task_type == 'regression':
        # Mean Squared Error
        mse = np.mean((y_true - y_pred) ** 2)
        metrics['mse'] = mse
        metrics['rmse'] = np.sqrt(mse)

        # Mean Absolute Error
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))

        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        metrics['r_squared'] = r_squared

        # Explained variance
        metrics['explained_variance'] = 1 - np.var(y_true - y_pred) / np.var(y_true)

    return metrics


def compare_algorithms(
    algorithms: list[Any],
    dataset: Any,
    metrics: list[str] = None,
    n_runs: int = 3,
    verbose: bool = True
) -> dict[str, Any]:
    """Compare multiple algorithms on the same dataset.

    Args:
        algorithms: List of algorithm instances
        dataset: Dataset to use for comparison
        metrics: Metrics to compare
        n_runs: Number of runs for averaging
        verbose: Whether to print progress

    Returns:
        Comparison results dictionary

    """
    if metrics is None:
        metrics = ['accuracy', 'execution_time']
    comparison_results = {
        'algorithms': [],
        'metrics': metrics,
        'n_runs': n_runs,
        'results': {}
    }

    for algorithm in algorithms:
        algorithm_name = algorithm.__class__.__name__
        comparison_results['algorithms'].append(algorithm_name)

        if verbose:
            print(f"Running {algorithm_name}...")

        # Run benchmark
        benchmark_results = benchmark_algorithm(
            algorithm,
            [('comparison_dataset', dataset)],
            metrics=metrics,
            n_runs=n_runs,
            verbose=False
        )

        # Extract averaged metrics
        result = benchmark_results[0]
        comparison_results['results'][algorithm_name] = {
            'execution_time': result.execution_time,
            'accuracy': result.accuracy,
            'memory_usage': result.memory_usage,
            'success': result.success,
            'error_message': result.error_message
        }

    # Find best performing algorithm for each metric
    comparison_results['best_algorithm'] = {}
    for metric in metrics:
        if metric == 'execution_time' or metric == 'memory_usage':
            # Lower is better
            best_value = float('inf')
            best_algorithm = None
            for alg_name, results in comparison_results['results'].items():
                if results.get(metric) and results[metric] < best_value:
                    best_value = results[metric]
                    best_algorithm = alg_name
        else:
            # Higher is better
            best_value = float('-inf')
            best_algorithm = None
            for alg_name, results in comparison_results['results'].items():
                if results.get(metric) and results[metric] > best_value:
                    best_value = results[metric]
                    best_algorithm = alg_name

        comparison_results['best_algorithm'][metric] = best_algorithm

    return comparison_results


def generate_benchmark_report(
    results: list[BenchmarkResult],
    output_path: str | None = None
) -> dict[str, Any]:
    """Generate comprehensive benchmark report.

    Args:
        results: List of benchmark results
        output_path: Optional path to save report

    Returns:
        Report dictionary

    """
    report = {
        'summary': {
            'total_benchmarks': len(results),
            'successful_runs': sum(1 for r in results if r.success),
            'failed_runs': sum(1 for r in results if not r.success),
            'algorithms_tested': list(set(r.algorithm_name for r in results)),
            'backends_tested': list(set(r.backend_name for r in results)),
            'datasets_tested': list(set(r.dataset_name for r in results))
        },
        'performance_analysis': {},
        'detailed_results': []
    }

    # Performance analysis
    successful_results = [r for r in results if r.success]

    if successful_results:
        execution_times = [r.execution_time for r in successful_results]
        accuracies = [r.accuracy for r in successful_results if r.accuracy is not None]
        memory_usage = [r.memory_usage for r in successful_results if r.memory_usage is not None]

        report['performance_analysis'] = {
            'execution_time': {
                'mean': np.mean(execution_times),
                'std': np.std(execution_times),
                'min': np.min(execution_times),
                'max': np.max(execution_times)
            },
            'accuracy': {
                'mean': np.mean(accuracies) if accuracies else None,
                'std': np.std(accuracies) if accuracies else None,
                'min': np.min(accuracies) if accuracies else None,
                'max': np.max(accuracies) if accuracies else None
            },
            'memory_usage': {
                'mean': np.mean(memory_usage) if memory_usage else None,
                'std': np.std(memory_usage) if memory_usage else None,
                'min': np.min(memory_usage) if memory_usage else None,
                'max': np.max(memory_usage) if memory_usage else None
            }
        }

    # Detailed results
    for result in results:
        result_dict = {
            'algorithm': result.algorithm_name,
            'backend': result.backend_name,
            'dataset': result.dataset_name,
            'execution_time': result.execution_time,
            'memory_usage': result.memory_usage,
            'accuracy': result.accuracy,
            'loss': result.loss,
            'n_parameters': result.n_parameters,
            'n_qubits': result.n_qubits,
            'n_iterations': result.n_iterations,
            'success': result.success,
            'error_message': result.error_message,
            'metadata': result.metadata
        }
        report['detailed_results'].append(result_dict)

    # Save report if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

    return report


def _run_single_benchmark(
    algorithm: Any,
    dataset_name: str,
    dataset: Any,
    metrics: list[str]
) -> BenchmarkResult:
    """Run single benchmark iteration."""
    try:
        start_time = time.time()
        start_memory = _get_memory_usage()

        # Fit algorithm
        if hasattr(algorithm, 'fit'):
            algorithm.fit(dataset['X_train'], dataset['y_train'])

        # Make predictions
        if hasattr(algorithm, 'predict'):
            y_pred = algorithm.predict(dataset.get('X_test', dataset['X_train']))
            y_true = dataset.get('y_test', dataset['y_train'])
        else:
            y_pred = None
            y_true = None

        execution_time = time.time() - start_time
        memory_usage = _get_memory_usage() - start_memory if start_memory else None

        # Compute metrics
        accuracy = None
        loss = None

        if y_pred is not None and y_true is not None:
            if 'accuracy' in metrics:
                accuracy = np.mean(y_true == y_pred)

            if 'loss' in metrics and hasattr(algorithm, 'loss'):
                loss = algorithm.loss(y_true, y_pred)

        # Extract algorithm info
        n_parameters = getattr(algorithm, 'n_parameters', None)
        n_qubits = getattr(algorithm, 'n_qubits', None)
        n_iterations = getattr(algorithm, 'n_iterations', None)

        result = BenchmarkResult(
            algorithm_name=algorithm.__class__.__name__,
            backend_name=getattr(algorithm, 'backend', {}).get('name', 'unknown'),
            dataset_name=dataset_name,
            execution_time=execution_time,
            memory_usage=memory_usage,
            accuracy=accuracy,
            loss=loss,
            n_parameters=n_parameters,
            n_qubits=n_qubits,
            n_iterations=n_iterations,
            success=True,
            error_message=None,
            metadata={}
        )

    except Exception as e:
        result = BenchmarkResult(
            algorithm_name=algorithm.__class__.__name__,
            backend_name=getattr(algorithm, 'backend', {}).get('name', 'unknown'),
            dataset_name=dataset_name,
            execution_time=0,
            memory_usage=None,
            accuracy=None,
            loss=None,
            n_parameters=None,
            n_qubits=None,
            n_iterations=None,
            success=False,
            error_message=str(e),
            metadata={}
        )

    return result


def _average_benchmark_results(results: list[BenchmarkResult]) -> BenchmarkResult:
    """Average multiple benchmark results."""
    # Take the first result as template
    template = results[0]

    # Average numerical values
    execution_times = [r.execution_time for r in results if r.success]
    accuracies = [r.accuracy for r in results if r.success and r.accuracy is not None]
    losses = [r.loss for r in results if r.success and r.loss is not None]
    memory_usages = [r.memory_usage for r in results if r.success and r.memory_usage is not None]

    return BenchmarkResult(
        algorithm_name=template.algorithm_name,
        backend_name=template.backend_name,
        dataset_name=template.dataset_name,
        execution_time=np.mean(execution_times) if execution_times else 0,
        memory_usage=np.mean(memory_usages) if memory_usages else None,
        accuracy=np.mean(accuracies) if accuracies else None,
        loss=np.mean(losses) if losses else None,
        n_parameters=template.n_parameters,
        n_qubits=template.n_qubits,
        n_iterations=template.n_iterations,
        success=all(r.success for r in results),
        error_message=None if all(r.success for r in results) else "Some runs failed",
        metadata={'n_runs': len(results)}
    )


def _get_memory_usage() -> float | None:
    """Get current memory usage in MB."""
    if not HAS_PSUTIL:
        return None

    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except (ImportError, AttributeError, Exception):
        return None
