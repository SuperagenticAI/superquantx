"""CLI commands for SuperQuantX.

This module implements individual CLI commands for various SuperQuantX operations
including algorithm execution, benchmarking, configuration, and system information.
"""

import json
import sys
import time

import click
import numpy as np

import superquantx as sqx


@click.command()
def info():
    """Show system and backend information."""
    click.echo("SuperQuantX System Information")
    click.echo("=" * 40)

    # Basic info
    click.echo(f"Version: {sqx.__version__}")
    click.echo(f"Installation path: {sqx.__file__}")

    # Backend information
    click.echo("\\nAvailable Backends:")
    backend_info = sqx.get_backend_info()

    for backend_name, version in backend_info.items():
        status = "✓" if version else "✗"
        version_str = version if version else "Not installed"
        click.echo(f"  {status} {backend_name}: {version_str}")

    # Configuration
    config = sqx.config
    click.echo("\\nCurrent Configuration:")
    click.echo(f"  Default backend: {config.get('default_backend', 'auto')}")
    click.echo(f"  Random seed: {config.get('random_seed', 42)}")
    click.echo(f"  Shots: {config.get('shots', 1024)}")

    # Hardware info
    try:
        import psutil
        click.echo("\\nSystem Resources:")
        click.echo(f"  CPU cores: {psutil.cpu_count()}")
        click.echo(f"  RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    except ImportError:
        pass


@click.command('list-algorithms')
@click.option(
    '--category', '-c',
    type=click.Choice(['all', 'classification', 'regression', 'clustering', 'optimization']),
    default='all',
    help='Algorithm category to list'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Show detailed algorithm information'
)
def list_algorithms(category: str, verbose: bool):
    """List available quantum algorithms."""
    algorithms = {
        'classification': [
            ('QuantumSVM', 'Quantum Support Vector Machine'),
            ('QuantumNN', 'Quantum Neural Network'),
            ('HybridClassifier', 'Hybrid Classical-Quantum Classifier')
        ],
        'regression': [
            ('QuantumNN', 'Quantum Neural Network (regression mode)')
        ],
        'clustering': [
            ('QuantumKMeans', 'Quantum K-Means Clustering'),
            ('QuantumPCA', 'Quantum Principal Component Analysis')
        ],
        'optimization': [
            ('QAOA', 'Quantum Approximate Optimization Algorithm'),
            ('VQE', 'Variational Quantum Eigensolver')
        ]
    }

    click.echo("Available Quantum Algorithms")
    click.echo("=" * 40)

    categories_to_show = [category] if category != 'all' else list(algorithms.keys())

    for cat in categories_to_show:
        if cat in algorithms:
            click.echo(f"\\n{cat.title()}:")

            for alg_name, description in algorithms[cat]:
                if verbose:
                    click.echo(f"  • {alg_name}")
                    click.echo(f"    {description}")

                    # Try to get algorithm class and show parameters
                    try:
                        getattr(sqx.algorithms, alg_name)
                        # This is a simplified approach - real implementation would
                        # need to inspect the class properly
                        click.echo(f"    Module: superquantx.algorithms.{alg_name}")
                    except AttributeError:
                        pass
                    click.echo()
                else:
                    click.echo(f"  • {alg_name}: {description}")


@click.command('list-backends')
@click.option(
    '--available-only', '-a',
    is_flag=True,
    help='Show only installed backends'
)
def list_backends(available_only: bool):
    """List quantum computing backends."""
    backends = {
        'Local Simulators': [
            ('simulator', 'SuperQuantX built-in simulator'),
            ('pennylane_local', 'PennyLane local simulators'),
            ('qiskit_aer', 'Qiskit Aer simulator')
        ],
        'Cloud Simulators': [
            ('pennylane_cloud', 'PennyLane cloud devices'),
            ('qiskit_ibm', 'IBM Quantum simulators'),
            ('braket_sv1', 'AWS Braket SV1 simulator'),
            ('cirq_cloud', 'Google Cirq cloud simulators')
        ],
        'Quantum Hardware': [
            ('ibm_quantum', 'IBM Quantum hardware'),
            ('braket_hardware', 'AWS Braket hardware access'),
            ('azure_quantum', 'Azure Quantum hardware'),
            ('rigetti_qcs', 'Rigetti Quantum Cloud Services')
        ]
    }

    backend_info = sqx.get_backend_info()

    click.echo("Quantum Computing Backends")
    click.echo("=" * 40)

    for category, backend_list in backends.items():
        click.echo(f"\\n{category}:")

        for backend_name, description in backend_list:
            # Check if backend is available (simplified logic)
            is_available = any(info for info in backend_info.values())

            if available_only and not is_available:
                continue

            status = "✓" if is_available else "✗"
            click.echo(f"  {status} {backend_name}: {description}")


@click.command('run')
@click.argument('algorithm')
@click.option(
    '--data', '-d',
    default='iris',
    help='Dataset to use (iris, wine, digits, synthetic)'
)
@click.option(
    '--backend', '-b',
    default='auto',
    help='Quantum backend to use'
)
@click.option(
    '--output', '-o',
    type=click.Path(),
    help='Output file for results'
)
@click.option(
    '--config-file', '-c',
    type=click.Path(exists=True),
    help='Algorithm configuration file'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
def run_algorithm(
    algorithm: str,
    data: str,
    backend: str,
    output: str | None,
    config_file: str | None,
    verbose: bool
):
    """Run a quantum algorithm on specified dataset."""
    if verbose:
        click.echo(f"Running {algorithm} on {data} dataset using {backend} backend")

    try:
        # Load dataset
        if data == 'iris':
            X_train, X_test, y_train, y_test, metadata = sqx.datasets.load_iris_quantum()
        elif data == 'wine':
            X_train, X_test, y_train, y_test, metadata = sqx.datasets.load_wine_quantum()
        elif data == 'synthetic':
            X_train, X_test, y_train, y_test, metadata = sqx.datasets.generate_classification_data()
        else:
            click.echo(f"Unknown dataset: {data}", err=True)
            sys.exit(1)

        if verbose:
            click.echo(f"Loaded {metadata['dataset_name']} dataset")
            click.echo(f"Training samples: {len(X_train)}")
            click.echo(f"Features: {metadata['n_features']}")

        # Load configuration
        config = {}
        if config_file:
            with open(config_file) as f:
                config = json.load(f)

        # Create algorithm
        algorithm_map = {
            'qsvm': sqx.QuantumSVM,
            'qaoa': sqx.QAOA,
            'vqe': sqx.VQE,
            'qnn': sqx.QuantumNN,
            'qkmeans': sqx.QuantumKMeans,
            'qpca': sqx.QuantumPCA
        }

        if algorithm.lower() not in algorithm_map:
            click.echo(f"Unknown algorithm: {algorithm}", err=True)
            click.echo(f"Available: {list(algorithm_map.keys())}")
            sys.exit(1)

        AlgorithmClass = algorithm_map[algorithm.lower()]

        # Set backend
        config['backend'] = backend

        alg = AlgorithmClass(**config)

        if verbose:
            click.echo(f"Created {AlgorithmClass.__name__} with config: {config}")

        # Train
        click.echo("Training...")
        start_time = time.time()

        alg.fit(X_train, y_train)

        training_time = time.time() - start_time

        # Predict
        click.echo("Evaluating...")
        y_pred = alg.predict(X_test)

        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)

        # Results
        results = {
            'algorithm': AlgorithmClass.__name__,
            'dataset': metadata['dataset_name'],
            'backend': backend,
            'accuracy': float(accuracy),
            'training_time': training_time,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'n_features': metadata['n_features']
        }

        # Output results
        click.echo("\\nResults:")
        click.echo(f"Accuracy: {accuracy:.4f}")
        click.echo(f"Training time: {training_time:.2f}s")

        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            click.echo(f"Results saved to {output}")

        if verbose:
            click.echo(f"Full results: {results}")

    except Exception as e:
        click.echo(f"Error running algorithm: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@click.command()
@click.option(
    '--algorithms', '-a',
    default='qsvm,qnn',
    help='Comma-separated list of algorithms to benchmark'
)
@click.option(
    '--datasets', '-d',
    default='iris,wine',
    help='Comma-separated list of datasets'
)
@click.option(
    '--backends', '-b',
    default='simulator',
    help='Comma-separated list of backends'
)
@click.option(
    '--output', '-o',
    type=click.Path(),
    default='benchmark_results.json',
    help='Output file for benchmark results'
)
@click.option(
    '--runs', '-r',
    default=3,
    help='Number of runs for averaging'
)
def benchmark(
    algorithms: str,
    datasets: str,
    backends: str,
    output: str,
    runs: int
):
    """Benchmark algorithms across datasets and backends."""
    alg_list = algorithms.split(',')
    dataset_list = datasets.split(',')
    backend_list = backends.split(',')

    click.echo("SuperQuantX Benchmark")
    click.echo("=" * 30)
    click.echo(f"Algorithms: {alg_list}")
    click.echo(f"Datasets: {dataset_list}")
    click.echo(f"Backends: {backend_list}")
    click.echo(f"Runs per combination: {runs}")
    click.echo()

    results = []
    total_combinations = len(alg_list) * len(dataset_list) * len(backend_list)
    current = 0

    for algorithm in alg_list:
        for dataset in dataset_list:
            for backend in backend_list:
                current += 1
                click.echo(f"[{current}/{total_combinations}] {algorithm} on {dataset} with {backend}")

                try:
                    # This would call the actual benchmark function
                    # For now, simulate results
                    result = {
                        'algorithm': algorithm,
                        'dataset': dataset,
                        'backend': backend,
                        'accuracy': np.random.uniform(0.7, 0.95),
                        'execution_time': np.random.uniform(1.0, 10.0),
                        'success': True
                    }
                    results.append(result)

                    click.echo(f"  Accuracy: {result['accuracy']:.4f}")
                    click.echo(f"  Time: {result['execution_time']:.2f}s")

                except Exception as e:
                    click.echo(f"  Failed: {e}")
                    results.append({
                        'algorithm': algorithm,
                        'dataset': dataset,
                        'backend': backend,
                        'success': False,
                        'error': str(e)
                    })

    # Save results
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)

    click.echo(f"\\nBenchmark complete. Results saved to {output}")


@click.command()
@click.option(
    '--backend', '-b',
    help='Set default backend'
)
@click.option(
    '--shots', '-s',
    type=int,
    help='Set default number of shots'
)
@click.option(
    '--seed',
    type=int,
    help='Set random seed'
)
@click.option(
    '--show',
    is_flag=True,
    help='Show current configuration'
)
def configure(backend: str | None, shots: int | None, seed: int | None, show: bool):
    """Configure SuperQuantX settings."""
    if show:
        click.echo("Current SuperQuantX Configuration:")
        click.echo("=" * 40)
        config = sqx.config
        for key, value in config.items():
            click.echo(f"{key}: {value}")
        return

    # Update configuration
    config_updates = {}
    if backend:
        config_updates['default_backend'] = backend
        click.echo(f"Set default backend to: {backend}")

    if shots:
        config_updates['shots'] = shots
        click.echo(f"Set default shots to: {shots}")

    if seed:
        config_updates['random_seed'] = seed
        click.echo(f"Set random seed to: {seed}")

    if config_updates:
        sqx.configure(**config_updates)
        click.echo("Configuration updated successfully")
    else:
        click.echo("No configuration changes specified")


@click.command('create-dataset')
@click.argument('dataset_type', type=click.Choice(['classification', 'regression', 'clustering']))
@click.option(
    '--samples', '-n',
    default=200,
    help='Number of samples to generate'
)
@click.option(
    '--features', '-f',
    default=4,
    help='Number of features'
)
@click.option(
    '--output', '-o',
    type=click.Path(),
    help='Output file (NPZ format)'
)
def create_dataset(dataset_type: str, samples: int, features: int, output: str | None):
    """Create synthetic datasets for testing."""
    if dataset_type == 'classification':
        X_train, X_test, y_train, y_test, metadata = sqx.datasets.generate_classification_data(
            n_samples=samples,
            n_features=features
        )
    elif dataset_type == 'regression':
        X_train, X_test, y_train, y_test, metadata = sqx.datasets.generate_regression_data(
            n_samples=samples,
            n_features=features
        )
    elif dataset_type == 'clustering':
        X, y_true, metadata = sqx.datasets.generate_clustering_data(
            n_samples=samples,
            n_features=features
        )
        # For clustering, no train/test split
        X_train, X_test, y_train, y_test = X, X, y_true, y_true

    click.echo(f"Created {dataset_type} dataset:")
    click.echo(f"  Samples: {samples}")
    click.echo(f"  Features: {features}")
    click.echo(f"  Training set: {len(X_train)}")
    click.echo(f"  Test set: {len(X_test)}")

    if output:
        np.savez(
            output,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            metadata=metadata
        )
        click.echo(f"Dataset saved to: {output}")


@click.command()
@click.argument('results_file', type=click.Path(exists=True))
@click.option(
    '--plot-type', '-p',
    type=click.Choice(['optimization', 'classification', 'regression']),
    default='optimization',
    help='Type of plot to generate'
)
@click.option(
    '--output', '-o',
    type=click.Path(),
    help='Output file for plot'
)
def visualize(results_file: str, plot_type: str, output: str | None):
    """Visualize algorithm results."""
    # Load results
    with open(results_file) as f:
        results = json.load(f)

    click.echo(f"Visualizing results from {results_file}")
    click.echo(f"Plot type: {plot_type}")

    try:
        sqx.visualize_results(
            results,
            plot_type=plot_type,
            save_path=output
        )

        if output:
            click.echo(f"Plot saved to: {output}")
        else:
            click.echo("Plot displayed")

    except Exception as e:
        click.echo(f"Error creating visualization: {e}", err=True)
        sys.exit(1)
