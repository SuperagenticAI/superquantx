"""Main CLI entry point for SuperQuantX.

This module provides the main CLI application using Click framework,
with subcommands for various quantum machine learning operations.
"""

import sys
from pathlib import Path

import click

# Import version directly to avoid circular import
from superquantx.version import __version__

from .commands import (
    benchmark,
    configure,
    create_dataset,
    info,
    list_algorithms,
    list_backends,
    run_algorithm,
    visualize,
)


@click.group()
@click.version_option(version=__version__, prog_name='SuperQuantX')
@click.option(
    '--config', '-c',
    type=click.Path(exists=True),
    help='Path to configuration file'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
@click.pass_context
def cli(ctx: click.Context, config: str | None, verbose: bool):
    """SuperQuantX: Building the Foundation for Quantum Agentic AI

    Deploy quantum-enhanced autonomous agents and AI systems in minutes.
    From quantum circuits to intelligent agents across all quantum platforms.

    Examples:
        sqx create-agent trading               # Deploy quantum trading agent
        sqx run automl --data portfolio        # Quantum AutoML optimization
        sqx run qsvm --data iris               # Traditional quantum algorithm
        sqx benchmark quantum-vs-classical     # Performance comparison
        sqx benchmark --backend all # Benchmark all backends

    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Store global options
    ctx.obj['config'] = config
    ctx.obj['verbose'] = verbose

    # Load configuration if specified
    if config:
        try:
            sqx.configure(config_file=config)
            if verbose:
                click.echo(f"Loaded configuration from {config}")
        except Exception as e:
            click.echo(f"Error loading configuration: {e}", err=True)
            sys.exit(1)


# Add command groups
cli.add_command(info)
cli.add_command(list_algorithms)
cli.add_command(list_backends)
cli.add_command(run_algorithm)
cli.add_command(benchmark)
cli.add_command(configure)
cli.add_command(create_dataset)
cli.add_command(visualize)


@cli.command()
@click.pass_context
def version(ctx: click.Context):
    """Show detailed version information."""
    click.echo(f"SuperQuantX version: {sqx.__version__}")
    click.echo(f"Python version: {sys.version}")

    if ctx.obj.get('verbose'):
        click.echo("\nBackend versions:")
        backend_info = sqx.get_backend_info()
        for backend, version in backend_info.items():
            status = version if version else "Not installed"
            click.echo(f"  {backend}: {status}")


@cli.command()
def shell():
    """Start interactive SuperQuantX shell."""
    try:
        import matplotlib.pyplot as plt

        # Import common modules for convenience
        import numpy as np
        from IPython import embed

        banner = """
        SuperQuantX Interactive Shell
        =============================

        Available imports:
        - superquantx as sqx
        - numpy as np
        - matplotlib.pyplot as plt

        Try: sqx.algorithms.QuantumSVM()
        """

        embed(banner1=banner, exit_msg="Goodbye!")

    except ImportError:
        click.echo("IPython not available. Install with: pip install ipython")
        sys.exit(1)


@cli.command()
@click.option(
    '--output', '-o',
    type=click.Path(),
    default='superquantx_examples',
    help='Output directory for examples'
)
def examples(output: str):
    """Generate example scripts and notebooks."""
    output_path = Path(output)
    output_path.mkdir(exist_ok=True)

    # Basic example
    basic_example = '''#!/usr/bin/env python3
"""
Basic SuperQuantX Example: Quantum SVM Classification
"""

import numpy as np
import superquantx as sqx

# Load quantum-adapted Iris dataset
X_train, X_test, y_train, y_test, metadata = sqx.datasets.load_iris_quantum(
    n_features=4, encoding='amplitude'
)

print(f"Dataset: {metadata['dataset_name']}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {metadata['n_features']}")
print(f"Classes: {metadata['n_classes']}")

# Create quantum SVM with automatic backend selection
qsvm = sqx.QuantumSVM(
    backend='auto',
    feature_map='ZZFeatureMap',
    quantum_kernel=True
)

# Train the model
print("\\nTraining Quantum SVM...")
qsvm.fit(X_train, y_train)

# Make predictions
y_pred = qsvm.predict(X_test)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Visualize results
sqx.visualize_results({
    'y_true': y_test,
    'y_pred': y_pred,
    'algorithm': 'QuantumSVM'
}, plot_type='classification')
'''

    # VQE example
    vqe_example = '''#!/usr/bin/env python3
"""
SuperQuantX VQE Example: H2 Molecule Ground State
"""

import superquantx as sqx

# Load H2 molecule
molecule, metadata = sqx.datasets.load_h2_molecule(bond_length=0.735)

print(f"Molecule: {molecule.name}")
print(f"Bond length: {metadata['bond_length']} Å")
print(f"Expected ground state energy: {metadata['ground_state_energy']} Ha")

# Create VQE algorithm
vqe = sqx.VQE(
    backend='auto',
    ansatz='UCCSD',
    optimizer='Adam'
)

# Run VQE optimization
print("\\nRunning VQE optimization...")
result = vqe.compute_minimum_eigenvalue(molecule)

print(f"VQE ground state energy: {result['eigenvalue']:.6f} Ha")
print(f"Number of iterations: {result['n_iterations']}")
print(f"Optimization time: {result['optimization_time']:.2f} s")

# Plot optimization history
sqx.plot_optimization_history(result)
'''

    # Write examples
    examples_to_create = [
        ('basic_qsvm.py', basic_example),
        ('vqe_h2.py', vqe_example)
    ]

    for filename, content in examples_to_create:
        example_path = output_path / filename
        with open(example_path, 'w') as f:
            f.write(content)
        click.echo(f"Created: {example_path}")

    click.echo(f"\\nExamples created in: {output_path}")
    click.echo("Run with: python basic_qsvm.py")


def create_app():
    """Create and return the CLI application."""
    return cli


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
