"""Main CLI entry point for SuperQuantX.

This module provides the main CLI application using Click framework,
with subcommands for various quantum machine learning operations.
"""

import sys
from pathlib import Path

import click

import superquantx as sqx
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
    """SuperQuantX: experimental quantum machine learning toolkit.

    One API over PennyLane, Qiskit, Cirq, Braket, tket and a built-in
    simulator, with classical baselines for every quantum result.

    \b
    Examples:
        superquantx -v version                  # versions of installed SDKs
        superquantx list-backends               # Bell-state check per backend
        superquantx benchmark -d moons          # quantum kernel vs tuned classical
        superquantx run qsvm --data iris        # train and score one algorithm
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
        import matplotlib.pyplot as plt  # noqa: F401 - exposed in the shell

        # Import common modules for convenience
        import numpy as np  # noqa: F401 - exposed in the shell
        from IPython import embed

        banner = """
        SuperQuantX Interactive Shell
        =============================

        Available imports:
        - superquantx as sqx
        - numpy as np
        - matplotlib.pyplot as plt

        Try: sqx.QuantumSVM(backend='simulator')
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
"""Quantum SVM on two Iris classes, checked against tuned classical baselines."""

import numpy as np
from sklearn.datasets import load_iris

import superquantx as sqx

X, y = load_iris(return_X_y=True)
X, y = X[y < 2], y[y < 2]

result = sqx.compare_to_classical(X, y, sqx.QuantumSVM(backend="simulator"), cv=5)
print(result.table())
print("Verdict:", result.verdict)
'''

    # VQE example
    vqe_example = '''#!/usr/bin/env python3
"""VQE for the H2 ground state (STO-3G, 0.735 Angstrom), checked against exact."""

from superquantx.algorithms import create_vqe_for_molecule

vqe = create_vqe_for_molecule("H2", backend="simulator", seed=0)
energy = vqe.find_ground_state()
exact = vqe.exact_ground_energy()
print(f"VQE energy:   {energy:.6f} Ha")
print(f"Exact energy: {exact:.6f} Ha")
print(f"Error:        {abs(energy - exact):.2e} Ha")
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

    click.echo(f"\nExamples created in: {output_path}")
    click.echo("Run with: python basic_qsvm.py")


def create_app():
    """Create and return the CLI application."""
    return cli


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
