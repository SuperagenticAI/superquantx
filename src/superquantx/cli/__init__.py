"""SuperQuantX Command Line Interface.

This module provides a comprehensive CLI for SuperQuantX, enabling users
to run quantum machine learning algorithms, manage configurations,
and perform various quantum computing tasks from the command line.
"""

from .commands import (
    benchmark,
    configure,
    info,
    list_algorithms,
    list_backends,
    run_algorithm,
)
from .main import create_app, main


__all__ = [
    'main',
    'create_app',
    'run_algorithm',
    'list_algorithms',
    'list_backends',
    'benchmark',
    'configure',
    'info'
]
