"""Quantum computing backends for SuperQuantX.

This module provides a unified interface to various quantum computing
platforms and simulators, including PennyLane, Qiskit, Cirq, and others.
"""

import logging
from typing import Any, Dict, Optional, Union

from .base_backend import BaseBackend
from .simulator_backend import SimulatorBackend


# Import backends with graceful failure handling
try:
    from .pennylane_backend import PennyLaneBackend
except ImportError:
    PennyLaneBackend = None

try:
    from .qiskit_backend import QiskitBackend
except ImportError:
    QiskitBackend = None

try:
    from .cirq_backend import CirqBackend
except ImportError:
    CirqBackend = None

try:
    from .braket_backend import BraketBackend
except ImportError:
    BraketBackend = None

try:
    from .tket_backend import TKETBackend
except ImportError:
    TKETBackend = None

try:
    from .ocean_backend import OceanBackend
except ImportError:
    OceanBackend = None

logger = logging.getLogger(__name__)

# Registry of available backends
BACKEND_REGISTRY = {
    'simulator': SimulatorBackend,
    'auto': None,  # Auto-selection
}

# Add backends if they were imported successfully
if PennyLaneBackend is not None:
    BACKEND_REGISTRY['pennylane'] = PennyLaneBackend
if QiskitBackend is not None:
    BACKEND_REGISTRY['qiskit'] = QiskitBackend
if CirqBackend is not None:
    BACKEND_REGISTRY['cirq'] = CirqBackend
if BraketBackend is not None:
    BACKEND_REGISTRY['braket'] = BraketBackend
if TKETBackend is not None:
    BACKEND_REGISTRY['tket'] = TKETBackend
    BACKEND_REGISTRY['quantinuum'] = TKETBackend  # Alias for Quantinuum
if OceanBackend is not None:
    BACKEND_REGISTRY['ocean'] = OceanBackend
    BACKEND_REGISTRY['dwave'] = OceanBackend  # Alias for D-Wave

# Aliases for common backends
BACKEND_ALIASES = {
    'pl': 'pennylane',
    'penny': 'pennylane',
    'qk': 'qiskit',
    'ibm': 'qiskit',
    'google': 'cirq',
    'aws': 'braket',
    'amazon': 'braket',
    'tk': 'tket',
    'pytket': 'tket',
    'h1': 'quantinuum',  # Quantinuum H-Series
    'h2': 'quantinuum',
    'annealing': 'ocean',
    'quantum_annealing': 'ocean',
    'sim': 'simulator',
    'local': 'simulator',
}

def get_backend(backend: str | BaseBackend, **kwargs) -> BaseBackend:
    """Get a quantum backend instance.

    Args:
        backend: Backend name or instance
        **kwargs: Backend configuration parameters

    Returns:
        Backend instance

    Raises:
        ValueError: If backend is not supported
        ImportError: If backend dependencies are missing

    Example:
        >>> backend = get_backend('pennylane', device='default.qubit')
        >>> backend = get_backend('qiskit', provider='IBMQ')

    """
    if isinstance(backend, BaseBackend):
        return backend

    if not isinstance(backend, str):
        raise ValueError(f"Backend must be string or BaseBackend instance, got {type(backend)}")

    # Resolve aliases
    backend_name = BACKEND_ALIASES.get(backend.lower(), backend.lower())

    # Auto-select backend if requested
    if backend_name == 'auto':
        backend_name = _auto_select_backend()

    # Get backend class
    if backend_name not in BACKEND_REGISTRY:
        available = list(BACKEND_REGISTRY.keys())
        raise ValueError(f"Backend '{backend_name}' not supported. Available: {available}")

    backend_class = BACKEND_REGISTRY[backend_name]

    try:
        logger.info(f"Initializing {backend_name} backend")
        return backend_class(**kwargs)
    except ImportError as e:
        logger.error(f"Failed to import {backend_name} backend: {e}")
        raise ImportError(f"Backend '{backend_name}' requires additional dependencies: {e}")
    except Exception as e:
        logger.error(f"Failed to initialize {backend_name} backend: {e}")
        raise

def _auto_select_backend() -> str:
    """Automatically select the best available backend.

    Returns:
        Backend name

    """
    # Try backends in order of preference
    preference_order = ['pennylane', 'qiskit', 'cirq', 'braket', 'tket', 'simulator']

    for backend_name in preference_order:
        try:
            backend_class = BACKEND_REGISTRY[backend_name]
            # Try to instantiate with minimal config
            backend_class()
            logger.info(f"Auto-selected backend: {backend_name}")
            return backend_name
        except (ImportError, Exception) as e:
            logger.debug(f"Backend {backend_name} not available: {e}")
            continue

    # Fallback to simulator
    logger.warning("No preferred backends available, falling back to simulator")
    return 'simulator'

def list_available_backends() -> dict[str, dict[str, Any]]:
    """List all available backends and their status.

    Returns:
        Dictionary with backend information

    """
    backend_info = {}

    for name, backend_class in BACKEND_REGISTRY.items():
        if name == 'auto' or backend_class is None:
            continue

        try:
            # Try to instantiate to check availability
            test_instance = backend_class()
            backend_info[name] = {
                'available': True,
                'class': backend_class.__name__,
                'description': getattr(backend_class, '__doc__', '').split('\n')[0] if backend_class.__doc__ else '',
                'capabilities': getattr(test_instance, 'capabilities', {}),
            }
        except ImportError:
            backend_info[name] = {
                'available': False,
                'class': backend_class.__name__,
                'reason': 'Missing dependencies',
            }
        except Exception as e:
            backend_info[name] = {
                'available': False,
                'class': backend_class.__name__,
                'reason': str(e),
            }

    return backend_info

def check_backend_compatibility(backend_name: str) -> dict[str, Any]:
    """Check compatibility and requirements for a specific backend.

    Args:
        backend_name: Name of the backend to check

    Returns:
        Compatibility information

    """
    backend_name = BACKEND_ALIASES.get(backend_name.lower(), backend_name.lower())

    if backend_name not in BACKEND_REGISTRY:
        return {'compatible': False, 'reason': 'Backend not supported'}

    backend_class = BACKEND_REGISTRY[backend_name]

    try:
        # Try basic instantiation
        test_backend = backend_class()

        return {
            'compatible': True,
            'backend_class': backend_class.__name__,
            'requirements_met': True,
            'capabilities': getattr(test_backend, 'capabilities', {}),
            'version_info': getattr(test_backend, 'get_version_info', lambda: {})(),
        }

    except ImportError as e:
        return {
            'compatible': False,
            'reason': 'Missing dependencies',
            'missing_packages': str(e),
            'requirements_met': False,
        }
    except Exception as e:
        return {
            'compatible': False,
            'reason': str(e),
            'requirements_met': False,
        }

# Make backend classes available at module level
__all__ = [
    'BaseBackend',
    'SimulatorBackend',
    'get_backend',
    'list_available_backends',
    'check_backend_compatibility',
    'BACKEND_REGISTRY',
    'BACKEND_ALIASES',
]

# Add available backends to __all__
if PennyLaneBackend is not None:
    __all__.append('PennyLaneBackend')
if QiskitBackend is not None:
    __all__.append('QiskitBackend')
if CirqBackend is not None:
    __all__.append('CirqBackend')
if BraketBackend is not None:
    __all__.append('BraketBackend')
if TKETBackend is not None:
    __all__.append('TKETBackend')
if OceanBackend is not None:
    __all__.append('OceanBackend')
