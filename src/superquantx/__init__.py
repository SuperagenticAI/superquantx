"""SuperQuantX: Experimental Quantum AI Research Platform.

⚠️ RESEARCH SOFTWARE WARNING: This is experimental research software developed by
SuperXLab (Superagentic AI Research Division). NOT intended for production use.
For research and educational purposes only.

Part of SuperXLab's comprehensive quantum research program - the practical implementation
platform for validating theoretical research in:
    🔬 Quantum-Inspired Agentic Systems: Superposition, interference, entanglement in agents
    🔬 Quantum Neural Networks (QNNs): Hardware-validated quantum neural architectures
    🔬 QuantumML for AI Training: Quantum-accelerated machine learning techniques
    🔬 Quantinuum Integration: Real hardware validation on H-Series quantum computers

Research Examples:
    Experimental quantum agent research:
        >>> import superquantx as sqx  # EXPERIMENTAL
        >>> agent = sqx.QuantumTradingAgent(strategy="quantum_portfolio", backend="simulator")
        >>> results = agent.solve(research_data)  # Research use only
        >>> print(f"Research findings: {results.metadata}")

    Quantum neural network experiments:
        >>> qnn = sqx.QuantumNN(architecture='hybrid', backend='pennylane')  # EXPERIMENTAL
        >>> qnn.fit(X_research, y_research)  # Research data only
        >>> analysis = qnn.analyze_expressivity()  # Research analysis

    Quantum algorithm benchmarking:
        >>> qsvm = sqx.QuantumSVM(backend='simulator')  # EXPERIMENTAL
        >>> benchmark = sqx.benchmark_algorithm(qsvm, classical_baseline)
"""

import logging
from typing import Any

# Core imports - make available at top level
from . import algorithms, backends, cli, utils

# Core circuit and client classes
from .circuits import QuantumCircuit
from .client import SuperQuantXClient
from .gates import Hamiltonian, PauliString


# Import datasets lazily to avoid circular imports
try:
    from . import datasets
except ImportError:
    # If datasets import fails, create a placeholder
    import sys
    import types
    datasets = types.ModuleType('datasets')
    sys.modules[f'{__name__}.datasets'] = datasets
from .config import (
    config, configure, create_default_config, configure_interactive, 
    load_config, configure_logging, configure_simulation, config_context,
    get_default_backend, get_config, get_config_search_paths, 
    get_active_config_path, validate_config, validate_config_schema,
    configure_performance, configure_experiments, experiment,
    create_profile, activate_profile, profile, ConfigValidationError
)
from .version import __version__


# Quantum AI Agents (High-level interface)
try:
    from .algorithms.quantum_agents import (
        QuantumClassificationAgent,
        QuantumOptimizationAgent,
        QuantumPortfolioAgent,
        QuantumResearchAgent,
        QuantumTradingAgent,
    )
except (ImportError, AttributeError):
    # Fallback classes for when agents module isn't fully implemented
    QuantumTradingAgent = None
    QuantumResearchAgent = None
    QuantumOptimizationAgent = None
    QuantumClassificationAgent = None
    QuantumPortfolioAgent = None

# Quantum AutoML
try:
    from .ml import QuantumAutoML
except (ImportError, AttributeError, NameError):
    # QuantumAutoML temporarily disabled due to dependency issues
    QuantumAutoML = None

# Core quantum algorithms
from .algorithms import (
    QAOA,
    VQE,
    HybridClassifier,
    QuantumKMeans,
    QuantumNN,
    QuantumPCA,
    QuantumSVM,
)
from .backends import (
    BaseBackend,
    SimulatorBackend,
    check_backend_compatibility,
    get_backend,
    list_available_backends,
)


# Import available backends gracefully
try:
    from .backends import QiskitBackend
except (ImportError, AttributeError):
    QiskitBackend = None

try:
    from .backends import PennyLaneBackend
except (ImportError, AttributeError):
    PennyLaneBackend = None

try:
    from .backends import CirqBackend
except (ImportError, AttributeError):
    CirqBackend = None

try:
    from .backends import BraketBackend
except (ImportError, AttributeError):
    BraketBackend = None

try:
    from .backends import TKETBackend
except (ImportError, AttributeError):
    TKETBackend = None

try:
    from .backends import OceanBackend
except (ImportError, AttributeError):
    OceanBackend = None

# Create dummy classes for non-existent backends
class AutoBackend:
    """Auto backend selection (wrapper around get_backend)."""

    def __new__(cls, **kwargs):
        return get_backend('auto', **kwargs)

# Aliases for compatibility
QuantinuumBackend = TKETBackend  # Quantinuum uses TKET
DWaveBackend = OceanBackend      # D-Wave uses Ocean SDK
AWSBackend = BraketBackend       # AWS uses Braket

# Not yet implemented
AzureBackend = None
CuQuantumBackend = None
ForestBackend = None
TensorFlowQuantumBackend = None
ClassicalBackend = SimulatorBackend

try:
    from .datasets import (
        generate_portfolio_data,
        load_iris_quantum,
        load_molecule,
    )
except ImportError:
    # Fallback if datasets module has issues
    load_iris_quantum = None
    generate_portfolio_data = None
    load_molecule = None

try:
    from .utils import (
        QuantumFeatureMap,
        benchmark_algorithm,
        optimize_circuit,
        visualize_results,
    )
except ImportError:
    # Fallback if utils module has issues
    optimize_circuit = None
    visualize_results = None
    benchmark_algorithm = None
    QuantumFeatureMap = None

# Set up logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Package metadata
__title__ = "SuperQuantX"
__description__ = "Experimental Quantum AI Research Platform - NOT for production use"
__author__ = "SuperXLab - Superagentic AI Research Division"
__author_email__ = "research@superagentic.ai"
__license__ = "Apache-2.0"
__url__ = "https://github.com/superagentic/superquantx"

# Version information
def get_version() -> str:
    """Get SuperQuantX version."""
    return __version__

def get_backend_info() -> dict[str, Any]:
    """Get information about available backends."""
    info = {}

    # Check which backends are available
    try:
        import pennylane
        info['pennylane'] = pennylane.__version__
    except ImportError:
        info['pennylane'] = None

    try:
        import qiskit
        info['qiskit'] = qiskit.__version__
    except ImportError:
        info['qiskit'] = None

    try:
        import cirq
        info['cirq'] = cirq.__version__
    except ImportError:
        info['cirq'] = None

    try:
        import braket
        info['braket'] = braket.__version__
    except ImportError:
        info['braket'] = None

    try:
        import azure.quantum
        info['azure_quantum'] = azure.quantum.__version__
    except (ImportError, AttributeError):
        info['azure_quantum'] = None

    try:
        import pytket
        info['pytket'] = pytket.__version__
    except ImportError:
        info['pytket'] = None

    try:
        import dwave
        info['dwave'] = dwave.ocean.__version__
    except (ImportError, AttributeError):
        info['dwave'] = None

    try:
        import pyquil
        info['pyquil'] = pyquil.__version__
    except ImportError:
        info['pyquil'] = None

    try:
        import tensorflow_quantum
        info['tensorflow_quantum'] = tensorflow_quantum.__version__
    except ImportError:
        info['tensorflow_quantum'] = None

    try:
        import cuquantum
        info['cuquantum'] = cuquantum.__version__
    except (ImportError, AttributeError):
        info['cuquantum'] = None

    return info

def print_system_info() -> None:
    """Print system and backend information."""
    import platform
    import sys

    print(f"SuperQuantX version: {__version__}")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print("\nBackend versions:")

    backend_info = get_backend_info()
    for backend, version in backend_info.items():
        status = version if version else "Not installed"
        print(f"  {backend}: {status}")

def run_diagnostics() -> dict[str, dict[str, Any]]:
    """Run comprehensive diagnostics on all backends.
    
    Returns:
        Dictionary with backend names as keys and diagnostic info as values
    """
    diagnostics = {}
    
    # Test each backend
    backend_names = ['simulator', 'pennylane', 'qiskit', 'cirq', 'braket']
    
    for backend_name in backend_names:
        try:
            backend = get_backend(backend_name)
            diagnostics[backend_name] = {
                "available": True,
                "message": f"{backend_name} backend is working correctly",
                "backend_type": type(backend).__name__
            }
        except ImportError as e:
            diagnostics[backend_name] = {
                "available": False,
                "message": f"Backend not available: {e}",
                "backend_type": None
            }
        except Exception as e:
            diagnostics[backend_name] = {
                "available": False,
                "message": f"Backend error: {e}",
                "backend_type": None
            }
    
    return diagnostics

# Make commonly used functions available at package level
__all__ = [
    # Version and info
    "__version__",
    "get_version",
    "get_backend_info",
    "print_system_info",
    "run_diagnostics",

    # Core classes
    "QuantumCircuit",
    "SuperQuantXClient",
    "Hamiltonian",
    "PauliString",

    # Configuration
    "config",
    "configure",
    "create_default_config",
    "configure_interactive", 
    "load_config",
    "configure_logging",
    "configure_simulation",
    "config_context",
    "get_default_backend",
    "get_config",
    "get_config_search_paths", 
    "get_active_config_path",
    "validate_config",
    "validate_config_schema",
    "configure_performance",
    "configure_experiments",
    "experiment",
    "create_profile",
    "activate_profile",
    "profile",
    "ConfigValidationError",

    # Backend functions
    "get_backend",
    "list_available_backends",
    "check_backend_compatibility",

    # Modules
    "algorithms",
    "backends",
    "datasets",
    "utils",
    "cli",

    # Common algorithms
    "QuantumSVM",
    "QAOA",
    "VQE",
    "QuantumNN",
    "QuantumPCA",
    "QuantumKMeans",
    "HybridClassifier",

    # Common backends
    "BaseBackend",
    "AutoBackend",
    "QiskitBackend",
    "PennyLaneBackend",
    "CirqBackend",
    "BraketBackend",
    "TKETBackend",
    "OceanBackend",

    # Aliases
    "QuantinuumBackend",  # -> TKETBackend
    "DWaveBackend",       # -> OceanBackend
    "AWSBackend",         # -> BraketBackend

    # Not yet implemented
    "AzureBackend",
    "CuQuantumBackend",
    "ForestBackend",
    "TensorFlowQuantumBackend",
    "ClassicalBackend",

    # Common datasets
    "load_iris_quantum",
    "generate_portfolio_data",
    "load_molecule",

    # Common utils
    "optimize_circuit",
    "visualize_results",
    "benchmark_algorithm",
    "QuantumFeatureMap",
]
