"""SuperQuantX Utilities Module.

This module provides utility functions and classes for quantum machine learning,
including circuit optimization, visualization, benchmarking, and feature mapping.
"""

from .benchmarking import (
    benchmark_algorithm,
    benchmark_backend,
    compare_algorithms,
    performance_metrics,
)
from .classical_utils import (
    cross_validation,
    data_splitting,
    hyperparameter_search,
    model_selection,
)
from .feature_mapping import (
    QuantumFeatureMap,
    create_feature_map,
    pauli_feature_map,
    zz_feature_map,
)
from .optimization import (
    adam_optimizer,
    gradient_descent,
    optimize_circuit,
    optimize_parameters,
)
from .quantum_utils import (
    entanglement_measure,
    fidelity,
    quantum_mutual_information,
    trace_distance,
)
from .visualization import (
    plot_bloch_sphere,
    plot_circuit,
    plot_optimization_history,
    plot_quantum_state,
    visualize_results,
)


__all__ = [
    # Optimization
    "optimize_circuit",
    "optimize_parameters",
    "gradient_descent",
    "adam_optimizer",

    # Visualization
    "visualize_results",
    "plot_optimization_history",
    "plot_circuit",
    "plot_quantum_state",
    "plot_bloch_sphere",

    # Benchmarking
    "benchmark_algorithm",
    "benchmark_backend",
    "performance_metrics",
    "compare_algorithms",

    # Feature mapping
    "QuantumFeatureMap",
    "create_feature_map",
    "pauli_feature_map",
    "zz_feature_map",

    # Quantum utilities
    "fidelity",
    "trace_distance",
    "quantum_mutual_information",
    "entanglement_measure",

    # Classical utilities
    "cross_validation",
    "hyperparameter_search",
    "model_selection",
    "data_splitting",
]
