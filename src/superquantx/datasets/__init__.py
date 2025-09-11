"""SuperQuantX Datasets Module.

This module provides datasets and data loading utilities for quantum machine learning.
It includes quantum-specific datasets, classical datasets adapted for quantum computing,
and synthetic data generators for testing quantum algorithms.
"""

from .molecular import (
    load_beh2_molecule,
    load_h2_molecule,
    load_lih_molecule,
    load_molecule,
)
from .preprocessing import (
    AmplitudeEncoder,
    AngleEncoder,
    QuantumFeatureEncoder,
    normalize_quantum_data,
)
from .quantum_datasets import (
    load_breast_cancer_quantum,
    load_digits_quantum,
    load_iris_quantum,
    load_wine_quantum,
)
from .synthetic import (
    generate_classification_data,
    generate_clustering_data,
    generate_portfolio_data,
    generate_regression_data,
)


__all__ = [
    # Quantum datasets
    "load_iris_quantum",
    "load_wine_quantum",
    "load_digits_quantum",
    "load_breast_cancer_quantum",

    # Synthetic data generators
    "generate_classification_data",
    "generate_regression_data",
    "generate_clustering_data",
    "generate_portfolio_data",

    # Molecular datasets
    "load_molecule",
    "load_h2_molecule",
    "load_lih_molecule",
    "load_beh2_molecule",

    # Preprocessing utilities
    "QuantumFeatureEncoder",
    "AmplitudeEncoder",
    "AngleEncoder",
    "normalize_quantum_data",
]
