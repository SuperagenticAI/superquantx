"""Quantum AI algorithms and autonomous agents.

This module provides the core algorithms powering quantum agentic AI systems,
from traditional quantum ML algorithms to autonomous quantum agents that can
make decisions and optimize complex problems independently.

The module is organized in layers:
- Quantum ML Algorithms: QSVM, VQE, QAOA, Quantum NN
- Quantum AI Models: Advanced neural networks and transformers
- Quantum Agents: Autonomous systems for trading, research, optimization
- Hybrid Intelligence: Quantum-classical integrated systems
"""

from .base_algorithm import BaseQuantumAlgorithm, QuantumResult
from .hybrid_classifier import HybridClassifier
from .qaoa import QAOA
from .quantum_agents import (
    QuantumAgent,
    QuantumClassificationAgent,
    QuantumOptimizationAgent,
    QuantumPortfolioAgent,
)
from .quantum_kmeans import QuantumKMeans
from .quantum_nn import QuantumNeuralNetwork, QuantumNN
from .quantum_pca import QuantumPCA
from .quantum_svm import QuantumSVM
from .vqe import VQE, create_vqe_for_molecule


__all__ = [
    # Base classes
    "BaseQuantumAlgorithm",
    "QuantumResult",

    # Classification algorithms
    "QuantumSVM",
    "QuantumNN",
    "QuantumNeuralNetwork",
    "HybridClassifier",

    # Optimization algorithms
    "QAOA",
    "VQE",
    "create_vqe_for_molecule",

    # Unsupervised learning
    "QuantumPCA",
    "QuantumKMeans",

    # Pre-built agents
    "QuantumAgent",
    "QuantumPortfolioAgent",
    "QuantumOptimizationAgent",
    "QuantumClassificationAgent",
]
