"""Quantum feature mapping utilities.

This module provides functions and classes for creating quantum feature maps,
which encode classical data into quantum states for machine learning algorithms.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FeatureMapConfig:
    """Configuration for quantum feature maps."""

    n_qubits: int
    n_features: int
    reps: int = 1
    entanglement: str = 'full'
    parameter_prefix: str = 'x'
    insert_barriers: bool = False
    data_map_func: Callable | None = None


class QuantumFeatureMap(ABC):
    """Abstract base class for quantum feature maps.

    Feature maps encode classical data into quantum states by applying
    parameterized quantum gates based on the input features.
    """

    def __init__(
        self,
        n_features: int,
        reps: int = 1,
        entanglement: str = 'full',
        parameter_prefix: str = 'x'
    ):
        self.n_features = n_features
        self.reps = reps
        self.entanglement = entanglement
        self.parameter_prefix = parameter_prefix
        self.n_qubits = n_features
        self._parameters = []

    @abstractmethod
    def _build_circuit(self, parameters: np.ndarray) -> dict[str, Any]:
        """Build the quantum circuit for the feature map."""
        pass

    def map_data_point(self, x: np.ndarray) -> dict[str, Any]:
        """Map a single data point to quantum circuit parameters.

        Args:
            x: Input data point of length n_features

        Returns:
            Circuit representation with parameters

        """
        if len(x) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(x)}")

        # Repeat the data point for each repetition
        parameters = np.tile(x, self.reps)

        return self._build_circuit(parameters)

    def map_data(self, X: np.ndarray) -> list[dict[str, Any]]:
        """Map multiple data points to quantum circuits.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            List of circuit representations

        """
        return [self.map_data_point(x) for x in X]

    @property
    def num_parameters(self) -> int:
        """Number of parameters in the feature map."""
        return self.n_features * self.reps


class ZFeatureMap(QuantumFeatureMap):
    """Z-axis rotation feature map.

    This feature map applies RZ rotations to encode features:
    RZ(2 * x_i) for each feature x_i
    """

    def __init__(
        self,
        n_features: int,
        reps: int = 1,
        parameter_prefix: str = 'x'
    ):
        super().__init__(n_features, reps, 'none', parameter_prefix)

    def _build_circuit(self, parameters: np.ndarray) -> dict[str, Any]:
        """Build Z feature map circuit."""
        gates = []

        for rep in range(self.reps):
            for i in range(self.n_features):
                param_idx = rep * self.n_features + i
                gates.append({
                    'type': 'RZ',
                    'qubit': i,
                    'parameter': 2.0 * parameters[param_idx],
                    'parameter_name': f'{self.parameter_prefix}_{param_idx}'
                })

        return {
            'n_qubits': self.n_qubits,
            'gates': gates,
            'parameters': parameters.tolist(),
            'feature_map_type': 'Z'
        }


class ZZFeatureMap(QuantumFeatureMap):
    """ZZ entangling feature map.

    This feature map uses both single-qubit Z rotations and two-qubit ZZ interactions:
    - Single qubit: RZ(2 * x_i)
    - Two qubit: RZZ(2 * x_i * x_j) for entangled qubits
    """

    def __init__(
        self,
        n_features: int,
        reps: int = 1,
        entanglement: str = 'linear',
        alpha: float = 2.0,
        parameter_prefix: str = 'x'
    ):
        super().__init__(n_features, reps, entanglement, parameter_prefix)
        self.alpha = alpha

    def _build_circuit(self, parameters: np.ndarray) -> dict[str, Any]:
        """Build ZZ feature map circuit."""
        gates = []

        for rep in range(self.reps):
            # Single qubit Z rotations
            for i in range(self.n_features):
                param_idx = rep * self.n_features + i
                gates.append({
                    'type': 'RZ',
                    'qubit': i,
                    'parameter': self.alpha * parameters[param_idx],
                    'parameter_name': f'{self.parameter_prefix}_{param_idx}'
                })

            # Two-qubit ZZ entangling gates
            entangling_pairs = self._get_entangling_pairs()

            for i, j in entangling_pairs:
                param_i = rep * self.n_features + i
                param_j = rep * self.n_features + j

                gates.append({
                    'type': 'RZZ',
                    'qubits': [i, j],
                    'parameter': self.alpha * parameters[param_i] * parameters[param_j],
                    'parameter_names': [f'{self.parameter_prefix}_{param_i}',
                                      f'{self.parameter_prefix}_{param_j}']
                })

        return {
            'n_qubits': self.n_qubits,
            'gates': gates,
            'parameters': parameters.tolist(),
            'feature_map_type': 'ZZ',
            'entanglement': self.entanglement
        }

    def _get_entangling_pairs(self) -> list[tuple[int, int]]:
        """Get pairs of qubits for entangling gates."""
        pairs = []

        if self.entanglement == 'linear':
            # Linear chain: (0,1), (1,2), (2,3), ...
            for i in range(self.n_qubits - 1):
                pairs.append((i, i + 1))

        elif self.entanglement == 'circular':
            # Circular: linear + (n-1, 0)
            pairs = self._get_linear_pairs()
            pairs.append((self.n_qubits - 1, 0))

        elif self.entanglement == 'full':
            # Full connectivity: all pairs
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    pairs.append((i, j))

        elif self.entanglement == 'sca':
            # Strongly correlated ansatz
            for i in range(0, self.n_qubits - 1, 2):
                if i + 1 < self.n_qubits:
                    pairs.append((i, i + 1))

        return pairs


class PauliFeatureMap(QuantumFeatureMap):
    """Pauli feature map with arbitrary Pauli strings.

    This feature map applies rotations based on Pauli operators:
    exp(i * alpha * phi * P) where P is a Pauli string and phi is the feature value.
    """

    def __init__(
        self,
        n_features: int,
        reps: int = 1,
        paulis: list[str] = None,
        entanglement: str = 'full',
        alpha: float = 2.0,
        parameter_prefix: str = 'x'
    ):
        if paulis is None:
            paulis = ['Z', 'ZZ']
        super().__init__(n_features, reps, entanglement, parameter_prefix)
        self.paulis = paulis
        self.alpha = alpha

    def _build_circuit(self, parameters: np.ndarray) -> dict[str, Any]:
        """Build Pauli feature map circuit."""
        gates = []

        for rep in range(self.reps):
            for pauli_string in self.paulis:
                if len(pauli_string) == 1:
                    # Single qubit Pauli
                    self._add_single_pauli_gates(
                        gates, pauli_string, parameters, rep
                    )
                else:
                    # Multi-qubit Pauli
                    self._add_multi_pauli_gates(
                        gates, pauli_string, parameters, rep
                    )

        return {
            'n_qubits': self.n_qubits,
            'gates': gates,
            'parameters': parameters.tolist(),
            'feature_map_type': 'Pauli',
            'paulis': self.paulis
        }

    def _add_single_pauli_gates(
        self,
        gates: list[dict],
        pauli: str,
        parameters: np.ndarray,
        rep: int
    ):
        """Add single-qubit Pauli rotation gates."""
        for i in range(self.n_features):
            param_idx = rep * self.n_features + i

            if pauli == 'X':
                gate_type = 'RX'
            elif pauli == 'Y':
                gate_type = 'RY'
            elif pauli == 'Z':
                gate_type = 'RZ'
            else:
                continue

            gates.append({
                'type': gate_type,
                'qubit': i,
                'parameter': self.alpha * parameters[param_idx],
                'parameter_name': f'{self.parameter_prefix}_{param_idx}'
            })

    def _add_multi_pauli_gates(
        self,
        gates: list[dict],
        pauli_string: str,
        parameters: np.ndarray,
        rep: int
    ):
        """Add multi-qubit Pauli rotation gates."""
        # For simplicity, handle ZZ case
        if pauli_string == 'ZZ':
            entangling_pairs = self._get_entangling_pairs()

            for i, j in entangling_pairs:
                param_i = rep * self.n_features + i
                param_j = rep * self.n_features + j

                gates.append({
                    'type': 'RZZ',
                    'qubits': [i, j],
                    'parameter': self.alpha * parameters[param_i] * parameters[param_j],
                    'parameter_names': [f'{self.parameter_prefix}_{param_i}',
                                      f'{self.parameter_prefix}_{param_j}']
                })


def create_feature_map(
    feature_map_type: str,
    n_features: int,
    **kwargs
) -> QuantumFeatureMap:
    """Factory function to create quantum feature maps.

    Args:
        feature_map_type: Type of feature map ('Z', 'ZZ', 'Pauli')
        n_features: Number of input features
        **kwargs: Additional arguments for specific feature maps

    Returns:
        QuantumFeatureMap instance

    """
    feature_map_type = feature_map_type.upper()

    if feature_map_type == 'Z':
        return ZFeatureMap(n_features, **kwargs)
    elif feature_map_type == 'ZZ':
        return ZZFeatureMap(n_features, **kwargs)
    elif feature_map_type == 'PAULI':
        return PauliFeatureMap(n_features, **kwargs)
    else:
        raise ValueError(f"Unknown feature map type: {feature_map_type}")


def pauli_feature_map(
    n_features: int,
    paulis: list[str] = None,
    reps: int = 1,
    alpha: float = 2.0,
    entanglement: str = 'full'
) -> PauliFeatureMap:
    """Create a Pauli feature map with specified Pauli strings.

    Args:
        n_features: Number of input features
        paulis: List of Pauli strings to use
        reps: Number of repetitions
        alpha: Scaling factor
        entanglement: Entanglement pattern

    Returns:
        PauliFeatureMap instance

    """
    if paulis is None:
        paulis = ['Z', 'ZZ']
    return PauliFeatureMap(
        n_features=n_features,
        paulis=paulis,
        reps=reps,
        alpha=alpha,
        entanglement=entanglement
    )


def zz_feature_map(
    n_features: int,
    reps: int = 1,
    entanglement: str = 'linear',
    alpha: float = 2.0
) -> ZZFeatureMap:
    """Create a ZZ feature map with specified parameters.

    Args:
        n_features: Number of input features
        reps: Number of repetitions
        entanglement: Entanglement pattern ('linear', 'circular', 'full')
        alpha: Scaling factor

    Returns:
        ZZFeatureMap instance

    """
    return ZZFeatureMap(
        n_features=n_features,
        reps=reps,
        entanglement=entanglement,
        alpha=alpha
    )


def feature_map_from_config(config: FeatureMapConfig) -> QuantumFeatureMap:
    """Create feature map from configuration.

    Args:
        config: FeatureMapConfig instance

    Returns:
        QuantumFeatureMap instance

    """
    # Determine feature map type based on config
    # This is a simplified approach - in practice, you'd need more logic
    if config.entanglement == 'none':
        return ZFeatureMap(
            n_features=config.n_features,
            reps=config.reps,
            parameter_prefix=config.parameter_prefix
        )
    else:
        return ZZFeatureMap(
            n_features=config.n_features,
            reps=config.reps,
            entanglement=config.entanglement,
            parameter_prefix=config.parameter_prefix
        )


def evaluate_feature_map_expressibility(
    feature_map: QuantumFeatureMap,
    n_samples: int = 1000,
    random_state: int | None = None
) -> dict[str, float]:
    """Evaluate the expressibility of a quantum feature map.

    Expressibility measures how well a feature map can generate
    diverse quantum states across the Hilbert space.

    Args:
        feature_map: QuantumFeatureMap to evaluate
        n_samples: Number of random data points to sample
        random_state: Random seed

    Returns:
        Dictionary with expressibility metrics

    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate random data points
    X_random = np.random.uniform(-1, 1, (n_samples, feature_map.n_features))

    # Map to quantum circuits
    circuits = feature_map.map_data(X_random)

    # Compute diversity metrics
    # This is a simplified version - real implementation would need
    # to simulate the circuits and compute state overlaps

    parameter_diversity = []
    for circuit in circuits:
        params = circuit['parameters']
        # Measure parameter diversity (standard deviation)
        param_std = np.std(params)
        parameter_diversity.append(param_std)

    metrics = {
        'parameter_diversity_mean': np.mean(parameter_diversity),
        'parameter_diversity_std': np.std(parameter_diversity),
        'n_samples': n_samples,
        'n_features': feature_map.n_features,
        'n_qubits': feature_map.n_qubits,
        'reps': feature_map.reps
    }

    return metrics
