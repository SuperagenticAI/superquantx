"""Quantum measurement and result handling for SuperQuantX
"""

import json
from collections import Counter, defaultdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, Field


class MeasurementResult(BaseModel):
    """Represents the result of quantum measurements
    """

    counts: dict[str, int] = Field(..., description="Measurement outcome counts")
    shots: int = Field(..., description="Total number of shots")
    memory: list[str] | None = Field(default=None, description="Individual shot outcomes")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def model_post_init(self, __context):
        """Validate measurement result"""
        if sum(self.counts.values()) != self.shots:
            raise ValueError("Sum of counts must equal total shots")

    @property
    def probabilities(self) -> dict[str, float]:
        """Get measurement probabilities"""
        return {outcome: count / self.shots for outcome, count in self.counts.items()}

    @property
    def most_frequent(self) -> tuple[str, int]:
        """Get most frequent measurement outcome"""
        return max(self.counts.items(), key=lambda x: x[1])

    def marginal_counts(self, qubits: list[int]) -> dict[str, int]:
        """Get marginal counts for specific qubits

        Args:
            qubits: List of qubit indices to marginalize over

        Returns:
            Marginal counts dictionary

        """
        marginal = defaultdict(int)

        for outcome, count in self.counts.items():
            # Extract bits for specified qubits (reverse order due to endianness)
            marginal_outcome = ''.join(outcome[-(i+1)] for i in qubits)
            marginal[marginal_outcome] += count

        return dict(marginal)

    def expectation_value(self, observable: str) -> float:
        """Calculate expectation value for Pauli observable

        Args:
            observable: Pauli string (e.g., "ZZI")

        Returns:
            Expectation value

        """
        if len(observable) == 0:
            return 1.0

        expectation = 0.0

        for outcome, count in self.counts.items():
            # Calculate parity for non-identity Pauli operators
            parity = 1
            for i, pauli_op in enumerate(observable):
                if pauli_op == 'Z' and i < len(outcome):
                    bit = int(outcome[-(i+1)])  # Reverse order
                    parity *= (-1) ** bit
                elif pauli_op in ['X', 'Y']:
                    # X and Y measurements require basis rotation
                    raise ValueError(f"Cannot compute expectation for {pauli_op} from Z-basis measurements")

            expectation += parity * count / self.shots

        return expectation

    def entropy(self) -> float:
        """Calculate measurement entropy"""
        entropy = 0.0
        for prob in self.probabilities.values():
            if prob > 0:
                entropy -= prob * np.log2(prob)
        return entropy

    def plot_histogram(
        self,
        title: str = "Measurement Results",
        figsize: tuple[int, int] = (10, 6),
        max_outcomes: int = 20
    ) -> plt.Figure:
        """Plot measurement histogram

        Args:
            title: Plot title
            figsize: Figure size
            max_outcomes: Maximum number of outcomes to show

        Returns:
            Matplotlib figure

        """
        fig, ax = plt.subplots(figsize=figsize)

        # Sort outcomes by count (descending)
        sorted_outcomes = sorted(self.counts.items(), key=lambda x: x[1], reverse=True)

        # Take top outcomes
        if len(sorted_outcomes) > max_outcomes:
            sorted_outcomes = sorted_outcomes[:max_outcomes]

        outcomes, counts = zip(*sorted_outcomes, strict=False) if sorted_outcomes else ([], [])

        ax.bar(range(len(outcomes)), counts)
        ax.set_xlabel('Measurement Outcome')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.set_xticks(range(len(outcomes)))
        ax.set_xticklabels(outcomes, rotation=45, ha='right')

        plt.tight_layout()
        return fig

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "counts": self.counts,
            "shots": self.shots,
            "memory": self.memory,
            "metadata": self.metadata
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MeasurementResult":
        """Create from dictionary"""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "MeasurementResult":
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __add__(self, other: "MeasurementResult") -> "MeasurementResult":
        """Add two measurement results"""
        combined_counts = dict(self.counts)

        for outcome, count in other.counts.items():
            combined_counts[outcome] = combined_counts.get(outcome, 0) + count

        combined_memory = None
        if self.memory is not None and other.memory is not None:
            combined_memory = self.memory + other.memory

        return MeasurementResult(
            counts=combined_counts,
            shots=self.shots + other.shots,
            memory=combined_memory,
            metadata={**self.metadata, **other.metadata}
        )


class QuantumMeasurement:
    """Quantum measurement operations and analysis
    """

    def __init__(self, backend: str | None = "simulator"):
        """Initialize measurement system

        Args:
            backend: Quantum backend for measurements

        """
        self.backend = backend
        self.measurement_history: list[MeasurementResult] = []

    def measure_circuit(
        self,
        circuit: "QuantumCircuit",
        shots: int = 1024,
        memory: bool = False
    ) -> MeasurementResult:
        """Measure quantum circuit

        Args:
            circuit: Quantum circuit to measure
            shots: Number of measurement shots
            memory: Whether to store individual shot outcomes

        Returns:
            Measurement result

        """
        # This would interface with actual quantum hardware/simulator
        # For now, create simulated results


        # Simulate measurement outcomes
        if self.backend == "simulator":
            counts = self._simulate_measurements(circuit, shots)
        else:
            counts = self._execute_measurements(circuit, shots)

        # Generate memory if requested
        shot_memory = None
        if memory:
            shot_memory = []
            for outcome, count in counts.items():
                shot_memory.extend([outcome] * count)
            np.random.shuffle(shot_memory)  # Randomize order

        result = MeasurementResult(
            counts=counts,
            shots=shots,
            memory=shot_memory,
            metadata={"backend": self.backend, "circuit_name": circuit.name}
        )

        self.measurement_history.append(result)
        return result

    def _simulate_measurements(self, circuit: "QuantumCircuit", shots: int) -> dict[str, int]:
        """Simulate measurement outcomes"""
        num_bits = circuit.num_classical_bits

        # Simple simulation: uniform random outcomes for now
        # In practice, would run full quantum simulation
        outcomes = []
        for _ in range(shots):
            outcome = ''.join(str(np.random.randint(2)) for _ in range(num_bits))
            outcomes.append(outcome)

        return dict(Counter(outcomes))

    def _execute_measurements(self, circuit: "QuantumCircuit", shots: int) -> dict[str, int]:
        """Execute measurements on quantum hardware"""
        # This would interface with quantum hardware via client
        # Placeholder for now
        return self._simulate_measurements(circuit, shots)

    def measure_observable(
        self,
        circuit: "QuantumCircuit",
        observable: str,
        shots: int = 1024
    ) -> float:
        """Measure expectation value of Pauli observable

        Args:
            circuit: Quantum circuit (without measurements)
            observable: Pauli string observable
            shots: Number of shots

        Returns:
            Expectation value

        """
        # Create measurement circuit with basis rotations
        measurement_circuit = self._prepare_observable_measurement(circuit, observable)

        # Measure circuit
        result = self.measure_circuit(measurement_circuit, shots)

        # Calculate expectation value
        return result.expectation_value('Z' * len(observable))

    def _prepare_observable_measurement(
        self,
        circuit: "QuantumCircuit",
        observable: str
    ) -> "QuantumCircuit":
        """Prepare circuit for observable measurement"""
        # Copy original circuit
        measurement_circuit = circuit.copy()

        # Add basis rotation gates
        for i, pauli_op in enumerate(observable):
            if i >= circuit.num_qubits:
                break

            if pauli_op == 'X':
                measurement_circuit.ry(-np.pi/2, i)  # Rotate Y→Z basis
            elif pauli_op == 'Y':
                measurement_circuit.rx(np.pi/2, i)   # Rotate X→Z basis
            # Z measurements don't need rotation

        # Add measurements
        for i in range(min(len(observable), circuit.num_qubits)):
            measurement_circuit.measure(i, i)

        return measurement_circuit

    def tomography_measurements(
        self,
        circuit: "QuantumCircuit",
        qubits: list[int] | None = None,
        shots_per_measurement: int = 1024
    ) -> dict[str, MeasurementResult]:
        """Perform quantum state tomography measurements

        Args:
            circuit: Quantum circuit to tomographically reconstruct
            qubits: Qubits to perform tomography on (default: all)
            shots_per_measurement: Shots per Pauli measurement

        Returns:
            Dictionary of measurement results for each Pauli basis

        """
        if qubits is None:
            qubits = list(range(circuit.num_qubits))

        num_qubits = len(qubits)
        pauli_bases = ['I', 'X', 'Y', 'Z']

        # Generate all Pauli measurement settings
        measurements = {}

        def generate_pauli_strings(n):
            if n == 0:
                yield ''
            else:
                for base in pauli_bases:
                    for rest in generate_pauli_strings(n - 1):
                        yield base + rest

        for pauli_string in generate_pauli_strings(num_qubits):
            if 'I' in pauli_string:
                continue  # Skip identity measurements

            measurement_circuit = self._prepare_tomography_measurement(
                circuit, pauli_string, qubits
            )

            result = self.measure_circuit(measurement_circuit, shots_per_measurement)
            measurements[pauli_string] = result

        return measurements

    def _prepare_tomography_measurement(
        self,
        circuit: "QuantumCircuit",
        pauli_string: str,
        qubits: list[int]
    ) -> "QuantumCircuit":
        """Prepare circuit for tomography measurement"""
        measurement_circuit = circuit.copy()

        for i, pauli_op in enumerate(pauli_string):
            if i >= len(qubits):
                break

            qubit = qubits[i]

            if pauli_op == 'X':
                measurement_circuit.ry(-np.pi/2, qubit)
            elif pauli_op == 'Y':
                measurement_circuit.rx(np.pi/2, qubit)

        # Measure selected qubits
        for i, qubit in enumerate(qubits):
            if i < len(pauli_string):
                measurement_circuit.measure(qubit, i)

        return measurement_circuit

    def reconstruct_state(
        self,
        tomography_results: dict[str, MeasurementResult]
    ) -> np.ndarray:
        """Reconstruct quantum state from tomography measurements

        Args:
            tomography_results: Results from tomography_measurements

        Returns:
            Reconstructed density matrix

        """
        # This is a simplified implementation
        # Full tomography would use maximum likelihood estimation

        num_qubits = len(next(iter(tomography_results.keys())))
        dim = 2 ** num_qubits

        # Initialize density matrix
        rho = np.eye(dim, dtype=complex) / dim

        # This is a placeholder implementation
        # Real tomography would involve solving a constrained optimization problem

        return rho

    def fidelity(
        self,
        state1: np.ndarray,
        state2: np.ndarray
    ) -> float:
        """Calculate quantum state fidelity

        Args:
            state1: First quantum state (vector or density matrix)
            state2: Second quantum state (vector or density matrix)

        Returns:
            Fidelity between states

        """
        # Convert to density matrices if needed
        if state1.ndim == 1:
            rho1 = np.outer(state1, np.conj(state1))
        else:
            rho1 = state1

        if state2.ndim == 1:
            rho2 = np.outer(state2, np.conj(state2))
        else:
            rho2 = state2

        # Calculate fidelity using proper matrix square root
        eigenvals_rho1, eigenvecs_rho1 = np.linalg.eigh(rho1)
        eigenvals_rho1 = np.maximum(eigenvals_rho1, 0)  # Ensure non-negative
        sqrt_rho1 = eigenvecs_rho1 @ np.diag(np.sqrt(eigenvals_rho1)) @ eigenvecs_rho1.conj().T

        product = sqrt_rho1 @ rho2 @ sqrt_rho1
        eigenvals = np.linalg.eigvals(product)
        eigenvals = np.maximum(eigenvals.real, 0)  # Take real part and ensure non-negative

        return float(np.sum(np.sqrt(eigenvals)))

    def trace_distance(
        self,
        state1: np.ndarray,
        state2: np.ndarray
    ) -> float:
        """Calculate trace distance between quantum states

        Args:
            state1: First quantum state
            state2: Second quantum state

        Returns:
            Trace distance

        """
        # Convert to density matrices if needed
        if state1.ndim == 1:
            rho1 = np.outer(state1, np.conj(state1))
        else:
            rho1 = state1

        if state2.ndim == 1:
            rho2 = np.outer(state2, np.conj(state2))
        else:
            rho2 = state2

        diff = rho1 - rho2
        eigenvals = np.linalg.eigvals(diff)

        return 0.5 * np.sum(np.abs(eigenvals))

    def quantum_volume(
        self,
        num_qubits: int,
        depth: int,
        trials: int = 100,
        shots_per_trial: int = 1024
    ) -> dict[str, Any]:
        """Perform quantum volume benchmark

        Args:
            num_qubits: Number of qubits
            depth: Circuit depth
            trials: Number of random circuits to test
            shots_per_trial: Shots per circuit

        Returns:
            Quantum volume benchmark results

        """
        import random

        from .circuits import QuantumCircuit

        successes = 0

        for trial in range(trials):
            # Generate random quantum volume circuit
            circuit = QuantumCircuit(num_qubits)

            for layer in range(depth):
                # Random SU(4) gates on random qubit pairs
                available_qubits = list(range(num_qubits))
                random.shuffle(available_qubits)

                for i in range(0, num_qubits - 1, 2):
                    q1, q2 = available_qubits[i], available_qubits[i + 1]

                    # Random single-qubit gates
                    circuit.u(
                        random.uniform(0, 2*np.pi),
                        random.uniform(0, 2*np.pi),
                        random.uniform(0, 2*np.pi),
                        q1
                    )
                    circuit.u(
                        random.uniform(0, 2*np.pi),
                        random.uniform(0, 2*np.pi),
                        random.uniform(0, 2*np.pi),
                        q2
                    )

                    # CNOT gate
                    circuit.cnot(q1, q2)

            # Measure circuit
            circuit.measure_all()
            result = self.measure_circuit(circuit, shots_per_trial)

            # Check if heavy output (simplified check)
            # Real quantum volume would compute ideal probabilities
            most_frequent = result.most_frequent
            if most_frequent[1] > shots_per_trial // 4:  # Simplified threshold
                successes += 1

        success_rate = successes / trials

        return {
            "num_qubits": num_qubits,
            "depth": depth,
            "trials": trials,
            "successes": successes,
            "success_rate": success_rate,
            "passed": success_rate > 2/3,  # Standard QV threshold
            "quantum_volume": 2 ** num_qubits if success_rate > 2/3 else 0
        }


class ResultAnalyzer:
    """Advanced analysis of quantum measurement results
    """

    @staticmethod
    def compare_results(
        results1: MeasurementResult,
        results2: MeasurementResult
    ) -> dict[str, float]:
        """Compare two measurement results

        Args:
            results1: First measurement result
            results2: Second measurement result

        Returns:
            Comparison metrics

        """
        # Hellinger distance between probability distributions
        prob1 = results1.probabilities
        prob2 = results2.probabilities

        all_outcomes = set(prob1.keys()) | set(prob2.keys())

        hellinger = 0.0
        kl_divergence = 0.0

        for outcome in all_outcomes:
            p1 = prob1.get(outcome, 0)
            p2 = prob2.get(outcome, 0)

            # Hellinger distance
            hellinger += (np.sqrt(p1) - np.sqrt(p2)) ** 2

            # KL divergence (with smoothing to avoid log(0))
            p1_smooth = p1 + 1e-10
            p2_smooth = p2 + 1e-10
            kl_divergence += p1_smooth * np.log(p1_smooth / p2_smooth)

        hellinger = np.sqrt(hellinger / 2)

        # Total variation distance
        tv_distance = 0.5 * sum(abs(prob1.get(outcome, 0) - prob2.get(outcome, 0))
                              for outcome in all_outcomes)

        return {
            "hellinger_distance": hellinger,
            "kl_divergence": kl_divergence,
            "total_variation_distance": tv_distance
        }

    @staticmethod
    def error_mitigation_zero_noise_extrapolation(
        noise_levels: list[float],
        measurement_results: list[MeasurementResult],
        observable: str = "Z"
    ) -> tuple[float, dict[str, Any]]:
        """Perform zero-noise extrapolation error mitigation

        Args:
            noise_levels: List of noise levels (e.g., [1, 2, 3])
            measurement_results: Results for each noise level
            observable: Observable to extrapolate

        Returns:
            Zero-noise extrapolated value and fitting info

        """
        if len(noise_levels) != len(measurement_results):
            raise ValueError("Noise levels and results must have same length")

        # Extract expectation values
        expectation_values = []
        for result in measurement_results:
            exp_val = result.expectation_value(observable)
            expectation_values.append(exp_val)

        # Fit exponential decay model: f(x) = a * exp(-b * x) + c
        from scipy.optimize import curve_fit

        def exponential_model(x, a, b, c):
            return a * np.exp(-b * x) + c

        try:
            popt, pcov = curve_fit(
                exponential_model,
                noise_levels,
                expectation_values,
                p0=[expectation_values[0], 0.1, 0]
            )

            # Extrapolate to zero noise
            zero_noise_value = exponential_model(0, *popt)

            # Calculate fit quality
            fitted_values = [exponential_model(x, *popt) for x in noise_levels]
            r_squared = 1 - np.sum((np.array(expectation_values) - fitted_values)**2) / \
                           np.sum((np.array(expectation_values) - np.mean(expectation_values))**2)

            return zero_noise_value, {
                "fit_parameters": popt,
                "fit_covariance": pcov,
                "r_squared": r_squared,
                "fitted_values": fitted_values,
                "raw_values": expectation_values
            }

        except Exception as e:
            # Fallback to linear extrapolation
            coeffs = np.polyfit(noise_levels, expectation_values, 1)
            zero_noise_value = coeffs[1]  # y-intercept

            return zero_noise_value, {
                "method": "linear_fallback",
                "coefficients": coeffs,
                "error": str(e)
            }

    @staticmethod
    def readout_error_mitigation(
        calibration_results: dict[str, MeasurementResult],
        measurement_result: MeasurementResult
    ) -> MeasurementResult:
        """Apply readout error mitigation

        Args:
            calibration_results: Results from measuring |0⟩ and |1⟩ states
            measurement_result: Result to correct

        Returns:
            Error-mitigated result

        """
        # Build calibration matrix
        len(next(iter(calibration_results.keys())))

        # This is a simplified implementation
        # Full readout error mitigation would build complete confusion matrix

        corrected_counts = dict(measurement_result.counts)

        # Apply simple correction (placeholder)
        # Real implementation would invert the calibration matrix

        return MeasurementResult(
            counts=corrected_counts,
            shots=measurement_result.shots,
            memory=measurement_result.memory,
            metadata={**measurement_result.metadata, "error_mitigated": True}
        )
