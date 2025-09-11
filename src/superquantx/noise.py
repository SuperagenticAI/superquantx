"""Quantum noise models and error correction for SuperQuantX
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from .circuits import QuantumCircuit, QuantumGate
from .gates import GateMatrix, PauliString


class NoiseChannel(ABC):
    """Abstract base class for quantum noise channels
    """

    def __init__(self, probability: float):
        """Initialize noise channel

        Args:
            probability: Noise probability (0 <= p <= 1)

        """
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1")
        self.probability = probability

    @abstractmethod
    def kraus_operators(self) -> list[np.ndarray]:
        """Return Kraus operators for the noise channel"""
        pass

    @abstractmethod
    def apply_to_density_matrix(self, rho: np.ndarray) -> np.ndarray:
        """Apply noise channel to density matrix"""
        pass

    def is_unital(self) -> bool:
        """Check if channel is unital (maps identity to identity)"""
        identity = np.eye(2, dtype=complex)
        noisy_identity = self.apply_to_density_matrix(identity)
        return np.allclose(noisy_identity, identity)


class BitFlipChannel(NoiseChannel):
    """Bit flip (X) noise channel
    """

    def kraus_operators(self) -> list[np.ndarray]:
        """Kraus operators for bit flip channel"""
        sqrt_p = np.sqrt(self.probability)
        sqrt_1_p = np.sqrt(1 - self.probability)

        return [
            sqrt_1_p * GateMatrix.I,  # No error
            sqrt_p * GateMatrix.X     # Bit flip
        ]

    def apply_to_density_matrix(self, rho: np.ndarray) -> np.ndarray:
        """Apply bit flip noise to density matrix"""
        kraus_ops = self.kraus_operators()
        result = np.zeros_like(rho, dtype=complex)

        for kraus in kraus_ops:
            result += kraus @ rho @ kraus.conj().T

        return result


class PhaseFlipChannel(NoiseChannel):
    """Phase flip (Z) noise channel
    """

    def kraus_operators(self) -> list[np.ndarray]:
        """Kraus operators for phase flip channel"""
        sqrt_p = np.sqrt(self.probability)
        sqrt_1_p = np.sqrt(1 - self.probability)

        return [
            sqrt_1_p * GateMatrix.I,  # No error
            sqrt_p * GateMatrix.Z     # Phase flip
        ]

    def apply_to_density_matrix(self, rho: np.ndarray) -> np.ndarray:
        """Apply phase flip noise to density matrix"""
        kraus_ops = self.kraus_operators()
        result = np.zeros_like(rho, dtype=complex)

        for kraus in kraus_ops:
            result += kraus @ rho @ kraus.conj().T

        return result


class BitPhaseFlipChannel(NoiseChannel):
    """Bit-phase flip (Y) noise channel
    """

    def kraus_operators(self) -> list[np.ndarray]:
        """Kraus operators for bit-phase flip channel"""
        sqrt_p = np.sqrt(self.probability)
        sqrt_1_p = np.sqrt(1 - self.probability)

        return [
            sqrt_1_p * GateMatrix.I,  # No error
            sqrt_p * GateMatrix.Y     # Bit-phase flip
        ]

    def apply_to_density_matrix(self, rho: np.ndarray) -> np.ndarray:
        """Apply bit-phase flip noise to density matrix"""
        kraus_ops = self.kraus_operators()
        result = np.zeros_like(rho, dtype=complex)

        for kraus in kraus_ops:
            result += kraus @ rho @ kraus.conj().T

        return result


class DepolarizingChannel(NoiseChannel):
    """Depolarizing noise channel
    """

    def kraus_operators(self) -> list[np.ndarray]:
        """Kraus operators for depolarizing channel"""
        p = self.probability

        return [
            np.sqrt(1 - 3*p/4) * GateMatrix.I,  # No error
            np.sqrt(p/4) * GateMatrix.X,        # X error
            np.sqrt(p/4) * GateMatrix.Y,        # Y error
            np.sqrt(p/4) * GateMatrix.Z         # Z error
        ]

    def apply_to_density_matrix(self, rho: np.ndarray) -> np.ndarray:
        """Apply depolarizing noise to density matrix"""
        p = self.probability

        # Direct formula: ρ → (1-p)ρ + p*I/2
        identity = np.eye(rho.shape[0], dtype=complex)
        return (1 - p) * rho + (p / rho.shape[0]) * identity


class AmplitudeDampingChannel(NoiseChannel):
    """Amplitude damping noise channel (T1 decay)
    """

    def kraus_operators(self) -> list[np.ndarray]:
        """Kraus operators for amplitude damping channel"""
        gamma = self.probability

        E0 = np.array([
            [1, 0],
            [0, np.sqrt(1 - gamma)]
        ], dtype=complex)

        E1 = np.array([
            [0, np.sqrt(gamma)],
            [0, 0]
        ], dtype=complex)

        return [E0, E1]

    def apply_to_density_matrix(self, rho: np.ndarray) -> np.ndarray:
        """Apply amplitude damping noise to density matrix"""
        kraus_ops = self.kraus_operators()
        result = np.zeros_like(rho, dtype=complex)

        for kraus in kraus_ops:
            result += kraus @ rho @ kraus.conj().T

        return result


class PhaseDampingChannel(NoiseChannel):
    """Phase damping noise channel (T2 dephasing)
    """

    def kraus_operators(self) -> list[np.ndarray]:
        """Kraus operators for phase damping channel"""
        gamma = self.probability

        E0 = np.array([
            [1, 0],
            [0, np.sqrt(1 - gamma)]
        ], dtype=complex)

        E1 = np.array([
            [0, 0],
            [0, np.sqrt(gamma)]
        ], dtype=complex)

        return [E0, E1]

    def apply_to_density_matrix(self, rho: np.ndarray) -> np.ndarray:
        """Apply phase damping noise to density matrix"""
        kraus_ops = self.kraus_operators()
        result = np.zeros_like(rho, dtype=complex)

        for kraus in kraus_ops:
            result += kraus @ rho @ kraus.conj().T

        return result


class TwoQubitDepolarizingChannel(NoiseChannel):
    """Two-qubit depolarizing noise channel
    """

    def kraus_operators(self) -> list[np.ndarray]:
        """Kraus operators for two-qubit depolarizing channel"""
        p = self.probability

        # Single-qubit Pauli operators
        pauli_ops = [GateMatrix.I, GateMatrix.X, GateMatrix.Y, GateMatrix.Z]

        kraus_ops = []

        # All combinations of Pauli operators on two qubits
        for i, p1 in enumerate(pauli_ops):
            for j, p2 in enumerate(pauli_ops):
                if i == 0 and j == 0:
                    # Identity case
                    coeff = np.sqrt(1 - 15*p/16)
                else:
                    # Error cases
                    coeff = np.sqrt(p/16)

                kraus_op = coeff * np.kron(p1, p2)
                kraus_ops.append(kraus_op)

        return kraus_ops

    def apply_to_density_matrix(self, rho: np.ndarray) -> np.ndarray:
        """Apply two-qubit depolarizing noise to density matrix"""
        kraus_ops = self.kraus_operators()
        result = np.zeros_like(rho, dtype=complex)

        for kraus in kraus_ops:
            result += kraus @ rho @ kraus.conj().T

        return result


class NoiseModel(BaseModel):
    """Comprehensive noise model for quantum circuits
    """

    model_config = {"arbitrary_types_allowed": True}

    single_qubit_error_rates: dict[str, float] = Field(
        default_factory=dict,
        description="Error rates for single-qubit gates"
    )

    two_qubit_error_rates: dict[str, float] = Field(
        default_factory=dict,
        description="Error rates for two-qubit gates"
    )

    readout_error_rates: dict[int, float] = Field(
        default_factory=dict,
        description="Readout error rates per qubit"
    )

    coherence_times: dict[str, dict[int, float]] = Field(
        default_factory=dict,
        description="T1 and T2 times per qubit"
    )

    crosstalk_matrix: np.ndarray | None = Field(
        default=None,
        description="Crosstalk coupling matrix"
    )

    def add_single_qubit_error(self, gate_name: str, error_rate: float) -> None:
        """Add single-qubit gate error rate"""
        self.single_qubit_error_rates[gate_name] = error_rate

    def add_two_qubit_error(self, gate_name: str, error_rate: float) -> None:
        """Add two-qubit gate error rate"""
        self.two_qubit_error_rates[gate_name] = error_rate

    def add_readout_error(self, qubit: int, error_rate: float) -> None:
        """Add readout error rate for qubit"""
        self.readout_error_rates[qubit] = error_rate

    def set_coherence_time(self, qubit: int, t1: float, t2: float) -> None:
        """Set T1 and T2 coherence times for qubit"""
        if "T1" not in self.coherence_times:
            self.coherence_times["T1"] = {}
        if "T2" not in self.coherence_times:
            self.coherence_times["T2"] = {}

        self.coherence_times["T1"][qubit] = t1
        self.coherence_times["T2"][qubit] = t2

    def apply_to_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Apply noise model to quantum circuit

        Args:
            circuit: Original circuit

        Returns:
            Noisy circuit with error channels

        """
        noisy_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_classical_bits)

        for gate in circuit.gates:
            # Add original gate
            noisy_circuit.gates.append(gate)

            # Add noise after gate
            self._add_gate_noise(noisy_circuit, gate)

        # Add measurements with readout errors
        for qubit, cbit in circuit.measurements:
            self._add_readout_noise(noisy_circuit, qubit, cbit)

        return noisy_circuit

    def _add_gate_noise(self, circuit: QuantumCircuit, gate: QuantumGate) -> None:
        """Add noise after a gate operation"""
        if len(gate.qubits) == 1:
            # Single-qubit gate noise
            error_rate = self.single_qubit_error_rates.get(gate.name, 0.0)
            if error_rate > 0:
                qubit = gate.qubits[0]
                # Add depolarizing noise (simplified)
                circuit.gates.append(
                    QuantumGate(name="DEPOL", qubits=[qubit], parameters=[error_rate])
                )

        elif len(gate.qubits) == 2:
            # Two-qubit gate noise
            error_rate = self.two_qubit_error_rates.get(gate.name, 0.0)
            if error_rate > 0:
                qubits = gate.qubits
                circuit.gates.append(
                    QuantumGate(name="DEPOL2", qubits=qubits, parameters=[error_rate])
                )

    def _add_readout_noise(self, circuit: QuantumCircuit, qubit: int, cbit: int) -> None:
        """Add readout error to measurement"""
        error_rate = self.readout_error_rates.get(qubit, 0.0)

        # Add measurement with potential readout error
        if error_rate > 0:
            circuit.gates.append(
                QuantumGate(name="READOUT_ERROR", qubits=[qubit], parameters=[error_rate])
            )

        circuit.measure(qubit, cbit)

    @classmethod
    def from_device_properties(
        cls,
        device_props: dict[str, Any]
    ) -> "NoiseModel":
        """Create noise model from device properties

        Args:
            device_props: Device property dictionary

        Returns:
            Noise model based on device properties

        """
        noise_model = cls()

        # Extract gate error rates
        if "gates" in device_props:
            for gate_info in device_props["gates"]:
                gate_name = gate_info.get("gate")
                error_rate = gate_info.get("error_rate", 0.0)
                qubits = gate_info.get("qubits", [])

                if len(qubits) == 1:
                    noise_model.add_single_qubit_error(gate_name, error_rate)
                elif len(qubits) == 2:
                    noise_model.add_two_qubit_error(gate_name, error_rate)

        # Extract readout errors
        if "readout_errors" in device_props:
            for qubit, error_rate in enumerate(device_props["readout_errors"]):
                noise_model.add_readout_error(qubit, error_rate)

        # Extract coherence times
        if "coherence_times" in device_props:
            t1_times = device_props["coherence_times"].get("T1", [])
            t2_times = device_props["coherence_times"].get("T2", [])

            for qubit, (t1, t2) in enumerate(zip(t1_times, t2_times, strict=False)):
                noise_model.set_coherence_time(qubit, t1, t2)

        return noise_model

    @classmethod
    def ideal(cls) -> "NoiseModel":
        """Create ideal (noiseless) noise model"""
        return cls()

    @classmethod
    def basic_device_noise(
        cls,
        single_qubit_error: float = 1e-3,
        two_qubit_error: float = 1e-2,
        readout_error: float = 1e-2
    ) -> "NoiseModel":
        """Create basic device noise model

        Args:
            single_qubit_error: Single-qubit gate error rate
            two_qubit_error: Two-qubit gate error rate
            readout_error: Readout error rate

        Returns:
            Basic noise model

        """
        noise_model = cls()

        # Common single-qubit gates
        for gate in ["H", "X", "Y", "Z", "RX", "RY", "RZ", "U"]:
            noise_model.add_single_qubit_error(gate, single_qubit_error)

        # Common two-qubit gates
        for gate in ["CNOT", "CZ", "SWAP"]:
            noise_model.add_two_qubit_error(gate, two_qubit_error)

        return noise_model


class QuantumErrorCorrection:
    """Quantum error correction codes and syndromes
    """

    @staticmethod
    def three_qubit_bit_flip_code() -> dict[str, Any]:
        """Three-qubit repetition code for bit flip errors

        Returns:
            Code properties and circuits

        """
        # Encoding circuit: |0⟩ → |000⟩, |1⟩ → |111⟩
        encoding_circuit = QuantumCircuit(3)
        encoding_circuit.cnot(0, 1)
        encoding_circuit.cnot(0, 2)

        # Syndrome measurement circuit
        syndrome_circuit = QuantumCircuit(5, 2)  # 3 data + 2 ancilla qubits
        # Measure Z₀Z₁ and Z₁Z₂
        syndrome_circuit.cnot(0, 3)
        syndrome_circuit.cnot(1, 3)
        syndrome_circuit.cnot(1, 4)
        syndrome_circuit.cnot(2, 4)
        syndrome_circuit.measure(3, 0)
        syndrome_circuit.measure(4, 1)

        # Error correction lookup table
        correction_table = {
            "00": None,     # No error
            "10": "X0",     # Error on qubit 0
            "11": "X1",     # Error on qubit 1
            "01": "X2"      # Error on qubit 2
        }

        return {
            "encoding_circuit": encoding_circuit,
            "syndrome_circuit": syndrome_circuit,
            "correction_table": correction_table,
            "code_distance": 3,
            "correctable_errors": 1
        }

    @staticmethod
    def three_qubit_phase_flip_code() -> dict[str, Any]:
        """Three-qubit code for phase flip errors

        Returns:
            Code properties and circuits

        """
        # Encoding: |+⟩ → |+++⟩, |-⟩ → |---⟩
        encoding_circuit = QuantumCircuit(3)
        encoding_circuit.h(0)
        encoding_circuit.h(1)
        encoding_circuit.h(2)
        encoding_circuit.cnot(0, 1)
        encoding_circuit.cnot(0, 2)
        encoding_circuit.h(0)
        encoding_circuit.h(1)
        encoding_circuit.h(2)

        # Syndrome measurement in X basis
        syndrome_circuit = QuantumCircuit(5, 2)
        # Rotate to X basis
        for i in range(3):
            syndrome_circuit.h(i)

        # Measure X₀X₁ and X₁X₂
        syndrome_circuit.cnot(0, 3)
        syndrome_circuit.cnot(1, 3)
        syndrome_circuit.cnot(1, 4)
        syndrome_circuit.cnot(2, 4)
        syndrome_circuit.measure(3, 0)
        syndrome_circuit.measure(4, 1)

        correction_table = {
            "00": None,     # No error
            "10": "Z0",     # Error on qubit 0
            "11": "Z1",     # Error on qubit 1
            "01": "Z2"      # Error on qubit 2
        }

        return {
            "encoding_circuit": encoding_circuit,
            "syndrome_circuit": syndrome_circuit,
            "correction_table": correction_table,
            "code_distance": 3,
            "correctable_errors": 1
        }

    @staticmethod
    def nine_qubit_shor_code() -> dict[str, Any]:
        """Nine-qubit Shor code (corrects arbitrary single-qubit errors)

        Returns:
            Code properties and circuits

        """
        # Encoding circuit
        encoding_circuit = QuantumCircuit(9)

        # First level: bit flip encoding
        encoding_circuit.cnot(0, 3)
        encoding_circuit.cnot(0, 6)

        # Second level: phase flip encoding within each block
        for block_start in [0, 3, 6]:
            encoding_circuit.h(block_start)
            encoding_circuit.h(block_start + 1)
            encoding_circuit.h(block_start + 2)
            encoding_circuit.cnot(block_start, block_start + 1)
            encoding_circuit.cnot(block_start, block_start + 2)
            encoding_circuit.h(block_start)
            encoding_circuit.h(block_start + 1)
            encoding_circuit.h(block_start + 2)

        # Syndrome measurement circuit (simplified)
        syndrome_circuit = QuantumCircuit(15, 6)  # 9 data + 6 ancilla

        correction_table = {}  # Would need full syndrome table

        return {
            "encoding_circuit": encoding_circuit,
            "syndrome_circuit": syndrome_circuit,
            "correction_table": correction_table,
            "code_distance": 3,
            "correctable_errors": 1,
            "logical_qubits": 1,
            "physical_qubits": 9
        }

    @staticmethod
    def steane_code() -> dict[str, Any]:
        """7-qubit Steane code

        Returns:
            Code properties

        """
        # Generator matrix for Steane code
        generator_matrix = np.array([
            [1, 0, 0, 1, 0, 1, 1],
            [0, 1, 0, 1, 1, 0, 1],
            [0, 0, 1, 0, 1, 1, 1]
        ])

        # Parity check matrix
        parity_check_matrix = np.array([
            [1, 1, 0, 1, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 0],
            [1, 0, 1, 0, 0, 1, 0],
            [1, 1, 1, 0, 0, 0, 1]
        ])

        return {
            "generator_matrix": generator_matrix,
            "parity_check_matrix": parity_check_matrix,
            "code_distance": 3,
            "correctable_errors": 1,
            "logical_qubits": 1,
            "physical_qubits": 7
        }

    @staticmethod
    def decode_syndrome(syndrome: str, correction_table: dict[str, str]) -> str | None:
        """Decode error syndrome to determine correction

        Args:
            syndrome: Measured syndrome bitstring
            correction_table: Syndrome to correction mapping

        Returns:
            Correction operation or None if no error

        """
        return correction_table.get(syndrome)

    @staticmethod
    def apply_correction(
        circuit: QuantumCircuit,
        correction: str
    ) -> QuantumCircuit:
        """Apply error correction to circuit

        Args:
            circuit: Circuit to correct
            correction: Correction operation (e.g., "X0", "Z2")

        Returns:
            Corrected circuit

        """
        if correction is None:
            return circuit

        corrected_circuit = circuit.copy()

        # Parse correction operation
        if correction.startswith("X"):
            qubit = int(correction[1:])
            corrected_circuit.x(qubit)
        elif correction.startswith("Z"):
            qubit = int(correction[1:])
            corrected_circuit.z(qubit)
        elif correction.startswith("Y"):
            qubit = int(correction[1:])
            corrected_circuit.y(qubit)

        return corrected_circuit


class ErrorMitigation:
    """Quantum error mitigation techniques
    """

    @staticmethod
    def randomized_compiling(
        circuit: QuantumCircuit,
        num_random_circuits: int = 10,
        random_seed: int | None = None
    ) -> list[QuantumCircuit]:
        """Generate randomly compiled circuits for error mitigation

        Args:
            circuit: Original circuit
            num_random_circuits: Number of random compilations
            random_seed: Random seed for reproducibility

        Returns:
            List of randomly compiled circuits

        """
        if random_seed is not None:
            np.random.seed(random_seed)

        random_circuits = []

        for _ in range(num_random_circuits):
            # Create copy of circuit
            random_circuit = circuit.copy()

            # Apply random Pauli twirling
            for gate in random_circuit.gates:
                if len(gate.qubits) == 1:
                    # Add random Pauli before and after
                    gate.qubits[0]

                    # Random Pauli group element
                    pauli_choice = np.random.choice(['I', 'X', 'Y', 'Z'])

                    if pauli_choice == 'X':
                        # Add X before gate, X after gate (cancels out)
                        pass  # Would add X gates in practice
                    elif pauli_choice == 'Y':
                        pass  # Would add Y gates
                    elif pauli_choice == 'Z':
                        pass  # Would add Z gates

            random_circuits.append(random_circuit)

        return random_circuits

    @staticmethod
    def clifford_data_regression(
        noisy_results: list[float],
        clifford_expectation_values: list[float]
    ) -> float:
        """Perform Clifford data regression for error mitigation

        Args:
            noisy_results: Noisy measurement results
            clifford_expectation_values: Expected values from Clifford simulation

        Returns:
            Error-mitigated expectation value

        """
        # Simple linear regression to extrapolate ideal result
        # In practice, would use more sophisticated regression

        if len(noisy_results) != len(clifford_expectation_values):
            raise ValueError("Result arrays must have same length")

        # Fit linear model: noisy = a * ideal + b
        ideal_values = np.array(clifford_expectation_values)
        noisy_values = np.array(noisy_results)

        # Least squares fit
        A = np.vstack([ideal_values, np.ones(len(ideal_values))]).T
        slope, intercept = np.linalg.lstsq(A, noisy_values, rcond=None)[0]

        # Extrapolate: if noisy = a * ideal + b, then ideal = (noisy - b) / a
        # But we want to correct the average
        corrected_average = (np.mean(noisy_values) - intercept) / slope if slope != 0 else np.mean(noisy_values)

        return float(corrected_average)

    @staticmethod
    def symmetry_verification(
        circuit: QuantumCircuit,
        symmetry_generators: list[PauliString]
    ) -> dict[str, Any]:
        """Verify circuit preserves expected symmetries

        Args:
            circuit: Quantum circuit
            symmetry_generators: List of Pauli symmetries to check

        Returns:
            Symmetry verification results

        """
        verification_results = {}

        for i, generator in enumerate(symmetry_generators):
            # In practice, would measure commutator [H, generator]
            # For now, return placeholder
            verification_results[f"symmetry_{i}"] = {
                "generator": str(generator),
                "violation": 0.0,  # Placeholder
                "verified": True
            }

        return verification_results
