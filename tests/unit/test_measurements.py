"""
Unit tests for quantum measurements
"""


import numpy as np
import pytest

from superquantx.circuits import QuantumCircuit
from superquantx.measurements import (
    MeasurementResult,
    QuantumMeasurement,
    ResultAnalyzer,
)


class TestMeasurementResult:
    """Test MeasurementResult class"""

    def test_measurement_result_creation(self):
        """Test basic measurement result creation"""
        result = MeasurementResult(
            counts={"00": 256, "01": 256, "10": 256, "11": 256},
            shots=1024,
            metadata={"backend": "simulator"}
        )

        assert result.counts == {"00": 256, "01": 256, "10": 256, "11": 256}
        assert result.shots == 1024
        assert result.metadata == {"backend": "simulator"}
        assert result.memory is None

    def test_measurement_result_with_memory(self):
        """Test measurement result with memory"""
        memory = ["00", "01", "00", "11"]
        result = MeasurementResult(
            counts={"00": 2, "01": 1, "11": 1},
            shots=4,
            memory=memory
        )

        assert result.memory == memory

    def test_probabilities_property(self):
        """Test probabilities calculation"""
        result = MeasurementResult(
            counts={"00": 500, "01": 300, "10": 200, "11": 24},
            shots=1024
        )

        probabilities = result.probabilities

        assert probabilities["00"] == 500 / 1024
        assert probabilities["01"] == 300 / 1024
        assert probabilities["10"] == 200 / 1024
        assert probabilities["11"] == 24 / 1024

        # Probabilities should sum to 1
        assert abs(sum(probabilities.values()) - 1.0) < 1e-10

    def test_most_frequent_property(self):
        """Test most frequent outcome"""
        result = MeasurementResult(
            counts={"00": 100, "01": 500, "10": 200, "11": 224},
            shots=1024
        )

        most_frequent = result.most_frequent
        assert most_frequent == ("01", 500)

    def test_marginal_counts(self):
        """Test marginal counts calculation"""
        result = MeasurementResult(
            counts={"00": 256, "01": 256, "10": 256, "11": 256},
            shots=1024
        )

        # Marginal on first qubit (qubit 0)
        marginal_0 = result.marginal_counts([0])
        assert marginal_0 == {"0": 512, "1": 512}  # 00,01 -> 0; 10,11 -> 1

        # Marginal on second qubit (qubit 1)
        marginal_1 = result.marginal_counts([1])
        assert marginal_1 == {"0": 512, "1": 512}  # 00,10 -> 0; 01,11 -> 1

        # Marginal on both qubits (should be same as original)
        marginal_both = result.marginal_counts([0, 1])
        assert marginal_both == result.counts

    def test_expectation_value_z_basis(self):
        """Test expectation value calculation for Z-basis measurements"""
        result = MeasurementResult(
            counts={"00": 300, "01": 200, "10": 200, "11": 324},
            shots=1024
        )

        # Expectation value of Z on first qubit
        exp_z0 = result.expectation_value("Z")
        # P(0) = (300 + 200) / 1024 = 500/1024, P(1) = (200 + 324) / 1024 = 524/1024
        # ⟨Z⟩ = P(0) - P(1) = 500/1024 - 524/1024 = -24/1024
        expected = (300 + 200 - 200 - 324) / 1024
        assert abs(exp_z0 - expected) < 1e-10

        # Expectation value of ZZ
        exp_zz = result.expectation_value("ZZ")
        # ZZ: 00 -> +1, 01 -> -1, 10 -> -1, 11 -> +1
        expected_zz = (300 - 200 - 200 + 324) / 1024
        assert abs(exp_zz - expected_zz) < 1e-10

    def test_expectation_value_invalid_observable(self):
        """Test expectation value with invalid observable"""
        result = MeasurementResult(
            counts={"00": 512, "11": 512},
            shots=1024
        )

        # X and Y measurements require basis rotation
        with pytest.raises(ValueError):
            result.expectation_value("X")

        with pytest.raises(ValueError):
            result.expectation_value("Y")

    def test_entropy_calculation(self):
        """Test measurement entropy calculation"""
        # Uniform distribution - maximum entropy
        result_uniform = MeasurementResult(
            counts={"00": 256, "01": 256, "10": 256, "11": 256},
            shots=1024
        )

        entropy_uniform = result_uniform.entropy()
        expected_uniform = -4 * (0.25 * np.log2(0.25))  # 2 bits
        assert abs(entropy_uniform - expected_uniform) < 1e-10

        # Deterministic outcome - minimum entropy
        result_deterministic = MeasurementResult(
            counts={"00": 1024},
            shots=1024
        )

        entropy_deterministic = result_deterministic.entropy()
        assert abs(entropy_deterministic - 0.0) < 1e-10

    def test_result_addition(self):
        """Test adding measurement results"""
        result1 = MeasurementResult(
            counts={"00": 256, "11": 256},
            shots=512,
            memory=["00", "11"],
            metadata={"backend": "sim1"}
        )

        result2 = MeasurementResult(
            counts={"00": 200, "01": 312},
            shots=512,
            memory=["00", "01"],
            metadata={"backend": "sim2"}
        )

        combined = result1 + result2

        assert combined.counts == {"00": 456, "11": 256, "01": 312}
        assert combined.shots == 1024
        assert combined.memory == ["00", "11", "00", "01"]
        assert "backend" in combined.metadata  # Should combine metadata

    def test_serialization(self):
        """Test result serialization"""
        original = MeasurementResult(
            counts={"00": 512, "11": 512},
            shots=1024,
            memory=["00", "11", "00", "11"],
            metadata={"backend": "simulator", "job_id": "test-123"}
        )

        # Test to_dict
        result_dict = original.to_dict()
        assert result_dict["counts"] == original.counts
        assert result_dict["shots"] == original.shots
        assert result_dict["memory"] == original.memory
        assert result_dict["metadata"] == original.metadata

        # Test from_dict
        restored = MeasurementResult.from_dict(result_dict)
        assert restored.counts == original.counts
        assert restored.shots == original.shots
        assert restored.memory == original.memory
        assert restored.metadata == original.metadata

        # Test JSON serialization
        json_str = original.to_json()
        json_restored = MeasurementResult.from_json(json_str)
        assert json_restored.counts == original.counts
        assert json_restored.shots == original.shots


class TestQuantumMeasurement:
    """Test QuantumMeasurement class"""

    def test_measurement_initialization(self):
        """Test measurement system initialization"""
        measurement = QuantumMeasurement(backend="simulator")

        assert measurement.backend == "simulator"
        assert len(measurement.measurement_history) == 0

    def test_measure_circuit_simulation(self, sample_circuit):
        """Test circuit measurement with simulation"""
        measurement = QuantumMeasurement(backend="simulator")

        result = measurement.measure_circuit(sample_circuit, shots=1024)

        assert isinstance(result, MeasurementResult)
        assert result.shots == 1024
        assert sum(result.counts.values()) == 1024
        assert result.metadata["backend"] == "simulator"
        assert len(measurement.measurement_history) == 1

    def test_measure_circuit_with_memory(self, sample_circuit):
        """Test circuit measurement with memory storage"""
        measurement = QuantumMeasurement(backend="simulator")

        result = measurement.measure_circuit(sample_circuit, shots=100, memory=True)

        assert result.memory is not None
        assert len(result.memory) == 100
        # All memory entries should be valid bitstrings
        for memory_entry in result.memory:
            assert isinstance(memory_entry, str)
            assert all(bit in '01' for bit in memory_entry)

    def test_measure_observable(self, sample_circuit):
        """Test observable measurement"""
        measurement = QuantumMeasurement(backend="simulator")

        # Remove measurements from sample circuit for observable measurement
        circuit_no_measure = QuantumCircuit(sample_circuit.num_qubits)
        circuit_no_measure.gates = sample_circuit.gates.copy()

        expectation = measurement.measure_observable(
            circuit_no_measure,
            observable="ZZ",
            shots=1024
        )

        assert isinstance(expectation, float)
        assert -1 <= expectation <= 1  # Expectation values are bounded

    def test_tomography_measurements(self, sample_circuit):
        """Test quantum state tomography measurements"""
        measurement = QuantumMeasurement(backend="simulator")

        # Remove measurements from sample circuit
        circuit_no_measure = QuantumCircuit(sample_circuit.num_qubits)
        circuit_no_measure.gates = sample_circuit.gates.copy()

        tomography_results = measurement.tomography_measurements(
            circuit_no_measure,
            qubits=[0, 1],
            shots_per_measurement=100
        )

        # Should have measurements for all non-identity Pauli combinations
        # For 2 qubits: XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ
        expected_measurements = [
            pauli1 + pauli2
            for pauli1 in ['X', 'Y', 'Z']
            for pauli2 in ['X', 'Y', 'Z']
        ]

        assert len(tomography_results) == len(expected_measurements)

        for pauli_string in expected_measurements:
            assert pauli_string in tomography_results
            assert isinstance(tomography_results[pauli_string], MeasurementResult)

    def test_fidelity_calculation(self):
        """Test quantum state fidelity calculation"""
        measurement = QuantumMeasurement()

        # Test with state vectors
        state1 = np.array([1, 0], dtype=complex)  # |0⟩
        state2 = np.array([0, 1], dtype=complex)  # |1⟩

        fidelity_orthogonal = measurement.fidelity(state1, state2)
        assert abs(fidelity_orthogonal - 0.0) < 1e-10

        # Test with identical states
        fidelity_identical = measurement.fidelity(state1, state1)
        assert abs(fidelity_identical - 1.0) < 1e-10

        # Test with density matrices
        rho1 = np.outer(state1, state1.conj())
        rho2 = np.outer(state2, state2.conj())

        fidelity_dm = measurement.fidelity(rho1, rho2)
        assert abs(fidelity_dm - 0.0) < 1e-10

    def test_trace_distance(self):
        """Test trace distance calculation"""
        measurement = QuantumMeasurement()

        state1 = np.array([1, 0], dtype=complex)  # |0⟩
        state2 = np.array([0, 1], dtype=complex)  # |1⟩

        # Orthogonal states have maximum trace distance
        distance_orthogonal = measurement.trace_distance(state1, state2)
        assert abs(distance_orthogonal - 1.0) < 1e-10

        # Identical states have zero trace distance
        distance_identical = measurement.trace_distance(state1, state1)
        assert abs(distance_identical - 0.0) < 1e-10

    def test_quantum_volume_benchmark(self):
        """Test quantum volume benchmark"""
        measurement = QuantumMeasurement(backend="simulator")

        qv_results = measurement.quantum_volume(
            num_qubits=2,
            depth=2,
            trials=5,
            shots_per_trial=100
        )

        assert qv_results["num_qubits"] == 2
        assert qv_results["depth"] == 2
        assert qv_results["trials"] == 5
        assert 0 <= qv_results["successes"] <= 5
        assert 0 <= qv_results["success_rate"] <= 1
        assert isinstance(qv_results["passed"], bool)

        # If passed, quantum volume should be 2^num_qubits
        if qv_results["passed"]:
            assert qv_results["quantum_volume"] == 4  # 2^2
        else:
            assert qv_results["quantum_volume"] == 0


class TestResultAnalyzer:
    """Test ResultAnalyzer class"""

    def test_compare_results(self):
        """Test result comparison metrics"""
        result1 = MeasurementResult(
            counts={"00": 500, "11": 524},
            shots=1024
        )

        result2 = MeasurementResult(
            counts={"00": 400, "01": 100, "10": 100, "11": 424},
            shots=1024
        )

        comparison = ResultAnalyzer.compare_results(result1, result2)

        assert "hellinger_distance" in comparison
        assert "kl_divergence" in comparison
        assert "total_variation_distance" in comparison

        # All metrics should be non-negative
        assert comparison["hellinger_distance"] >= 0
        assert comparison["kl_divergence"] >= 0
        assert comparison["total_variation_distance"] >= 0

        # Hellinger and TV distances should be <= 1
        assert comparison["hellinger_distance"] <= 1
        assert comparison["total_variation_distance"] <= 1

    def test_zero_noise_extrapolation(self):
        """Test zero-noise extrapolation error mitigation"""
        # Create mock results with decreasing expectation values
        noise_levels = [1.0, 2.0, 3.0]

        results = []
        for noise in noise_levels:
            # Simulate decreasing expectation with noise
            expectation = 1.0 * np.exp(-0.1 * noise)  # Exponential decay

            # Create counts that give this expectation value
            shots = 1000
            prob_0 = (expectation + 1) / 2  # Convert from [-1,1] to [0,1]
            count_0 = int(prob_0 * shots)
            count_1 = shots - count_0

            result = MeasurementResult(
                counts={"0": count_0, "1": count_1},
                shots=shots
            )
            results.append(result)

        zero_noise_value, fit_info = ResultAnalyzer.error_mitigation_zero_noise_extrapolation(
            noise_levels, results, observable="Z"
        )

        assert isinstance(zero_noise_value, float)
        assert isinstance(fit_info, dict)

        # Zero noise value should be closer to ideal (1.0) than noisy values
        noisy_expectations = [r.expectation_value("Z") for r in results]
        assert zero_noise_value >= max(noisy_expectations)  # Should extrapolate upward

    def test_readout_error_mitigation(self):
        """Test readout error mitigation"""
        # Mock calibration results
        calibration_results = {
            "0": MeasurementResult(counts={"0": 950, "1": 50}, shots=1000),  # 5% error when preparing |0⟩
            "1": MeasurementResult(counts={"0": 70, "1": 930}, shots=1000)   # 7% error when preparing |1⟩
        }

        # Mock measurement result with readout errors
        measurement_result = MeasurementResult(
            counts={"0": 600, "1": 400},
            shots=1000
        )

        corrected_result = ResultAnalyzer.readout_error_mitigation(
            calibration_results, measurement_result
        )

        assert isinstance(corrected_result, MeasurementResult)
        assert corrected_result.shots == measurement_result.shots
        assert corrected_result.metadata.get("error_mitigated") is True
