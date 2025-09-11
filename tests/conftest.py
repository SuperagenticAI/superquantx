"""
Test configuration and fixtures for SuperQuantX tests
"""

from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from superquantx.circuits import QuantumCircuit
from superquantx.client import SuperQuantXClient
from superquantx.gates import Hamiltonian, PauliString
from superquantx.measurements import MeasurementResult
from superquantx.noise import NoiseModel


@pytest.fixture
def mock_client():
    """Mock SuperQuantX client for testing"""
    client = Mock(spec=SuperQuantXClient)

    # Mock job submission
    mock_job = MagicMock()
    mock_job.job_id = "test-job-123"
    mock_job.status = "completed"

    client.submit_job_sync.return_value = mock_job
    client.wait_for_job_sync.return_value = mock_job

    # Mock job results
    mock_job.results = {
        "counts": {"00": 512, "11": 512},
        "shots": 1024
    }

    return client


@pytest.fixture
def sample_circuit():
    """Simple quantum circuit for testing"""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cnot(0, 1)
    circuit.measure_all()
    return circuit


@pytest.fixture
def sample_hamiltonian():
    """Sample Hamiltonian for testing"""
    pauli_strings = [
        PauliString("ZZ", -1.0),
        PauliString("XX", 0.5),
        PauliString("YY", 0.5)
    ]
    return Hamiltonian(pauli_strings)


@pytest.fixture
def sample_measurement_result():
    """Sample measurement result for testing"""
    return MeasurementResult(
        counts={"00": 256, "01": 256, "10": 256, "11": 256},
        shots=1024,
        metadata={"backend": "simulator"}
    )


@pytest.fixture
def basic_noise_model():
    """Basic noise model for testing"""
    noise_model = NoiseModel()
    noise_model.add_single_qubit_error("H", 0.001)
    noise_model.add_two_qubit_error("CNOT", 0.01)
    return noise_model


@pytest.fixture
def random_data():
    """Random test data"""
    np.random.seed(42)  # For reproducibility
    X = np.random.rand(20, 4)
    y = np.random.randint(0, 2, 20)
    return X, y


@pytest.fixture
def quantum_feature_data():
    """Quantum-suitable feature data"""
    np.random.seed(42)
    # Features normalized to [0, 2π] for quantum encoding
    X = np.random.uniform(0, 2*np.pi, (10, 3))
    y = np.random.choice([0, 1], 10)
    return X, y
