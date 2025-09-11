"""
Integration tests for quantum algorithms
"""


import numpy as np
import pytest

from superquantx.algorithms import (
    QAOA,
    VQE,
    QuantumNeuralNetwork,
    create_vqe_for_molecule,
)
from superquantx.circuits import QuantumCircuit
from superquantx.gates import Hamiltonian


class TestVQEIntegration:
    """Integration tests for VQE algorithm"""

    def test_vqe_h2_molecule(self, mock_client):
        """Test VQE for H2 molecule"""
        # Create simple H2 Hamiltonian
        hamiltonian = Hamiltonian.from_dict({
            "ZZ": -1.0523732,
            "ZI": -0.39793742,
            "IZ": -0.39793742,
            "XX": -0.01128010,
            "YY": 0.01128010
        })

        def simple_ansatz(params):
            circuit = QuantumCircuit(2)
            circuit.ry(params[0], 0)
            circuit.ry(params[1], 1)
            circuit.cnot(0, 1)
            return circuit

        # Test without client (simulation mode)
        vqe = VQE(hamiltonian, simple_ansatz, client=None)

        initial_params = np.array([0.1, 0.1])
        results = vqe.run(initial_params)

        assert "optimal_energy" in results
        assert "optimal_parameters" in results
        assert "optimization_history" in results
        assert "converged" in results
        assert isinstance(results["optimal_energy"], float)
        assert isinstance(results["optimal_parameters"], np.ndarray)

    def test_vqe_with_mock_client(self, mock_client):
        """Test VQE with mock client"""
        hamiltonian = Hamiltonian.from_dict({"ZZ": -1.0})

        def simple_ansatz(params):
            circuit = QuantumCircuit(2)
            circuit.ry(params[0], 0)
            return circuit

        vqe = VQE(hamiltonian, simple_ansatz, client=mock_client)

        # Mock client should return consistent results
        results = vqe.run(np.array([0.1]))

        assert isinstance(results, dict)
        assert mock_client.submit_job_sync.called
        assert mock_client.wait_for_job_sync.called

    def test_vqe_molecule_factory(self):
        """Test VQE molecule factory function"""
        vqe = create_vqe_for_molecule("H2", client=None)

        assert isinstance(vqe, VQE)
        assert vqe.hamiltonian.num_qubits == 2

        # Test unsupported molecule
        with pytest.raises(ValueError):
            create_vqe_for_molecule("unknown_molecule")


class TestQAOAIntegration:
    """Integration tests for QAOA algorithm"""

    def test_qaoa_max_cut(self):
        """Test QAOA for Max-Cut problem"""
        # Simple triangle graph: edges (0,1), (1,2), (0,2)
        from superquantx.algorithms import create_qaoa_for_max_cut

        graph_edges = [(0, 1), (1, 2), (0, 2)]
        qaoa = create_qaoa_for_max_cut(graph_edges, num_nodes=3, p=1, client=None)

        assert isinstance(qaoa, QAOA)
        assert qaoa.cost_hamiltonian.num_qubits == 3
        assert qaoa.p == 1

    def test_qaoa_circuit_creation(self):
        """Test QAOA circuit creation"""
        cost_hamiltonian = Hamiltonian.from_dict({"ZZ": 0.5, "ZI": -0.5})
        qaoa = QAOA(cost_hamiltonian, p=2, client=None)

        # Parameters for 2 layers: [gamma1, beta1, gamma2, beta2]
        parameters = np.array([0.1, 0.2, 0.3, 0.4])
        circuit = qaoa.create_qaoa_circuit(parameters)

        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 2  # From Hamiltonian
        assert len(circuit.gates) > 0  # Should have gates

        # Should start with Hadamards for equal superposition
        h_gates = [g for g in circuit.gates if g.name == "H"]
        assert len(h_gates) == 2  # One H per qubit

    def test_qaoa_optimization(self):
        """Test QAOA optimization"""
        cost_hamiltonian = Hamiltonian.from_dict({"Z": 1.0})
        qaoa = QAOA(cost_hamiltonian, p=1, client=None)

        results = qaoa.run(initial_parameters=np.array([0.1, 0.2]))

        assert "optimal_energy" in results
        assert "optimal_parameters" in results
        assert "optimal_circuit" in results
        assert "converged" in results

        optimal_circuit = results["optimal_circuit"]
        assert isinstance(optimal_circuit, QuantumCircuit)


class TestQuantumNeuralNetworkIntegration:
    """Integration tests for Quantum Neural Network"""

    def test_qnn_binary_classification(self, mock_client, quantum_feature_data):
        """Test QNN for binary classification"""
        X, y = quantum_feature_data

        qnn = QuantumNeuralNetwork(
            num_qubits=3,
            num_layers=2,
            client=None  # Use simulation mode
        )

        # Fit the model
        qnn.fit(X, y)

        assert qnn.is_fitted_
        assert qnn.parameters is not None
        assert len(qnn.parameters) == qnn.num_parameters

        # Make predictions
        predictions = qnn.predict(X)
        assert len(predictions) == len(y)
        assert all(pred in [-1, 1] for pred in predictions)

    def test_qnn_circuit_creation(self):
        """Test QNN circuit creation"""
        qnn = QuantumNeuralNetwork(num_qubits=2, num_layers=1, client=None)

        parameters = np.random.uniform(0, 2*np.pi, qnn.num_parameters)
        x = np.array([0.5, 1.5])  # Input data

        circuit = qnn.create_ansatz(parameters, x)

        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 2
        assert len(circuit.gates) > 0

    def test_qnn_with_mock_client(self, mock_client, quantum_feature_data):
        """Test QNN with mock client"""
        X, y = quantum_feature_data

        qnn = QuantumNeuralNetwork(
            num_qubits=2,
            num_layers=1,
            client=mock_client
        )

        # Should work with mock client
        qnn.fit(X, y)
        predictions = qnn.predict(X)

        assert len(predictions) == len(y)
        assert mock_client.submit_job_sync.called


class TestAlgorithmComposition:
    """Test composition of multiple algorithms"""

    def test_vqe_then_qaoa(self):
        """Test using VQE result to initialize QAOA"""
        # Use same Hamiltonian for both
        hamiltonian = Hamiltonian.from_dict({"ZZ": 1.0, "XX": -0.5})

        # Run VQE first
        def vqe_ansatz(params):
            circuit = QuantumCircuit(2)
            circuit.ry(params[0], 0)
            circuit.ry(params[1], 1)
            return circuit

        vqe = VQE(hamiltonian, vqe_ansatz, client=None)
        vqe_results = vqe.run(np.array([0.1, 0.2]))

        # Use VQE energy as reference for QAOA
        qaoa = QAOA(hamiltonian, p=1, client=None)
        qaoa_results = qaoa.run(np.array([0.1, 0.2]))

        # Both should provide energy estimates
        assert "optimal_energy" in vqe_results
        assert "optimal_energy" in qaoa_results

        # QAOA might find better or similar solution
        vqe_energy = vqe_results["optimal_energy"]
        qaoa_energy = qaoa_results["optimal_energy"]

        # Both should be reasonable energy values
        assert isinstance(vqe_energy, float)
        assert isinstance(qaoa_energy, float)

    def test_algorithm_parameter_estimation(self):
        """Test parameter counting for different algorithms"""
        hamiltonian = Hamiltonian.from_dict({"ZZ": 1.0})

        # VQE parameter estimation
        def ansatz(params):
            circuit = QuantumCircuit(2)
            for i in range(len(params)):
                circuit.ry(params[i], i % 2)
            return circuit

        vqe = VQE(hamiltonian, ansatz, client=None)
        estimated_params = vqe._estimate_parameter_count()

        # Should estimate based on number of qubits
        assert estimated_params == 4  # 2 qubits * 2

        # QAOA parameter count is deterministic
        qaoa = QAOA(hamiltonian, p=3, client=None)
        # QAOA has 2*p parameters
        assert qaoa.p == 3

        # QNN parameter count
        qnn = QuantumNeuralNetwork(num_qubits=3, num_layers=2, client=None)
        assert qnn.num_parameters == 3 * 2 * 3  # num_qubits * num_layers * 3
