"""
Unit tests for SuperQuantX backend implementations.

Tests the unified API across all quantum computing backends,
including the new AWS Braket, TKET/Quantinuum, and D-Wave Ocean backends.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

import superquantx as sqx
from superquantx.backends import (
    BACKEND_ALIASES,
    BACKEND_REGISTRY,
    check_backend_compatibility,
    get_backend,
    list_available_backends,
)


class TestBackendRegistry:
    """Test the backend registry and discovery system."""

    def test_backend_registry_structure(self):
        """Test that backend registry is properly structured."""
        assert isinstance(BACKEND_REGISTRY, dict)
        assert 'simulator' in BACKEND_REGISTRY
        assert 'auto' in BACKEND_REGISTRY

    def test_backend_aliases(self):
        """Test backend alias resolution."""
        assert isinstance(BACKEND_ALIASES, dict)

        # Test common aliases
        assert BACKEND_ALIASES['ibm'] == 'qiskit'
        assert BACKEND_ALIASES['google'] == 'cirq'
        assert BACKEND_ALIASES['aws'] == 'braket'
        assert BACKEND_ALIASES['h1'] == 'quantinuum'
        assert BACKEND_ALIASES['annealing'] == 'ocean'

    def test_list_available_backends(self):
        """Test backend availability listing."""
        backends = list_available_backends()
        assert isinstance(backends, dict)

        # Should always have simulator
        assert 'simulator' in backends
        assert backends['simulator']['available'] == True

    def test_check_backend_compatibility(self):
        """Test individual backend compatibility checking."""
        # Test known available backend
        compat = check_backend_compatibility('simulator')
        assert compat['compatible'] == True

        # Test unknown backend
        compat = check_backend_compatibility('nonexistent_backend')
        assert compat['compatible'] == False


class TestBackendCreation:
    """Test backend creation and auto-selection."""

    def test_get_backend_simulator(self):
        """Test getting simulator backend."""
        backend = get_backend('simulator')
        assert backend is not None
        assert hasattr(backend, 'execute_circuit')

    def test_get_backend_auto_selection(self):
        """Test automatic backend selection."""
        backend = get_backend('auto')
        assert backend is not None

    def test_get_backend_with_alias(self):
        """Test backend creation with aliases."""
        # These should work if backends are available
        try:
            backend = get_backend('sim')  # -> simulator
            assert backend is not None
        except ImportError:
            pass  # Expected if dependencies not installed

    def test_get_backend_invalid(self):
        """Test invalid backend handling."""
        with pytest.raises(ValueError):
            get_backend('totally_invalid_backend')


class TestBraketBackend:
    """Test AWS Braket backend functionality."""

    @pytest.fixture
    def mock_braket(self):
        """Mock Braket SDK components."""
        with patch('superquantx.backends.braket_backend.BRAKET_AVAILABLE', True), \
             patch('superquantx.backends.braket_backend.BraketCircuit') as mock_circuit, \
             patch('superquantx.backends.braket_backend.LocalSimulator') as mock_sim:

            mock_circuit.return_value = Mock()
            mock_sim.return_value = Mock()
            yield mock_circuit, mock_sim

    def test_braket_backend_creation(self, mock_braket):
        """Test Braket backend creation."""
        try:
            from superquantx.backends.braket_backend import BraketBackend
            backend = BraketBackend(device='local:braket/braket_sv')
            assert backend.device_name == 'local:braket/braket_sv'
        except ImportError:
            pytest.skip("Braket not available")

    def test_braket_circuit_creation(self, mock_braket):
        """Test Braket circuit creation."""
        try:
            from superquantx.backends.braket_backend import BraketBackend
            backend = BraketBackend()
            circuit = backend.create_circuit(4)
            assert circuit is not None
        except ImportError:
            pytest.skip("Braket not available")

    def test_braket_gate_mapping(self, mock_braket):
        """Test Braket gate operations."""
        try:
            from superquantx.backends.braket_backend import BraketBackend
            backend = BraketBackend()
            circuit = backend.create_circuit(2)

            # Test basic gates
            backend.add_gate(circuit, 'h', 0)
            backend.add_gate(circuit, 'cnot', [0, 1])
            backend.add_gate(circuit, 'rx', 0, [np.pi/4])
        except ImportError:
            pytest.skip("Braket not available")


class TestTKETBackend:
    """Test TKET/Quantinuum backend functionality."""

    @pytest.fixture
    def mock_tket(self):
        """Mock TKET components."""
        with patch('superquantx.backends.tket_backend.TKET_AVAILABLE', True), \
             patch('superquantx.backends.tket_backend.Circuit') as mock_circuit, \
             patch('superquantx.backends.tket_backend.OpType') as mock_optype:

            mock_circuit.return_value = Mock()
            mock_circuit.return_value.qubits = [Mock() for _ in range(10)]
            mock_circuit.return_value.bits = [Mock() for _ in range(10)]
            mock_circuit.return_value.n_qubits = 4
            mock_circuit.return_value.n_bits = 0
            mock_circuit.return_value.n_gates = 5

            yield mock_circuit, mock_optype

    def test_tket_backend_creation(self, mock_tket):
        """Test TKET backend creation."""
        try:
            from superquantx.backends.tket_backend import TKETBackend
            backend = TKETBackend(device='aer_simulator')
            assert backend.device_name == 'aer_simulator'
        except ImportError:
            pytest.skip("TKET not available")

    def test_tket_circuit_optimization(self, mock_tket):
        """Test TKET circuit optimization."""
        try:
            from superquantx.backends.tket_backend import TKETBackend
            backend = TKETBackend()

            mock_circuit = Mock()
            mock_circuit.copy.return_value = mock_circuit
            mock_circuit.n_gates = 10

            # Should not crash even without real TKET
            optimized = backend.optimize_circuit(mock_circuit, optimization_level=1)
            assert optimized is not None
        except ImportError:
            pytest.skip("TKET not available")

    def test_tket_parameterized_circuits(self, mock_tket):
        """Test TKET parameterized circuit creation."""
        try:
            from superquantx.backends.tket_backend import TKETBackend
            backend = TKETBackend()
            circuit, param_names = backend.create_parameterized_circuit(4, 8)

            assert len(param_names) == 8
            assert all(name.startswith('theta_') for name in param_names)
        except ImportError:
            pytest.skip("TKET not available")


class TestOceanBackend:
    """Test D-Wave Ocean backend functionality."""

    @pytest.fixture
    def mock_ocean(self):
        """Mock Ocean SDK components."""
        with patch('superquantx.backends.ocean_backend.OCEAN_AVAILABLE', True), \
             patch('superquantx.backends.ocean_backend.dimod') as mock_dimod, \
             patch('superquantx.backends.ocean_backend.SimulatedAnnealingSampler') as mock_sampler:

            # Mock BQM and sampleset
            mock_bqm = Mock()
            mock_dimod.BinaryQuadraticModel.from_qubo.return_value = mock_bqm
            mock_dimod.BinaryQuadraticModel.from_ising.return_value = mock_bqm

            mock_sampleset = Mock()
            mock_sampleset.data.return_value = [
                ({'x0': 1, 'x1': 0}, -1.0, 1),
                ({'x0': 0, 'x1': 1}, -0.5, 1)
            ]
            mock_sampleset.__len__.return_value = 2
            mock_sampleset.info = {'timing': {}}

            mock_sampler_instance = Mock()
            mock_sampler_instance.sample.return_value = mock_sampleset
            mock_sampler.return_value = mock_sampler_instance

            yield mock_dimod, mock_sampler, mock_sampleset

    def test_ocean_backend_creation(self, mock_ocean):
        """Test Ocean backend creation."""
        try:
            from superquantx.backends.ocean_backend import OceanBackend
            backend = OceanBackend(device='simulator')
            assert backend.device_name == 'simulator'
            assert backend._is_quantum_annealing == True
        except ImportError:
            pytest.skip("Ocean not available")

    def test_ocean_qubo_solving(self, mock_ocean):
        """Test QUBO problem solving."""
        try:
            from superquantx.backends.ocean_backend import OceanBackend
            backend = OceanBackend()

            Q = {(0, 0): -1, (1, 1): -1, (0, 1): 2}
            result = backend.solve_qubo(Q, num_reads=100)

            assert result['success'] == True
            assert 'samples' in result
            assert 'energies' in result
            assert result['problem_type'] == 'QUBO'
        except ImportError:
            pytest.skip("Ocean not available")

    def test_ocean_ising_solving(self, mock_ocean):
        """Test Ising model solving."""
        try:
            from superquantx.backends.ocean_backend import OceanBackend
            backend = OceanBackend()

            h = {0: -1, 1: 1}
            J = {(0, 1): -1}
            result = backend.solve_ising(h, J, num_reads=50)

            assert result['success'] == True
            assert result['problem_type'] == 'Ising'
        except ImportError:
            pytest.skip("Ocean not available")

    def test_ocean_max_cut(self, mock_ocean):
        """Test Max Cut problem solving."""
        try:
            from superquantx.backends.ocean_backend import OceanBackend
            backend = OceanBackend()

            # Simple graph as edge list
            edges = [(0, 1), (1, 2), (2, 0)]
            result = backend.solve_max_cut(edges, num_reads=100)

            assert 'samples' in result
        except ImportError:
            pytest.skip("Ocean not available")

    def test_ocean_circuit_compatibility(self, mock_ocean):
        """Test Ocean backend circuit compatibility warnings."""
        try:
            from superquantx.backends.ocean_backend import OceanBackend
            backend = OceanBackend()

            # Should return placeholder
            circuit = backend.create_circuit(4)
            assert circuit['type'] == 'optimization_placeholder'

            # Should warn about gate operations
            backend.add_gate(circuit, 'h', 0)  # Should warn
            result = backend.execute_circuit(circuit)  # Should warn
            assert result['success'] == False
        except ImportError:
            pytest.skip("Ocean not available")


class TestUnifiedAPIIntegration:
    """Test unified API integration across all backends."""

    def test_algorithm_backend_compatibility(self):
        """Test that algorithms work with different backends."""
        # Test with simulator (always available)
        qsvm = sqx.QuantumSVM(backend='simulator', shots=100)
        assert qsvm.backend is not None

    def test_backend_info_retrieval(self):
        """Test backend information retrieval."""
        backend = get_backend('simulator')
        info = backend.get_backend_info()

        assert isinstance(info, dict)
        assert 'backend_name' in info
        assert 'capabilities' in info

    def test_cross_backend_algorithm_creation(self):
        """Test creating algorithms with different backends."""
        backends_to_test = ['simulator']

        # Add available backends
        available = list_available_backends()
        for backend_name, info in available.items():
            if info.get('available') and backend_name not in ['simulator', 'auto']:
                backends_to_test.append(backend_name)

        for backend_name in backends_to_test:
            try:
                # Test basic algorithm creation
                qnn = sqx.QuantumNN(backend=backend_name, n_layers=2, shots=100)
                assert qnn.backend is not None

                # Test agent creation (if supported)
                if backend_name != 'ocean':  # Ocean is annealing-only
                    agent = sqx.QuantumOptimizationAgent(backend=backend_name)
                    assert agent.backend is not None

            except (ImportError, ValueError):
                # Expected for backends without installed dependencies
                continue

    def test_backend_error_handling(self):
        """Test proper error handling for backend issues."""
        # Test invalid backend
        with pytest.raises(ValueError):
            get_backend('nonexistent_backend')

        # Test backend with missing dependencies should be handled gracefully
        available = list_available_backends()
        for backend_name, info in available.items():
            if not info.get('available'):
                # Should raise ImportError with helpful message
                with pytest.raises(ImportError):
                    get_backend(backend_name)


class TestBackendSpecificFeatures:
    """Test backend-specific advanced features."""

    def test_quantum_vs_annealing_backend_distinction(self):
        """Test that quantum annealing backends are properly distinguished."""
        try:
            from superquantx.backends.ocean_backend import OceanBackend
            ocean = OceanBackend()
            info = ocean.get_backend_info()
            assert info['quantum_annealing'] == True
            assert info['gate_model'] == False
        except ImportError:
            pytest.skip("Ocean not available")

        # Compare with gate-model backend
        simulator = get_backend('simulator')
        sim_info = simulator.get_backend_info()
        # Simulator should support gate model
        circuit = simulator.create_circuit(2)
        assert circuit is not None

    def test_backend_capabilities_reporting(self):
        """Test that backends correctly report their capabilities."""
        backend = get_backend('simulator')
        capabilities = backend.capabilities

        assert isinstance(capabilities, dict)
        assert 'supports_measurements' in capabilities
        assert 'supports_parameterized_circuits' in capabilities

        # Test D-Wave specific capabilities
        try:
            from superquantx.backends.ocean_backend import OceanBackend
            ocean = OceanBackend()
            assert ocean.capabilities['annealing_backend'] == True
            assert ocean.capabilities['supports_optimization'] == True
        except ImportError:
            pass

    def test_version_information(self):
        """Test version information retrieval."""
        backend = get_backend('simulator')
        version_info = backend.get_version_info()

        assert isinstance(version_info, dict)
        assert 'backend_version' in version_info


if __name__ == '__main__':
    # Run specific test
    pytest.main([__file__, '-v'])
