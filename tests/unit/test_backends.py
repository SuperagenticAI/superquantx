"""
Unit tests for SuperQuantX backend implementations.

Tests the unified API across all quantum computing backends,
including the new AWS Braket, TKET/Quantinuum, and D-Wave Ocean backends.
"""

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
        assert backends['simulator']['available']

    def test_check_backend_compatibility(self):
        """Test individual backend compatibility checking."""
        # Test known available backend
        compat = check_backend_compatibility('simulator')
        assert compat['compatible']

        # Test unknown backend
        compat = check_backend_compatibility('nonexistent_backend')
        assert not compat['compatible']


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


GATE_MODEL_BACKENDS = [
    ('simulator', {}),
    ('qiskit', {}),
    ('pennylane', {}),
    ('cirq', {}),
    ('braket', {'device': 'local:braket_sv'}),
    ('tket', {}),
]


def _backend_or_skip(name, **kwargs):
    try:
        return get_backend(name, **kwargs)
    except ImportError as exc:
        pytest.skip(f"{name} not installed: {exc}")


@pytest.mark.parametrize("name,kwargs", GATE_MODEL_BACKENDS)
class TestGateModelBackends:
    """Every gate-model backend must agree on real circuit results."""

    def test_bell_state_counts(self, name, kwargs):
        backend = _backend_or_skip(name, **kwargs)
        circuit = backend.create_circuit(2)
        circuit = backend.add_gate(circuit, 'H', 0)
        circuit = backend.add_gate(circuit, 'CNOT', [0, 1])
        circuit = backend.add_measurement(circuit)
        counts = backend.execute_circuit(circuit, shots=400)['counts']
        assert sum(counts.values()) == 400
        assert set(counts) <= {'00', '11'}
        assert min(counts.get('00', 0), counts.get('11', 0)) > 100

    def test_qubit_order_is_normalised(self, name, kwargs):
        backend = _backend_or_skip(name, **kwargs)
        circuit = backend.create_circuit(3)
        circuit = backend.add_gate(circuit, 'X', 0)
        sv = backend.statevector_big_endian(circuit)
        assert np.argmax(np.abs(sv)) == 0b100
        circuit = backend.add_measurement(circuit)
        counts = backend.counts_big_endian(backend.execute_circuit(circuit, shots=20)['counts'])
        assert counts == {'100': 20}

    def test_exact_expectation(self, name, kwargs):
        backend = _backend_or_skip(name, **kwargs)
        circuit = backend.create_circuit(1)
        circuit = backend.add_gate(circuit, 'RY', 0, [np.pi])
        assert backend.compute_expectation(circuit, {'Z': 1.0}) == pytest.approx(-1.0, abs=1e-6)

    def test_fidelity_kernel_matches_simulator(self, name, kwargs):
        backend = _backend_or_skip(name, **kwargs)
        X = np.array([[0.1, 0.2], [0.5, 0.9], [1.3, -0.4]])
        K = backend.compute_kernel_matrix(X, X)
        K_ref = get_backend('simulator').compute_kernel_matrix(X, X)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-6)
        np.testing.assert_allclose(K, K_ref, atol=1e-6)

    def test_unknown_gate_raises(self, name, kwargs):
        backend = _backend_or_skip(name, **kwargs)
        circuit = backend.create_circuit(1)
        with pytest.raises(ValueError):
            backend.add_gate(circuit, 'NOT_A_GATE', 0)


class TestBraketBackend:
    """AWS Braket specifics (local simulator only)."""

    def test_device_name_accepts_braket_prefix(self):
        pytest.importorskip("braket.circuits")
        from superquantx.backends.braket_backend import BraketBackend
        backend = BraketBackend(device='local:braket/braket_sv')
        assert backend.device_name == 'local:braket/braket_sv'
        assert backend.is_available()

    def test_register_size_is_fixed(self):
        pytest.importorskip("braket.circuits")
        from superquantx.backends.braket_backend import BraketBackend
        backend = BraketBackend()
        circuit = backend.create_circuit(4)
        backend.add_gate(circuit, 'rx', 0, [np.pi / 4])
        assert circuit.qubit_count == 4


class TestTKETBackend:
    """TKET specifics."""

    def test_rotation_angles_are_radians(self):
        pytest.importorskip("pytket.extensions.qiskit")
        from superquantx.backends.tket_backend import TKETBackend
        backend = TKETBackend(device='aer_simulator')
        circuit = backend.add_gate(backend.create_circuit(1), 'RX', 0, [np.pi])
        counts = backend.execute_circuit(circuit, shots=50)['counts']
        assert counts == {'1': 50}

    def test_circuit_optimization(self):
        pytest.importorskip("pytket.extensions.qiskit")
        from superquantx.backends.tket_backend import TKETBackend
        backend = TKETBackend()
        circuit = backend.create_circuit(2)
        for _ in range(2):
            backend.add_gate(circuit, 'H', 0)
        optimized = backend.optimize_circuit(circuit, optimization_level=1)
        assert optimized.n_gates <= circuit.n_gates


class TestOceanBackend:
    """D-Wave Ocean backend (simulated annealing, no hardware)."""

    @pytest.fixture
    def ocean(self):
        pytest.importorskip("dimod")
        from superquantx.backends.ocean_backend import OceanBackend
        return OceanBackend(device='simulator')

    def test_ocean_backend_creation(self, ocean):
        assert ocean.device_name == 'simulator'
        assert ocean._is_quantum_annealing
        assert ocean.gate_model is False

    def test_ocean_qubo_solving(self, ocean):
        Q = {(0, 0): -1, (1, 1): -1, (0, 1): 2}
        result = ocean.solve_qubo(Q, num_reads=50)
        assert result['success']
        assert result['problem_type'] == 'QUBO'
        assert min(result['energies']) == pytest.approx(-1.0)

    def test_ocean_ising_solving(self, ocean):
        result = ocean.solve_ising({0: -1, 1: 1}, {(0, 1): -1}, num_reads=50)
        assert result['success']
        assert result['problem_type'] == 'Ising'

    def test_ocean_max_cut_finds_maximum(self, ocean):
        # Square 0-1-2-3-0: maximum cut is 4 (alternate colouring)
        result = ocean.solve_max_cut([(0, 1), (1, 2), (2, 3), (3, 0)], num_reads=50)
        assert result['best_cut_value'] == pytest.approx(4.0)

    def test_ocean_rejects_gate_model_calls(self, ocean):
        with pytest.raises(NotImplementedError, match="annealer"):
            ocean.create_circuit(4)
        with pytest.raises(NotImplementedError):
            ocean.execute_circuit({})


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
            assert info['quantum_annealing']
            assert not info['gate_model']
        except ImportError:
            pytest.skip("Ocean not available")

        # Compare with gate-model backend
        simulator = get_backend('simulator')
        simulator.get_backend_info()
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
            assert ocean.capabilities['annealing_backend']
            assert ocean.capabilities['supports_optimization']
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
