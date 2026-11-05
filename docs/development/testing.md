# Testing Guide

Comprehensive testing strategy for SuperQuantX to ensure reliability, correctness, and performance of quantum algorithms and backend integrations.

## 🧪 Testing Philosophy

### Core Principles

1. **Quantum Correctness**: Validate quantum algorithms produce expected results
2. **Backend Compatibility**: Ensure consistent behavior across quantum frameworks
3. **Performance Validation**: Benchmark execution time and resource usage
4. **Error Handling**: Test failure modes and error recovery
5. **Reproducibility**: Ensure deterministic test results

## 📁 Test Structure

```
tests/
├── unit/                    # Fast, isolated tests
│   ├── test_gates.py       # Gate matrix operations
│   ├── test_circuits.py    # Circuit construction
│   ├── test_algorithms.py  # Algorithm logic
│   ├── test_measurements.py # Measurement processing
│   └── test_noise.py       # Noise modeling
├── integration/             # Multi-component tests
│   ├── test_backends.py    # Backend integrations
│   ├── test_workflows.py   # End-to-end workflows
│   └── test_examples.py    # Documentation examples
├── performance/             # Benchmarking tests
│   ├── test_scalability.py # Scaling behavior
│   └── test_benchmarks.py  # Performance regression
└── fixtures/               # Test data and utilities
    ├── quantum_states.py   # Predefined quantum states
    ├── test_circuits.py    # Standard test circuits
    └── sample_data.py      # ML datasets
```

## 🚀 Running Tests

### Quick Test Run

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run with coverage
pytest --cov=superquantx --cov-report=html

# Run specific test file
pytest tests/unit/test_algorithms.py

# Run specific test function
pytest tests/unit/test_algorithms.py::test_qsvm_training
```

### Test Configuration

```bash
# Parallel execution
pytest -n auto

# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l

# Run only failed tests from last run
pytest --lf
```

## 🔬 Unit Testing

### Testing Quantum Gates

```python
import numpy as np
import pytest
from superquantx.gates import GateMatrix

class TestQuantumGates:
    def test_pauli_gates_hermitian(self):
        """Test that Pauli gates are Hermitian"""
        gates = [GateMatrix.X, GateMatrix.Y, GateMatrix.Z]
        
        for gate in gates:
            # A gate is Hermitian if A† = A
            assert np.allclose(gate, gate.conj().T)
    
    def test_pauli_gates_unitary(self):
        """Test that Pauli gates are unitary"""
        gates = [GateMatrix.X, GateMatrix.Y, GateMatrix.Z]
        
        for gate in gates:
            # A gate is unitary if A†A = I
            identity = np.eye(2, dtype=complex)
            assert np.allclose(gate @ gate.conj().T, identity)
    
    def test_rotation_gates(self):
        """Test rotation gate properties"""
        theta = np.pi / 4
        
        # Test Rx gate
        rx = GateMatrix.rx(theta)
        assert np.allclose(np.linalg.det(rx), 1.0)  # Determinant = 1
        
        # Test specific rotation
        rx_pi = GateMatrix.rx(np.pi)
        expected = -1j * GateMatrix.X
        assert np.allclose(rx_pi, expected)
    
    @pytest.mark.parametrize("theta", [0, np.pi/2, np.pi, 2*np.pi])
    def test_rotation_angles(self, theta):
        """Test rotation gates at specific angles"""
        rx = GateMatrix.rx(theta)
        ry = GateMatrix.ry(theta)
        rz = GateMatrix.rz(theta)
        
        # All should be unitary
        I = np.eye(2, dtype=complex)
        assert np.allclose(rx @ rx.conj().T, I)
        assert np.allclose(ry @ ry.conj().T, I)
        assert np.allclose(rz @ rz.conj().T, I)
```

### Testing Quantum Circuits

```python
import numpy as np
from superquantx.circuits import QuantumCircuit


class TestQuantumCircuits:
    """Tests for the backend-independent circuit description."""

    def test_circuit_creation(self):
        circuit = QuantumCircuit(3, 3)

        assert circuit.num_qubits == 3
        assert circuit.num_classical_bits == 3
        assert len(circuit.gates) == 0

    def test_gate_application(self):
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        assert [g.name for g in circuit.gates] == ['H', 'CNOT']
        assert circuit.gates[1].qubits == [0, 1]

    def test_bell_state_circuit(self):
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()

        assert len(circuit.gates) == 2
        assert circuit.measurements == [(0, 0), (1, 1)]

    def test_rotation_parameters(self):
        # There is no symbolic Parameter class; rotations take numbers
        circuit = QuantumCircuit(1)
        circuit.ry(np.pi / 2, 0)

        assert circuit.gates[0].parameters == [np.pi / 2]

    def test_json_round_trip(self):
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        assert QuantumCircuit.from_json(circuit.to_json()).gates == circuit.gates
```

### Testing Algorithms

```python
from sklearn.datasets import make_classification
from superquantx.algorithms import QuantumSVM
from superquantx.backends import SimulatorBackend


class TestQuantumAlgorithms:
    def test_qsvm_initialization(self):
        qsvm = QuantumSVM(backend='simulator', feature_map='ZZFeatureMap')

        assert isinstance(qsvm.backend, SimulatorBackend)   # names resolve to backend objects
        assert qsvm.feature_map == 'ZZFeatureMap'
        assert not qsvm.is_fitted

    def test_qsvm_small_dataset(self):
        X, y = make_classification(
            n_samples=20, n_features=2, n_redundant=0,
            n_informative=2, n_clusters_per_class=1, random_state=42,
        )
        qsvm = QuantumSVM(backend='simulator')
        predictions = qsvm.fit(X, y).predict(X)

        assert len(predictions) == len(y)
        assert set(predictions) <= {0, 1}
        assert np.mean(predictions == y) >= 0.6   # training accuracy floor

    @pytest.mark.parametrize("backend", ["simulator", "pennylane"])
    def test_qsvm_backend_consistency(self, backend):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])  # XOR pattern

        predictions = QuantumSVM(backend=backend).fit(X, y).predict(X)

        assert len(predictions) == len(y)
        assert all(isinstance(pred, (int, np.integer)) for pred in predictions)
```

## 🔗 Integration Testing

### Backend Integration Tests

```python
import pytest
from superquantx import get_backend


def bell_counts(backend_name, shots=1000):
    backend = get_backend(backend_name)
    circuit = backend.create_circuit(2)
    backend.add_gate(circuit, 'H', 0)
    backend.add_gate(circuit, 'CNOT', [0, 1])
    backend.add_measurement(circuit)
    return backend.execute_circuit(circuit, shots=shots)['counts']


class TestBackendIntegration:
    @pytest.mark.parametrize("backend_name", ["simulator", "pennylane", "qiskit", "cirq"])
    def test_backend_availability(self, backend_name):
        try:
            backend = get_backend(backend_name)
        except ImportError:
            pytest.skip(f"Backend {backend_name} not available")
        assert backend is not None

    def test_cross_backend_consistency(self):
        """The same Bell circuit gives similar distributions on every backend."""
        results = {}
        for backend_name in ["simulator", "qiskit", "cirq", "pennylane"]:
            try:
                results[backend_name] = bell_counts(backend_name)
            except ImportError:
                continue

        reference = results.pop('simulator')
        for counts in results.values():
            for outcome in ('00', '11'):
                diff = abs(reference.get(outcome, 0) - counts.get(outcome, 0)) / 1000
                assert diff < 0.1   # within 10%

    def test_bell_state_measurement(self):
        counts = bell_counts('simulator')

        assert set(counts) <= {'00', '11'}   # only correlated outcomes
        assert 0.3 < counts.get('00', 0) / 1000 < 0.7
        assert 0.3 < counts.get('11', 0) / 1000 < 0.7
```

### End-to-End Workflow Tests

```python
class TestWorkflows:
    def test_complete_qml_workflow(self):
        """Data preparation, training, prediction and evaluation."""
        from sklearn.datasets import make_blobs
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split

        X, y = make_blobs(n_samples=40, centers=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        qsvm = QuantumSVM(backend='simulator', feature_map='ZFeatureMap')
        y_pred = qsvm.fit(X_train, y_train).predict(X_test)

        assert accuracy_score(y_test, y_pred) > 0.7
        assert len(y_pred) == len(y_test)

    def test_readout_noise_workflow(self):
        """Compare ideal counts with counts after simulated readout noise."""
        from superquantx.measurements import MeasurementResult, ResultAnalyzer

        backend = get_backend('simulator')
        circuit = backend.create_circuit(2)
        backend.add_gate(circuit, 'H', 0)
        backend.add_gate(circuit, 'CNOT', [0, 1])
        backend.add_measurement(circuit)
        clean = backend.execute_circuit(circuit, shots=1000)['counts']

        # Flip each measured bit with probability 5% (classical readout noise)
        rng = np.random.default_rng(0)
        noisy = {}
        for bits, count in clean.items():
            for _ in range(count):
                out = ''.join(b if rng.random() > 0.05 else str(1 - int(b)) for b in bits)
                noisy[out] = noisy.get(out, 0) + 1

        comparison = ResultAnalyzer.compare_results(
            MeasurementResult(counts=clean, shots=1000),
            MeasurementResult(counts=noisy, shots=1000),
        )
        assert comparison['total_variation_distance'] > 0
```

## ⚡ Performance Testing

### Benchmarking Framework

```python
import time

import pytest
from superquantx import get_backend


class TestPerformance:
    @pytest.mark.performance
    def test_circuit_execution_performance(self):
        """A 10-qubit, 190-gate circuit should run quickly on the simulator."""
        backend = get_backend('simulator')
        circuit = backend.create_circuit(10)
        for i in range(100):
            backend.add_gate(circuit, 'H', i % 10)
            if i % 10 < 9:
                backend.add_gate(circuit, 'CNOT', [i % 10, i % 10 + 1])
        backend.add_measurement(circuit)

        start_time = time.time()
        result = backend.execute_circuit(circuit, shots=1000)
        assert time.time() - start_time < 5.0
        assert sum(result['counts'].values()) == 1000

    @pytest.mark.performance
    def test_algorithm_scaling(self):
        """QSVM training time should not explode with dataset size."""
        from sklearn.datasets import make_classification

        times = []
        for size in [10, 20, 40]:
            X, y = make_classification(
                n_samples=size, n_features=2, n_redundant=0, n_informative=2, random_state=42
            )
            start_time = time.time()
            QuantumSVM(backend='simulator').fit(X, y)
            times.append(time.time() - start_time)

        assert times[-1] < max(times[0], 0.05) * 50   # kernel cost is O(n^2), with headroom

    @pytest.mark.performance
    def test_memory_usage(self):
        """Executing a 12-qubit circuit should not use much memory."""
        psutil = pytest.importorskip("psutil")
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        backend = get_backend('simulator', max_qubits=12)
        circuit = backend.create_circuit(12)
        for i in range(12):
            backend.add_gate(circuit, 'H', i)
        backend.add_measurement(circuit)
        backend.execute_circuit(circuit, shots=1000)

        assert process.memory_info().rss - initial_memory < 100 * 1024 * 1024
```

### Regression Testing

```python
class TestRegression:
    """Pin results recorded on SuperQuantX 0.3.0 so changes are noticed."""

    def test_iris_qsvm_accuracy(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        iris = load_iris()
        X, y = iris.data[:100, :2], iris.target[:100]   # 2 classes, 2 features
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        qsvm = QuantumSVM(backend='simulator', feature_map='ZFeatureMap', feature_map_reps=1)
        accuracy = np.mean(qsvm.fit(X_train, y_train).predict(X_test) == y_test)

        assert abs(accuracy - 0.77) < 0.05   # baseline recorded on 0.3.0

    def test_vqe_h2_energy(self):
        from superquantx import VQE, Hamiltonian

        h2 = Hamiltonian.from_dict({
            'II': -1.0523732, 'ZI': 0.39793742, 'IZ': -0.39793742,
            'ZZ': -0.01128010, 'XX': 0.18093119,
        })
        energy = VQE(hamiltonian=h2, backend='simulator', optimizer='COBYLA', seed=0).find_ground_state()

        assert abs(energy - (-1.857275)) < 1e-3   # exact: h2.ground_state_energy()
```

## 🛠️ Test Utilities

### Quantum State Fixtures

```python
# tests/fixtures/quantum_states.py
import numpy as np

def bell_state_00():
    """Return |Φ+⟩ = (|00⟩ + |11⟩)/√2"""
    return np.array([1, 0, 0, 1]) / np.sqrt(2)

def bell_state_01():
    """Return |Φ-⟩ = (|00⟩ - |11⟩)/√2"""
    return np.array([1, 0, 0, -1]) / np.sqrt(2)

def ghz_state(n_qubits):
    """Return n-qubit GHZ state"""
    state = np.zeros(2**n_qubits)
    state[0] = 1/np.sqrt(2)  # |00...0⟩
    state[-1] = 1/np.sqrt(2)  # |11...1⟩
    return state
```

### Test Circuit Library

```python
# tests/fixtures/test_circuits.py
import numpy as np
from superquantx import get_backend


def bell_circuit(backend):
    """Bell state preparation on a backend circuit."""
    circuit = backend.create_circuit(2)
    backend.add_gate(circuit, 'H', 0)
    backend.add_gate(circuit, 'CNOT', [0, 1])
    backend.add_measurement(circuit)
    return circuit


def qft_circuit(backend, n_qubits):
    """Quantum Fourier Transform built from H, controlled-RZ and SWAP.

    CRZ(theta) equals the controlled-phase gate up to a phase on the control,
    which is enough for tests that compare measurement statistics.
    """
    circuit = backend.create_circuit(n_qubits)
    for i in range(n_qubits):
        backend.add_gate(circuit, 'H', i)
        for j in range(i + 1, n_qubits):
            backend.add_gate(circuit, 'CRZ', [j, i], [np.pi / 2 ** (j - i)])
    for i in range(n_qubits // 2):
        backend.add_gate(circuit, 'SWAP', [i, n_qubits - 1 - i])
    return circuit


sim = get_backend('simulator')
print(sim.execute_circuit(bell_circuit(sim), shots=100)['counts'])
print(qft_circuit(sim, 3) is not None)
```

## 📊 Test Reporting

### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=superquantx --cov-report=html

# Generate XML coverage report (for CI)
pytest --cov=superquantx --cov-report=xml

# Show missing lines
pytest --cov=superquantx --cov-report=term-missing
```

### Performance Reports

```bash
# Generate performance benchmark report
pytest --benchmark-only --benchmark-json=benchmark_results.json

# Compare with previous benchmarks
pytest --benchmark-compare=0001 --benchmark-compare-fail=mean:10%
```

## 🔧 Continuous Integration

### GitHub Actions Configuration

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12, 3.13]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        pytest --cov=superquantx --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## 📝 Testing Best Practices

### Writing Good Tests

1. **Test One Thing**: Each test should verify one specific behavior
2. **Use Descriptive Names**: Test names should explain what is being tested
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Use Fixtures**: Share common test setup
5. **Parameterize**: Test multiple inputs efficiently

### Quantum-Specific Testing

1. **Statistical Testing**: Account for quantum measurement randomness
2. **Fidelity Checks**: Verify quantum state preparation accuracy
3. **Unitarity Tests**: Ensure quantum operations preserve unitarity
4. **Backend Consistency**: Test across different quantum backends

### Debugging Failed Tests

```bash
# Run specific failing test with detailed output
pytest tests/unit/test_algorithms.py::test_qsvm_training -v -s --tb=long

# Start debugger on test failure
pytest --pdb tests/unit/test_algorithms.py::test_qsvm_training

# Create test report
pytest --html=report.html --self-contained-html
```

---

This comprehensive testing framework ensures SuperQuantX maintains high quality and reliability across all quantum computing backends and algorithms.