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
from superquantx.circuits import QuantumCircuit
from superquantx.measurements import MeasurementResult

class TestQuantumCircuits:
    def test_circuit_creation(self):
        """Test basic circuit creation"""
        circuit = QuantumCircuit(n_qubits=3, n_classical=3)
        
        assert circuit.num_qubits == 3
        assert circuit.num_classical_bits == 3
        assert len(circuit.data) == 0
    
    def test_gate_application(self):
        """Test applying gates to circuit"""
        circuit = QuantumCircuit(2)
        
        # Apply gates
        circuit.h(0)
        circuit.cx(0, 1)
        
        assert len(circuit.data) == 2
        assert circuit.data[0].name == 'h'
        assert circuit.data[1].name == 'cx'
    
    def test_bell_state_circuit(self):
        """Test Bell state preparation circuit"""
        circuit = QuantumCircuit(2, 2)
        
        # Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        
        # Verify circuit structure
        assert circuit.depth() == 2  # H gate, then CX gate
        assert circuit.count_ops()['h'] == 1
        assert circuit.count_ops()['cx'] == 1
        assert circuit.count_ops()['measure'] == 2
    
    def test_parameterized_circuit(self):
        """Test circuits with parameters"""
        from superquantx.circuits import Parameter
        
        theta = Parameter('theta')
        circuit = QuantumCircuit(1)
        circuit.ry(theta, 0)
        
        # Bind parameter
        bound_circuit = circuit.bind_parameters({theta: np.pi/2})
        
        assert len(bound_circuit.parameters) == 0
        assert bound_circuit.data[0].params[0] == np.pi/2
```

### Testing Algorithms

```python
from superquantx.algorithms import QuantumSVM
from sklearn.datasets import make_classification

class TestQuantumAlgorithms:
    def test_qsvm_initialization(self):
        """Test QSVM initialization"""
        backend = 'simulator'
        qsvm = QuantumSVM(backend=backend)
        
        assert qsvm.backend == backend
        assert qsvm.feature_map is not None
        assert qsvm.quantum_kernel is not None
    
    def test_qsvm_small_dataset(self):
        """Test QSVM on small synthetic dataset"""
        # Create simple 2D dataset
        X, y = make_classification(
            n_samples=20, n_features=2, n_redundant=0, 
            n_informative=2, n_clusters_per_class=1, random_state=42
        )
        
        qsvm = QuantumSVM(backend='simulator')
        
        # Fit and predict
        qsvm.fit(X, y)
        predictions = qsvm.predict(X)
        
        # Basic validations
        assert len(predictions) == len(y)
        assert all(pred in [0, 1] for pred in predictions)
        
        # Should achieve reasonable accuracy on training data
        accuracy = np.mean(predictions == y)
        assert accuracy > 0.7  # At least 70% accuracy
    
    @pytest.mark.parametrize("backend", ["simulator", "pennylane"])
    def test_qsvm_backend_consistency(self, backend):
        """Test QSVM consistency across backends"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])  # XOR pattern
        
        qsvm = QuantumSVM(backend=backend)
        
        # Should not crash and produce valid output
        qsvm.fit(X, y)
        predictions = qsvm.predict(X)
        
        assert len(predictions) == len(y)
        assert all(isinstance(pred, (int, np.integer)) for pred in predictions)
```

## 🔗 Integration Testing

### Backend Integration Tests

```python
import pytest
from superquantx import get_backend

class TestBackendIntegration:
    @pytest.mark.parametrize("backend_name", [
        "simulator", "pennylane", "qiskit", "cirq"
    ])
    def test_backend_availability(self, backend_name):
        """Test that backends can be loaded"""
        try:
            backend = get_backend(backend_name)
            assert backend is not None
        except ImportError:
            pytest.skip(f"Backend {backend_name} not available")
    
    def test_cross_backend_consistency(self):
        """Test that same circuit produces similar results across backends"""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        
        backends = ["simulator"]  # Add more as available
        results = {}
        
        for backend_name in backends:
            try:
                backend = get_backend(backend_name)
                result = backend.execute(circuit, shots=1000)
                results[backend_name] = result
            except ImportError:
                continue
        
        # Compare probability distributions
        if len(results) > 1:
            backend_names = list(results.keys())
            ref_probs = results[backend_names[0]].probabilities
            
            for backend_name in backend_names[1:]:
                probs = results[backend_name].probabilities
                
                # Check that probability distributions are similar
                for outcome in ref_probs:
                    if outcome in probs:
                        diff = abs(ref_probs[outcome] - probs[outcome])
                        assert diff < 0.1  # Within 10%
    
    def test_bell_state_measurement(self):
        """Test Bell state measurement across backends"""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        
        backend = get_backend('simulator')
        result = backend.execute(circuit, shots=1000)
        
        # Bell state should only produce |00⟩ and |11⟩
        valid_outcomes = {'00', '11'}
        for outcome in result.counts:
            assert outcome in valid_outcomes
        
        # Should be roughly 50/50 distribution
        if '00' in result.counts and '11' in result.counts:
            prob_00 = result.counts['00'] / 1000
            prob_11 = result.counts['11'] / 1000
            
            assert 0.3 < prob_00 < 0.7  # Rough statistical test
            assert 0.3 < prob_11 < 0.7
```

### End-to-End Workflow Tests

```python
class TestWorkflows:
    def test_complete_qml_workflow(self):
        """Test complete quantum machine learning workflow"""
        from sklearn.datasets import make_blobs
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        # 1. Data preparation
        X, y = make_blobs(n_samples=40, centers=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # 2. Model training
        qsvm = QuantumSVM(backend='simulator')
        qsvm.fit(X_train, y_train)
        
        # 3. Prediction
        y_pred = qsvm.predict(X_test)
        
        # 4. Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        
        # Should achieve reasonable performance
        assert accuracy > 0.6
        assert len(y_pred) == len(y_test)
    
    def test_noise_modeling_workflow(self):
        """Test noise modeling and error mitigation workflow"""
        from superquantx.noise import NoiseModel, DepolarizingChannel
        from superquantx.measurements import ResultAnalyzer
        
        # 1. Create noisy backend
        noise_model = NoiseModel()
        noise_channel = DepolarizingChannel(probability=0.01)
        noise_model.add_gate_noise('h', noise_channel)
        
        # 2. Create test circuit
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        
        # 3. Execute with and without noise
        clean_backend = get_backend('simulator')
        noisy_backend = get_backend('simulator', noise_model=noise_model)
        
        clean_result = clean_backend.execute(circuit, shots=1000)
        noisy_result = noisy_backend.execute(circuit, shots=1000)
        
        # 4. Compare results
        comparison = ResultAnalyzer.compare_results(clean_result, noisy_result)
        
        # Noise should create some difference
        assert comparison['total_variation_distance'] > 0
```

## ⚡ Performance Testing

### Benchmarking Framework

```python
import time
import pytest
from superquantx.benchmarks import QuantumBenchmark

class TestPerformance:
    @pytest.mark.performance
    def test_circuit_compilation_performance(self):
        """Benchmark circuit compilation speed"""
        circuit = QuantumCircuit(10)
        
        # Add many gates
        for i in range(100):
            circuit.h(i % 10)
            if i % 10 < 9:
                circuit.cx(i % 10, (i % 10) + 1)
        
        backend = get_backend('simulator')
        
        # Measure compilation time
        start_time = time.time()
        compiled_circuit = backend.compile_circuit(circuit)
        compilation_time = time.time() - start_time
        
        # Should compile reasonably quickly
        assert compilation_time < 5.0  # Less than 5 seconds
        assert compiled_circuit is not None
    
    @pytest.mark.performance
    def test_algorithm_scaling(self):
        """Test algorithm performance scaling"""
        from sklearn.datasets import make_classification
        
        sizes = [10, 20, 40]
        times = []
        
        for size in sizes:
            X, y = make_classification(n_samples=size, n_features=2, random_state=42)
            
            qsvm = QuantumSVM(backend='simulator')
            
            start_time = time.time()
            qsvm.fit(X, y)
            training_time = time.time() - start_time
            
            times.append(training_time)
        
        # Training time shouldn't increase exponentially
        assert times[-1] < times[0] * 10  # No more than 10x increase
    
    @pytest.mark.performance
    def test_memory_usage(self):
        """Test memory usage during execution"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create and execute large circuit
        circuit = QuantumCircuit(12)
        for i in range(12):
            circuit.h(i)
        circuit.measure_all()
        
        backend = get_backend('simulator')
        result = backend.execute(circuit, shots=1000)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024
```

### Regression Testing

```python
class TestRegression:
    def test_known_algorithm_results(self):
        """Test that algorithms produce known results on standard datasets"""
        # Store expected results for regression testing
        expected_results = {
            'iris_qsvm_accuracy': 0.95,
            'bell_state_fidelity': 0.99,
            'vqe_h2_energy': -1.137
        }
        
        # Test each known result
        for test_name, expected_value in expected_results.items():
            if test_name == 'iris_qsvm_accuracy':
                actual_value = self._test_iris_qsvm()
                assert abs(actual_value - expected_value) < 0.05
    
    def _test_iris_qsvm(self):
        """Helper method for QSVM on Iris dataset"""
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        
        # Use subset of Iris dataset (2 classes, 2 features)
        iris = load_iris()
        X = iris.data[:100, :2]  # First 2 features, first 2 classes
        y = iris.target[:100]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        qsvm = QuantumSVM(backend='simulator')
        qsvm.fit(X_train, y_train)
        predictions = qsvm.predict(X_test)
        
        return np.mean(predictions == y_test)
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
from superquantx.circuits import QuantumCircuit

def bell_circuit():
    """Create Bell state preparation circuit"""
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    return circuit

def qft_circuit(n_qubits):
    """Create Quantum Fourier Transform circuit"""
    circuit = QuantumCircuit(n_qubits)
    
    for i in range(n_qubits):
        circuit.h(i)
        for j in range(i + 1, n_qubits):
            circuit.cp(np.pi / 2**(j - i), j, i)
    
    # Reverse qubit order
    for i in range(n_qubits // 2):
        circuit.swap(i, n_qubits - 1 - i)
    
    return circuit
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
        python-version: [3.9, 3.10, 3.11]
    
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