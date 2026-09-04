# Code Standards

Comprehensive coding standards and guidelines for SuperQuantX development to ensure consistency, maintainability, and quality across the codebase.

## 🎯 General Principles

### Code Philosophy

1. **Readability First**: Code is read more often than written
2. **Explicit is Better**: Avoid implicit behavior and magic
3. **Consistency**: Follow established patterns throughout the codebase
4. **Performance Awareness**: Optimize for quantum computing workloads
5. **Research-Friendly**: Support experimentation and reproducibility

### Design Patterns

- **Strategy Pattern**: For backend implementations
- **Factory Pattern**: For algorithm and backend creation
- **Observer Pattern**: For experiment tracking
- **Command Pattern**: For quantum operations

## 🐍 Python Style Guide

### PEP 8 Compliance

Follow [PEP 8](https://pep8.org/) with the following specifics:

```python
# Line length: 88 characters (Black formatter default)
# Indentation: 4 spaces (no tabs)
# Quotes: Double quotes for strings, single for internal

class QuantumAlgorithm:
    """
    Base class for quantum algorithms.
    
    This class provides the foundation for implementing quantum
    machine learning and optimization algorithms.
    """
    
    def __init__(self, backend: str = "simulator") -> None:
        """Initialize algorithm with specified backend."""
        self.backend = backend
        self._is_trained = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumAlgorithm":
        """
        Train the quantum algorithm.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)
            
        Returns:
            Self for method chaining
        """
        self._validate_input(X, y)
        self._fit_impl(X, y)
        self._is_trained = True
        return self
```

### Naming Conventions

#### Variables and Functions
```python
# Snake case for variables and functions
learning_rate = 0.01
num_qubits = 4

def calculate_expectation_value(observable: str) -> float:
    """Calculate expectation value of observable."""
    pass

def get_backend_name() -> str:
    """Get the name of the current backend."""
    pass
```

#### Classes
```python
# PascalCase for classes
class QuantumCircuit:
    pass

class VarationalQuantumEigensolver:
    pass

class BackendNotFoundError(Exception):
    pass
```

#### Constants
```python
# UPPER_SNAKE_CASE for constants
DEFAULT_SHOTS = 1024
MAX_QUBITS = 32
SUPPORTED_BACKENDS = ["simulator", "pennylane", "qiskit", "cirq"]

# Module-level constants
PI = np.pi
SQRT_2 = np.sqrt(2)
```

#### Private Members
```python
class QuantumAlgorithm:
    def __init__(self):
        self._internal_state = {}    # Single underscore for internal use
        self.__private_data = []     # Double underscore for name mangling
    
    def _helper_method(self):        # Internal method
        """Private helper method."""
        pass
    
    def __special_method__(self):    # Magic method
        """Special dunder method."""
        pass
```

### Type Hints

Use comprehensive type hints:

```python
from typing import List, Dict, Optional, Union, Callable, Tuple, Any
import numpy as np

# Basic types
def process_data(data: List[float]) -> Dict[str, float]:
    """Process numerical data."""
    pass

# NumPy arrays
def quantum_operation(
    state: np.ndarray,
    gate: np.ndarray
) -> np.ndarray:
    """Apply quantum gate to state."""
    pass

# Optional and Union types
def get_backend(
    name: str,
    config: Optional[Dict[str, Any]] = None
) -> Union[QuantumBackend, None]:
    """Get quantum backend by name."""
    pass

# Callable types
def optimize(
    cost_function: Callable[[np.ndarray], float],
    initial_params: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Optimize parameters using cost function."""
    pass

# Generic types
from typing import TypeVar, Generic

T = TypeVar('T')

class QuantumResult(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value
```

## 📝 Documentation Standards

### Docstring Format

Use Google-style docstrings:

```python
def quantum_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    feature_map: str = "ZZFeatureMap",
    backend: str = "simulator"
) -> float:
    """
    Compute quantum kernel between two data points.
    
    This function calculates the quantum kernel using a specified
    feature map and quantum backend. The kernel represents the
    inner product in the quantum feature space.
    
    Args:
        x1: First data point of shape (n_features,)
        x2: Second data point of shape (n_features,)
        feature_map: Type of quantum feature map to use.
            Options: "ZZFeatureMap", "PauliFeatureMap"
        backend: Quantum backend for execution
        
    Returns:
        Kernel value as a float between 0 and 1
        
    Raises:
        ValueError: If data points have different dimensions
        BackendError: If backend is not available
        
    Example:
        >>> x1 = np.array([0.1, 0.5])
        >>> x2 = np.array([0.2, 0.3])
        >>> kernel_value = quantum_kernel(x1, x2)
        >>> print(f"Kernel: {kernel_value:.4f}")
        Kernel: 0.8765
        
    Note:
        This function requires a quantum backend to be available.
        For large datasets, consider using batch computation.
        
    References:
        Havlíček, V., et al. "Supervised learning with quantum-enhanced
        feature spaces." Nature 567.7747 (2019): 209-212.
    """
```

### Class Documentation

```python
class QuantumSupportVectorMachine:
    """
    Quantum Support Vector Machine for binary classification.
    
    This class implements a quantum version of Support Vector Machine
    using quantum kernels for feature mapping. It leverages quantum
    computers to potentially achieve quantum advantage in machine learning.
    
    The algorithm encodes classical data into quantum states using
    feature maps, computes quantum kernels, and uses classical SVM
    optimization for training.
    
    Attributes:
        backend: Quantum backend for execution
        feature_map: Quantum feature map for data encoding
        C: Regularization parameter
        kernel_matrix_: Computed quantum kernel matrix (set after fit)
        support_vectors_: Support vectors found during training
        
    Example:
        Basic usage for binary classification:
        
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=20, n_features=2)
        >>> qsvm = QuantumSupportVectorMachine(backend="simulator")
        >>> qsvm.fit(X, y)
        >>> predictions = qsvm.predict(X)
        >>> accuracy = np.mean(predictions == y)
        >>> print(f"Accuracy: {accuracy:.2f}")
        
    Note:
        This is experimental software designed for research purposes.
        Performance may vary significantly across different quantum
        backends and problem sizes.
    """
    
    def __init__(
        self,
        backend: str = "simulator",
        feature_map: str = "ZZFeatureMap",
        C: float = 1.0
    ) -> None:
        """
        Initialize Quantum SVM.
        
        Args:
            backend: Quantum backend to use for computation
            feature_map: Type of quantum feature map
            C: Regularization parameter for SVM
        """
```

### Module Documentation

```python
"""
SuperQuantX Algorithms Module

This module contains implementations of quantum machine learning algorithms
and optimization routines. It provides high-level interfaces for quantum
algorithms while abstracting away backend-specific details.

The module is organized into several submodules:

- supervised: Quantum supervised learning algorithms
- unsupervised: Quantum clustering and dimensionality reduction
- optimization: Quantum optimization algorithms
- hybrid: Quantum-classical hybrid algorithms

Classes:
    QuantumSVM: Quantum Support Vector Machine
    VQE: Variational Quantum Eigensolver
    QAOA: Quantum Approximate Optimization Algorithm
    QuantumNeuralNetwork: Parameterized quantum circuits for ML

Functions:
    quantum_kernel: Compute quantum kernel between data points
    optimize_parameters: Classical optimization of quantum parameters

Example:
    Basic usage of quantum machine learning:
    
    >>> import superquantx as sqx
    >>> from sklearn.datasets import make_classification
    >>> 
    >>> # Create dataset
    >>> X, y = make_classification(n_samples=40, n_features=2)
    >>> 
    >>> # Train quantum SVM
    >>> qsvm = sqx.QuantumSVM(backend="simulator")
    >>> qsvm.fit(X, y)
    >>> 
    >>> # Make predictions
    >>> predictions = qsvm.predict(X)
    >>> accuracy = sqx.accuracy_score(y, predictions)
    >>> print(f"Accuracy: {accuracy:.2f}")

Note:
    All algorithms in this module are experimental and designed for
    research purposes. They may not be suitable for production use.
"""
```

## 🧪 Testing Standards

### Test Structure

```python
import pytest
import numpy as np
from unittest.mock import Mock, patch
from superquantx.algorithms import QuantumSVM

class TestQuantumSVM:
    """Test suite for Quantum Support Vector Machine."""
    
    @pytest.fixture
    def sample_data(self):
        """Provide sample dataset for testing."""
        np.random.seed(42)
        X = np.random.randn(20, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y
    
    @pytest.fixture
    def qsvm(self):
        """Provide QSVM instance for testing."""
        return QuantumSVM(backend="simulator")
    
    def test_initialization(self):
        """Test QSVM initialization with default parameters."""
        qsvm = QuantumSVM()
        
        assert qsvm.backend == "simulator"
        assert qsvm.C == 1.0
        assert not qsvm._is_trained
    
    def test_fit_with_valid_data(self, qsvm, sample_data):
        """Test fitting QSVM with valid training data."""
        X, y = sample_data
        
        result = qsvm.fit(X, y)
        
        # Should return self for method chaining
        assert result is qsvm
        assert qsvm._is_trained
        assert hasattr(qsvm, 'kernel_matrix_')
    
    def test_fit_with_invalid_data(self, qsvm):
        """Test fitting QSVM with invalid data raises ValueError."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1, 2])  # Wrong length
        
        with pytest.raises(ValueError, match="X and y must have same length"):
            qsvm.fit(X, y)
    
    @pytest.mark.parametrize("backend", ["simulator", "pennylane"])
    def test_different_backends(self, backend, sample_data):
        """Test QSVM works with different backends."""
        X, y = sample_data
        
        try:
            qsvm = QuantumSVM(backend=backend)
            qsvm.fit(X, y)
            predictions = qsvm.predict(X)
            
            assert len(predictions) == len(y)
            assert all(pred in [0, 1] for pred in predictions)
        except ImportError:
            pytest.skip(f"Backend {backend} not available")
    
    def test_predict_before_fit_raises_error(self, qsvm, sample_data):
        """Test that predict raises error before fit is called."""
        X, _ = sample_data
        
        with pytest.raises(RuntimeError, match="must be trained first"):
            qsvm.predict(X)
    
    @patch('superquantx.backends.get_backend')
    def test_backend_integration(self, mock_get_backend, qsvm, sample_data):
        """Test backend integration with mocking."""
        X, y = sample_data
        
        # Mock backend behavior
        mock_backend = Mock()
        mock_backend.execute.return_value = Mock(counts={'00': 50, '11': 50})
        mock_get_backend.return_value = mock_backend
        
        qsvm.fit(X, y)
        
        # Verify backend was called
        mock_get_backend.assert_called_once()
        mock_backend.execute.assert_called()
```

### Property-Based Testing

```python
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst

class TestQuantumGates:
    """Property-based tests for quantum gates."""
    
    @given(theta=st.floats(min_value=0, max_value=2*np.pi))
    def test_rotation_gates_unitary(self, theta):
        """Test that rotation gates are always unitary."""
        gates = [
            GateMatrix.rx(theta),
            GateMatrix.ry(theta),
            GateMatrix.rz(theta)
        ]
        
        for gate in gates:
            # Unitary condition: U†U = I
            identity = np.eye(2, dtype=complex)
            result = gate.conj().T @ gate
            assert np.allclose(result, identity, atol=1e-10)
    
    @given(
        state=npst.arrays(
            dtype=complex,
            shape=(2,),
            elements=st.complex_numbers(
                min_magnitude=0,
                max_magnitude=1,
                allow_infinity=False,
                allow_nan=False
            )
        )
    )
    def test_pauli_gates_preserve_norm(self, state):
        """Test that Pauli gates preserve state norm."""
        # Normalize input state
        if np.linalg.norm(state) > 0:
            state = state / np.linalg.norm(state)
            
            pauli_gates = [GateMatrix.X, GateMatrix.Y, GateMatrix.Z]
            
            for gate in pauli_gates:
                result_state = gate @ state
                assert np.allclose(
                    np.linalg.norm(result_state), 
                    1.0, 
                    atol=1e-10
                )
```

## 🔧 Code Quality Tools

### Formatter: Black

```toml
# pyproject.toml
[tool.black]
line-length = 88
target-version = ['py311', 'py312', 'py313']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''
```

### Linter: Flake8

```ini
# .flake8
[flake8]
max-line-length = 88
max-complexity = 10
ignore = 
    E203,  # whitespace before ':'
    W503,  # line break before binary operator
    E501,  # line too long (handled by black)
exclude =
    .git,
    __pycache__,
    .pytest_cache,
    .mypy_cache,
    build,
    dist
per-file-ignores =
    __init__.py:F401,F403  # Allow unused imports in __init__.py
    tests/*:D100,D101,D102,D103  # Don't require docstrings in tests
```

### Type Checker: mypy

```ini
# mypy.ini
[mypy]
python_version = 3.11
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True

[mypy-numpy.*]
ignore_missing_imports = True

[mypy-scipy.*]
ignore_missing_imports = True

[mypy-matplotlib.*]
ignore_missing_imports = True
```

### Import Sorting: isort

```toml
# pyproject.toml
[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["superquantx"]
known_third_party = ["numpy", "scipy", "matplotlib", "pytest"]
sections = ["FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"]
```

## 🚦 Performance Guidelines

### Quantum-Specific Optimizations

```python
# Good: Minimize quantum circuit depth
def efficient_ansatz(n_qubits: int, layers: int) -> QuantumCircuit:
    """Create efficient variational ansatz."""
    circuit = QuantumCircuit(n_qubits)
    
    for layer in range(layers):
        # Single-qubit rotations
        for qubit in range(n_qubits):
            circuit.ry(f"theta_{layer}_{qubit}", qubit)
        
        # Entangling gates (linear connectivity)
        for qubit in range(n_qubits - 1):
            circuit.cx(qubit, qubit + 1)
    
    return circuit

# Bad: Excessive circuit depth
def inefficient_ansatz(n_qubits: int, layers: int) -> QuantumCircuit:
    """Poor ansatz with unnecessary depth."""
    circuit = QuantumCircuit(n_qubits)
    
    for layer in range(layers):
        # Too many single-qubit gates
        for qubit in range(n_qubits):
            circuit.rx(f"rx_{layer}_{qubit}", qubit)
            circuit.ry(f"ry_{layer}_{qubit}", qubit)
            circuit.rz(f"rz_{layer}_{qubit}", qubit)
        
        # All-to-all connectivity (expensive)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                circuit.cx(i, j)
    
    return circuit
```

### Memory Management

```python
# Good: Use generators for large datasets
def batch_quantum_kernels(
    X1: np.ndarray, 
    X2: np.ndarray, 
    batch_size: int = 10
) -> np.ndarray:
    """Compute kernel matrix in batches to save memory."""
    n1, n2 = len(X1), len(X2)
    kernel_matrix = np.zeros((n1, n2))
    
    for i in range(0, n1, batch_size):
        for j in range(0, n2, batch_size):
            end_i = min(i + batch_size, n1)
            end_j = min(j + batch_size, n2)
            
            batch_kernels = compute_kernel_batch(
                X1[i:end_i], X2[j:end_j]
            )
            kernel_matrix[i:end_i, j:end_j] = batch_kernels
    
    return kernel_matrix

# Bad: Load everything into memory
def memory_inefficient_kernels(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Inefficient kernel computation."""
    all_pairs = [(x1, x2) for x1 in X1 for x2 in X2]  # Memory explosion
    return np.array([quantum_kernel(x1, x2) for x1, x2 in all_pairs])
```

### Caching Strategy

```python
from functools import lru_cache
import hashlib

class QuantumKernelCache:
    """Intelligent caching for quantum kernel computations."""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
    
    def _hash_array(self, arr: np.ndarray) -> str:
        """Create hash for numpy array."""
        return hashlib.md5(arr.tobytes()).hexdigest()
    
    def get_kernel(self, x1: np.ndarray, x2: np.ndarray) -> Optional[float]:
        """Get cached kernel value."""
        key = (self._hash_array(x1), self._hash_array(x2))
        return self.cache.get(key)
    
    def set_kernel(self, x1: np.ndarray, x2: np.ndarray, value: float) -> None:
        """Cache kernel value."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            self.cache.pop(next(iter(self.cache)))
        
        key = (self._hash_array(x1), self._hash_array(x2))
        self.cache[key] = value
```

## 🔒 Security Guidelines

### Input Validation

```python
def validate_quantum_data(data: np.ndarray) -> None:
    """Validate quantum data inputs."""
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be numpy array")
    
    if data.dtype not in [np.float32, np.float64, np.complex64, np.complex128]:
        raise ValueError("Data must be numeric")
    
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values")
    
    if data.size == 0:
        raise ValueError("Data cannot be empty")

def sanitize_backend_name(name: str) -> str:
    """Sanitize backend name to prevent injection."""
    allowed_chars = set("abcdefghijklmnopqrstuvwxyz0123456789_-")
    
    if not name.isascii():
        raise ValueError("Backend name must be ASCII")
    
    if not all(c.lower() in allowed_chars for c in name):
        raise ValueError("Backend name contains invalid characters")
    
    return name.lower()
```

### Error Handling

```python
class SuperQuantXError(Exception):
    """Base exception for SuperQuantX."""
    pass

class BackendError(SuperQuantXError):
    """Error related to quantum backend operations."""
    pass

class CircuitError(SuperQuantXError):
    """Error in quantum circuit construction or execution."""
    pass

class AlgorithmError(SuperQuantXError):
    """Error in quantum algorithm execution."""
    pass

# Good error handling
def execute_quantum_circuit(circuit: QuantumCircuit) -> MeasurementResult:
    """Execute quantum circuit with proper error handling."""
    try:
        validate_circuit(circuit)
        backend = get_backend()
        result = backend.execute(circuit)
        return process_result(result)
        
    except BackendError as e:
        logger.error(f"Backend execution failed: {e}")
        raise AlgorithmError(f"Circuit execution failed: {e}") from e
    
    except Exception as e:
        logger.error(f"Unexpected error in circuit execution: {e}")
        raise AlgorithmError("Unexpected error during execution") from e
```

## 📊 Logging Standards

### Logging Configuration

```python
import logging
from typing import Optional

def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[str] = None
) -> None:
    """Configure logging for SuperQuantX."""
    
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        filename=log_file
    )
    
    # Quantum-specific logger
    quantum_logger = logging.getLogger("superquantx.quantum")
    quantum_logger.setLevel(logging.DEBUG)

# Usage in modules
logger = logging.getLogger(__name__)

def quantum_algorithm_step(params: np.ndarray) -> float:
    """Execute one step of quantum algorithm."""
    logger.debug(f"Starting algorithm step with params shape: {params.shape}")
    
    try:
        cost = compute_cost_function(params)
        logger.info(f"Cost function value: {cost:.6f}")
        return cost
        
    except Exception as e:
        logger.error(f"Algorithm step failed: {e}", exc_info=True)
        raise
```

## 🔄 Git Workflow

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(algorithms): add quantum neural network implementation

- Implement parameterized quantum circuit for ML
- Add gradient computation using parameter shift rule
- Include example usage in documentation

Closes #123

fix(backends): handle simulator timeout gracefully

The simulator backend now catches timeout exceptions and provides
a meaningful error message to the user.

Fixes #456

docs(tutorials): update quantum ML tutorial with latest API

- Update code examples to use new QuantumSVM interface
- Add performance benchmarking section
- Fix typos in mathematical expressions
```

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test improvements

---

These standards ensure SuperQuantX maintains high code quality, consistency, and reliability while supporting quantum computing research and development.