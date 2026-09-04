# SuperQuantX Backend Documentation

## Overview

SuperQuantX provides a unified API that works across all major quantum computing platforms. This documentation covers all supported backends and their specific features.

## Supported Backends

### Gate-Model Quantum Computers

#### 1. PennyLane Backend
**Device Name**: `'pennylane'`, `'pl'`, `'penny'`
**Provider**: Xanadu Quantum Technologies
**Type**: Quantum Machine Learning focused framework

```python
import superquantx as sqx

# Basic usage
qnn = sqx.QuantumNN(backend='pennylane', device='default.qubit')

# With specific device
backend = sqx.get_backend('pennylane', device='lightning.qubit', shots=1000)
```

**Features**:
- Automatic differentiation for quantum circuits
- Extensive ML library integration
- Support for multiple hardware providers through PennyLane plugins
- Optimized for variational quantum algorithms

**Supported Devices**:
- `default.qubit`: CPU simulator
- `lightning.qubit`: Fast C++ simulator  
- `strawberryfields.fock`: Continuous-variable quantum computing
- Hardware devices via plugins (IBM, Rigetti, etc.)

#### 2. Qiskit Backend  
**Device Name**: `'qiskit'`, `'qk'`, `'ibm'`
**Provider**: IBM Quantum
**Type**: Gate-model quantum circuits

```python
# Local simulator
qsvm = sqx.QuantumSVM(backend='qiskit', device='aer_simulator')

# IBM hardware (requires IBM account)
backend = sqx.get_backend('qiskit', 
                         device='ibmq_manila',  # Specific IBM device
                         hub='ibm-q',
                         group='open', 
                         project='main')
```

**Features**:
- Direct access to IBM Quantum Network
- Advanced noise modeling
- Quantum circuit transpilation and optimization
- Extensive gate set and circuit control

**Supported Devices**:
- `aer_simulator`: Local quantum simulator
- `statevector_simulator`: Statevector simulation
- `ibmq_*`: IBM quantum hardware (e.g., `ibmq_manila`, `ibmq_bogota`)

#### 3. Cirq Backend
**Device Name**: `'cirq'`, `'google'` 
**Provider**: Google Quantum AI
**Type**: NISQ-focused quantum circuits

```python
# Local simulator
qaoa = sqx.QAOA(backend='cirq', device='cirq_simulator')

# Google hardware (requires access)
backend = sqx.get_backend('cirq', device='weber', project_id='your-project')
```

**Features**:
- Google Quantum AI hardware access
- NISQ algorithm optimization
- Advanced circuit scheduling
- Native support for Google's quantum processors

**Supported Devices**:
- `cirq_simulator`: Local Cirq simulator
- `weber`, `rainbow`, `sycamore`: Google quantum processors (requires access)

#### 4. AWS Braket Backend ⭐ *NEW*
**Device Name**: `'braket'`, `'aws'`, `'amazon'`
**Provider**: Amazon Web Services
**Type**: Multi-vendor quantum access

```python
# Local simulator
backend = sqx.get_backend('braket', device='local:braket/braket_sv')

# IonQ hardware via AWS
backend = sqx.get_backend('braket', 
                         device='arn:aws:braket::device/qpu/ionq/ionQdevice',
                         s3_folder=('my-bucket', 'quantum-results'))

# Rigetti hardware via AWS  
backend = sqx.get_backend('braket', 
                         device='arn:aws:braket::device/qpu/rigetti/Aspen-M-3')
```

**Features**:
- Access to multiple quantum hardware providers
- Managed simulators with high performance
- Automatic result storage in S3
- Pay-per-use pricing model

**Supported Devices**:
- `local:braket/braket_sv`: Local state vector simulator
- IonQ trapped-ion processors
- Rigetti superconducting processors  
- D-Wave quantum annealers (via Braket)

#### 5. TKET/Quantinuum Backend ⭐ *NEW*
**Device Name**: `'tket'`, `'quantinuum'`, `'pytket'`, `'h1'`, `'h2'`
**Provider**: Quantinuum/Cambridge Quantum Computing
**Type**: Advanced compilation + trapped-ion hardware

```python
# Local simulator
backend = sqx.get_backend('tket', device='aer_simulator')

# Quantinuum H1-1E hardware
backend = sqx.get_backend('quantinuum', 
                         device='H1-1E',
                         api_key='your-quantinuum-key')

# With circuit optimization
circuit = backend.create_circuit(4)
optimized = backend.optimize_circuit(circuit, optimization_level=3)
```

**Features**:
- Advanced quantum circuit compilation and optimization
- Direct access to Quantinuum H-Series trapped-ion systems
- Industry-leading gate fidelities
- Sophisticated error mitigation

**Supported Devices**:
- `H1-1E`: 20-qubit trapped-ion processor
- `H1-2E`: 56-qubit trapped-ion processor  
- `H2-1`: 32-qubit second-generation system
- Various simulators through TKET

### Quantum Annealing Systems

#### 6. D-Wave Ocean Backend ⭐ *NEW* 
**Device Name**: `'ocean'`, `'dwave'`, `'annealing'`
**Provider**: D-Wave Systems
**Type**: Quantum annealing for optimization

```python
# Simulated annealing
backend = sqx.get_backend('ocean', device='simulator')

# D-Wave Advantage hardware
backend = sqx.get_backend('dwave', 
                         device='advantage',
                         token='your-dwave-token')

# Solve optimization problem
Q = {(0, 0): -1, (1, 1): -1, (0, 1): 2}  # QUBO formulation
result = backend.solve_qubo(Q, num_reads=1000)
```

**Features**:
- Quantum annealing for optimization problems
- QUBO and Ising model solving
- Hybrid classical-quantum algorithms
- Large problem sizes (5000+ variables)

**Supported Problems**:
- Quadratic Unconstrained Binary Optimization (QUBO)
- Ising model problems
- Maximum Cut, TSP, portfolio optimization
- Constrained optimization via penalty methods

## Backend Comparison

| Backend | Hardware Access | Max Qubits | Specialization | Cost Model |
|---------|----------------|------------|----------------|------------|
| **PennyLane** | Multi-vendor | Varies | Quantum ML | Provider-dependent |
| **Qiskit** | IBM Quantum | 127+ | General purpose | IBM pricing |
| **Cirq** | Google Quantum | 70+ | NISQ algorithms | Google Cloud |
| **Braket** | Multi-vendor | Varies | Cloud access | AWS pay-per-use |
| **Quantinuum** | H-Series | 56 | High fidelity | Premium pricing |
| **D-Wave** | Advantage | 5000+ | Optimization | Annealing-specific |

## Advanced Usage

### Backend Auto-Selection
```python
# Automatically select best available backend
backend = sqx.get_backend('auto')

# Custom preference order
for backend_name in ['quantinuum', 'braket', 'pennylane']:
    try:
        backend = sqx.get_backend(backend_name)
        break
    except ImportError:
        continue
```

### Multi-Backend Benchmarking
```python
backends = ['pennylane', 'qiskit', 'cirq', 'braket']
results = {}

for backend_name in backends:
    try:
        qnn = sqx.QuantumNN(backend=backend_name, n_layers=2)
        qnn.fit(X_train, y_train)
        accuracy = qnn.score(X_test, y_test)
        results[backend_name] = accuracy
    except Exception as e:
        results[backend_name] = f"Error: {e}"

print("Cross-platform results:", results)
```

### Hardware-Specific Optimization
```python
# TKET advanced optimization
tket_backend = sqx.get_backend('tket')
circuit = tket_backend.create_circuit(4)
# ... add gates ...
optimized_circuit = tket_backend.optimize_circuit(circuit, optimization_level=3)

# D-Wave problem-specific solving
ocean_backend = sqx.get_backend('ocean')
# Max Cut problem
import networkx as nx
G = nx.complete_graph(4)
result = ocean_backend.solve_max_cut(G, num_reads=1000)
```


## Backend Version Compatibility

Declared optional-extra floors (PyPI majors verified 2026-09-04). Installs target **Python ≥3.11**.

| Extra | Packages (declared floor) | Latest verified | Notes |
|-------|---------------------------|-----------------|-------|
| `pennylane` | `pennylane>=0.45.0`, `pennylane-lightning>=0.45.0` | 0.45.1 / 0.45.0 | Requires Python ≥3.11 |
| `qiskit` | `qiskit>=2.0,<3`, `qiskit-aer>=0.17.0`, `qiskit-ibm-runtime>=0.49.0` | 2.5.2 / 0.17.2 / 0.49.0 | Explicit Qiskit **2.x** policy (not open-ended ≥1.0) |
| `cirq` | `cirq>=1.7.0` | 1.7.0 | Requires Python ≥3.11 |
| `braket` | `amazon-braket-sdk>=1.127.0` | 1.127.0 | Requires Python ≥3.11 |
| `tket` | `pytket>=2.18.0`, `pytket-qiskit>=0.78.0` | 2.18.1 / 0.78.0 | pytket 2.x |
| `ocean` | `dwave-ocean-sdk>=9.0.0`, `dimod>=0.12.22` | 9.4.0 / 0.12.22 | Ocean 9.x |

Core scientific stack floors stay **3.11-compatible** (NumPy/SciPy latest majors that need Python ≥3.12 are intentionally not forced).

## Installation Requirements

### Core Backends (Included)
- PennyLane: `pip install pennylane`
- Qiskit: `pip install qiskit`
- Cirq: `pip install cirq`

### New Backends (Additional Install)
```bash
# AWS Braket
pip install amazon-braket-sdk boto3

# TKET/Quantinuum  
pip install pytket pytket-quantinuum

# D-Wave Ocean
pip install dwave-ocean-sdk dwave-system
```

### Authentication Setup

#### IBM Quantum
```python
from qiskit import IBMQ
IBMQ.save_account('your-ibm-token')
```

#### AWS Braket
```bash
aws configure  # Set up AWS credentials
```

#### Quantinuum
```python
# API key passed directly to backend
backend = sqx.get_backend('quantinuum', api_key='your-key')
```

#### D-Wave
```python
# Set environment variable
export DWAVE_API_TOKEN='your-dwave-token'
```

## Error Handling

All backends include graceful error handling:

```python
# Check backend availability
available = sqx.list_available_backends()
print("Available backends:", available)

# Validate specific backend
compatibility = sqx.check_backend_compatibility('quantinuum')
if compatibility['compatible']:
    backend = sqx.get_backend('quantinuum')
else:
    print("Issue:", compatibility['reason'])
```

## Research Applications

### For SuperXLab Research
- **Cross-platform validation**: Test quantum algorithms on all major platforms
- **Hardware comparison**: Benchmark quantum advantages across different architectures  
- **Quantinuum H-Series**: Direct access for trapped-ion research
- **D-Wave optimization**: Quantum annealing for complex optimization problems
- **Multi-vendor access**: AWS Braket provides access to multiple hardware types

This unified backend system enables comprehensive quantum computing research across the entire landscape of available quantum hardware and simulators.