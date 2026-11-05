# Qiskit Backend Integration

Qiskit is IBM's open-source quantum computing framework that provides access to quantum simulators and real IBM quantum hardware. SuperQuantX integrates seamlessly with Qiskit to provide access to IBM's quantum ecosystem.

## Overview

The Qiskit backend in SuperQuantX offers:

- **IBM Quantum Hardware Access**: Connect to real IBM quantum computers
- **High-Performance Simulators**: Use Qiskit Aer simulators
- **Noise Models**: Simulate realistic quantum device noise
- **Quantum Circuits**: Native Qiskit circuit manipulation
- **Job Management**: Automatic job queue handling

## Installation

```bash
# Install SuperQuantX with Qiskit support
uv add "superquantx[qiskit]==0.3.0"

# Or install Qiskit directly
pip install qiskit qiskit-aer qiskit-ibm-runtime
```

## Quick Start

### Basic Usage

```python
import superquantx as sqx

# Get the Qiskit backend (local simulator by default)
backend = sqx.get_backend("qiskit")

# Bell state through the unified circuit API
circuit = backend.create_circuit(2)
circuit = backend.add_gate(circuit, "H", 0)
circuit = backend.add_gate(circuit, "CNOT", [0, 1])
circuit = backend.add_measurement(circuit)

result = backend.execute_circuit(circuit, shots=1000)
print(result["counts"])  # about 500 each of '00' and '11'
```

### IBM Quantum Hardware

!!! note "Requires credentials"
    Needs an IBM Quantum token; checked against the 0.3.0 constructor, not run.

<!-- docs-check: skip -->
```python
from qiskit_ibm_runtime import QiskitRuntimeService

# Save your credentials (one-time setup)
QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="YOUR_IBM_QUANTUM_TOKEN",
    overwrite=True,
)

# Any device name starting with "ibm" goes to IBM Quantum
backend = sqx.QiskitBackend(device='ibm_brisbane', shots=1024)
result = backend.execute_circuit(circuit, shots=1024)

# Or pass the service arguments directly
backend = sqx.QiskitBackend(
    device='ibm_brisbane',
    provider={'channel': 'ibm_quantum_platform', 'token': 'YOUR_TOKEN', 'instance': 'YOUR_CRN'},
)
```

## Configuration Options

### Backend Selection

```python
# Devices understood by QiskitBackend(device=...)
devices = {
    'aer_simulator': 'Local Aer simulator (default; also qasm_simulator, statevector_simulator)',
    'statevector': 'Aer with a fixed method (also density_matrix, stabilizer, matrix_product_state, ...)',
    'ibm_brisbane': 'IBM Quantum device by name (requires credentials)',
}

backend = sqx.QiskitBackend(device='statevector', shots=2048)
print(backend.backend.options.method)
```

### Advanced Configuration

```python
# Custom configuration
backend = sqx.QiskitBackend(device='aer_simulator', shots=1000)

# Access the underlying Qiskit backend (an AerSimulator) to change its options
backend.backend.set_options(seed_simulator=42, max_parallel_threads=4)
print(backend.backend.options.seed_simulator)
```

`QiskitBackend` takes `device`, `provider` and `shots`. Options such as an optimisation level, noise model or memory are set on the Qiskit objects (`backend.backend`, `qiskit.transpile`) as shown below.

## Working with Circuits

### Circuit Creation

```python
import numpy as np


def demonstrate_qiskit_circuits():
    """Backend circuits are qiskit.QuantumCircuit objects, so native methods work."""
    backend = sqx.get_backend('qiskit')
    circuit = backend.create_circuit(4)

    circuit.h(0)              # Hadamard
    circuit.x(1)              # Pauli-X
    circuit.y(2)              # Pauli-Y
    circuit.z(3)              # Pauli-Z
    circuit.rx(np.pi / 4, 0)  # rotations
    circuit.ry(np.pi / 3, 1)
    circuit.rz(np.pi / 6, 2)
    circuit.cx(0, 1)          # CNOT
    circuit.cz(1, 2)          # controlled-Z
    circuit.swap(2, 3)        # SWAP
    circuit.ccx(0, 1, 2)      # Toffoli
    circuit.s(0)              # S
    circuit.t(1)              # T

    # The same gates through the portable API: backend.add_gate(circuit, 'H', 0), ...
    circuit = backend.add_measurement(circuit)
    result = backend.execute_circuit(circuit, shots=1000)
    print(f"Counts (qubit 0 is the rightmost bit): {result['counts']}")
    print(circuit.draw())


demonstrate_qiskit_circuits()
```

### Parameterized Circuits

```python
import numpy as np


def qiskit_parameterized_circuits(seed=0):
    """Hardware-efficient ansatz and a parameter sensitivity check."""
    backend = sqx.get_backend('qiskit')
    n_qubits, layers = 4, 2

    def create_ansatz(params):
        circuit = backend.create_circuit(n_qubits)
        idx = 0
        for q in range(n_qubits):
            circuit.ry(params[idx], q)
            idx += 1
        for _ in range(layers):
            for q in range(n_qubits - 1):
                circuit.cx(q, q + 1)
            for q in range(n_qubits):
                circuit.ry(params[idx], q)
                circuit.rz(params[idx + 1], q)
                idx += 2
        return circuit

    n_params = n_qubits + 2 * layers * n_qubits
    params = np.random.default_rng(seed).random(n_params) * 2 * np.pi

    # Exact probabilities from the statevector (qubit 0 is the leftmost bit here)
    def p_zero(p):
        return abs(backend.statevector_big_endian(create_ansatz(p))[0]) ** 2

    base = p_zero(params)
    print(f"P(0000) = {base:.4f}")
    for delta in [-0.1, -0.05, 0.05, 0.1]:
        shifted = params.copy()
        shifted[0] += delta
        print(f"  dtheta = {delta:+.2f}: P(0000) = {p_zero(shifted):.4f} "
              f"(change: {p_zero(shifted) - base:+.4f})")


qiskit_parameterized_circuits()
```

## IBM Quantum Hardware

### Real Hardware Execution

!!! note "Requires credentials"
    Needs an IBM Quantum account; checked against the 0.3.0 `QiskitBackend` constructor, not run.

<!-- docs-check: skip -->
```python
def run_on_ibm_hardware():
    """Run a Bell state on the least busy IBM device."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()
    device = service.least_busy(operational=True, simulator=False).name
    print(f"Using backend: {device}")

    # provider= takes the same keyword arguments as QiskitRuntimeService
    hw = sqx.QiskitBackend(device=device, shots=1024)
    circuit = hw.create_circuit(2)
    hw.add_gate(circuit, 'H', 0)
    hw.add_gate(circuit, 'CNOT', [0, 1])
    result = hw.execute_circuit(circuit)     # transpiled and run with SamplerV2

    counts = result['counts']
    print(f"Job {result['job_id']}: {counts}")
    print(f"Bell fidelity: {(counts.get('00', 0) + counts.get('11', 0)) / 1024:.3f}")


# run_on_ibm_hardware()
```

### Device Calibration Data

!!! note "Requires credentials"
    Needs an IBM Quantum account; checked against the `qiskit-ibm-runtime` API, not run.

<!-- docs-check: skip -->
```python
def analyze_device_properties(device='ibm_brisbane'):
    """Read calibration data and build a matching noisy simulator."""
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime import QiskitRuntimeService

    ibm_backend = QiskitRuntimeService().backend(device)
    target = ibm_backend.target
    print(f"Backend: {ibm_backend.name}, {ibm_backend.num_qubits} qubits")

    for q in range(5):
        props = target.qubit_properties[q]
        readout = target['measure'][(q,)].error
        print(f"  Qubit {q}: T1={props.t1 * 1e6:.1f}us, T2={props.t2 * 1e6:.1f}us, readout={readout:.4f}")

    # Noisy local simulation that mimics the device
    sim = sqx.QiskitBackend()
    sim.backend = AerSimulator.from_backend(ibm_backend)
    circuit = sim.create_circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    counts = sim.execute_circuit(circuit, shots=1000)['counts']
    print(f"Bell fidelity with device noise: {(counts.get('00', 0) + counts.get('11', 0)) / 1000:.3f}")
```

## Advanced Features

### Transpilation and Optimization

```python
import numpy as np
from qiskit import transpile


def demonstrate_transpilation():
    """Show Qiskit transpilation at each optimisation level."""
    backend = sqx.get_backend('qiskit')
    circuit = backend.create_circuit(5)
    for i in range(5):
        circuit.h(i)
    for _ in range(3):
        for i in range(4):
            circuit.cx(i, i + 1)
        for i in range(5):
            circuit.rz(np.pi / 4, i)
    circuit.measure_all(add_bits=False)
    print(f"Original: {circuit.size()} gates, depth {circuit.depth()}")

    # backend.backend is the underlying AerSimulator
    for level in range(4):
        compiled = transpile(circuit, backend.backend, optimization_level=level)
        counts = backend.execute_circuit(compiled, shots=1000)['counts']
        print(f"Level {level}: {compiled.size()} gates, depth {compiled.depth()}, "
              f"{len(counts)} outcomes")


demonstrate_transpilation()
```

`execute_circuit` itself transpiles with Qiskit's default settings; there is no `optimization_level` argument on `QiskitBackend` in 0.3.0.

### Custom Noise Models

```python
def create_custom_noise_models():
    """Build a Qiskit Aer noise model and compare clean and noisy results."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (
        NoiseModel,
        ReadoutError,
        depolarizing_error,
        thermal_relaxation_error,
    )

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.001, 1), ['h', 'x'])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ['cx'])
    for qubit in range(3):   # T1 = 50 us, T2 = 70 us, 35 ns gates
        noise_model.add_quantum_error(
            thermal_relaxation_error(50_000, 70_000, 35), ['id'], [qubit]
        )
    noise_model.add_all_qubit_readout_error(ReadoutError([[0.95, 0.05], [0.1, 0.9]]))
    print("Noisy instructions:", noise_model.noise_instructions)

    clean = sqx.QiskitBackend(shots=2000)
    noisy = sqx.QiskitBackend(shots=2000)
    noisy.backend = AerSimulator(noise_model=noise_model)   # swap in the noisy simulator

    for n in (2, 3):                                          # Bell and GHZ states
        fidelities = []
        for b in (clean, noisy):
            circuit = b.create_circuit(n)
            b.add_gate(circuit, 'H', 0)
            for q in range(n - 1):
                b.add_gate(circuit, 'CNOT', [q, q + 1])
            counts = b.execute_circuit(circuit)['counts']
            fidelities.append((counts.get('0' * n, 0) + counts.get('1' * n, 0)) / 2000)
        print(f"  {n}-qubit GHZ: clean {fidelities[0]:.3f}, noisy {fidelities[1]:.3f}")


create_custom_noise_models()
```

### Error Mitigation

```python
def demonstrate_error_mitigation():
    """Zero-noise extrapolation and readout correction on a noisy Aer simulator."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.02, 1), ['x'])
    noise_model.add_all_qubit_readout_error(ReadoutError([[0.97, 0.03], [0.06, 0.94]]))
    noisy = sqx.QiskitBackend(shots=4000)
    noisy.backend = AerSimulator(noise_model=noise_model)

    # 1. Zero-noise extrapolation: X^(2k+1) = X, but noise grows with k
    print("Zero-noise extrapolation (ideal <Z> of |1> is -1):")
    factors, values = [1, 3, 5], []
    for factor in factors:
        circuit = noisy.create_circuit(1)
        for _ in range(factor):
            circuit.x(0)
            circuit.barrier()   # stop the transpiler from cancelling X pairs
        circuit.measure_all(add_bits=False)
        counts = noisy.execute_circuit(circuit)['counts']
        z = (counts.get('0', 0) - counts.get('1', 0)) / 4000
        values.append(z)
        print(f"  noise factor {factor}: <Z> = {z:.3f}")
    zero_noise = np.polyfit(factors, values, 1)[1]
    print(f"  extrapolated: <Z> = {zero_noise:.3f}")

    # 2. Readout correction from a calibration matrix
    def measure(prep_one):
        circuit = noisy.create_circuit(1)
        if prep_one:
            circuit.x(0)
        circuit.measure_all(add_bits=False)
        counts = noisy.execute_circuit(circuit)['counts']
        return np.array([counts.get('0', 0), counts.get('1', 0)]) / 4000

    cal = np.column_stack([measure(False), measure(True)])   # columns: prepared 0, prepared 1
    raw = measure(True)
    corrected = np.linalg.solve(cal, raw)
    print(f"Readout correction: raw P(1)={raw[1]:.3f}, corrected P(1)={corrected[1]:.3f}")


demonstrate_error_mitigation()
```

SuperQuantX also has helpers for this in `superquantx.noise` (`ZeroNoiseExtrapolation`) and `superquantx.measurements.ResultAnalyzer`; note that `ResultAnalyzer.readout_error_mitigation` is a placeholder in 0.3.0 and returns the counts unchanged.

## Performance Optimization

### Execution Time Analysis

```python
import time

import numpy as np
from qiskit import transpile


def qiskit_performance_analysis(seed=0):
    """Execution time by circuit size, and shot count against accuracy."""
    rng = np.random.default_rng(seed)
    backend = sqx.QiskitBackend(device='aer_simulator')

    for n_qubits, n_layers in [(2, 1), (4, 2), (6, 3)]:
        circuit = backend.create_circuit(n_qubits)
        for _ in range(n_layers):
            for q in range(n_qubits):
                circuit.ry(rng.uniform(0, 2 * np.pi), q)
                circuit.rz(rng.uniform(0, 2 * np.pi), q)
            for q in range(n_qubits - 1):
                circuit.cx(q, q + 1)
        start = time.time()
        counts = backend.execute_circuit(circuit, shots=1000)['counts']
        print(f"  {n_qubits} qubits, {n_layers} layers: {time.time() - start:.3f}s, "
              f"{len(counts)} unique outcomes")

    # Shot count against sampling error for a Bell state
    bell = backend.create_circuit(2)
    bell.h(0)
    bell.cx(0, 1)
    for shots in [100, 1000, 10000]:
        p00 = backend.execute_circuit(bell, shots=shots)['counts'].get('00', 0) / shots
        print(f"  {shots:5d} shots: P(00)={p00:.3f}, error={abs(p00 - 0.5):.3f}")

    # Transpiler optimisation levels on a circuit with removable gates
    redundant = backend.create_circuit(4)
    for q in range(4):
        redundant.h(q)
        redundant.x(q)
        redundant.h(q)       # H X H = Z
    for q in range(3):
        redundant.cx(q, q + 1)
    for level in range(4):
        out = transpile(redundant, backend.backend, optimization_level=level)
        print(f"  Opt level {level}: {out.size()} gates, depth {out.depth()}")


qiskit_performance_analysis()
```

## Integration Examples

### Quantum Machine Learning with Qiskit

```python
def qiskit_quantum_ml_example():
    """Quantum SVM on the Qiskit backend."""
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = make_classification(n_samples=60, n_features=4, n_redundant=0, random_state=42)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Training: {len(X_train)}, Testing: {len(X_test)}")

    # Train on the Qiskit backend with each feature map
    for fm in ['ZFeatureMap', 'ZZFeatureMap', 'PauliFeatureMap']:
        fm_qsvm = sqx.QuantumSVM(backend=sqx.QiskitBackend(), feature_map=fm, feature_map_reps=1)
        fm_acc = accuracy_score(y_test, fm_qsvm.fit(X_train, y_train).predict(X_test))
        print(f"  {fm}: {fm_acc:.3f}")


qiskit_quantum_ml_example()
```

## Troubleshooting

### Common Issues

#### 1. IBM Quantum Authentication

!!! note "Requires credentials"
    Needs an IBM Quantum token; not run. Current `qiskit-ibm-runtime` accepts `channel="ibm_quantum_platform"` or `"ibm_cloud"`; the old `"ibm_quantum"` channel is rejected.

<!-- docs-check: skip -->
```python
from qiskit_ibm_runtime import QiskitRuntimeService

print("Saved accounts:", QiskitRuntimeService.saved_accounts())

QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="YOUR_TOKEN",
    overwrite=True,
)
```

#### 2. Backend Availability

!!! note "Requires credentials"
    Needs an IBM Quantum account; not run.

<!-- docs-check: skip -->
```python
from qiskit_ibm_runtime import QiskitRuntimeService

for ibm_backend in QiskitRuntimeService().backends():
    status = ibm_backend.status()
    print(f"{ibm_backend.name}: operational={status.operational}, "
          f"pending_jobs={status.pending_jobs}")
```

#### 3. Memory Issues

```python
# Large circuits: the matrix-product-state method needs far less memory
# than a full statevector for weakly entangled circuits
backend = sqx.QiskitBackend(device='matrix_product_state', shots=1000)

wide = backend.create_circuit(40)
backend.add_gate(wide, 'H', 0)
for q in range(39):
    backend.add_gate(wide, 'CNOT', [q, q + 1])
counts = backend.execute_circuit(wide)['counts']
print(sorted(counts))   # only all-zeros and all-ones
```

### Performance Tips

1. **Use appropriate simulators**: `statevector_simulator` for small circuits, `qasm_simulator` for general use
2. **Optimize circuits**: transpile with `qiskit.transpile(circuit, backend.backend, optimization_level=3)` before hardware execution
3. **Manage shots**: Balance accuracy vs execution time
4. **Batch jobs**: Submit multiple circuits together when possible
5. **Monitor queue**: Check IBM Quantum queue status before submission

## Best Practices

### Code Organization

```python
# qiskit_helpers.py
import time


class QiskitHelpers:
    @staticmethod
    def create_efficient_backend(n_qubits, ibm_device=None, provider=None):
        if ibm_device:   # requires credentials
            return sqx.QiskitBackend(device=ibm_device, provider=provider, shots=1024)
        method = 'statevector' if n_qubits <= 20 else 'matrix_product_state'
        return sqx.QiskitBackend(device=method, shots=1000)

    @staticmethod
    def submit_with_retry(backend, circuit, max_retries=3):
        for attempt in range(max_retries):
            try:
                return backend.execute_circuit(circuit)
            except Exception:
                if attempt == max_retries - 1:
                    raise
                print(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(2 ** attempt)  # exponential backoff


helper_backend = QiskitHelpers.create_efficient_backend(2)
bell = helper_backend.create_circuit(2)
bell.h(0)
bell.cx(0, 1)
print(QiskitHelpers.submit_with_retry(helper_backend, bell)['counts'])
```

### Error Handling

```python
def robust_qiskit_execution(circuit, device='ibm_brisbane', fallback='aer_simulator'):
    """Try a device and fall back to the local Aer simulator on failure."""
    try:
        return sqx.QiskitBackend(device=device).execute_circuit(circuit, shots=1000)
    except Exception as e:
        print(f"Execution failed with {device}: {e.__class__.__name__}")
        if device == fallback:
            raise
        print(f"Falling back to {fallback}...")
        return robust_qiskit_execution(circuit, fallback, fallback)


print(robust_qiskit_execution(bell)['counts'])
```

## Resources

### Documentation
- **Qiskit Documentation**: https://qiskit.org/documentation/
- **IBM Quantum**: https://quantum-computing.ibm.com/
- **Qiskit Textbook**: https://qiskit.org/textbook/

### Hardware Access
- **IBM Quantum Network**: Apply for access to premium quantum systems
- **Qiskit Runtime**: Optimized quantum program execution
- **IBM Quantum Lab**: Cloud-based development environment

---

*The Qiskit backend runs on Qiskit Aer and is tested on the simulator. IBM Quantum hardware access goes through qiskit-ibm-runtime with your own credentials and is untested.*

!!! tip "Optimization"
    For best performance on IBM hardware, transpile with `optimization_level=3` and minimize circuit depth. Consider using Qiskit Runtime for reduced latency.

!!! warning "Queue Times"
    Real IBM quantum hardware may have significant queue times. Check device status and consider using simulators for development.

!!! info "Hardware Access"
    IBM Quantum Network membership provides access to advanced quantum systems and priority queue access.