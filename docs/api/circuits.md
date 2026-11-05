# Circuits API Reference

This page covers circuit construction, gate matrices, Pauli strings and Hamiltonians, measurement results and noise channels. The usage examples at the end run on SuperQuantX 0.3.0.

!!! info "Two kinds of circuit"
    - **Backend circuits** come from `backend.create_circuit(n)` and are built with `backend.add_gate(...)` or, on the simulator, with methods such as `circuit.h(0)`. These are what `backend.execute_circuit` runs.
    - **`superquantx.QuantumCircuit`** is a backend-independent circuit description that can be drawn, composed, inverted and serialised to JSON. In 0.3.0 no backend executes it directly.

## Quantum Circuits

### Quantum Circuit

::: superquantx.circuits.QuantumCircuit
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

### Quantum Gate

::: superquantx.circuits.QuantumGate
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

### Classical Register

::: superquantx.circuits.ClassicalRegister
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

## Quantum Gates

### Gate Matrix Library

::: superquantx.gates.GateMatrix
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

### Pauli String Operations

::: superquantx.gates.PauliString
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

### Hamiltonian Operations

::: superquantx.gates.Hamiltonian
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

## Measurements

### Measurement Result

::: superquantx.measurements.MeasurementResult
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

### Quantum Measurement

::: superquantx.measurements.QuantumMeasurement
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

### Result Analyzer

::: superquantx.measurements.ResultAnalyzer
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

## Noise Models

### Noise Channel Base

::: superquantx.noise.NoiseChannel
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

### Specific Noise Channels

::: superquantx.noise.BitFlipChannel
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

::: superquantx.noise.PhaseFlipChannel
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

::: superquantx.noise.DepolarizingChannel
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

::: superquantx.noise.AmplitudeDampingChannel
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

### Noise Model

::: superquantx.noise.NoiseModel
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true

## Usage Examples

### Building and running a circuit

```python
import numpy as np
import superquantx as sqx

backend = sqx.get_backend('simulator')   # or 'qiskit', 'pennylane', 'cirq', 'braket', 'tket'

circuit = backend.create_circuit(3)
backend.add_gate(circuit, 'H', 0)
backend.add_gate(circuit, 'CNOT', [0, 1])
backend.add_gate(circuit, 'RY', 2, [np.pi / 4])
backend.add_measurement(circuit)

result = backend.execute_circuit(circuit, shots=1000)
print(result['counts'])                              # qubit 0 is the rightmost bit
print(backend.counts_big_endian(result['counts']))   # qubit 0 first
```

`add_gate(circuit, gate, qubits, params=None)` uses the same gate names on every gate-model backend, so this code runs unchanged on any of them.

### Describing a circuit with QuantumCircuit

```python
qc = sqx.QuantumCircuit(3, name='ghz')
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.ry(np.pi / 4, 2)
qc.measure_all()

print(qc.draw())
print("Gates:", len(qc.gates))

# Serialise and restore
restored = sqx.QuantumCircuit.from_json(qc.to_json())
print("Round trip keeps gates:", len(restored.gates) == len(qc.gates))

# Inverse (without measurements) and composition
unitary_part = sqx.QuantumCircuit(3)
unitary_part.h(0)
unitary_part.cx(0, 1)
print(unitary_part.compose(unitary_part.inverse()).draw())
```

### Gate matrices, Pauli strings and Hamiltonians

```python
from superquantx.gates import GateMatrix, Hamiltonian, PauliString

theta = np.pi / 3
print("RX(pi/3) =\n", np.round(GateMatrix.rx(theta), 3))
print("H is unitary:", np.allclose(GateMatrix.H @ GateMatrix.H.conj().T, np.eye(2)))

pauli = PauliString('XYZX')               # X (x) Y (x) Z (x) X
print("Pauli string matrix shape:", pauli.matrix().shape)
print("XYZX commutes with ZZZZ:", pauli.commutes_with(PauliString('ZZZZ')))

heisenberg = Hamiltonian.heisenberg_model(num_qubits=4, Jx=1.0, Jy=1.0, Jz=1.0)
print(f"Heisenberg ground-state energy: {heisenberg.ground_state_energy():.4f}")
U = heisenberg.time_evolution(time=0.1)
print("exp(-iHt) is unitary:", np.allclose(U @ U.conj().T, np.eye(16)))
```

### Measurement analysis

`MeasurementResult` wraps a counts dictionary from any backend.

```python
from superquantx.measurements import MeasurementResult

bell = backend.create_circuit(2)
backend.add_gate(bell, 'H', 0)
backend.add_gate(bell, 'CNOT', [0, 1])
backend.add_measurement(bell)
raw = backend.execute_circuit(bell, shots=1000)

measurement = MeasurementResult(counts=raw['counts'], shots=raw['shots'])
print("Most frequent:", measurement.most_frequent)
print("Probabilities:", measurement.probabilities)
print("Marginal on qubit 0:", measurement.marginal_counts([0]))
print(f"Entropy: {measurement.entropy():.4f} bits")
print(f"<ZZ> from counts: {measurement.expectation_value('ZZ'):.4f}")
```

### Exact expectation values

For exact values on the simulator, combine the statevector with a `Hamiltonian`:

```python
bell = backend.create_circuit(2)
backend.add_gate(bell, 'H', 0)
backend.add_gate(bell, 'CNOT', [0, 1])
state = backend.statevector_big_endian(bell)

for label in ['XX', 'YY', 'ZZ']:
    observable = Hamiltonian.from_dict({label: 1.0})
    print(f"<{label}> = {np.real(observable.expectation_value(state)):+.4f}")

target = np.array([1, 0, 0, 1]) / np.sqrt(2)
print(f"Fidelity with |Phi+>: {abs(np.vdot(target, state)) ** 2:.4f}")
```

`backend.compute_expectation(circuit, hamiltonian)` gives the same numbers through the backend interface.

### Noise channels

The channel classes give Kraus operators and act on density matrices.

```python
from superquantx.noise import (
    AmplitudeDampingChannel, DepolarizingChannel, NoiseModel, PhaseDampingChannel,
)

rho_plus = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)   # |+><+|
for channel in [DepolarizingChannel(0.1), AmplitudeDampingChannel(0.1), PhaseDampingChannel(0.1)]:
    out = channel.apply_to_density_matrix(rho_plus)
    purity = np.real(np.trace(out @ out))
    print(f"{type(channel).__name__:24s} purity {purity:.3f}, coherence {abs(out[0, 1]):.3f}")

# A NoiseModel records error rates, for example for documentation or a qiskit noise model
model = NoiseModel.basic_device_noise(single_qubit_error=0.001, two_qubit_error=0.01, readout_error=0.02)
print(model.single_qubit_error_rates, model.two_qubit_error_rates)
```

!!! bug "Known issue in 0.3.0"
    `BitFlipChannel` and `PhaseFlipChannel` raise `AttributeError` (they reference a missing `GateMatrix.I`). `DepolarizingChannel`, `AmplitudeDampingChannel` and `PhaseDampingChannel` work.

!!! note "Noisy simulation"
    The built-in simulator is noiseless and does not apply a `NoiseModel`. For noisy circuit simulation use the Qiskit backend with an Aer noise model (see the [Qiskit backend guide](../backends/qiskit.md)).

### Zero-noise extrapolation

`ResultAnalyzer.error_mitigation_zero_noise_extrapolation` fits measurements taken at several noise scale factors and extrapolates to zero noise.

```python
from superquantx.measurements import ResultAnalyzer

# <Z> measured at noise scale factors 1, 2 and 3 (0.8, 0.6, 0.4)
results = [
    MeasurementResult(counts={'0': 900, '1': 100}, shots=1000),
    MeasurementResult(counts={'0': 800, '1': 200}, shots=1000),
    MeasurementResult(counts={'0': 700, '1': 300}, shots=1000),
]
value, info = ResultAnalyzer.error_mitigation_zero_noise_extrapolation([1, 2, 3], results, observable='Z')
print(f"Zero-noise estimate of <Z>: {value:.3f} (fit: {info['method']})")
```

### Parameterised circuits

There is no symbolic `Parameter` type in 0.3.0. Pass numbers when you build the circuit, or use the ansatz helpers, which take a parameter array:

```python
n_qubits = 3
n_params = backend.ansatz_param_count('RealAmplitudes', n_qubits, reps=2)
params = np.random.default_rng(0).uniform(0, 2 * np.pi, n_params)

ansatz = backend.create_ansatz('RealAmplitudes', n_qubits, params, reps=2)
print(f"RealAmplitudes with {n_params} parameters, norm {np.linalg.norm(backend.get_statevector(ansatz)):.6f}")

feature_map = backend.create_feature_map(n_features=3, feature_map='ZZFeatureMap', reps=2)
encoded = backend.build_feature_map_circuit(np.array([0.1, 0.5, -0.3]), feature_map)
print("Encoded state norm:", round(float(np.linalg.norm(backend.get_statevector(encoded))), 6))
```

### Running many circuits

```python
def random_circuit(n_qubits, depth, rng):
    circuit = backend.create_circuit(n_qubits)
    for _ in range(depth):
        q = int(rng.integers(n_qubits))
        backend.add_gate(circuit, 'RY', q, [float(rng.uniform(0, np.pi))])
        backend.add_gate(circuit, 'CNOT', [q, (q + 1) % n_qubits])
    backend.add_measurement(circuit)
    return circuit


rng = np.random.default_rng(1)
circuits = [random_circuit(4, 5, rng) for _ in range(10)]
results = [backend.execute_circuit(c, shots=500) for c in circuits]
print("Executed", len(results), "circuits; first counts:", results[0]['counts'])
```

!!! note "Not available in 0.3.0"
    `CircuitOptimizer`, `QuantumCompiler`, `CircuitSimulator`, `Parameter` / `ParameterVector`, the `superquantx.circuits.library` module and `execute_circuits_parallel` do not exist. `QuantumMeasurement.measure_circuit`, `measure_observable` and the tomography helpers return placeholder outcomes rather than simulating the circuit, and `ResultAnalyzer.readout_error_mitigation` returns the counts unchanged. For circuit compilation use the TKET backend (see the [TKET guide](../backends/tket.md)).

## Best Practices

### Circuit Construction

1. **Use native gates**: prefer gates supported by the target backend
2. **Minimise depth**: shorter circuits keep more fidelity on hardware
3. **Mind connectivity**: two-qubit gates between distant qubits cost extra swaps on hardware

### Measurement Strategy

1. **Use enough shots**: the standard error of a probability p is sqrt(p(1-p)/shots)
2. **Report uncertainties** next to sampled values
3. **Compare with exact values** from the statevector when the circuit is small

---

For more examples see the [basic quantum computing tutorial](../tutorials/basic-quantum.md).
