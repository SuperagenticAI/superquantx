# Quantum Backends

SuperQuantX gives every backend the same small interface: build a circuit,
add gates, measure, run. The algorithms (VQE, QAOA, QSVM, QNN) are written
against that interface, so they run unchanged on every gate-model backend.

## What works today

Every row marked "tested" is exercised by the unit tests with a Bell state,
a qubit-order check, exact expectation values and the fidelity kernel.

| Backend name | SDK | Runs on | Status |
|---|---|---|---|
| `simulator` | none (NumPy) | local statevector | tested |
| `pennylane` | PennyLane | `default.qubit` | tested |
| `qiskit` | Qiskit + Aer | Aer simulator | tested |
| `cirq` | Cirq | Cirq simulator | tested |
| `braket` | Amazon Braket SDK | `LocalSimulator` (`device="local:braket_sv"`) | tested |
| `tket` | pytket | pytket local simulator | tested |
| `ocean` | D-Wave Ocean | QUBO/Ising only, needs a Leap token for hardware | annealing only, not a gate-model backend |
| IBM Quantum, Braket devices, Quantinuum | via `qiskit`, `braket`, `tket` | cloud hardware | untested, needs your own credentials |
| Azure Quantum, Rigetti Forest, cuQuantum, TensorFlow Quantum | | | planned, not implemented |

Check your own installation with a real Bell circuit on each backend:

```bash
superquantx list-backends
```

## Choosing a backend

```python
import superquantx as sqx

backend = sqx.get_backend("pennylane")      # or "simulator", "qiskit", "cirq", "tket"
braket = sqx.get_backend("braket", device="local:braket_sv")

print(sorted(sqx.list_available_backends()))
print(sqx.get_backend_info())               # installed SDK versions
```

Algorithms accept either a backend name or a backend object:

```python
import superquantx as sqx

vqe = sqx.VQE(hamiltonian={"Z": 1.0}, backend="qiskit")
print(vqe.find_ground_state())               # -1.0
```

## Building and running circuits

```python
import superquantx as sqx

backend = sqx.get_backend("cirq")
circuit = backend.create_circuit(3)
circuit = backend.add_gate(circuit, "H", 0)
circuit = backend.add_gate(circuit, "CNOT", [0, 1])
circuit = backend.add_gate(circuit, "CNOT", [1, 2])
circuit = backend.add_measurement(circuit)

result = backend.execute_circuit(circuit, shots=1000)
print(backend.counts_big_endian(result["counts"]))   # only '000' and '111'
```

Supported gates on every backend: `H`, `X`, `Y`, `Z`, `S`, `T`, `RX`, `RY`,
`RZ`, `CNOT`, `CZ`, `SWAP`, `CCX`. Controlled rotations (`CRX`, `CRY`,
`CRZ`) work everywhere except `braket`. An unknown gate raises an error
instead of being silently dropped.

### Bit order

Qiskit and the built-in simulator order bits little-endian; the other SDKs
are big-endian. `counts_big_endian` and `statevector_big_endian` convert any
backend's output to big-endian (qubit 0 is the leftmost bit), so results are
directly comparable across backends.

## Expectation values and kernels

```python
import numpy as np
import superquantx as sqx

backend = sqx.get_backend("pennylane")

circuit = backend.create_circuit(1)
circuit = backend.add_gate(circuit, "RY", 0, [np.pi])
print(backend.compute_expectation(circuit, {"Z": 1.0}))   # -1.0

X = np.array([[0.1, 0.2], [0.7, -0.4]])
K = backend.compute_kernel_matrix(X, X, feature_map="ZZFeatureMap")
print(np.diag(K))                                        # [1. 1.]
```

Feature maps: `ZZFeatureMap`, `ZFeatureMap`, `PauliFeatureMap`,
`AngleEncoding`. Kernels are computed exactly from simulated statevectors.
They are real quantum-circuit kernels, but classically simulated, so they say
nothing about quantum advantage on their own. Use
[`compare_to_classical`](benchmarking.md) to see how they stack up.

## Hardware

Running on cloud hardware goes through the underlying SDK and your own
account (IBM Quantum token for `qiskit`, AWS credentials for `braket`,
Quantinuum credentials for `tket`). These paths are not covered by the test
suite. Start on the local simulators and move to hardware only once a result
holds up against classical baselines.
