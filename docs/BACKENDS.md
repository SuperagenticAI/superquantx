# Backend Overview

SuperQuantX 0.3.0 exposes one portable gate-model API across several frameworks,
plus a separate Ocean backend for annealing. Prefer the per-backend guides for
detail:

| Backend | Guide | Local default | Notes |
|---|---|---|---|
| Simulator | [simulator](backends/simulator.md) | always | Pure Python, fluent `circuit.h` API |
| PennyLane | [pennylane](backends/pennylane.md) | `default.qubit` | Best for variational / QML work |
| Qiskit | [qiskit](backends/qiskit.md) | Aer simulator | IBM Quantum needs credentials |
| Cirq | [cirq](backends/cirq.md) | local Cirq simulator | Google hardware needs credentials |
| Braket | [braket](backends/braket.md) | local Braket SV | AWS / IonQ / Rigetti need credentials |
| TKET | [tket](backends/tket.md) | Aer via pytket-qiskit | Quantinuum hardware needs credentials |
| Ocean | [ocean](backends/ocean.md) | simulated annealing | QUBO / Ising, not gate circuits |

## Portable Gate-Model Pattern

```python
import numpy as np
import superquantx as sqx

for name in ["simulator", "pennylane", "qiskit", "cirq", "braket", "tket"]:
    backend = sqx.get_backend(name)
    circuit = backend.create_circuit(2)
    circuit = backend.add_gate(circuit, "H", 0)
    circuit = backend.add_gate(circuit, "CNOT", [0, 1])
    circuit = backend.add_measurement(circuit)
    counts = backend.execute_circuit(circuit, shots=200)["counts"]
    print(name, counts)
```

## Selecting a Backend

```python
import superquantx as sqx

print(sqx.list_available_backends())
backend = sqx.get_backend("auto")          # first available gate-model backend
print(type(backend).__name__)
```

## Hardware Access

Hardware constructors accept credentials but are not exercised in CI.

!!! note "Requires credentials"
    IBM Quantum, AWS Braket QPUs, Quantinuum and D-Wave Leap all need accounts.
    Snippets below are API-checked only.

<!-- docs-check: skip -->
```python
import superquantx as sqx

# IBM Quantum (Qiskit Runtime)
ibm = sqx.QiskitBackend(device="ibm_brisbane", shots=1024)

# Amazon Braket QPU
braket = sqx.BraketBackend(
    device="arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1",
    s3_folder=("my-braket-bucket", "runs"),
    shots=100,
)

# D-Wave Leap
ocean = sqx.OceanBackend(device="advantage", token="YOUR_DWAVE_LEAP_TOKEN")
```

There is no separate `quantinuum` backend name in 0.3.0; use `tket` locally.
Quantinuum cloud access needs `pytket-quantinuum`, which is not part of the
`superquantx[tket]` extra.

## Annealing (Ocean)

```python
import superquantx as sqx

backend = sqx.get_backend("ocean", device="simulator")
Q = {(0, 0): 1, (1, 1): -2, (0, 1): 1}
result = backend.solve_qubo(Q, num_reads=200)
print(result["samples"][0], result["energies"][0])
```

## Further Reading

- [User guide: backends](user-guide/backends.md)
- [API: backends](api/backends.md)
