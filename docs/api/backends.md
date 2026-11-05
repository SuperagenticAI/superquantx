# Backends API Reference

Backend classes wrap a quantum framework behind one portable interface.
Autodoc for the base class and factory is below; runnable examples follow.

## Base Backend

::: superquantx.backends.BaseBackend
    handler: python
    options:
      docstring_style: google
      show_source: false
      show_root_heading: true

## Factory

::: superquantx.get_backend
    handler: python
    options:
      docstring_style: google
      show_source: false
      show_root_heading: true

## Creating Backends

```python
import superquantx as sqx

sim = sqx.get_backend("simulator", max_qubits=10, shots=1000)
pl = sqx.get_backend("pennylane", device="default.qubit")
qk = sqx.get_backend("qiskit")
print(type(sim).__name__, type(pl).__name__, type(qk).__name__)
```

## Portable Circuit Execution

```python
import superquantx as sqx

def bell_counts(name, shots=300):
    backend = sqx.get_backend(name)
    circuit = backend.create_circuit(2)
    backend.add_gate(circuit, "H", 0)
    backend.add_gate(circuit, "CNOT", [0, 1])
    backend.add_measurement(circuit)
    return backend.execute_circuit(circuit, shots=shots)["counts"]

for name in ["simulator", "pennylane", "qiskit", "cirq", "braket", "tket"]:
    print(name, bell_counts(name))
```

## Fallback Selection

```python
import superquantx as sqx

def first_working(names):
    for name in names:
        try:
            return sqx.get_backend(name)
        except Exception as e:
            print(f"{name} unavailable: {type(e).__name__}")
    raise RuntimeError("no backend available")

backend = first_working(["pennylane", "qiskit", "simulator"])
print(type(backend).__name__)
```

## Hardware (credentials required)

!!! note "Requires credentials"
    Not run in the docs checker.

<!-- docs-check: skip -->
```python
import superquantx as sqx

ibm = sqx.get_backend("qiskit", device="ibm_brisbane")
braket = sqx.BraketBackend(
    device="arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1",
    s3_folder=("my-braket-bucket", "runs"),
)
ocean = sqx.get_backend("ocean", device="advantage", token="YOUR_DWAVE_LEAP_TOKEN")
```

## See Also

- [BACKENDS overview](../BACKENDS.md)
- [User guide](../user-guide/backends.md)
- Per-backend pages under `docs/backends/`
