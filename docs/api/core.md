# Core API Reference

The core API provides fundamental functionality for SuperQuantX, including backend management, configuration, and system information.

## Package-Level Functions

### Version Information

::: superquantx.get_version
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.get_backend_info
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.print_system_info
    handler: python
    options:
      docstring_style: google
      show_source: true

### Backend Management

::: superquantx.get_backend
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.list_available_backends
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.check_backend_compatibility
    handler: python
    options:
      docstring_style: google
      show_source: true

## Configuration

::: superquantx.config
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.configure
    handler: python
    options:
      docstring_style: google
      show_source: true

## Backend Classes

### Base Backend

::: superquantx.backends.BaseBackend
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

### Auto Backend

::: superquantx.AutoBackend
    handler: python
    options:
      docstring_style: google
      show_source: true

## Basic Usage

```python
import superquantx as sqx

print(sqx.get_version())
print(sqx.list_available_backends())
sqx.print_system_info()
```

### Backend Selection

```python
import superquantx as sqx

backend = sqx.get_backend("auto")
print(type(backend).__name__)

sim = sqx.get_backend("simulator", max_qubits=8, shots=512)
print(sim.get_backend_info()["backend_name"])
print(sqx.check_backend_compatibility("qiskit")["compatible"])
```

### Configuration

```python
import superquantx as sqx

cfg = sqx.get_config()
print(sorted(cfg.keys())[:8])

sqx.configure(backend="simulator", shots=512)
print(sqx.get_default_backend())
```

### Portable Circuit Across Backends

```python
import superquantx as sqx

results = {}
for name in ["simulator", "pennylane", "qiskit"]:
    backend = sqx.get_backend(name)
    circuit = backend.create_circuit(2)
    backend.add_gate(circuit, "H", 0)
    backend.add_gate(circuit, "CNOT", [0, 1])
    backend.add_measurement(circuit)
    results[name] = backend.execute_circuit(circuit, shots=200)["counts"]
print(results)
```

### Error Handling

```python
import superquantx as sqx
from superquantx.exceptions import BackendError, BackendNotAvailableError

try:
    backend = sqx.get_backend("simulator")
    print(backend.is_available() if hasattr(backend, "is_available") else True)
except (BackendError, BackendNotAvailableError, ImportError) as e:
    print(type(e).__name__, e)
    backend = sqx.get_backend("simulator")
```

### Hardware Backends (credentials required)

!!! note "Requires credentials"
    IBM, AWS, D-Wave and similar constructors need accounts; not run in CI.

<!-- docs-check: skip -->
```python
import superquantx as sqx

ibm = sqx.QiskitBackend(device="ibm_brisbane", shots=1024)
braket = sqx.BraketBackend(
    device="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
    s3_folder=("my-braket-bucket", "runs"),
)
ocean = sqx.OceanBackend(device="advantage", token="YOUR_DWAVE_LEAP_TOKEN")
```

## See Also

- [Backends guide](../user-guide/backends.md)
- [Per-backend pages](../backends/simulator.md)
- [BACKENDS overview](../BACKENDS.md)
