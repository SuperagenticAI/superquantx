# D-Wave Ocean Integration

D-Wave Ocean targets quantum annealing and classical simulated annealing.
Problems are QUBO or Ising models, not gate-model circuits.

## Overview

`OceanBackend` provides:

- Local simulated annealing (`device='simulator'`) for offline work
- D-Wave Advantage / hybrid solvers when a Leap token is configured
- Helpers for MaxCut, TSP and generic QUBO / Ising problems

Gate-model methods such as `add_gate` / `execute_circuit` are not meaningful
on this backend.

## Installation

```bash
uv add "superquantx[ocean]==0.3.0"
```

### Verify Installation

```python
import superquantx as sqx

backend = sqx.get_backend("ocean", device="simulator")
print(backend.get_version_info())
print(backend.is_available())
```

## Quick Start

### QUBO on the Local Simulator

```python
import superquantx as sqx

backend = sqx.get_backend("ocean", device="simulator")

# Minimise x0 - 2 x1 + x0 x1
Q = {
    (0, 0): 1,
    (1, 1): -2,
    (0, 1): 1,
}

result = backend.solve_qubo(Q, num_reads=1000)
print(result["samples"][0], result["energies"][0])
```

### Ising Model

```python
import superquantx as sqx

backend = sqx.get_backend("ocean", device="simulator")
h = {0: -1, 1: -1}
J = {(0, 1): -1}

result = backend.solve_ising(h, J, num_reads=1000)
for sample, energy in zip(result["samples"][:3], result["energies"][:3]):
    print(sample, energy)
```

## Configuration

### Local Simulator

```python
import superquantx as sqx

backend = sqx.OceanBackend(device="simulator", shots=1000)
print(backend.get_device_info())
```

### D-Wave Hardware and Hybrid

!!! note "Requires credentials"
    Needs a D-Wave Leap API token. Checked against the 0.3.0 constructor, not run.

<!-- docs-check: skip -->
```python
import superquantx as sqx

# Advantage QPU
advantage = sqx.OceanBackend(
    device="advantage",
    token="YOUR_DWAVE_LEAP_TOKEN",
    shots=100,
)

# Hybrid solver
hybrid = sqx.OceanBackend(
    device="hybrid",
    token="YOUR_DWAVE_LEAP_TOKEN",
    shots=1,
)
```

Set the token via the environment if you prefer:

```bash
export DWAVE_API_TOKEN="your-api-token-here"
```

Optional D-Wave config file (`~/.dwave/config`):

```ini
[defaults]
endpoint = https://cloud.dwavesys.com/sapi
token = your-api-token-here
solver = Advantage_system4.1
```

## Built-in Problems

### MaxCut

```python
import superquantx as sqx

backend = sqx.get_backend("ocean", device="simulator")
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
result = backend.solve_max_cut(edges, num_reads=500)
print(result["samples"][0], result["energies"][0])
```

### TSP

`OceanBackend.solve_tsp` raises `NotImplementedError` in 0.3.0. Build a QUBO
with one-hot city/position penalties and call `solve_qubo` instead.

## Tips

- Always develop against `device='simulator'` first; it needs no token.
- Annealers do not run Hadamard / CNOT circuits. Convert the problem to QUBO
  or Ising, or use a gate-model backend instead.
- Hardware and hybrid calls are skipped by `scripts/check_docs_examples.py`
  via `<!-- docs-check: skip -->`.
