# SuperQuantX MCP Server for AI Assistants

SuperQuantX ships a [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server so that
AI assistants such as Claude Desktop, Cursor, or any MCP client can build and run quantum circuits,
compare backends, and run small quantum machine learning and optimization experiments through one
unified API.

The server speaks MCP over **stdio**, uses the official
[`mcp` Python SDK](https://pypi.org/project/mcp/) (2.x), and runs on **simulators only by default**.

## Install

```bash
pip install "superquantx[mcp,qiskit,pennylane]"
# or, from a checkout
uv pip install ".[mcp,qiskit,pennylane]"
```

The `mcp` extra needs Python 3.10 or newer (SuperQuantX itself requires 3.11+). The built-in
`simulator` backend works with no extra SDK, so `pip install "superquantx[mcp]"` is enough to try it.

Two equivalent entry points start the server on stdio:

```bash
superquantx-mcp
python -m superquantx.mcp
```

Both are meant to be launched by an MCP client, not used interactively. `superquantx-mcp --help`
prints the supported environment variables.

## Tools

| Tool | What it does |
| --- | --- |
| `list_backends` | Every SuperQuantX backend with SDK install status, initialization, and a 2 qubit Bell smoke test. Status is `working`, `not_installed`, `init_failed`, `execution_failed`, or `wrong_results`. |
| `backend_versions` | Installed versions of SuperQuantX, Python, numpy/scipy/scikit-learn, every quantum SDK, and per-backend version info. |
| `run_circuit` | Run a circuit given as OpenQASM 2, OpenQASM 3, or a JSON gate list. Returns counts, probabilities, gate counts, and timing (parse, backend init, execute, total). |
| `bell_state_demo` | Run the same Bell circuit on several backends and report how many shots are correlated (`00` or `11`). |
| `qsvm_classify` | Quantum kernel SVM (SuperQuantX `QuantumSVM`) on a toy dataset or your own data, with kernel diagnostics and an RBF SVM baseline on the same split. |
| `qaoa_maxcut` | QAOA for MaxCut on a small graph with a classical optimizer, checked against exact brute force. |

Every tool returns structured JSON with `"ok": true` or `"ok": false` plus an `error` object. Tools
never raise: bad input comes back as `ValidationError`, and a backend failure is reported for that
backend only (for example `bell_state_demo` still returns results for the backends that work).

### Circuit input formats

OpenQASM 2:

```text
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
```

OpenQASM 3 (`qubit[n]` / `bit[n]` declarations, `c = measure q;`) is accepted too. The parser
supports a safe subset: standard gates (`id x y z h s sdg t tdg sx rx ry rz p u1 u2 u3 u cx cz swap ccx`
and common aliases such as `cnot`, `toffoli`, `phase`), register broadcast (`h q;`), parameter
expressions with `pi` and basic math (`rz(-3*pi/4) q[0];`). Gate definitions, classical control
flow, and `reset` are rejected with a clear message. `measure` and `barrier` are accepted and
ignored, since every qubit is measured at the end.

JSON gate list (a list, or an object with `n_qubits` and `gates`):

```json
{"n_qubits": 2, "gates": [
  {"gate": "ry", "qubits": [0], "params": ["pi/3"]},
  {"gate": "cx", "qubits": [0, 1]}
]}
```

Parameters may be numbers or expression strings such as `"pi/3"`, evaluated without `eval`.

**Bit order.** Counts are little-endian, the Qiskit convention: qubit 0 is the rightmost bit. All
backends are normalized to this order (PennyLane results are reversed internally).

### Notes on `qsvm_classify`

- Datasets: `moons`, `circles`, `blobs`, `xor` (2 features) and `iris` (versicolor vs virginica,
  4 features). Or pass `X`, `y` (and optionally `X_test`, `y_test`).
- `kernel="statevector"` (default) computes the exact fidelity kernel `|<phi(a)|phi(b)>|^2` from the
  statevectors of a Qiskit-convention `ZZFeatureMap` (or `ZFeatureMap`) produced by the selected
  backend SDK, then trains SuperQuantX `QuantumSVM` with it. Simulator, Qiskit, and PennyLane give
  identical kernels for the same data.
- `kernel="backend"` uses the backend's own `compute_kernel_matrix`. On SuperQuantX 0.2.0 this is a
  classical placeholder on `simulator` and is not a valid fidelity kernel on `qiskit`, so the tool
  reports `kernel_diagnostics.looks_valid` and a warning instead of silently returning numbers.
- `bandwidth` (default 0.5) scales features before encoding. Smaller bandwidths usually generalize
  better on these toy sets (see Shaydulin and Wild, "Importance of kernel bandwidth in quantum
  machine learning", Phys. Rev. A 106, 042407, 2022).
- A classical RBF SVM baseline is always reported unless `classical_baseline=false`. On these toy
  datasets the RBF baseline is often as good or better. Nothing here is evidence of quantum
  advantage.

### Notes on `qaoa_maxcut`

- `edges` is `[[u, v], ...]` or `[[u, v, weight], ...]`, nodes `0..n-1`.
- Circuit: Hadamards, then per layer `cx, rz(2*gamma*w), cx` for every edge and `rx(2*beta)` mixers,
  executed through the SuperQuantX unified backend API. Parameters come from a coarse grid search
  refined with COBYLA (scipy).
- The result includes the exact brute-force optimum, the approximation ratio (expected cut divided by
  max cut), the probability of sampling an optimal cut, and `found_optimum`.

## Limits and safety

Inputs are validated and capped. Defaults can be lowered or raised with environment variables, but
never above the hard ceiling:

| Variable | Default | Hard ceiling |
| --- | --- | --- |
| `SUPERQUANTX_MCP_MAX_QUBITS` | 10 | 16 |
| `SUPERQUANTX_MCP_MAX_SHOTS` | 10000 | 100000 |
| `SUPERQUANTX_MCP_MAX_GATES` | 500 | 2000 |
| `SUPERQUANTX_MCP_MAX_QASM_CHARS` | 20000 | 50000 |
| `SUPERQUANTX_MCP_MAX_SAMPLES` | 80 | 200 |
| `SUPERQUANTX_MCP_MAX_FEATURES` | 4 | 6 |
| `SUPERQUANTX_MCP_MAX_QAOA_NODES` | 10 | 12 |
| `SUPERQUANTX_MCP_MAX_QAOA_LAYERS` | 3 | 4 |

**Hardware is off by default.** Tools only use local simulators (`simulator`, Qiskit Aer, PennyLane
`default.qubit`, and the default local devices of the other SDKs). Passing a `device` to
`run_circuit` is refused unless the server was started with `SUPERQUANTX_MCP_ALLOW_HARDWARE=1`, and
even then the backend SDK's own credentials and configuration are required. Annealing-only backends
(D-Wave Ocean) are listed but not used for gate circuits.

All tools are annotated read-only and do not touch the filesystem. Logs go to stderr so they never
corrupt the stdio protocol stream.

## Client configuration

Use the absolute path of the `superquantx-mcp` script from the environment where you installed
SuperQuantX (`which superquantx-mcp`).

### Claude Desktop

Edit `claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/`, Windows:
`%APPDATA%\Claude\`):

```json
{
  "mcpServers": {
    "superquantx": {
      "command": "/absolute/path/to/venv/bin/superquantx-mcp",
      "args": [],
      "env": {
        "SUPERQUANTX_MCP_MAX_QUBITS": "10"
      }
    }
  }
}
```

### Cursor

Create `.cursor/mcp.json` in your project (or `~/.cursor/mcp.json` for all projects):

```json
{
  "mcpServers": {
    "superquantx": {
      "command": "/absolute/path/to/venv/bin/python",
      "args": ["-m", "superquantx.mcp"]
    }
  }
}
```

### Without a pre-made virtualenv (uvx)

Once this server is in a published release, `uvx` can run it in an isolated environment:

```json
{
  "mcpServers": {
    "superquantx": {
      "command": "uvx",
      "args": ["--from", "superquantx[mcp,qiskit,pennylane]", "superquantx-mcp"]
    }
  }
}
```

SuperQuantX 0.2.0 on PyPI does not yet include the server, so until the next release use a local
install as shown above.

### Programmatic client

`examples/mcp_client_demo.py` launches the server over stdio with the official SDK client and calls
every tool:

```python
from mcp import Client, StdioServerParameters

params = StdioServerParameters(command="python", args=["-m", "superquantx.mcp"])
async with Client(params) as client:
    result = await client.call_tool("bell_state_demo", {"shots": 1000})
    print(result.structured_content)
```

## How it relates to the IBM Qiskit MCP servers

IBM maintains [Qiskit MCP servers](https://github.com/Qiskit/mcp-servers) for the Qiskit stack
(Qiskit, Qiskit IBM Runtime, the IBM transpiler service, and Qiskit Code Assistant). Those are the
right choice for deep IBM Quantum workflows such as managing Runtime jobs on IBM hardware.

The SuperQuantX server is complementary: it is **multi-backend**. One circuit, one call, and the
assistant can compare the built-in simulator, Qiskit Aer, PennyLane, and (when installed) Cirq,
Braket, and TKET backends, plus ready-made QML and optimization experiments with classical
baselines. You can enable both servers in the same client.

## Verified end to end output

The output below is real. It was captured on 2 November 2026 from a fresh `uv venv` on Python
3.11.16 after `uv pip install ".[mcp,qiskit,pennylane]"` (mcp 2.3.0, qiskit 2.6.1, qiskit-aer
0.18.2, pennylane 0.45.1), by running `examples/mcp_client_demo.py` against both
`python -m superquantx.mcp` and the `superquantx-mcp` script over stdio. Both launch modes returned
the same results (up to shot noise). Timings are from a shared Linux cloud VM; JSON is trimmed for
length where marked.

```text
connected via /tmp/venv-e2e/bin/superquantx-mcp
tools: ['list_backends', 'backend_versions', 'run_circuit', 'bell_state_demo', 'qsvm_classify', 'qaoa_maxcut']
```

`list_backends` (trimmed):

```json
{
  "ok": true,
  "working_backends": ["simulator", "qiskit", "pennylane"],
  "backends": {
    "simulator": {"status": "working", "device": "simulator",
                  "smoke_test": {"circuit": "bell", "counts": {"11": 68, "00": 60}, "execute_ms": 0.34, "passed": true}},
    "qiskit": {"status": "working", "device": "aer_simulator",
               "smoke_test": {"circuit": "bell", "counts": {"11": 68, "00": 60}, "execute_ms": 338.66, "passed": true}},
    "pennylane": {"status": "working", "device": "default.qubit",
                  "smoke_test": {"circuit": "bell", "counts": {"11": 70, "00": 58}, "execute_ms": 2.82, "passed": true}},
    "cirq": {"sdk_installed": false, "status": "not_installed",
             "error": "ImportError: Backend 'cirq' requires additional dependencies: Cirq is required for CirqBackend. Install with: pip install cirq"},
    "braket": {"sdk_installed": false, "status": "not_installed", "error": "..."},
    "tket": {"sdk_installed": false, "status": "not_installed", "error": "..."},
    "ocean": {"sdk_installed": false, "gate_model": false, "status": "not_installed", "error": "..."}
  }
}
```

`backend_versions` (trimmed):

```json
{
  "ok": true,
  "superquantx": "0.2.0",
  "python": "3.11.16",
  "packages": {"qiskit": "2.6.1", "qiskit-aer": "0.18.2", "qiskit-ibm-runtime": "0.51.0",
               "pennylane": "0.45.1", "pennylane-lightning": "0.45.0", "numpy": "2.4.6",
               "scipy": "1.17.1", "scikit-learn": "1.9.1", "mcp": "2.3.0"},
  "not_installed": ["cirq", "cirq-core", "amazon-braket-sdk", "pytket", "pytket-qiskit", "dwave-ocean-sdk", "dimod"]
}
```

`bell_state_demo {"shots": 1000}`:

```json
{
  "ok": true,
  "shots": 1000,
  "results": {
    "simulator": {"ok": true, "counts": {"11": 511, "00": 489}, "correlated_fraction": 1.0, "execute_ms": 0.98},
    "qiskit": {"ok": true, "counts": {"11": 507, "00": 493}, "correlated_fraction": 1.0, "execute_ms": 76.15},
    "pennylane": {"ok": true, "counts": {"00": 502, "11": 498}, "correlated_fraction": 1.0, "execute_ms": 2.91}
  },
  "all_entangled": true
}
```

`run_circuit` with a 3 qubit GHZ circuit in OpenQASM 3 on `qiskit`:

```json
{
  "ok": true,
  "backend": "qiskit",
  "device": "aer_simulator",
  "n_qubits": 3,
  "input_format": "openqasm3",
  "gate_counts": {"h": 1, "cx": 2},
  "counts": {"000": 504, "111": 496},
  "probabilities": {"000": 0.504, "111": 0.496},
  "shots": 1000,
  "bit_order": "little-endian (qubit 0 is the rightmost bit)",
  "timing_ms": {"parse": 0.09, "backend_init_or_cache": 0.0, "execute": 66.15, "total": 66.29},
  "backend_warnings": []
}
```

Same circuit on `simulator`: `{"000": 513, "111": 487}` (execute 0.61 ms). On `pennylane`:
`{"000": 521, "111": 479}` (execute 3.38 ms).

`run_circuit` with the JSON gate list above (`ry(pi/3)` then `cx`, expected about 75% `00` and 25% `11`):

```json
{"ok": true, "input_format": "json", "gate_counts": {"ry": 1, "cx": 1},
 "counts": {"00": 734, "11": 266}, "shots": 1000}
```

`qsvm_classify {"dataset": "moons", "backend": "<b>", "n_samples": 40, "seed": 7}`, same result on
all three backends (only the time differs):

```json
{
  "ok": true,
  "backend": "simulator",
  "n_train": 28,
  "n_test": 12,
  "n_qubits": 2,
  "feature_map": "ZZFeatureMap",
  "kernel_method": "exact fidelity kernel from simulator statevectors of a ZZFeatureMap (reps=1, bandwidth=0.5), fed to SuperQuantX QuantumSVM",
  "quantum": {"test_accuracy": 0.8333, "train_accuracy": 0.8571, "time_ms": 16.4},
  "kernel_diagnostics": {"mean_diagonal": 1.0, "expected_diagonal": 1.0, "mean_off_diagonal": 0.2888,
                         "quantum_kernel_used": true, "looks_valid": true},
  "classical_rbf_baseline": {"test_accuracy": 0.8333, "time_ms": 2.1},
  "note": "Toy-scale comparison on a noiseless simulator. Matching or beating the RBF baseline here is not evidence of quantum advantage."
}
```

Kernel time: simulator 16.4 ms, qiskit 39.3 ms, pennylane 68.4 ms.

`qsvm_classify {"dataset": "iris", "backend": "simulator", "n_samples": 40, "seed": 7}` (4 qubits),
where the classical baseline wins:

```json
{"ok": true, "n_qubits": 4,
 "quantum": {"test_accuracy": 0.6667, "train_accuracy": 0.9643, "time_ms": 298.0},
 "kernel_diagnostics": {"mean_diagonal": 1.0, "mean_off_diagonal": 0.1189, "looks_valid": true},
 "classical_rbf_baseline": {"test_accuracy": 0.9167, "time_ms": 2.3}}
```

`qaoa_maxcut {"edges": [[0,1],[1,2],[2,3],[3,0],[0,2]], "p": 1, "seed": 0}` (square with one diagonal,
max cut 4) on `simulator` (trimmed):

```json
{
  "ok": true,
  "n_nodes": 4,
  "p": 1,
  "shots": 1024,
  "qaoa": {
    "best_sampled_cut": 4.0,
    "best_bitstring": "0101",
    "partition": {"S": [1, 3], "T": [0, 2]},
    "expected_cut": 3.2109,
    "approximation_ratio": 0.8027,
    "probability_of_optimal_cut": 0.3115,
    "gammas": [2.8797],
    "betas": [0.3483],
    "circuit_executions": 89,
    "time_ms": 404.1
  },
  "classical_brute_force": {"max_cut": 4.0, "optimal_assignments": [[0, 1, 0, 1], [1, 0, 1, 0]], "n_optimal_assignments": 2, "time_ms": 0.03},
  "found_optimum": true,
  "note": "Brute force is exact and faster at this size. The QAOA run demonstrates the hybrid workflow on a noiseless simulator, not a speedup."
}
```

| Backend | best cut / optimum | approximation ratio | P(optimal) | time |
| --- | --- | --- | --- | --- |
| simulator | 4 / 4 | 0.803 | 0.312 | 0.40 s |
| qiskit (Aer) | 4 / 4 | 0.808 | 0.308 | 5.56 s |
| pennylane | 4 / 4 | 0.807 | 0.312 | 0.35 s |

Qiskit is slower here because each of the roughly 90 optimizer steps builds and runs a fresh Aer job.

Input validation, a 64 qubit circuit:

```json
{"ok": false, "error": {"type": "ValidationError",
 "message": "Circuit uses 64 qubits; the limit is 10 (set SUPERQUANTX_MCP_MAX_QUBITS to change, hard ceiling applies)"}}
```

## Known limitations

- Everything runs on noiseless local simulators at toy scale (at most 16 qubits). Results show the
  workflow, not a quantum advantage.
- Tested backends: built-in simulator, Qiskit Aer, PennyLane `default.qubit`. Cirq, Braket, and TKET
  go through the same unified backend code path but were not installed in the verification run.
- The QAOA in this server is implemented on top of the unified circuit API because the
  `superquantx.algorithms.QAOA` class in 0.2.0 has placeholder fallbacks. The `statevector` QSVM kernel
  is computed here for the same reason (see `kernel="backend"` above).
