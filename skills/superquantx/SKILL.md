---
name: superquantx
description: Use when the user wants to run, compare, or explain quantum circuits, Bell or GHZ states, quantum kernel SVM classification, or QAOA MaxCut across quantum SDKs (built-in simulator, Qiskit, PennyLane, Cirq, Braket, TKET) through the SuperQuantX MCP server.
---

# SuperQuantX quantum workflows (MCP)

The `superquantx` MCP server exposes six read-only tools. They run on local noiseless simulators
unless the server operator set `SUPERQUANTX_MCP_ALLOW_HARDWARE=1`.

If the tools are not available, ask the user to install and register the server:

```bash
pip install "superquantx[mcp,qiskit,pennylane]"
# then add {"command": "<venv>/bin/superquantx-mcp"} under mcpServers in the client config
```

See `docs/mcp.md` in the SuperQuantX repository for Claude Desktop and Cursor snippets.

## Workflow

1. **Start with `list_backends`.** Use only backends whose `status` is `working`. Tell the user
   which ones are `not_installed` and the install hint from `error`, rather than guessing.
2. **Pick the tool:**
   - Run a circuit: `run_circuit` with OpenQASM 2, OpenQASM 3, or a JSON gate list
     (`{"n_qubits": 2, "gates": [{"gate": "h", "qubits": [0]}, {"gate": "cx", "qubits": [0, 1]}]}`).
     Parameters can be numbers or strings like `"pi/2"`.
   - Show entanglement across SDKs: `bell_state_demo`.
   - Classification: `qsvm_classify` (datasets `moons`, `circles`, `blobs`, `xor`, `iris`, or `X`/`y`).
   - Optimization: `qaoa_maxcut` with `edges` like `[[0,1],[1,2],[2,0]]`.
   - Environment report: `backend_versions`.
3. **Compare backends** by calling the same tool with `backend` set to each working backend.
4. **Read results correctly:**
   - Counts are little-endian: qubit 0 is the rightmost bit (`"01"` means qubit 0 is 1).
   - Always check `"ok"`. On `false`, show `error.message`; for `ValidationError` fix the input
     (limits are in the message) instead of retrying unchanged.
   - For `qsvm_classify`, report the quantum accuracy next to `classical_rbf_baseline` and check
     `kernel_diagnostics.looks_valid`.
   - For `qaoa_maxcut`, report `best_sampled_cut` against `classical_brute_force.max_cut`, plus
     `approximation_ratio`.

## Honesty rules

- Never claim quantum advantage or speedup from these results. They are toy-scale, noiseless
  simulations; brute force and classical SVMs are faster and often as accurate. Repeat the tool's
  `note` field when summarizing.
- Do not claim hardware execution unless the result's `device` is a real device.
- Simulator results are exact up to shot noise; mention shot counts when comparing distributions.

## Limits (defaults, configurable by the operator)

10 qubits, 10000 shots, 500 gates, 80 QSVM samples with up to 4 features, QAOA up to 10 nodes and
3 layers. Hard ceilings: 16 qubits, 100000 shots.
