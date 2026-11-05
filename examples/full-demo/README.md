# Try SuperQuantX locally

These examples target **SuperQuantX 0.3.0**, with Python **3.12** recommended.
They run local simulators without provider credentials, cloud resources, or
quantum hardware. `demo.py` checks the installed SDK version before starting.

## Try it yourself

Clone the repository to get the scripts, then install the SDK and optional
backends using the existing lockfile. These commands require
[uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone https://github.com/SuperagenticAI/superquantx.git
cd superquantx
uv sync --locked --python 3.12 --extra all-backends --extra mcp
uv run --no-sync python examples/full-demo/demo.py
uv run --no-sync python examples/full-demo/mcp_demo.py
```

From an existing checkout, start at the `uv sync` command. For a smaller
installation, use only `--extra mcp` and request the built-in simulator:

```bash
uv sync --locked --python 3.12 --extra mcp
uv run --no-sync python examples/full-demo/demo.py --section backends --backends simulator
uv run --no-sync python examples/full-demo/demo.py --section algorithms
uv run --no-sync python examples/full-demo/mcp_demo.py
```

With all backend extras installed, individual stages and fresh JSON output can
be selected explicitly:

```bash
uv run --no-sync python examples/full-demo/demo.py --section backends --backends qiskit ocean
uv run --no-sync python examples/full-demo/demo.py --output my-results.json
```

An existing output file is never overwritten. Requested stages that fail,
including missing optional dependencies, are reported as `FAILED`; the script
continues through the other requested stages and exits with status 1. CLI usage
errors exit with status 2. `--backends` selects backend stages only; use
`--section backends` to exclude the algorithms.

## Run against the published package

Installing the SDK does **not** create these scripts, and the package's source
distribution excludes examples. First obtain this directory from a repository
checkout. To test the published package separately from an editable install,
copy the two scripts into a new directory outside the checkout. For example,
from the repository root, using a Unix shell:

```bash
mkdir ../sqx-full-demo
cp examples/full-demo/demo.py examples/full-demo/mcp_demo.py ../sqx-full-demo/
cd ../sqx-full-demo
uv venv --python 3.12
uv pip install --python .venv/bin/python "superquantx[all-backends,mcp]==0.3.0"
.venv/bin/python demo.py
.venv/bin/python mcp_demo.py
```

On Windows, the virtual environment's interpreter is `.venv\Scripts\python.exe`.
Optional SDK availability varies by platform and Python version. Package
versions in the sample output describe that example run; they are not additional
installation requirements.

## Checks and interpretation

| Stage | Actual check |
|---|---|
| Six gate-model backends | Built-in simulator, PennyLane `default.qubit`, Qiskit Aer, Cirq simulator, Braket LocalSimulator and pytket Aer each return 1,024 Bell shots with only correlated `00`/`11` outcomes. An asymmetric `X(0)` circuit then checks normalized qubit ordering with 32 shots. |
| Ocean | Simulated annealing solves a two-variable QUBO; the returned sample's energy is recomputed and compared with the exhaustive minimum, -2. This is separate from gate-model simulation. |
| H2 VQE | The built-in H2 STO-3G Hamiltonian at 0.735 Å is evaluated with VQE and exact diagonalization; absolute energy difference must be at most 0.001 Hartree. |
| Five-node QAOA | Prints the sampled cut, expected cut, exact optimum and approximation ratio. Checks that the reference optimum is 6 and that the sampled cut does not exceed it; finding an optimal sample is not a required assertion. |
| MCP | Launches a real local server over stdio, verifies the exact six-tool inventory, and checks a successful `bell_state_demo` response on the built-in simulator with 128 shots. The other five tools are discovered, not exercised. No AI model is involved. |

Raw counts retain each backend's native ordering. The SDK example explicitly
calls `counts_big_endian`, whose convention puts qubit 0 on the **left**. This
does not assert that raw `execute_circuit` bitstrings all use the same ordering.
The MCP tool's result documents its own convention separately.

The QAOA approximation ratio uses the **expected** cut divided by the exact
optimum; it need not be 1 even when an optimal cut is sampled. Measurements,
annealing outcomes, optimization results and timings can vary. These are small
local correctness checks, not a complete conformance suite, hardware validation
or evidence of quantum advantage. These scripts do not exercise classification
benchmarks.

## Historical sample and local configuration

[`sample-results.json`](sample-results.json) contains historical example output
from a local Linux/Python 3.12.14 run of the published 0.3.0 SDK on 5 November
2026, including its package versions.
It reports nine passing stages, VQE error about 1.2e-9 Hartree, sampled MaxCut 6
and expected/optimal ratio about 0.844. Counts and timings vary between runs and
environments. The sample covers `demo.py`'s nine stages; run `mcp_demo.py`
separately to check MCP. Use `--output` with a new filename to save your own
results. Warnings from upstream libraries are not suppressed.

On headless machines with a read-only home, set writable paths before running
either script (Unix shell):

```bash
export XDG_CONFIG_HOME="$PWD/.cache/config"
export XDG_CACHE_HOME="$PWD/.cache"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export MPLBACKEND=Agg
```

The MCP example forwards these explicit settings to its subprocess, along with
optional `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` limits.
