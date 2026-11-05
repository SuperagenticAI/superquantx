[![PyPI - Version](https://img.shields.io/pypi/v/superquantx)](https://pypi.org/project/superquantx/)
[![Python Version](https://img.shields.io/pypi/pyversions/superquantx)](https://pypi.org/project/superquantx/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://github.com/SuperagenticAI/superquantx/actions/workflows/ci.yml/badge.svg)](https://github.com/SuperagenticAI/superquantx/actions/workflows/ci.yml)
[![Website](https://img.shields.io/badge/website-superquantx.dev-7C3AED)](https://superquantx.dev)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://superagenticai.github.io/superquantx)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# SuperQuantX

**The agentic layer for Quantum AI**

An open software layer across quantum SDKs, with a classical baseline for every result.

SuperQuantX 0.3.0 is an open-source toolkit for quantum machine learning and
optimisation research by [Superagentic AI](https://super-agentic.ai).

[Website](https://superquantx.dev) · [Documentation](https://superagenticai.github.io/superquantx/) · [PyPI](https://pypi.org/project/superquantx/) · [Showcase](docs/showcase.md)

[Watch the demo](https://www.youtube.com/watch?v=QwPBKtpNSKU) · SuperQuantX 0.3.0 · Local simulation · No quantum hardware used

> **Research software.** Examples use local simulation, not quantum hardware.
> They do not demonstrate quantum advantage; compare results with classical baselines.

## Backends

| Backend | Runs on | Status |
|---------|---------|--------|
| `simulator` | built-in NumPy statevector | ✅ tested |
| `pennylane` | `default.qubit` | ✅ tested |
| `qiskit` | Qiskit Aer | ✅ tested |
| `cirq` | Cirq simulator | ✅ tested |
| `braket` | Amazon Braket `LocalSimulator` | ✅ tested |
| `tket` | pytket local simulator | ✅ tested |
| `ocean` | D-Wave Ocean (QUBO/Ising only) | Annealing only; hardware untested, requires a Leap token |
| Remote devices | via `qiskit`, `braket`, `tket` | Untested; requires provider credentials |

"Tested" means the unit tests run a Bell state, a qubit-order check, exact
expectation values and the fidelity kernel on that backend. Check your own
install with `superquantx list-backends`.

## Install

Install the 0.3.0 release from PyPI (Python 3.11+):

```bash
uv add "superquantx==0.3.0"                       # built-in simulator
uv add "superquantx[mcp,qiskit,pennylane]==0.3.0" # MCP and SDKs
uvx --from "superquantx[mcp]==0.3.0" superquantx-mcp  # run the MCP server, no install
```

Not using a uv project? `uv pip install "superquantx==0.3.0"` or
`pip install "superquantx==0.3.0"` work too. Each backend has an extra:
`pennylane`, `qiskit`, `cirq`, `braket`, `tket`, `ocean`, or `all-backends`.

## Compare a quantum classifier with classical models

```python
import superquantx as sqx
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, noise=0.2, random_state=0)
result = sqx.compare_to_classical(X, y, sqx.QuantumSVM(backend="pennylane"), cv=5)
print(result.table())
```

The comparison uses identical folds, equal tuning budgets and per-fold
preprocessing. `result.table()` reports accuracy, timing and a statistical
comparison. See [benchmarking](docs/user-guide/benchmarking.md) for the protocol.

VQE, MaxCut QAOA and portfolio agents also report classical references.
See the [showcase](docs/showcase.md) and [SDK examples](examples/full-demo/README.md).
The [MCP server](docs/mcp.md) exposes circuits, backend checks, classification
and MaxCut to AI assistants.

## Command line

```bash
superquantx -v version        # versions of installed backend SDKs
superquantx list-backends     # real Bell-state check per backend
superquantx benchmark -d moons -b pennylane   # quantum kernel vs tuned classical
superquantx run qsvm --data iris -b simulator
```

## Development

```bash
git clone https://github.com/SuperagenticAI/superquantx.git
cd superquantx
uv sync --extra all-backends --extra dev
uv run pytest tests/unit tests/integration
uv run ruff check src tests
```

Contributions are welcome: [open an issue](https://github.com/SuperagenticAI/superquantx/issues)
or read the [contributing guide](docs/development/contributing.md).
Python 3.11+ is required; see [backend compatibility](docs/BACKENDS.md#backend-version-compatibility)
for version floors.

## License

Apache License 2.0. See [LICENSE](LICENSE).
