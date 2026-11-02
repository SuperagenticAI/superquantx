"""Safety limits and environment configuration for the SuperQuantX MCP server.

All limits can be lowered (never raised above the hard ceilings) with
environment variables, so an operator can keep an AI assistant inside a
small, cheap, simulator-only sandbox.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


# Hard ceilings. Environment variables can lower these but never exceed them.
HARD_MAX_QUBITS = 16
HARD_MAX_SHOTS = 100_000
HARD_MAX_GATES = 2_000
HARD_MAX_QASM_CHARS = 50_000
HARD_MAX_SAMPLES = 200
HARD_MAX_FEATURES = 6
HARD_MAX_QAOA_NODES = 12
HARD_MAX_QAOA_LAYERS = 4

# Backends that run on local simulators with their default configuration.
SIMULATOR_BACKENDS = ("simulator", "qiskit", "pennylane", "cirq", "braket", "tket")

# Backends preferred for demos because they work end to end today.
PREFERRED_BACKENDS = ("simulator", "qiskit", "pennylane")

HARDWARE_ENV = "SUPERQUANTX_MCP_ALLOW_HARDWARE"


def _env_int(name: str, default: int, ceiling: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return min(default, ceiling)
    try:
        value = int(raw)
    except ValueError:
        return min(default, ceiling)
    return max(1, min(value, ceiling))


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Limits:
    """Resolved runtime limits."""

    max_qubits: int
    max_shots: int
    max_gates: int
    max_qasm_chars: int
    max_samples: int
    max_features: int
    max_qaoa_nodes: int
    max_qaoa_layers: int
    allow_hardware: bool

    def as_dict(self) -> dict:
        return {
            "max_qubits": self.max_qubits,
            "max_shots": self.max_shots,
            "max_gates": self.max_gates,
            "max_qasm_chars": self.max_qasm_chars,
            "max_samples": self.max_samples,
            "max_features": self.max_features,
            "max_qaoa_nodes": self.max_qaoa_nodes,
            "max_qaoa_layers": self.max_qaoa_layers,
            "allow_hardware": self.allow_hardware,
        }


def get_limits() -> Limits:
    """Read limits from the environment (evaluated per call, cheap)."""
    return Limits(
        max_qubits=_env_int("SUPERQUANTX_MCP_MAX_QUBITS", 10, HARD_MAX_QUBITS),
        max_shots=_env_int("SUPERQUANTX_MCP_MAX_SHOTS", 10_000, HARD_MAX_SHOTS),
        max_gates=_env_int("SUPERQUANTX_MCP_MAX_GATES", 500, HARD_MAX_GATES),
        max_qasm_chars=_env_int(
            "SUPERQUANTX_MCP_MAX_QASM_CHARS", 20_000, HARD_MAX_QASM_CHARS
        ),
        max_samples=_env_int("SUPERQUANTX_MCP_MAX_SAMPLES", 80, HARD_MAX_SAMPLES),
        max_features=_env_int("SUPERQUANTX_MCP_MAX_FEATURES", 4, HARD_MAX_FEATURES),
        max_qaoa_nodes=_env_int(
            "SUPERQUANTX_MCP_MAX_QAOA_NODES", 10, HARD_MAX_QAOA_NODES
        ),
        max_qaoa_layers=_env_int(
            "SUPERQUANTX_MCP_MAX_QAOA_LAYERS", 3, HARD_MAX_QAOA_LAYERS
        ),
        allow_hardware=_env_flag(HARDWARE_ENV),
    )
