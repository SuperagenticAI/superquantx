"""Portable circuit representation, OpenQASM / JSON parsing, and execution.

The MCP tools accept circuits as a small OpenQASM 2 or 3 subset or as a JSON
gate list. Both are parsed into a backend-neutral list of :class:`Gate`
objects, validated against the configured limits, and then executed through
the SuperQuantX backend layer.

Bit order of returned counts is always little-endian, the Qiskit convention:
qubit 0 is the rightmost character of every bitstring.
"""

from __future__ import annotations

import ast
import logging
import math
import operator
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .config import Limits


class CircuitError(ValueError):
    """Raised for invalid or unsupported circuit input."""


@dataclass(frozen=True)
class Gate:
    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"gate": self.name, "qubits": list(self.qubits)}
        if self.params:
            d["params"] = list(self.params)
        return d


@dataclass
class Circuit:
    n_qubits: int
    gates: list[Gate] = field(default_factory=list)
    source_format: str = "json"

    def gate_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for g in self.gates:
            out[g.name] = out.get(g.name, 0) + 1
        return out


# name -> (number of qubits, number of params)
GATE_SPECS: dict[str, tuple[int, int]] = {
    "id": (1, 0),
    "x": (1, 0),
    "y": (1, 0),
    "z": (1, 0),
    "h": (1, 0),
    "s": (1, 0),
    "sdg": (1, 0),
    "t": (1, 0),
    "tdg": (1, 0),
    "sx": (1, 0),
    "rx": (1, 1),
    "ry": (1, 1),
    "rz": (1, 1),
    "p": (1, 1),
    "u1": (1, 1),
    "u2": (1, 2),
    "u3": (1, 3),
    "u": (1, 3),
    "cx": (2, 0),
    "cz": (2, 0),
    "swap": (2, 0),
    "ccx": (3, 0),
}

GATE_ALIASES = {
    "i": "id",
    "cnot": "cx",
    "toffoli": "ccx",
    "hadamard": "h",
    "pauli_x": "x",
    "pauli_y": "y",
    "pauli_z": "z",
    "phase": "p",
    "u3": "u3",
}

SUPPORTED_GATES = sorted(GATE_SPECS)


def _canonical_gate_name(name: str) -> str:
    key = str(name).strip().lower()
    key = GATE_ALIASES.get(key, key)
    if key not in GATE_SPECS:
        raise CircuitError(
            f"Unsupported gate '{name}'. Supported gates: {', '.join(SUPPORTED_GATES)}"
        )
    return key


def _make_gate(
    name: str, qubits: list[int], params: list[float], n_qubits: int
) -> Gate:
    canon = _canonical_gate_name(name)
    nq, np_ = GATE_SPECS[canon]
    if len(qubits) != nq:
        raise CircuitError(f"Gate '{canon}' needs {nq} qubit(s), got {len(qubits)}")
    if len(params) != np_:
        raise CircuitError(
            f"Gate '{canon}' needs {np_} parameter(s), got {len(params)}"
        )
    for q in qubits:
        if not isinstance(q, int) or isinstance(q, bool):
            raise CircuitError(f"Qubit index must be an integer, got {q!r}")
        if q < 0 or q >= n_qubits:
            raise CircuitError(f"Qubit index {q} out of range for {n_qubits} qubit(s)")
    if len(set(qubits)) != len(qubits):
        raise CircuitError(
            f"Gate '{canon}' uses the same qubit more than once: {qubits}"
        )
    clean_params = []
    for p in params:
        if isinstance(p, str):
            # Allow expressions like "pi/3" (evaluated safely, no code execution).
            p = safe_eval_expr(p)
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise CircuitError(
                f"Gate parameter must be a number or an expression like 'pi/2', got {p!r}"
            )
        if not math.isfinite(float(p)):
            raise CircuitError("Gate parameters must be finite numbers")
        clean_params.append(float(p))
    return Gate(canon, tuple(qubits), tuple(clean_params))


def _check_size(n_qubits: int, n_gates: int, limits: Limits) -> None:
    if n_qubits < 1:
        raise CircuitError("Circuit needs at least 1 qubit")
    if n_qubits > limits.max_qubits:
        raise CircuitError(
            f"Circuit uses {n_qubits} qubits; the limit is {limits.max_qubits} "
            "(set SUPERQUANTX_MCP_MAX_QUBITS to change, hard ceiling applies)"
        )
    if n_gates > limits.max_gates:
        raise CircuitError(
            f"Circuit has {n_gates} gates; the limit is {limits.max_gates}"
        )


# --------------------------------------------------------------------------
# JSON gate list
# --------------------------------------------------------------------------


def parse_gate_list(spec: Any, limits: Limits, n_qubits: int | None = None) -> Circuit:
    """Parse a JSON gate list.

    Accepted shapes::

        {"n_qubits": 2, "gates": [{"gate": "h", "qubits": [0]},
                                  {"gate": "cx", "qubits": [0, 1]}]}
        [{"gate": "h", "qubits": [0]}, ...]          # n_qubits inferred

    A gate may use ``"name"`` instead of ``"gate"``. ``"qubits"`` may be a
    single integer for one-qubit gates. Parameters are radians.
    """
    if isinstance(spec, dict):
        gates_raw = spec.get("gates")
        if n_qubits is None:
            n_qubits = spec.get("n_qubits")
    elif isinstance(spec, list):
        gates_raw = spec
    else:
        raise CircuitError(
            "Gate list must be a list of gates or an object with 'gates'"
        )
    if not isinstance(gates_raw, list):
        raise CircuitError("'gates' must be a list")

    parsed: list[tuple[str, list[int], list[float]]] = []
    max_q = -1
    for i, g in enumerate(gates_raw):
        if not isinstance(g, dict):
            raise CircuitError(f"Gate #{i} must be an object, got {type(g).__name__}")
        name = g.get("gate", g.get("name"))
        if not isinstance(name, str):
            raise CircuitError(f"Gate #{i} is missing a 'gate' name")
        qubits = g.get("qubits")
        if isinstance(qubits, int) and not isinstance(qubits, bool):
            qubits = [qubits]
        if not isinstance(qubits, list) or not qubits:
            raise CircuitError(f"Gate #{i} needs a non-empty 'qubits' list")
        params = g.get("params", [])
        if isinstance(params, (int, float)) and not isinstance(params, bool):
            params = [params]
        if not isinstance(params, list):
            raise CircuitError(f"Gate #{i} 'params' must be a list of numbers")
        for q in qubits:
            if isinstance(q, int) and not isinstance(q, bool):
                max_q = max(max_q, q)
        parsed.append((name, qubits, params))

    if n_qubits is None:
        n_qubits = max_q + 1
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int):
        raise CircuitError("'n_qubits' must be an integer")
    _check_size(n_qubits, len(parsed), limits)
    circuit = Circuit(n_qubits=n_qubits, source_format="json")
    for i, (name, qubits, params) in enumerate(parsed):
        try:
            circuit.gates.append(_make_gate(name, qubits, params, n_qubits))
        except CircuitError as exc:
            raise CircuitError(f"Gate #{i}: {exc}") from None
    return circuit


# --------------------------------------------------------------------------
# OpenQASM 2 / 3 subset
# --------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_CONSTS = {
    "pi": math.pi,
    "π": math.pi,
    "tau": math.tau,
    "τ": math.tau,
    "e": math.e,
    "euler": math.e,
}
_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "ln": math.log,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
}


def safe_eval_expr(expr: str) -> float:
    """Evaluate a numeric QASM parameter expression without ``eval``."""
    text = expr.strip().replace("π", "pi").replace("τ", "tau")
    if len(text) > 200:
        raise CircuitError("Parameter expression too long")
    try:
        tree = ast.parse(text.replace("^", "**"), mode="eval")
    except SyntaxError:
        raise CircuitError(f"Cannot parse parameter expression '{expr}'") from None

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 16:
                raise CircuitError("Exponent too large in parameter expression")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](ev(node.operand))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and len(node.args) == 1
            and not node.keywords
        ):
            return _FUNCS[node.func.id](ev(node.args[0]))
        raise CircuitError(f"Unsupported parameter expression '{expr}'")

    try:
        value = ev(tree)
    except ZeroDivisionError:
        raise CircuitError(f"Division by zero in '{expr}'") from None
    if not math.isfinite(value):
        raise CircuitError(f"Parameter '{expr}' is not finite")
    return value


_COMMENT_LINE = re.compile(r"//[^\n]*")
_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.S)
_QREG2 = re.compile(r"^qreg\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]$")
_QREG3 = re.compile(r"^qubit\s*\[\s*(\d+)\s*\]\s*([A-Za-z_]\w*)$")
_QREG3_SINGLE = re.compile(r"^qubit\s+([A-Za-z_]\w*)$")
_GATE_STMT = re.compile(r"^([A-Za-z_]\w*)\s*(?:\(([^)]*)\))?\s+(.+)$")
_OPERAND = re.compile(r"^([A-Za-z_]\w*)\s*(?:\[\s*(\d+)\s*\])?$")
_SKIP_PREFIXES = ("include", "creg", "bit", "barrier", "measure", "output", "input")


def parse_qasm(source: str, limits: Limits) -> Circuit:
    """Parse a practical OpenQASM 2.0 / 3.0 subset.

    Supported: version header, ``include``, ``qreg``/``qubit[n]`` declarations
    (several registers are flattened in declaration order), standard gates in
    :data:`SUPPORTED_GATES` with numeric or ``pi`` expressions, register
    broadcast (``h q;``), ``barrier``, classical registers and ``measure``
    statements (ignored; every qubit is measured at the end).

    Not supported: custom ``gate`` definitions, classical control, loops,
    ``reset``, mid-circuit measurement semantics.
    """
    if not isinstance(source, str) or not source.strip():
        raise CircuitError("QASM source must be a non-empty string")
    if len(source) > limits.max_qasm_chars:
        raise CircuitError(f"QASM source exceeds {limits.max_qasm_chars} characters")

    text = _COMMENT_BLOCK.sub(" ", source)
    text = _COMMENT_LINE.sub("", text)
    statements = [s.strip() for s in text.split(";")]
    version = None
    registers: dict[str, tuple[int, int]] = {}  # name -> (offset, size)
    total = 0
    pending: list[tuple[str, list[float], list[str]]] = []

    for stmt in statements:
        if not stmt:
            continue
        stmt = " ".join(stmt.split())
        low = stmt.lower()
        if low.startswith("openqasm"):
            version = stmt.split()[1] if len(stmt.split()) > 1 else "?"
            if not (version.startswith("2") or version.startswith("3")):
                raise CircuitError(f"Unsupported OpenQASM version {version}")
            continue
        if (
            "{" in stmt
            or "}" in stmt
            or low.startswith(("gate ", "if", "for ", "while", "def "))
        ):
            raise CircuitError(
                "Custom gate definitions and classical control flow are not supported; "
                f"inline the gates instead (statement: '{stmt[:60]}')"
            )
        if low.startswith("reset"):
            raise CircuitError("'reset' is not supported")
        m = _QREG2.match(stmt) or None
        if m:
            name, size = m.group(1), int(m.group(2))
            registers[name] = (total, size)
            total += size
            continue
        m = _QREG3.match(stmt)
        if m:
            size, name = int(m.group(1)), m.group(2)
            registers[name] = (total, size)
            total += size
            continue
        m = _QREG3_SINGLE.match(stmt)
        if m:
            registers[m.group(1)] = (total, 1)
            total += 1
            continue
        if (
            low.startswith(_SKIP_PREFIXES)
            or "= measure" in low
            or re.match(r"^\w+\s*\[\s*\d+\s*\]\s*=\s*measure", low)
        ):
            continue
        m = _GATE_STMT.match(stmt)
        if not m:
            raise CircuitError(f"Cannot parse QASM statement '{stmt[:80]}'")
        gname, pstr, operands = m.group(1), m.group(2), m.group(3)
        params = (
            [safe_eval_expr(p) for p in pstr.split(",")]
            if pstr and pstr.strip()
            else []
        )
        ops = [o.strip() for o in operands.split(",")]
        pending.append((gname, params, ops))
        if len(pending) > limits.max_gates:
            raise CircuitError(f"Circuit has more than {limits.max_gates} gates")

    if version is None:
        raise CircuitError(
            "QASM source must start with 'OPENQASM 2.0;' or 'OPENQASM 3.0;'"
        )
    if not registers:
        raise CircuitError(
            "No quantum register declared (use 'qreg q[n];' or 'qubit[n] q;')"
        )
    _check_size(total, 0, limits)

    circuit = Circuit(n_qubits=total, source_format=f"openqasm{version.split('.')[0]}")
    for gname, params, ops in pending:
        resolved: list[list[int]] = []
        for op in ops:
            om = _OPERAND.match(op)
            if not om or om.group(1) not in registers:
                raise CircuitError(f"Unknown qubit operand '{op}' in gate '{gname}'")
            offset, size = registers[om.group(1)]
            if om.group(2) is None:
                resolved.append([offset + k for k in range(size)])
            else:
                idx = int(om.group(2))
                if idx >= size:
                    raise CircuitError(
                        f"Index {idx} out of range for register '{om.group(1)}'"
                    )
                resolved.append([offset + idx])
        widths = {len(r) for r in resolved if len(r) > 1}
        if len(widths) > 1:
            raise CircuitError(f"Mismatched register sizes in broadcast '{gname}'")
        width = widths.pop() if widths else 1
        for k in range(width):
            qubits = [r[k] if len(r) > 1 else r[0] for r in resolved]
            circuit.gates.append(_make_gate(gname, qubits, params, total))
            if len(circuit.gates) > limits.max_gates:
                raise CircuitError(f"Circuit has more than {limits.max_gates} gates")
    return circuit


def parse_circuit(circuit: Any, limits: Limits) -> Circuit:
    """Parse either an OpenQASM string or a JSON gate list (dict, list, or JSON text)."""
    if isinstance(circuit, str):
        stripped = circuit.lstrip()
        if stripped.startswith(("{", "[")):
            import json

            try:
                return parse_gate_list(json.loads(stripped), limits)
            except json.JSONDecodeError as exc:
                raise CircuitError(f"Invalid JSON gate list: {exc}") from None
        return parse_qasm(circuit, limits)
    return parse_gate_list(circuit, limits)


# --------------------------------------------------------------------------
# Lowering to the gate set every SuperQuantX backend implements
# --------------------------------------------------------------------------

_PI = math.pi


def lower(circuit: Circuit, native_ccx: bool) -> list[Gate]:
    """Rewrite gates into {h, x, y, z, rx, ry, rz, cx, cz, swap[, ccx]}.

    Rewrites are exact up to a global phase, which does not affect
    measurement statistics.
    """
    out: list[Gate] = []
    for g in circuit.gates:
        q = g.qubits
        n = g.name
        if n == "id":
            continue
        if n in {"x", "y", "z", "h", "rx", "ry", "rz", "cx", "cz", "swap"}:
            out.append(g)
        elif n == "s":
            out.append(Gate("rz", q, (_PI / 2,)))
        elif n == "sdg":
            out.append(Gate("rz", q, (-_PI / 2,)))
        elif n == "t":
            out.append(Gate("rz", q, (_PI / 4,)))
        elif n == "tdg":
            out.append(Gate("rz", q, (-_PI / 4,)))
        elif n == "sx":
            out.append(Gate("rx", q, (_PI / 2,)))
        elif n in {"p", "u1"}:
            out.append(Gate("rz", q, g.params))
        elif n == "u2":
            phi, lam = g.params
            out += [
                Gate("rz", q, (lam,)),
                Gate("ry", q, (_PI / 2,)),
                Gate("rz", q, (phi,)),
            ]
        elif n in {"u3", "u"}:
            theta, phi, lam = g.params
            out += [
                Gate("rz", q, (lam,)),
                Gate("ry", q, (theta,)),
                Gate("rz", q, (phi,)),
            ]
        elif n == "ccx":
            if native_ccx:
                out.append(g)
            else:
                out += _ccx_decomposition(*q)
        else:  # pragma: no cover - guarded by GATE_SPECS
            raise CircuitError(f"No lowering for gate '{n}'")
    return out


def _ccx_decomposition(a: int, b: int, c: int) -> list[Gate]:
    t = (_PI / 4,)
    td = (-_PI / 4,)
    return [
        Gate("h", (c,)),
        Gate("cx", (b, c)),
        Gate("rz", (c,), td),
        Gate("cx", (a, c)),
        Gate("rz", (c,), t),
        Gate("cx", (b, c)),
        Gate("rz", (c,), td),
        Gate("cx", (a, c)),
        Gate("rz", (b,), t),
        Gate("rz", (c,), t),
        Gate("h", (c,)),
        Gate("cx", (a, b)),
        Gate("rz", (a,), t),
        Gate("rz", (b,), td),
        Gate("cx", (a, b)),
    ]


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
        try:
            self.records.append(f"{record.levelname}: {record.getMessage()}")
        except Exception:
            pass


@contextmanager
def capture_backend_logs():
    """Collect WARNING+ log lines from SuperQuantX during a call."""
    handler = _CaptureHandler()
    root = logging.getLogger("superquantx")
    root.addHandler(handler)
    try:
        yield handler.records
    finally:
        root.removeHandler(handler)


_SILENT_FAILURE_MARKERS = ("Unknown gate", "Failed to add gate")


def _normalize_counts(counts: dict[Any, Any], n_qubits: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, val in counts.items():
        k = str(key).replace(" ", "")
        if k.startswith("0x"):
            k = format(int(k, 16), f"0{n_qubits}b")
        k = k[-n_qubits:].zfill(n_qubits)
        out[k] = out.get(k, 0) + int(val)
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _run_unified(
    backend: Any, gates: list[Gate], n_qubits: int, shots: int
) -> dict[str, int]:
    """Build and run through the SuperQuantX BaseBackend gate API."""
    name_map = {"cx": "CNOT", "ccx": "CCX"}
    circ = backend.create_circuit(n_qubits)
    for g in gates:
        circ = backend.add_gate(
            circ,
            name_map.get(g.name, g.name.upper()),
            list(g.qubits),
            list(g.params) or None,
        )
    circ = backend.add_measurement(circ, list(range(n_qubits)))
    result = backend.execute_circuit(circ, shots=shots)
    counts = result.get("counts") if isinstance(result, dict) else None
    if not counts:
        raise RuntimeError("Backend returned no counts")
    return counts


def _run_pennylane(
    backend: Any, gates: list[Gate], n_qubits: int, shots: int
) -> dict[str, int]:
    """PennyLane builds circuits as quantum functions; construct one directly."""
    import pennylane as qml

    ops = {
        "x": qml.PauliX,
        "y": qml.PauliY,
        "z": qml.PauliZ,
        "h": qml.Hadamard,
        "rx": qml.RX,
        "ry": qml.RY,
        "rz": qml.RZ,
        "cx": qml.CNOT,
        "cz": qml.CZ,
        "swap": qml.SWAP,
        "ccx": qml.Toffoli,
    }
    device_name = getattr(backend, "device", None) or "default.qubit"
    dev = qml.device(device_name, wires=n_qubits)

    def qfunc():
        for g in gates:
            op = ops[g.name]
            if g.params:
                op(*g.params, wires=list(g.qubits))
            else:
                op(wires=list(g.qubits))
        return qml.counts(wires=list(range(n_qubits)))

    qnode = qml.QNode(qfunc, dev)
    if hasattr(qml, "set_shots"):
        qnode = qml.set_shots(qnode, shots=shots)
        raw = qnode()
    else:  # pragma: no cover - older PennyLane
        raw = qml.QNode(qfunc, qml.device(device_name, wires=n_qubits, shots=shots))()
    # PennyLane bitstrings are big-endian (wire 0 leftmost); flip to little-endian.
    return {str(k)[::-1]: int(v) for k, v in dict(raw).items()}


# Backends whose bit order SuperQuantX has been verified to return little-endian.
_VERIFIED_LITTLE_ENDIAN = {"simulator", "qiskit", "pennylane"}


def execute(
    circuit: Circuit, backend_name: str, backend: Any, shots: int
) -> dict[str, Any]:
    """Execute a parsed circuit on an initialised SuperQuantX backend."""
    native_ccx = backend_name in {"qiskit", "pennylane"}
    gates = lower(circuit, native_ccx=native_ccx)
    t0 = time.perf_counter()
    with capture_backend_logs() as logs:
        if backend_name == "pennylane":
            counts = _run_pennylane(backend, gates, circuit.n_qubits, shots)
        else:
            counts = _run_unified(backend, gates, circuit.n_qubits, shots)
    elapsed = (time.perf_counter() - t0) * 1000
    silent = [m for m in logs if any(s in m for s in _SILENT_FAILURE_MARKERS)]
    if silent:
        raise RuntimeError(
            "Backend dropped gates while building the circuit: " + "; ".join(silent[:3])
        )
    counts = _normalize_counts(counts, circuit.n_qubits)
    return {
        "counts": counts,
        "shots": int(sum(counts.values())),
        "execute_ms": round(elapsed, 2),
        "bit_order": (
            "little-endian (qubit 0 is the rightmost bit)"
            if backend_name in _VERIFIED_LITTLE_ENDIAN
            else "backend-native (not verified by the MCP server)"
        ),
        "lowered_gate_count": len(gates),
        "backend_warnings": logs[:10],
    }


# --------------------------------------------------------------------------
# Exact statevectors (used for fidelity quantum kernels)
# --------------------------------------------------------------------------


def statevector(circuit: Circuit, backend_name: str, backend: Any) -> Any:
    """Return the exact (noiseless) statevector of ``circuit`` from one SDK.

    The global phase and qubit ordering may differ between SDKs; fidelities
    ``|<a|b>|^2`` computed within one backend are unaffected.
    """
    import numpy as np

    gates = lower(circuit, native_ccx=backend_name in {"qiskit", "pennylane"})
    n = circuit.n_qubits
    with capture_backend_logs() as logs:
        if backend_name == "pennylane":
            import pennylane as qml

            ops = {
                "x": qml.PauliX,
                "y": qml.PauliY,
                "z": qml.PauliZ,
                "h": qml.Hadamard,
                "rx": qml.RX,
                "ry": qml.RY,
                "rz": qml.RZ,
                "cx": qml.CNOT,
                "cz": qml.CZ,
                "swap": qml.SWAP,
                "ccx": qml.Toffoli,
            }
            dev = qml.device("default.qubit", wires=n)

            @qml.qnode(dev)
            def qfunc():
                for g in gates:
                    op = ops[g.name]
                    if g.params:
                        op(*g.params, wires=list(g.qubits))
                    else:
                        op(wires=list(g.qubits))
                return qml.state()

            state = np.asarray(qfunc(), dtype=complex)
        else:
            name_map = {"cx": "CNOT", "ccx": "CCX"}
            circ = backend.create_circuit(n)
            for g in gates:
                circ = backend.add_gate(
                    circ,
                    name_map.get(g.name, g.name.upper()),
                    list(g.qubits),
                    list(g.params) or None,
                )
            if backend_name == "qiskit":
                from qiskit.quantum_info import Statevector

                bare = circ.copy()
                bare.remove_final_measurements()
                state = np.asarray(Statevector(bare).data, dtype=complex)
            else:
                state = np.asarray(backend.get_statevector(circ), dtype=complex)
    silent = [m for m in logs if any(s in m for s in _SILENT_FAILURE_MARKERS)]
    if silent:
        raise RuntimeError("Backend dropped gates: " + "; ".join(silent[:3]))
    if state.shape != (2**n,):
        raise RuntimeError(f"Backend returned a statevector of shape {state.shape}")
    return state


def feature_map_circuit(x: Any, feature_map: str, reps: int) -> Circuit:
    """Qiskit-convention ZZFeatureMap / ZFeatureMap as a portable circuit.

    ZZFeatureMap (full entanglement): per repetition, H on every qubit,
    P(2 x_i) on qubit i, then for every pair i < j: CX(i, j), P(2 (pi - x_i)(pi - x_j))
    on j, CX(i, j). ZFeatureMap drops the entangling block. P is applied as RZ,
    which differs only by a global phase.
    """
    x = [float(v) for v in x]
    n = len(x)
    gates: list[Gate] = []
    for _ in range(reps):
        gates += [Gate("h", (i,)) for i in range(n)]
        gates += [Gate("rz", (i,), (2.0 * x[i],)) for i in range(n)]
        if feature_map == "ZZFeatureMap":
            for i in range(n):
                for j in range(i + 1, n):
                    phi = 2.0 * (math.pi - x[i]) * (math.pi - x[j])
                    gates += [
                        Gate("cx", (i, j)),
                        Gate("rz", (j,), (phi,)),
                        Gate("cx", (i, j)),
                    ]
    return Circuit(n_qubits=n, gates=gates, source_format="feature_map")
