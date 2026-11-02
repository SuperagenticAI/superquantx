"""Plain-Python implementations of the SuperQuantX MCP tools.

Every function returns a JSON-serialisable ``dict`` with an ``ok`` flag. They
never raise for bad input or a broken backend; errors come back as
``{"ok": False, "error": {...}}`` so an AI assistant can read and react to
them. The MCP layer in :mod:`superquantx.mcp.server` only wraps these.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import importlib.util
import itertools
import logging
import math
import platform
import sys
import time
import traceback
from typing import Any

import numpy as np

from .circuits import (
    Circuit,
    CircuitError,
    Gate,
    capture_backend_logs,
    execute,
    feature_map_circuit,
    parse_circuit,
    statevector,
)
from .config import PREFERRED_BACKENDS, SIMULATOR_BACKENDS, Limits, get_limits


logger = logging.getLogger(__name__)

# Canonical backend name -> (python module that must import, superquantx module)
BACKEND_SDKS: dict[str, tuple[str, str]] = {
    "simulator": ("numpy", "superquantx.backends.simulator_backend"),
    "qiskit": ("qiskit", "superquantx.backends.qiskit_backend"),
    "pennylane": ("pennylane", "superquantx.backends.pennylane_backend"),
    "cirq": ("cirq", "superquantx.backends.cirq_backend"),
    "braket": ("braket", "superquantx.backends.braket_backend"),
    "tket": ("pytket", "superquantx.backends.tket_backend"),
    "ocean": ("dimod", "superquantx.backends.ocean_backend"),
}
GATE_MODEL_BACKENDS = ("simulator", "qiskit", "pennylane", "cirq", "braket", "tket")

SDK_PACKAGES = (
    "qiskit",
    "qiskit-aer",
    "qiskit-ibm-runtime",
    "pennylane",
    "pennylane-lightning",
    "cirq",
    "cirq-core",
    "amazon-braket-sdk",
    "pytket",
    "pytket-qiskit",
    "dwave-ocean-sdk",
    "dimod",
    "numpy",
    "scipy",
    "scikit-learn",
    "mcp",
)

_BACKEND_CACHE: dict[tuple[str, str | None], Any] = {}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _error(kind: str, message: str, **extra: Any) -> dict[str, Any]:
    err = {"type": kind, "message": message}
    err.update(extra)
    return {"ok": False, "error": err}


def _exc_error(exc: BaseException, stage: str, **extra: Any) -> dict[str, Any]:
    tb = traceback.extract_tb(exc.__traceback__)
    where = f"{tb[-1].filename.rsplit('/', 1)[-1]}:{tb[-1].lineno}" if tb else None
    return _error(
        type(exc).__name__, str(exc) or repr(exc), stage=stage, where=where, **extra
    )


def _canonical_backend(name: str) -> str:
    from superquantx.backends import BACKEND_ALIASES

    if not isinstance(name, str) or not name.strip():
        raise CircuitError("Backend name must be a non-empty string")
    key = name.strip().lower()
    key = BACKEND_ALIASES.get(key, key)
    if key == "quantinuum":
        key = "tket"
    if key == "dwave":
        key = "ocean"
    return key


def _check_backend_allowed(name: str, device: str | None, limits: Limits) -> None:
    if name not in BACKEND_SDKS:
        raise CircuitError(
            f"Unknown backend '{name}'. Known backends: {', '.join(BACKEND_SDKS)}"
        )
    if name not in GATE_MODEL_BACKENDS:
        raise CircuitError(f"Backend '{name}' is not a gate-model backend")
    if device is not None and not limits.allow_hardware:
        raise CircuitError(
            "Custom devices are disabled. Only local simulators run by default. "
            "Set SUPERQUANTX_MCP_ALLOW_HARDWARE=1 in the server environment to allow "
            "a 'device' argument (this may incur cost on cloud hardware)."
        )


def _get_backend(name: str, device: str | None = None, shots: int = 1024) -> Any:
    from superquantx.backends import get_backend

    key = (name, device)
    if key not in _BACKEND_CACHE:
        kwargs: dict[str, Any] = {"shots": shots}
        if device is not None:
            kwargs["device"] = device
        _BACKEND_CACHE[key] = get_backend(name, **kwargs)
    return _BACKEND_CACHE[key]


def _check_shots(shots: Any, limits: Limits) -> int:
    if isinstance(shots, bool) or not isinstance(shots, int):
        raise CircuitError("'shots' must be an integer")
    if shots < 1:
        raise CircuitError("'shots' must be at least 1")
    if shots > limits.max_shots:
        raise CircuitError(
            f"'shots' is {shots}; the limit is {limits.max_shots} "
            "(set SUPERQUANTX_MCP_MAX_SHOTS to change, hard ceiling applies)"
        )
    return shots


def _version() -> str:
    try:
        from superquantx import __version__

        return __version__
    except Exception:  # pragma: no cover
        return "unknown"


BELL_QASM = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q -> c;\n'


# --------------------------------------------------------------------------
# list_backends / backend_versions
# --------------------------------------------------------------------------


def list_backends(smoke_test: bool = True) -> dict[str, Any]:
    """Report which SuperQuantX backends are installed and actually work."""
    from superquantx.backends import BACKEND_REGISTRY

    limits = get_limits()
    results: dict[str, Any] = {}
    working: list[str] = []
    for name, (sdk_module, sqx_module) in BACKEND_SDKS.items():
        info: dict[str, Any] = {
            "sdk_installed": importlib.util.find_spec(sdk_module) is not None,
            "registered": name in BACKEND_REGISTRY
            and BACKEND_REGISTRY.get(name) is not None,
            "gate_model": name in GATE_MODEL_BACKENDS,
            "simulator_only_by_default": name in SIMULATOR_BACKENDS,
        }
        if not info["registered"]:
            if info["sdk_installed"]:
                try:
                    importlib.import_module(sqx_module)
                    info["error"] = "Backend module imported but is not registered"
                except Exception as exc:  # report the real import failure
                    info["error"] = f"{type(exc).__name__}: {exc}"
            else:
                info["error"] = f"SDK '{sdk_module}' not installed"
            info["status"] = "unavailable"
            results[name] = info
            continue
        t0 = time.perf_counter()
        try:
            with capture_backend_logs() as logs:
                backend = _get_backend(name)
            info["init_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            info["device"] = str(getattr(backend, "device", None))
            if logs:
                info["init_warnings"] = logs[:5]
        except Exception as exc:
            info["status"] = (
                "not_installed" if not info["sdk_installed"] else "init_failed"
            )
            info["error"] = f"{type(exc).__name__}: {exc}"
            results[name] = info
            continue
        if smoke_test and info["gate_model"]:
            try:
                circuit = parse_circuit(BELL_QASM, limits)
                run = execute(circuit, name, backend, shots=128)
                bad = {k: v for k, v in run["counts"].items() if k not in ("00", "11")}
                info["smoke_test"] = {
                    "circuit": "bell",
                    "counts": run["counts"],
                    "execute_ms": run["execute_ms"],
                    "passed": not bad,
                }
                info["status"] = "working" if not bad else "wrong_results"
            except Exception as exc:
                info["smoke_test"] = {
                    "circuit": "bell",
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                info["status"] = "execution_failed"
        else:
            info["status"] = "initialized"
        if info["status"] in ("working", "initialized") and info["gate_model"]:
            working.append(name)
        results[name] = info
    return {
        "ok": True,
        "working_backends": working,
        "preferred_backends": [b for b in PREFERRED_BACKENDS if b in working],
        "backends": results,
        "limits": limits.as_dict(),
        "note": "Only local simulators are used unless SUPERQUANTX_MCP_ALLOW_HARDWARE=1.",
    }


def backend_versions() -> dict[str, Any]:
    """Versions of SuperQuantX, Python, the MCP SDK, and every quantum SDK."""
    packages: dict[str, str | None] = {}
    for pkg in SDK_PACKAGES:
        try:
            packages[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            packages[pkg] = None
    backends: dict[str, Any] = {}
    from superquantx.backends import BACKEND_REGISTRY

    for name in BACKEND_SDKS:
        if BACKEND_REGISTRY.get(name) is None:
            continue
        try:
            info = _get_backend(name).get_version_info()
            clean: dict[str, Any] = {}
            for k, v in info.items():
                if k == "capabilities":
                    continue
                if isinstance(v, (str, int, float, bool, type(None))):
                    clean[k] = v
                elif isinstance(v, dict):
                    clean[k] = sorted(str(x) for x in v)
                elif isinstance(v, (list, tuple, set)):
                    clean[k] = [str(x) for x in v]
                else:
                    clean[k] = str(v)
            backends[name] = clean
        except Exception as exc:
            backends[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": True,
        "superquantx": _version(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {k: v for k, v in packages.items() if v is not None},
        "not_installed": [k for k, v in packages.items() if v is None],
        "backends": backends,
    }


# --------------------------------------------------------------------------
# run_circuit / bell_state_demo
# --------------------------------------------------------------------------


def run_circuit(
    circuit: Any,
    backend: str = "simulator",
    shots: int = 1024,
    device: str | None = None,
) -> dict[str, Any]:
    """Run an OpenQASM 2/3 string or JSON gate list on one backend."""
    limits = get_limits()
    t_total = time.perf_counter()
    try:
        name = _canonical_backend(backend)
        _check_backend_allowed(name, device, limits)
        shots = _check_shots(shots, limits)
        t0 = time.perf_counter()
        parsed = parse_circuit(circuit, limits)
        parse_ms = (time.perf_counter() - t0) * 1000
    except CircuitError as exc:
        return _error("ValidationError", str(exc))
    try:
        t0 = time.perf_counter()
        b = _get_backend(name, device)
        init_ms = (time.perf_counter() - t0) * 1000
    except Exception as exc:
        return _exc_error(exc, "backend_init", backend=name)
    try:
        run = execute(parsed, name, b, shots)
    except Exception as exc:
        return _exc_error(exc, "execution", backend=name)
    total = shots and sum(run["counts"].values())
    return {
        "ok": True,
        "backend": name,
        "device": str(getattr(b, "device", None)),
        "n_qubits": parsed.n_qubits,
        "input_format": parsed.source_format,
        "gate_counts": parsed.gate_counts(),
        "counts": run["counts"],
        "probabilities": {k: round(v / total, 6) for k, v in run["counts"].items()},
        "shots": run["shots"],
        "bit_order": run["bit_order"],
        "timing_ms": {
            "parse": round(parse_ms, 2),
            "backend_init_or_cache": round(init_ms, 2),
            "execute": run["execute_ms"],
            "total": round((time.perf_counter() - t_total) * 1000, 2),
        },
        "backend_warnings": run["backend_warnings"],
    }


def bell_state_demo(
    backends: list[str] | None = None, shots: int = 1024
) -> dict[str, Any]:
    """Prepare (|00> + |11>)/sqrt(2) on several SDKs and compare."""
    limits = get_limits()
    try:
        shots = _check_shots(shots, limits)
    except CircuitError as exc:
        return _error("ValidationError", str(exc))
    if backends is None:
        backends = list(PREFERRED_BACKENDS)
    if (
        not isinstance(backends, list)
        or not backends
        or len(backends) > len(BACKEND_SDKS)
    ):
        return _error(
            "ValidationError", "'backends' must be a short non-empty list of names"
        )
    out: dict[str, Any] = {}
    for b in backends:
        res = run_circuit(BELL_QASM, backend=b, shots=shots)
        if res["ok"]:
            counts = res["counts"]
            corr = (counts.get("00", 0) + counts.get("11", 0)) / max(
                1, sum(counts.values())
            )
            out[res["backend"]] = {
                "ok": True,
                "counts": counts,
                "correlated_fraction": round(corr, 4),
                "p00": res["probabilities"].get("00", 0.0),
                "p11": res["probabilities"].get("11", 0.0),
                "execute_ms": res["timing_ms"]["execute"],
            }
        else:
            out[str(b)] = res
    ok_runs = [v for v in out.values() if v.get("ok")]
    return {
        "ok": bool(ok_runs),
        "circuit_qasm": BELL_QASM,
        "expected": "about 50% '00' and 50% '11', never '01' or '10' (ideal simulator)",
        "shots": shots,
        "results": out,
        "all_entangled": bool(ok_runs)
        and all(v["correlated_fraction"] == 1.0 for v in ok_runs),
    }


# --------------------------------------------------------------------------
# qsvm_classify
# --------------------------------------------------------------------------

TOY_DATASETS = ("moons", "circles", "blobs", "xor", "iris")
FEATURE_MAPS = ("ZZFeatureMap", "ZFeatureMap", "PauliFeatureMap")
KERNEL_MODES = ("statevector", "backend")


def _toy_dataset(
    name: str, n_samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray, str]:
    from sklearn import datasets as skd

    rng = np.random.default_rng(seed)
    if name == "moons":
        X, y = skd.make_moons(n_samples=n_samples, noise=0.15, random_state=seed)
        desc = "two interleaving half circles, 2 features"
    elif name == "circles":
        X, y = skd.make_circles(
            n_samples=n_samples, noise=0.08, factor=0.5, random_state=seed
        )
        desc = "two concentric circles, 2 features"
    elif name == "blobs":
        X, y = skd.make_blobs(
            n_samples=n_samples, centers=2, n_features=2, random_state=seed
        )
        desc = "two Gaussian blobs, 2 features"
    elif name == "xor":
        X = rng.uniform(-1, 1, size=(n_samples, 2))
        y = (np.sign(X[:, 0]) != np.sign(X[:, 1])).astype(int)
        desc = "XOR quadrants, 2 features"
    elif name == "iris":
        X_all, y_all = skd.load_iris(return_X_y=True)
        mask = y_all > 0  # versicolor vs virginica: the non-trivial pair
        X_all, y_all = X_all[mask], y_all[mask] - 1
        idx = rng.permutation(len(y_all))[:n_samples]
        X, y = X_all[idx], y_all[idx]
        desc = "Iris versicolor vs virginica, 4 features"
    else:
        raise CircuitError(
            f"Unknown dataset '{name}'. Choose one of: {', '.join(TOY_DATASETS)}"
        )
    return np.asarray(X, float), np.asarray(y, int), desc


def _as_array(name: str, value: Any, ndim: int) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise CircuitError(f"'{name}' must be a numeric array") from None
    if arr.ndim != ndim:
        raise CircuitError(
            f"'{name}' must be {ndim}-dimensional, got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise CircuitError(f"'{name}' contains NaN or infinite values")
    return arr


def qsvm_classify(
    dataset: str | None = "moons",
    backend: str = "simulator",
    X: list[list[float]] | None = None,
    y: list[int] | None = None,
    X_test: list[list[float]] | None = None,
    y_test: list[int] | None = None,
    n_samples: int = 40,
    test_size: float = 0.3,
    feature_map: str = "ZZFeatureMap",
    feature_map_reps: int = 1,
    shots: int = 256,
    seed: int = 0,
    classical_baseline: bool = True,
    kernel: str = "statevector",
    bandwidth: float = 0.5,
) -> dict[str, Any]:
    """Train and score a quantum-kernel SVM, with a classical RBF baseline.

    ``kernel="statevector"`` (default) computes the fidelity kernel
    ``|<phi(x)|phi(x')>|^2`` from exact statevectors that the chosen SDK
    produces for a Qiskit-convention feature map, and hands it to
    :class:`superquantx.algorithms.QuantumSVM` as its ``quantum_kernel``.
    ``kernel="backend"`` uses the backend's own ``compute_kernel_matrix``
    instead, using the library's fidelity kernel implementation.
    ``bandwidth`` scales the standardised features before encoding (statevector
    mode); values below 1 reduce exponential kernel concentration.
    """
    from sklearn.model_selection import train_test_split

    limits = get_limits()
    try:
        name = _canonical_backend(backend)
        _check_backend_allowed(name, None, limits)
        shots = _check_shots(shots, limits)
        if (
            isinstance(bandwidth, bool)
            or not isinstance(bandwidth, (int, float))
            or not 0.01 <= float(bandwidth) <= 10
        ):
            raise CircuitError("'bandwidth' must be a number between 0.01 and 10")
        if kernel not in KERNEL_MODES:
            raise CircuitError(f"'kernel' must be one of {', '.join(KERNEL_MODES)}")
        if feature_map not in FEATURE_MAPS:
            raise CircuitError(
                f"'feature_map' must be one of {', '.join(FEATURE_MAPS)}"
            )
        if kernel == "statevector" and feature_map == "PauliFeatureMap":
            feature_map = "ZZFeatureMap"  # Qiskit's default PauliFeatureMap(Z, ZZ) is ZZFeatureMap
        if not isinstance(feature_map_reps, int) or not 1 <= feature_map_reps <= 3:
            raise CircuitError("'feature_map_reps' must be 1, 2 or 3")
        if (
            not isinstance(test_size, (int, float))
            or not 0.1 <= float(test_size) <= 0.5
        ):
            raise CircuitError("'test_size' must be between 0.1 and 0.5")
        if X is not None or y is not None:
            if X is None or y is None:
                raise CircuitError("Provide both 'X' and 'y'")
            Xa = _as_array("X", X, 2)
            ya = _as_array("y", y, 1).astype(int)
            if len(Xa) != len(ya):
                raise CircuitError("'X' and 'y' have different lengths")
            source = "provided arrays"
            desc = f"user data, {Xa.shape[1]} features"
        else:
            if not isinstance(n_samples, int) or n_samples < 10:
                raise CircuitError("'n_samples' must be an integer of at least 10")
            if n_samples > limits.max_samples:
                raise CircuitError(
                    f"'n_samples' is {n_samples}; the limit is {limits.max_samples}"
                )
            Xa, ya, desc = _toy_dataset(str(dataset).lower(), n_samples, seed)
            source = f"toy dataset '{str(dataset).lower()}'"
        if len(Xa) > limits.max_samples:
            raise CircuitError(
                f"{len(Xa)} training samples; the limit is {limits.max_samples}"
            )
        if Xa.shape[1] > limits.max_features:
            raise CircuitError(
                f"{Xa.shape[1]} features means {Xa.shape[1]} qubits; the limit is "
                f"{limits.max_features}. Reduce features first (for example PCA)."
            )
        if len(np.unique(ya)) < 2:
            raise CircuitError("Labels must contain at least two classes")
        if X_test is not None or y_test is not None:
            if X_test is None or y_test is None:
                raise CircuitError("Provide both 'X_test' and 'y_test'")
            Xtr, ytr = Xa, ya
            Xte = _as_array("X_test", X_test, 2)
            yte = _as_array("y_test", y_test, 1).astype(int)
            if Xte.shape[1] != Xtr.shape[1] or len(Xte) != len(yte):
                raise CircuitError("Test arrays do not match training shape")
            if len(Xte) > limits.max_samples:
                raise CircuitError(
                    f"{len(Xte)} test samples; the limit is {limits.max_samples}"
                )
        else:
            Xtr, Xte, ytr, yte = train_test_split(
                Xa, ya, test_size=float(test_size), random_state=seed, stratify=ya
            )
    except CircuitError as exc:
        return _error("ValidationError", str(exc))

    from superquantx.algorithms import QuantumSVM

    np.random.seed(seed)
    t0 = time.perf_counter()
    try:
        b = _get_backend(name)
    except Exception as exc:
        return _exc_error(exc, "backend_init", backend=name)
    quantum_kernel = None
    state_cache: dict[bytes, Any] = {}
    if kernel == "statevector":

        def _state(x: np.ndarray) -> Any:
            key = np.asarray(x, dtype=float).tobytes()
            if key not in state_cache:
                xs = np.asarray(x, dtype=float) * float(bandwidth)
                state_cache[key] = statevector(
                    feature_map_circuit(xs, feature_map, feature_map_reps), name, b
                )
            return state_cache[key]

        def quantum_kernel(x1: np.ndarray, x2: np.ndarray) -> float:
            return float(abs(np.vdot(_state(x1), _state(x2))) ** 2)

    try:
        with capture_backend_logs() as logs:
            model = QuantumSVM(
                backend=b,
                feature_map=feature_map if kernel == "backend" else "ZZFeatureMap",
                feature_map_reps=feature_map_reps,
                shots=shots,
                quantum_kernel=quantum_kernel,
            )
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            train_pred = model.svm.predict(model.kernel_matrix_)
    except Exception as exc:
        return _exc_error(exc, "qsvm", backend=name)
    q_ms = (time.perf_counter() - t0) * 1000
    fell_back = any("RBF kernel as fallback" in m for m in logs)
    K = np.asarray(model.kernel_matrix_, dtype=float)
    diag = float(np.mean(np.diag(K)))
    off = K[~np.eye(len(K), dtype=bool)]
    kernel_ok = diag > 0.8 and not fell_back
    if kernel == "statevector":
        method = (
            f"exact fidelity kernel from {name} statevectors of a "
            f"{feature_map} (reps={feature_map_reps}, bandwidth={float(bandwidth)}), "
            "fed to SuperQuantX QuantumSVM"
        )
    else:
        method = f"{name} backend compute_kernel_matrix (library built-in)"
    result: dict[str, Any] = {
        "ok": True,
        "backend": name,
        "data": source,
        "description": desc,
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "n_qubits": int(Xtr.shape[1]),
        "feature_map": feature_map,
        "feature_map_reps": feature_map_reps,
        "kernel_method": method,
        "shots": shots if kernel == "backend" else None,
        "quantum": {
            "test_accuracy": round(float(np.mean(pred == yte)), 4),
            "train_accuracy": round(float(np.mean(train_pred == ytr)), 4),
            "time_ms": round(q_ms, 1),
        },
        "kernel_diagnostics": {
            "mean_diagonal": round(diag, 4),
            "expected_diagonal": 1.0,
            "mean_off_diagonal": round(float(off.mean()), 4) if off.size else None,
            "quantum_kernel_used": not fell_back,
            "looks_valid": kernel_ok,
        },
        "backend_warnings": logs[:5],
    }
    if not kernel_ok:
        result["warning"] = (
            "The kernel diagonal is far from 1 or the library fell back to a classical "
            "RBF kernel, so this backend's quantum kernel result is not trustworthy. "
            "Use kernel='statevector'."
        )
    if classical_baseline:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        t1 = time.perf_counter()
        clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale"))
        clf.fit(Xtr, ytr)
        result["classical_rbf_baseline"] = {
            "test_accuracy": round(float(clf.score(Xte, yte)), 4),
            "time_ms": round((time.perf_counter() - t1) * 1000, 1),
        }
        result["note"] = (
            "Toy-scale comparison on a noiseless simulator. Matching or beating the RBF "
            "baseline here is not evidence of quantum advantage."
        )
    return result


# --------------------------------------------------------------------------
# qaoa_maxcut
# --------------------------------------------------------------------------


def _parse_edges(
    edges: Any, n_nodes: int | None, limits: Limits
) -> tuple[int, list[tuple[int, int, float]]]:
    if not isinstance(edges, list) or not edges:
        raise CircuitError(
            "'edges' must be a non-empty list like [[0, 1], [1, 2, 0.5]]"
        )
    parsed: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()
    for e in edges:
        if not isinstance(e, (list, tuple)) or len(e) not in (2, 3):
            raise CircuitError(f"Edge {e!r} must be [u, v] or [u, v, weight]")
        u, v = e[0], e[1]
        # JSON clients (and schema coercion) may send 1.0 for node 1.
        u, v = (
            int(t) if isinstance(t, float) and t.is_integer() else t for t in (u, v)
        )
        if (
            any(isinstance(t, bool) or not isinstance(t, int) for t in (u, v))
            or u < 0
            or v < 0
        ):
            raise CircuitError(f"Edge {e!r} needs non-negative integer node ids")
        if u == v:
            raise CircuitError(f"Self-loop {e!r} is not allowed")
        w = e[2] if len(e) == 3 else 1.0
        if (
            isinstance(w, bool)
            or not isinstance(w, (int, float))
            or not math.isfinite(w)
            or w <= 0
        ):
            raise CircuitError(f"Edge weight in {e!r} must be a positive number")
        key = (min(u, v), max(u, v))
        if key in seen:
            raise CircuitError(f"Duplicate edge {list(key)}")
        seen.add(key)
        parsed.append((key[0], key[1], float(w)))
    max_node = max(max(u, v) for u, v, _ in parsed)
    n = n_nodes if n_nodes is not None else max_node + 1
    if isinstance(n, bool) or not isinstance(n, int) or n <= max_node:
        raise CircuitError("'n_nodes' must be an integer larger than every node id")
    if n > limits.max_qaoa_nodes:
        raise CircuitError(
            f"{n} nodes means {n} qubits; the limit is {limits.max_qaoa_nodes}"
        )
    if n < 2:
        raise CircuitError("Need at least 2 nodes")
    return n, parsed


def _cut_value(assign: list[int], edges: list[tuple[int, int, float]]) -> float:
    return float(sum(w for u, v, w in edges if assign[u] != assign[v]))


def _bits_to_assign(bitstring: str, n: int) -> list[int]:
    # little-endian: node i is bitstring[-(i + 1)]
    return [int(bitstring[-(i + 1)]) for i in range(n)]


def _qaoa_circuit(
    n: int, edges: list[tuple[int, int, float]], gammas, betas
) -> Circuit:
    gates = [Gate("h", (q,)) for q in range(n)]
    for gamma, beta in zip(gammas, betas, strict=True):
        for u, v, w in edges:
            gates += [
                Gate("cx", (u, v)),
                Gate("rz", (v,), (2.0 * gamma * w,)),
                Gate("cx", (u, v)),
            ]
        gates += [Gate("rx", (q,), (2.0 * beta,)) for q in range(n)]
    return Circuit(n_qubits=n, gates=gates, source_format="qaoa")


def qaoa_maxcut(
    edges: list[list[float]],
    n_nodes: int | None = None,
    backend: str = "simulator",
    p: int = 1,
    shots: int = 1024,
    grid_points: int = 8,
    refine_iterations: int = 30,
    seed: int = 0,
) -> dict[str, Any]:
    """Solve a small Max-Cut with QAOA and verify against brute force."""
    from scipy.optimize import minimize

    limits = get_limits()
    try:
        name = _canonical_backend(backend)
        _check_backend_allowed(name, None, limits)
        shots = _check_shots(shots, limits)
        n, E = _parse_edges(edges, n_nodes, limits)
        if not isinstance(p, int) or not 1 <= p <= limits.max_qaoa_layers:
            raise CircuitError(
                f"'p' must be an integer from 1 to {limits.max_qaoa_layers}"
            )
        if not isinstance(grid_points, int) or not 2 <= grid_points <= 16:
            raise CircuitError("'grid_points' must be an integer from 2 to 16")
        if not isinstance(refine_iterations, int) or not 0 <= refine_iterations <= 100:
            raise CircuitError("'refine_iterations' must be an integer from 0 to 100")
        gates_per_circuit = n + p * (3 * len(E) + n)
        if gates_per_circuit > limits.max_gates:
            raise CircuitError(
                f"QAOA circuit would have {gates_per_circuit} gates; limit {limits.max_gates}"
            )
    except CircuitError as exc:
        return _error("ValidationError", str(exc))

    # Classical brute force (exact for these sizes).
    t0 = time.perf_counter()
    best_val, best_assigns = -1.0, []
    for bits in itertools.product((0, 1), repeat=n):
        val = _cut_value(list(bits), E)
        if val > best_val + 1e-12:
            best_val, best_assigns = val, [list(bits)]
        elif abs(val - best_val) <= 1e-12:
            best_assigns.append(list(bits))
    brute_ms = (time.perf_counter() - t0) * 1000

    try:
        b = _get_backend(name)
    except Exception as exc:
        return _exc_error(exc, "backend_init", backend=name)

    np.random.seed(seed)
    evals = {"count": 0}

    def expected_cut(params: np.ndarray) -> tuple[float, dict[str, int]]:
        gammas, betas = params[:p], params[p:]
        run = execute(_qaoa_circuit(n, E, gammas, betas), name, b, shots)
        evals["count"] += 1
        counts = run["counts"]
        tot = sum(counts.values())
        exp = (
            sum(c * _cut_value(_bits_to_assign(k, n), E) for k, c in counts.items())
            / tot
        )
        return exp, counts

    t1 = time.perf_counter()
    try:
        # Grid search over (gamma, beta) for layer 1, linear ramp for deeper layers.
        best = (-1.0, None)
        for g in np.linspace(0.0, math.pi, grid_points, endpoint=False) + math.pi / (
            2 * grid_points
        ):
            for bt in np.linspace(
                0.0, math.pi / 2, grid_points, endpoint=False
            ) + math.pi / (4 * grid_points):
                ramp = np.linspace(1.0, 1.0 / p, p) if p > 1 else np.array([1.0])
                params = np.concatenate([g * ramp[::-1], bt * ramp])
                val, _ = expected_cut(params)
                if val > best[0]:
                    best = (val, params)
        params0 = best[1]
        if refine_iterations > 0:
            res = minimize(
                lambda x: -expected_cut(x)[0],
                params0,
                method="COBYLA",
                options={"maxiter": refine_iterations, "rhobeg": 0.2},
            )
            params_opt = res.x if -res.fun >= best[0] else params0
        else:
            params_opt = params0
        final_exp, final_counts = expected_cut(params_opt)
    except Exception as exc:
        return _exc_error(exc, "qaoa_execution", backend=name)
    q_ms = (time.perf_counter() - t1) * 1000

    sampled = sorted(
        ((k, c, _cut_value(_bits_to_assign(k, n), E)) for k, c in final_counts.items()),
        key=lambda t: (-t[2], -t[1]),
    )
    best_bits, best_count, best_cut = sampled[0]
    assign = _bits_to_assign(best_bits, n)
    most_frequent = max(final_counts.items(), key=lambda kv: kv[1])
    p_opt = sum(c for k, c, v in sampled if abs(v - best_val) <= 1e-9) / sum(
        final_counts.values()
    )
    return {
        "ok": True,
        "backend": name,
        "n_nodes": n,
        "edges": [[u, v, w] for u, v, w in E],
        "p": p,
        "shots": shots,
        "qaoa": {
            "best_sampled_cut": best_cut,
            "best_bitstring": best_bits,
            "partition": {
                "S": [i for i in range(n) if assign[i] == 0],
                "T": [i for i in range(n) if assign[i] == 1],
            },
            "most_frequent_bitstring": most_frequent[0],
            "expected_cut": round(final_exp, 4),
            "approximation_ratio": round(final_exp / best_val, 4)
            if best_val > 0
            else None,
            "probability_of_optimal_cut": round(p_opt, 4),
            "gammas": [round(float(x), 4) for x in params_opt[:p]],
            "betas": [round(float(x), 4) for x in params_opt[p:]],
            "circuit_executions": evals["count"],
            "time_ms": round(q_ms, 1),
        },
        "classical_brute_force": {
            "max_cut": best_val,
            "optimal_assignments": best_assigns[:8],
            "n_optimal_assignments": len(best_assigns),
            "time_ms": round(brute_ms, 3),
        },
        "found_optimum": abs(best_cut - best_val) <= 1e-9,
        "bit_order": "little-endian (node 0 is the rightmost bit)",
        "note": (
            "Brute force is exact and faster at this size. The QAOA run demonstrates the "
            "hybrid workflow on a noiseless simulator, not a speedup."
        ),
    }
