"""SuperQuantX 0.3.0: local SDK interoperability and classical checks.

From the repository root, run:
    uv run --no-sync python examples/full-demo/demo.py --output my-results.json

Runs entirely on local simulators; no provider credentials or hardware required.
"""

import argparse
import importlib.metadata as metadata
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter


RELEASE = "0.3.0"
SEED = 7
GATE_BACKENDS = {
    "simulator": {},
    "pennylane": {"device": "default.qubit"},
    "qiskit": {"device": "aer_simulator"},
    "cirq": {"device": "simulator"},
    "braket": {"device": "local:braket/braket_sv"},
    "tket": {"device": "aer_simulator"},
}


def bell_backend(name):
    import superquantx as sqx

    backend = sqx.get_backend(name, **GATE_BACKENDS[name])
    # Identical public API across all six gate-model SDK backends.
    circuit = backend.create_circuit(2)
    circuit = backend.add_gate(circuit, "H", 0)
    circuit = backend.add_gate(circuit, "CNOT", [0, 1])
    circuit = backend.add_measurement(circuit)
    raw_counts = backend.execute_circuit(circuit, shots=1024)["counts"]
    counts = backend.counts_big_endian(raw_counts)
    total = int(sum(counts.values()))
    correlated = int(counts.get("00", 0) + counts.get("11", 0))
    if total != 1024 or correlated != total:
        raise ValueError(f"Unexpected Bell outcomes: {counts}")
    # Bell outcomes alone cannot detect reversed bit order. This asymmetric
    # circuit checks the public helper's convention: qubit 0 is the leftmost bit.
    order = backend.create_circuit(2)
    order = backend.add_gate(order, "X", 0)
    order = backend.add_measurement(order)
    raw_order_counts = backend.execute_circuit(order, shots=32)["counts"]
    order_counts = backend.counts_big_endian(raw_order_counts)
    if order_counts.get("10", 0) != 32 or sum(order_counts.values()) != 32:
        raise ValueError(f"Unexpected qubit ordering: {order_counts}")
    print(f"  Bell counts: {counts}")
    print(f"  Correlated outcomes: {correlated}/{total}")
    print(f"  Raw qubit-order counts: {raw_order_counts}")
    print(f"  Normalized (qubit 0 on left): {order_counts}")
    return {
        "counts": counts,
        "shots": total,
        "correlated_shots": correlated,
        "raw_qubit_order_counts": raw_order_counts,
        "qubit_order_counts": order_counts,
    }


def ocean():
    import superquantx as sqx

    backend = sqx.get_backend("ocean", device="simulator")
    # E(x0,x1) = x0 - 2*x1 + x0*x1. Exhaustive enumeration is the reference.
    Q = {(0, 0): 1, (1, 1): -2, (0, 1): 1}
    reference = min((a - 2 * b + a * b, a, b) for a in [0, 1] for b in [0, 1])
    result = backend.solve_qubo(Q, num_reads=1000)
    if not result["energies"]:
        raise ValueError(f"Ocean returned no energies: {result}")
    index = min(range(len(result["energies"])), key=result["energies"].__getitem__)
    sample = result["samples"][index]
    energy = float(result["energies"][index])
    a, b = int(sample[0]), int(sample[1])
    if abs(energy - (a - 2 * b + a * b)) > 1e-8:
        raise ValueError("Returned energy does not match the returned sample")
    matched = abs(energy - reference[0]) < 1e-8
    print(f"  Best returned sample: {sample}; energy: {energy}")
    print(f"  Exact minimum: {reference[0]} at x0={reference[1]}, x1={reference[2]}")
    print(f"  Matches exact minimum: {matched}")
    print("  Simulated annealing, not a gate circuit or D-Wave QPU call.")
    if not matched:
        raise ValueError("This annealing run did not find the exact minimum")
    return {
        "best_sample": {str(k): int(v) for k, v in sample.items()},
        "energy": energy,
        "exact_minimum": reference[0],
        "matches_optimum": matched,
    }


def vqe():
    import superquantx as sqx

    model = sqx.create_vqe_for_molecule("H2", backend="simulator", seed=SEED)
    estimate = float(model.find_ground_state())
    exact = float(model.exact_ground_energy())
    error = abs(estimate - exact)
    print(f"  VQE energy:          {estimate:.8f} Hartree")
    print(f"  Exact ground energy: {exact:.8f} Hartree")
    print(f"  Absolute difference: {error:.8f} Hartree")
    print("  Built-in H2 STO-3G Hamiltonian at 0.735 Angstrom.")
    if error > 1e-3:
        raise ValueError("VQE difference exceeds this demo's 0.001 Hartree check")
    return {
        "estimated_energy": estimate,
        "exact_energy": exact,
        "absolute_error": error,
    }


def qaoa():
    import numpy as np

    import superquantx as sqx

    W = np.array(
        [
            [0, 1, 1, 0, 1],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 1, 0],
            [0, 1, 1, 0, 1],
            [1, 0, 0, 1, 0],
        ],
        dtype=float,
    )
    result = sqx.QAOA(backend="simulator", p=2, seed=SEED).solve(W)
    keys = [
        "bitstring",
        "cut_value",
        "classical_optimum",
        "expected_cut",
        "approximation_ratio",
    ]
    for key in keys:
        print(f"  {key}: {result[key]}")
    print("  The sampled cut and the expected cut are different quantities.")
    print("  Brute force is easy for five nodes. No quantum advantage is demonstrated.")
    if not np.isclose(result["classical_optimum"], 6) or result["cut_value"] > 6:
        raise ValueError("Unexpected MaxCut reference or sample")
    return {key: result[key] for key in keys}


def package_versions():
    names = [
        "superquantx",
        "numpy",
        "scipy",
        "scikit-learn",
        "pennylane",
        "qiskit",
        "qiskit-aer",
        "cirq",
        "amazon-braket-sdk",
        "pytket",
        "pytket-qiskit",
        "dwave-ocean-sdk",
    ]
    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def json_value(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section", choices=["all", "backends", "algorithms"], default="all"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=[*GATE_BACKENDS, "ocean"],
        help="Run only selected backends during the backend section",
    )
    parser.add_argument(
        "--output", type=Path, help="Optional new JSON evidence file; never overwritten"
    )
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output already exists. Choose a new filename.")
    if args.backends and args.section == "algorithms":
        parser.error("--backends applies to the all/backends section only")
    try:
        installed = metadata.version("superquantx")
    except metadata.PackageNotFoundError:
        parser.error(
            'Install the SDK first: uv pip install "superquantx[all-backends]==0.3.0"'
        )
    if installed != RELEASE:
        parser.error(f"Demo targets SDK {RELEASE}, installed: {installed}")

    print(f"SuperQuantX {installed}: multi-SDK research demo", flush=True)
    print("Local simulators only. No provider account, API key or quantum hardware.")
    print(
        "Measurement counts vary; a PASS is a small correctness check, not advantage."
    )
    stages = []
    if args.section in ["all", "backends"]:
        for name in args.backends or [*GATE_BACKENDS, "ocean"]:
            stages.append(
                (name, ocean if name == "ocean" else lambda n=name: bell_backend(n))
            )
    if args.section in ["all", "algorithms"]:
        stages.extend([("vqe", vqe), ("qaoa", qaoa)])
    evidence = {
        "sdk": RELEASE,
        "python": platform.python_version(),
        "run_at": datetime.now(UTC).isoformat(),
        "scope": "local simulators only",
        "algorithm_seed": SEED,
        "packages": package_versions(),
        "stages": {},
    }
    failed = []
    for name, run in stages:
        print(f"\n--- {name} ---", flush=True)
        start = perf_counter()
        try:
            result = run()
            evidence["stages"][name] = {"status": "PASS", "result": result}
            print("  PASS", flush=True)
        except Exception as error:
            failed.append(name)
            message = f"{type(error).__name__}: {error}"
            evidence["stages"][name] = {"status": "FAILED", "error": message}
            print(f"  FAILED: {message}", flush=True)
        evidence["stages"][name]["seconds"] = round(perf_counter() - start, 3)

    print("\nSummary")
    for name, result in evidence["stages"].items():
        print(f"  {name:12} {result['status']}")
    print(f"{len(stages) - len(failed)}/{len(stages)} requested checks passed.")
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(evidence, handle, indent=2, default=json_value)
            handle.write("\n")
        print(f"Saved actual results: {args.output}")
    print("\nSDK source: https://github.com/SuperagenticAI/superquantx")
    print("Docs: https://superagenticai.github.io/superquantx/")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
