#!/usr/bin/env python3
"""SuperQuantX showcase: real circuits, honest baselines, exact checks.

Three parts, each checked against something classical:

1. Bell and GHZ circuits on every installed gate-model backend. Only the
   correct outcomes (00/11, 000/111) may appear.
2. A quantum kernel SVM on make_moons, compared with a tuned RBF SVM and
   logistic regression using the protocol of arXiv:2610.42029 (same folds,
   same tuning budget, corrected t-test).
3. QAOA on a small MaxCut instance, checked against brute force.

Everything runs on local simulators in well under a minute.

Run:
    python examples/showcase.py
    SQX_QML_BACKEND=qiskit python examples/showcase.py
"""

import logging
import os
import time
import warnings


warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import numpy as np  # noqa: E402
from sklearn.datasets import make_moons  # noqa: E402

import superquantx as sqx  # noqa: E402
from superquantx.algorithms.qaoa import brute_force_maxcut  # noqa: E402


SHOTS = 1000
GATE_BACKENDS = [
    ("simulator", {}),
    ("pennylane", {}),
    ("qiskit", {}),
    ("cirq", {}),
    ("braket", {"device": "local:braket_sv"}),
    ("tket", {}),
]


def rule(title: str) -> None:
    print(f"\n{title}\n{'=' * len(title)}")


def entangled_circuit(backend, n: int):
    circuit = backend.create_circuit(n)
    circuit = backend.add_gate(circuit, "H", 0)
    for q in range(n - 1):
        circuit = backend.add_gate(circuit, "CNOT", [q, q + 1])
    return backend.add_measurement(circuit)


def part_backends() -> list[str]:
    rule("1. Bell and GHZ states on every backend (1000 shots each)")
    print(f"{'backend':10s} {'Bell 00/11':>12s} {'GHZ 000/111':>13s} {'correct':>8s} {'time':>7s}")
    working = []
    for name, kwargs in GATE_BACKENDS:
        try:
            backend = sqx.get_backend(name, **kwargs)
            t0 = time.time()
            cells, correct = [], True
            for n in (2, 3):
                raw = backend.execute_circuit(entangled_circuit(backend, n), shots=SHOTS)
                counts = backend.counts_big_endian(raw["counts"])
                zeros, ones = counts.get("0" * n, 0), counts.get("1" * n, 0)
                correct &= zeros + ones == SHOTS
                cells.append(f"{zeros}/{ones}")
            elapsed = time.time() - t0
            print(f"{name:10s} {cells[0]:>12s} {cells[1]:>13s} {'yes' if correct else 'NO':>8s} {elapsed:6.2f}s")
            if correct:
                working.append(name)
        except Exception as e:  # noqa: BLE001
            print(f"{name:10s} unavailable: {str(e).splitlines()[0][:60]}")
    print(f"Working: {', '.join(working)}")
    return working


def part_qml(backend: str) -> None:
    rule(f"2. Quantum kernel SVM vs tuned classical models (make_moons, {backend})")
    X, y = make_moons(n_samples=100, noise=0.2, random_state=0)
    result = sqx.compare_to_classical(X, y, sqx.QuantumSVM(backend=backend), cv=5, seed=0)
    print(result.table())


def part_qaoa(backend: str) -> None:
    W = np.array(
        [[0, 1, 1, 0, 1],
         [1, 0, 1, 1, 0],
         [1, 1, 0, 1, 0],
         [0, 1, 1, 0, 1],
         [1, 0, 0, 1, 0]],
        dtype=float,
    )
    n_edges = int(W.sum() / 2)
    rule(f"3. QAOA MaxCut vs brute force ({len(W)} nodes, {n_edges} edges, p=2, {backend})")
    t0 = time.time()
    result = sqx.QAOA(backend=backend, p=2, seed=7, maxiter=200).solve(W)
    _, optimum = brute_force_maxcut(W)
    print(f"QAOA best sampled cut : {result['bitstring']}  value {result['cut_value']:.0f}")
    print(f"Brute-force optimum   : {result['classical_bitstring']}  value {optimum:.0f}")
    print(f"Expected cut of state : {result['expected_cut']:.3f} "
          f"(approximation ratio {result['approximation_ratio']:.3f}; "
          f"a random cut averages {n_edges / 2:.1f})")
    print(f"Matches optimum       : {'yes' if result['cut_value'] == optimum else 'NO'}"
          f"   ({result['n_evaluations']} circuit evaluations, {time.time() - t0:.1f}s)")


def main() -> int:
    start = time.time()
    print(f"SuperQuantX {sqx.__version__} showcase. All runs are local classical simulations.")
    working = part_backends()
    if not working:
        print("No working backend found.")
        return 1
    preferred = os.environ.get("SQX_QML_BACKEND", "pennylane")
    qml_backend = preferred if preferred in working else working[0]
    part_qml(qml_backend)
    part_qaoa(qml_backend)
    print(f"\nTotal time {time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
