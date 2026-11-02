#!/usr/bin/env python3
"""VQE for the H2 molecule, checked against exact diagonalisation.

Uses the built-in two-qubit H2 Hamiltonian (STO-3G, 0.735 Angstrom, parity
mapping with two-qubit reduction). Every energy is printed next to the exact
ground energy, so you can see the error directly.

Run:
    python examples/vqe_molecule.py
    SQX_BACKENDS=simulator,pennylane,qiskit python examples/vqe_molecule.py
"""

import os
import time

import superquantx as sqx
from superquantx.algorithms import create_vqe_for_molecule


CHEMICAL_ACCURACY = 1.6e-3  # Hartree, about 1 kcal/mol


def main() -> int:
    backends = os.environ.get("SQX_BACKENDS", "simulator,pennylane").split(",")
    print("H2 ground state with VQE (electronic energy, Hartree)")
    print(f"{'backend':10s} {'ansatz':15s} {'VQE':>11s} {'exact':>11s} {'error':>9s} {'time':>6s}")

    failures = 0
    for backend in backends:
        try:
            sqx.get_backend(backend)
        except Exception as e:  # noqa: BLE001
            print(f"{backend:10s} skipped: {e}")
            continue
        for ansatz in ("RealAmplitudes", "EfficientSU2"):
            t0 = time.time()
            vqe = create_vqe_for_molecule("H2", backend=backend, ansatz=ansatz, seed=0)
            energy = vqe.find_ground_state()
            exact = vqe.exact_ground_energy()
            error = abs(energy - exact)
            failures += error > CHEMICAL_ACCURACY
            print(f"{backend:10s} {ansatz:15s} {energy:11.6f} {exact:11.6f} "
                  f"{error:9.1e} {time.time() - t0:5.1f}s")

    print("\nTotal energy = electronic + 0.719969 Ha nuclear repulsion (exact -1.137306 Ha).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
