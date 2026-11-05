# Showcase

`examples/showcase.py` is the five-minute demo of what SuperQuantX does
today. It runs in about 10 seconds on a laptop, entirely on local
simulators, and checks every quantum result against something classical.

```bash
uv sync --extra all-backends
uv run python examples/showcase.py
```

1. **Backends.** Bell and GHZ circuits on every installed gate-model
   backend. Only the correct outcomes may appear.
2. **Quantum machine learning.** A quantum kernel SVM on `make_moons`,
   compared with a tuned RBF SVM and logistic regression using the protocol
   of [arXiv:2610.42029](https://arxiv.org/abs/2610.42029). See
   [Benchmarking](user-guide/benchmarking.md).
3. **Optimisation.** QAOA on a 5-node MaxCut instance, checked against
   brute force.

## Real output

Captured on 5 November 2026 with SuperQuantX 0.3.0 (Python 3.12, PennyLane 0.45,
Qiskit 2.6, Cirq 1.7, Braket SDK 1.127, pytket 2.18). Timings vary by machine; shot
counts vary with sampling.

```text
SuperQuantX 0.3.0 showcase. All runs are local classical simulations.

1. Bell and GHZ states on every backend (1000 shots each)
=========================================================
backend      Bell 00/11   GHZ 000/111  correct    time
simulator       528/472       488/512      yes   0.00s
pennylane       489/511       498/502      yes   0.01s
qiskit          509/491       525/475      yes   0.28s
cirq            498/502       484/516      yes   0.01s
braket          505/495       487/513      yes   0.03s
tket            492/508       493/507      yes   0.05s
Working: simulator, pennylane, qiskit, cirq, braket, tket

2. Quantum kernel SVM vs tuned classical models (make_moons, pennylane)
=======================================================================
model                                      accuracy  wall time  grid  QPU circuits
----------------------------------------------------------------------------------
QuantumSVM (PennyLane), tuned       0.930 +/- 0.051      3.15s    25       119,000
QuantumSVM (PennyLane), untuned     0.670 +/- 0.125      0.55s     1             -
SVC (RBF), tuned C and gamma        0.950 +/- 0.055      0.53s    25             -
Logistic regression, tuned C        0.860 +/- 0.097      0.14s     4             -
----------------------------------------------------------------------------------
100 samples, 2 features, 5-fold stratified CV (identical folds for every model), seed=0
quantum minus best classical: -0.020; no significant difference from SVC (RBF), tuned C and gamma (p=0.54)
Protocol: arXiv:2610.42029. Quantum kernels are classically simulated.

3. QAOA MaxCut vs brute force (5 nodes, 7 edges, p=2, pennylane)
================================================================
QAOA best sampled cut : 01101  value 6
Brute-force optimum   : 01101  value 6
Expected cut of state : 5.019 (approximation ratio 0.837; a random cut averages 3.5)
Matches optimum       : yes   (484 circuit evaluations, 2.1s)

Total time 7.1s
```

## What this shows, and what it does not

- Six SDKs run the same circuits through one API and agree, after bit-order
  normalisation.
- The tuned quantum kernel is competitive with classical models on this
  dataset, but not better: the difference from a tuned RBF SVM is not
  statistically significant. That matches the published evidence.
  Bandwidth tuning, not the circuit, is what lifts it from 0.67 to 0.93.
- QAOA at depth 2 finds the optimal cut among its samples. Its expected cut
  (approximation ratio about 0.84) is what would matter on hardware, and
  brute force is instant at this size.
- None of this is quantum advantage. All circuits are classically simulated.

## Agents demo

`examples/quantum_agents_demo.py` runs the portfolio agent (CVaR-VQE on a
QUBO, checked against brute-force enumeration and a Markowitz reference) and
the classification agent (which wraps `compare_to_classical`).

```bash
uv run python examples/quantum_agents_demo.py
```
