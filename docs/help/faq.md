# FAQ

**Is SuperQuantX production software?**
No. It is a research toolkit. Everything in the [showcase](../showcase.md)
runs on local classical simulators.

**Does it show quantum advantage?**
No, and it is built to tell you so. Every algorithm reports a classical
reference: exact diagonalisation for VQE, brute force for QAOA and the
portfolio agent, and tuned classical models for classifiers via
[`compare_to_classical`](../user-guide/benchmarking.md).

**Which backends work?**
`simulator`, `pennylane`, `qiskit`, `cirq`, `braket` (local simulator) and
`tket` are tested. D-Wave `ocean` handles QUBO/Ising problems only. Cloud
hardware through those SDKs is untested. Azure Quantum, Rigetti Forest,
cuQuantum and TensorFlow Quantum are planned, not implemented. Run
`superquantx list-backends` to check your installation.

**Why do all backends give the same accuracy?**
They compute the same exact statevector kernel. Identical results are a
correctness check.

**Why is the untuned quantum kernel so weak?**
A fidelity kernel with bandwidth 1 is often too narrow or too wide for the
data. Tuning the bandwidth (the `gamma` argument of `QuantumSVM`) matters
more than the choice of feature map; see arXiv:2610.42029.

**Does the library configure logging?**
No. It only adds a `NullHandler`. Configure logging in your application.
