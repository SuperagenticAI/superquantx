# Changelog

All notable changes to SuperQuantX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Azure Quantum, Rigetti Forest (pyQuil), NVIDIA cuQuantum and TensorFlow
  Quantum backends. These are not implemented yet.

## [0.3.0] - 2026-11-02

Demo-readiness release: every result is computed, and every result is
checked against a classical reference.

### Added
- MCP server (`superquantx[mcp]`) with six tools for quantum circuits, backend
  checks, classification and MaxCut over stdio.
- Automatic selection skips unimplemented encodings and records the choice.
- Spike: `QuantumSVM(feature_map="auto")` Qmes-style encoding selection with
  optional runtime Qmes import (installed manually from git, not a
  package extra) and a classical heuristic fallback
  (`docs/design/qmes-feature-map-auto.md`).
- `compare_to_classical` and `ComparisonResult`: benchmark any quantum
  classifier against a tuned RBF SVM and logistic regression on identical
  folds, with equal tuning budgets, per-fold preprocessing, hardware circuit
  counts and the Nadeau-Bengio corrected t-test (protocol of arXiv:2610.42029).
- `QAOA.solve(adjacency)` for MaxCut with brute-force optimum and
  approximation ratio; `brute_force_maxcut` and `cut_value` helpers.
- Built-in H2 (STO-3G, 0.735 Angstrom) Hamiltonian; VQE reports the exact
  ground energy.
- `QuantumPortfolioAgent` using CVaR-VQE (Barkoutsos et al. 2020), checked
  against brute-force selection and a Markowitz reference.
- `ZFeatureMap`.
- `examples/showcase.py` and `docs/showcase.md` with real output.
- `superquantx benchmark` runs `compare_to_classical`.
- Documentation: benchmarking guide, FAQ, troubleshooting, "What works today"
  backend table.

### Changed
- All six gate-model backends (simulator, PennyLane, Qiskit, Cirq, Braket
  local, pytket) run real circuits; bit order is normalised to big-endian.
- Fidelity kernels use the requested feature map on every backend and give
  K(x, x) = 1.
- `QuantumSVM(gamma=...)` is the kernel bandwidth.
- `QuantumNN` trains with parameter-shift gradients and a softmax readout.
- Agents rewritten around real computation; unsupported agent features
  raise `NotImplementedError` instead of returning made-up numbers.
- `superquantx list-backends` runs a Bell circuit per backend; versions are
  read with `importlib.metadata`.
- Qiskit floor raised; `pytest-asyncio` added to the dev extra.

### Removed
- Random and RBF fallbacks that silently replaced quantum results.
- Dead `src/superquantx/algorithms.py` (shadowed by the `algorithms` package).
- Interactive `input()` prompts from examples.
- Claims of quantum reinforcement learning, NLP and computer vision support.

### Fixed
- The library no longer configures the root logger on import.
- CLI `version` crash, `braket.__version__` crash, literal `\n` output and
  help examples that used a nonexistent `sqx` command.

## [0.2.0] - 2026-09-04

### Changed
- Raised supported Python floor to **3.11+** (3.10 no longer supported).
- Bumped optional quantum backend extras to current majors (PennyLane 0.45, Qiskit 2.x, Cirq 1.7, Braket 1.127, pytket 2.18, D-Wave Ocean 9).
- Qiskit backend wrapper updated for Qiskit 2.x (removed `execute` / legacy bind APIs).
- Docs and install guides updated for the new Python and backend floors.

### Added
- Backend compatibility table in `docs/BACKENDS.md`.

## [0.1.1] - 2025-09-11

### Added
- Initial release of SuperQuantX
- Complete project structure with modern Python packaging
- Support for all major quantum computing platforms:
  - IBM Qiskit
  - Xanadu PennyLane
  - Google Cirq
  - AWS Braket
  - Microsoft Azure Quantum
  - Quantinuum TKET
  - D-Wave Ocean SDK
  - Rigetti Forest
  - NVIDIA cuQuantum
  - TensorFlow Quantum
- Core quantum machine learning algorithms:
  - Quantum Support Vector Machine (QSVM)
  - Quantum Approximate Optimization Algorithm (QAOA)
  - Variational Quantum Eigensolver (VQE)
  - Quantum Neural Networks (QNN)
  - Quantum Principal Component Analysis (QPCA)
  - Quantum K-Means Clustering
  - Hybrid Quantum-Classical Classifiers
- Automatic backend selection and optimization
- Comprehensive utilities for optimization, visualization, and benchmarking
- Command-line interface for easy experimentation
- Extensive documentation and examples
- Complete test suite with coverage
- Development tools and pre-commit hooks

### Features
- Unified API across all quantum platforms
- Smart backend auto-selection based on problem requirements
- Quantum dataset generators and loaders
- Performance benchmarking tools
- Circuit optimization utilities
- Noise modeling and simulation
- Model serialization and loading
- Comprehensive visualization tools
- CLI for configuration, benchmarking, and examples

### Documentation
- Complete README with installation and usage examples
- API documentation structure
- Contributing guidelines
- Example notebooks and tutorials
- Performance benchmarks
