# Changelog

All notable changes to SuperQuantX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Quantum reinforcement learning algorithms
- Quantum natural language processing
- Fault-tolerant quantum algorithms
- Quantum federated learning

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
