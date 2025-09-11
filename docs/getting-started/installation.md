# Installation Guide

This guide provides detailed instructions for installing SuperQuantX in various environments and configurations.

## 🖥️ System Requirements

### Operating System
- **Linux**: Ubuntu 18.04+, CentOS 7+, or any recent distribution
- **macOS**: 10.14 (Mojave) or later
- **Windows**: Windows 10 or Windows 11

### Python Requirements
- **Python**: 3.10, 3.11, or 3.12
- **pip**: 21.0 or later (for proper dependency resolution)

!!! warning "Python Version Important"
    SuperQuantX requires Python 3.10+ due to dependencies in quantum frameworks. Python 3.9 and earlier are not supported.

### Hardware Requirements
- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: At least 2GB free space for full installation
- **CPU**: Any modern x64 processor
- **GPU**: Optional, but recommended for large quantum simulations

## 🚀 Installation Methods

### Method 1: Basic Installation (Recommended for Beginners)

This installs SuperQuantX with the built-in simulator backend:

```bash
pip install superquantx
```

**What you get:**
- Core SuperQuantX functionality
- Built-in quantum simulator
- Basic quantum algorithms
- Visualization tools

**Perfect for:**
- Learning quantum computing
- Small experiments
- Testing SuperQuantX features

### Method 2: Backend-Specific Installation

Install SuperQuantX with specific quantum computing frameworks:

=== "PennyLane (Recommended for ML)"

    ```bash
    pip install superquantx[pennylane]
    ```

    **Includes:**
    - Core SuperQuantX
    - PennyLane quantum ML framework
    - Lightning simulator (fast C++ backend)
    - Automatic differentiation support

=== "Qiskit (IBM's Framework)"

    ```bash
    pip install superquantx[qiskit]
    ```

    **Includes:**
    - Core SuperQuantX
    - Qiskit quantum framework
    - Qiskit Aer simulator
    - IBM Quantum runtime support

=== "Cirq (Google's Framework)"

    ```bash
    pip install superquantx[cirq]
    ```

    **Includes:**
    - Core SuperQuantX
    - Cirq quantum framework
    - Cirq simulators

=== "Amazon Braket"

    ```bash
    pip install superquantx[braket]
    ```

    **Includes:**
    - Core SuperQuantX
    - Amazon Braket SDK
    - AWS quantum hardware access

=== "Multiple Backends"

    ```bash
    # Install multiple backends at once
    pip install superquantx[pennylane,qiskit,cirq]

    # Or install all quantum backends
    pip install superquantx[all-backends]
    ```

### Method 3: Development Installation

For contributing to SuperQuantX or customizing the framework:

```bash
# Clone the repository
git clone https://github.com/SuperagenticAI/superquantx.git
cd superquantx

# Install in development mode
pip install -e .[dev]

# Or with all features for development
pip install -e .[full-dev]
```

**What you get:**
- Editable installation
- All development tools (testing, linting, etc.)
- All quantum backends
- Documentation building tools

## 🔧 Advanced Installation Options

### Using Virtual Environments (Recommended)

Always use virtual environments to avoid package conflicts:

=== "venv (Built-in)"

    ```bash
    # Create virtual environment
    python -m venv superquantx-env

    # Activate (Linux/macOS)
    source superquantx-env/bin/activate

    # Activate (Windows)
    superquantx-env\Scripts\activate

    # Install SuperQuantX
    pip install superquantx[pennylane]
    ```

=== "conda"

    ```bash
    # Create conda environment
    conda create -n superquantx python=3.11
    conda activate superquantx

    # Install SuperQuantX
    pip install superquantx[pennylane]
    ```

=== "uv (Fast Python Package Manager)"

    ```bash
    # Install uv (if not installed)
    pip install uv

    # Create project with uv
    uv init my-quantum-project
    cd my-quantum-project

    # Add SuperQuantX as dependency
    uv add superquantx[pennylane]

    # Activate environment and run
    uv run python your_script.py
    ```

### Docker Installation

Run SuperQuantX in a containerized environment:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install SuperQuantX
RUN pip install superquantx[pennylane,qiskit]

# Set working directory
WORKDIR /workspace

# Copy your code
COPY . .

# Run your quantum program
CMD ["python", "your_quantum_script.py"]
```

```bash
# Build and run
docker build -t superquantx-app .
docker run -it superquantx-app
```

### Jupyter Notebook Installation

For interactive quantum computing:

```bash
# Install SuperQuantX with Jupyter support
pip install superquantx[pennylane] jupyter

# Start Jupyter
jupyter notebook

# Or use JupyterLab
pip install jupyterlab
jupyter lab
```

## ✅ Verify Installation

After installation, verify everything works correctly:

### Basic Verification

```python
import superquantx as sqx

# Check version
print(f"SuperQuantX version: {sqx.__version__}")

# List available backends
backends = sqx.list_backends()
print(f"Available backends: {backends}")

# Test basic functionality
backend = sqx.get_backend('simulator')
circuit = backend.create_circuit(2)
circuit.h(0)
circuit.cx(0, 1)
result = backend.run(circuit, shots=100)
print(f"Test successful! Results: {result.get_counts()}")
```

### Backend-Specific Tests

=== "PennyLane Test"

    ```python
    try:
        import superquantx as sqx
        backend = sqx.get_backend('pennylane')
        print("✅ PennyLane backend available")
        
        # Test QML functionality
        qsvm = sqx.QuantumSVM(backend='pennylane')
        print("✅ PennyLane QML algorithms working")
    except ImportError as e:
        print(f"❌ PennyLane not available: {e}")
    ```

=== "Qiskit Test"

    ```python
    try:
        import superquantx as sqx
        backend = sqx.get_backend('qiskit')
        print("✅ Qiskit backend available")
        
        # Test circuit functionality
        circuit = backend.create_circuit(2)
        circuit.h(0)
        print("✅ Qiskit circuits working")
    except ImportError as e:
        print(f"❌ Qiskit not available: {e}")
    ```

=== "Full Test"

    ```python
    import superquantx as sqx
    
    # Run comprehensive test
    test_results = sqx.run_diagnostics()
    print("🔍 Diagnostic Results:")
    for backend_name, status in test_results.items():
        status_icon = "✅" if status["available"] else "❌"
        print(f"{status_icon} {backend_name}: {status['message']}")
    ```

## 🐛 Troubleshooting Installation

### Common Issues and Solutions

#### Python Version Issues

**Problem**: `SuperQuantX requires Python 3.10+`

**Solutions**:
```bash
# Check Python version
python --version

# Install Python 3.11 using pyenv
curl https://pyenv.run | bash
pyenv install 3.11.0
pyenv global 3.11.0

# Or use conda
conda install python=3.11
```

#### Package Conflicts

**Problem**: `Conflicting dependencies detected`

**Solutions**:
```bash
# Create fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate  # or fresh_env\Scripts\activate on Windows

# Install with no-cache to ensure fresh packages
pip install --no-cache-dir superquantx[pennylane]
```

#### Backend Installation Issues

**Problem**: `Failed to install quantum backend`

**Solutions**:
```bash
# Update pip first
pip install --upgrade pip

# Install backends individually
pip install pennylane
pip install superquantx[pennylane]

# For system-specific issues, install system dependencies
# Ubuntu/Debian:
sudo apt-get install build-essential

# macOS:
xcode-select --install

# Windows: Install Visual Studio Build Tools
```

#### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'superquantx'`

**Solutions**:
```bash
# Verify installation
pip list | grep superquantx

# Reinstall if needed
pip uninstall superquantx
pip install superquantx

# Check Python path
python -c "import sys; print(sys.path)"
```

### Performance Optimization

#### For Large Simulations

```bash
# Install with optimized backends
pip install superquantx[pennylane]

# For Intel CPUs, install Intel MKL
pip install mkl mkl-service

# For NVIDIA GPUs with CUDA
pip install cupy-cuda11x  # or cupy-cuda12x
```

#### Memory Optimization

```python
import superquantx as sqx

# Configure for memory-efficient simulations
sqx.configure(
    max_qubits=20,          # Limit qubit count
    memory_limit="4GB",     # Set memory limit
    optimization_level=2    # Enable optimizations
)
```

## 🚀 Next Steps

Once installation is complete:

1. **[Try the Quick Start Guide](quickstart.md)** - Get running in 5 minutes
2. **[Build Your First Program](first-program.md)** - Step-by-step tutorial
3. **[Configure SuperQuantX](configuration.md)** - Customize your setup
4. **[Explore Tutorials](../tutorials/basic-quantum.md)** - Learn quantum computing

## 📞 Getting Help

If you encounter issues:

- **[FAQ](../help/faq.md)**: Check common questions
- **[Troubleshooting Guide](../help/troubleshooting.md)**: Detailed problem solving
- **[GitHub Issues](https://github.com/SuperagenticAI/superquantx/issues)**: Report bugs
- **Email**: [research@super-agentic.ai](mailto:research@super-agentic.ai)

---

!!! tip "Installation Success!"
    Great job installing SuperQuantX! You're now ready to explore the fascinating world of quantum computing. Start with our [Quick Start Guide](quickstart.md) for immediate hands-on experience.