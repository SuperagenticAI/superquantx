# Contributing to SuperQuantX 🤝

Thank you for your interest in contributing to SuperQuantX! We welcome contributions from researchers, developers, and educators in the quantum computing community.

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Types of Contributions](#types-of-contributions)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Research Contributions](#research-contributions)

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Basic understanding of quantum computing concepts

### Development Environment Setup

#### Option 1: Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/SuperagenticAI/superquantx.git
cd superquantx

# Create and activate virtual environment with all dev dependencies
uv sync --extra full-dev

# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/SuperagenticAI/superquantx.git
cd superquantx

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[full-dev]"
```

### Verify Installation

```bash
# Run tests to verify everything works
pytest

# Check code formatting
black --check src/ tests/
ruff check src/ tests/

# Run type checking
mypy src/
```

## 🎯 Types of Contributions

### 🔬 Research Contributions
- New quantum algorithms or improvements to existing ones
- Novel quantum-agentic system architectures
- Performance benchmarks and comparisons
- Research examples and tutorials

### 🛠️ Technical Contributions  
- Bug fixes and performance improvements
- New quantum backend integrations
- Enhanced error handling and logging
- Code optimizations

### 📚 Documentation
- API documentation improvements
- Tutorial and example updates
- Research paper implementations
- Translation improvements

### 🧪 Testing
- Unit test improvements
- Integration tests for new backends
- Performance benchmarks
- Edge case coverage

## 🔄 Development Workflow

### 1. Fork and Clone
```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR-USERNAME/superquantx.git
cd superquantx
git remote add upstream https://github.com/SuperagenticAI/superquantx.git
```

### 2. Create a Feature Branch
```bash
git checkout -b feature/quantum-awesome-algorithm
# or
git checkout -b fix/backend-connection-issue
# or  
git checkout -b docs/improve-getting-started
```

### 3. Make Your Changes
- Follow our [code standards](#code-standards)
- Write tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 4. Commit Your Changes
```bash
# Stage your changes
git add .

# Commit with a clear message
git commit -m "feat: add quantum awesome algorithm

- Implement quantum awesome algorithm for classification
- Add comprehensive tests and benchmarks
- Update documentation with usage examples
- Resolves #123"
```

### Commit Message Format
We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `test`: Test additions/changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `ci`: CI/CD changes

## 📏 Code Standards

### Code Formatting
We use modern Python formatting tools:

```bash
# Format code with Black
black src/ tests/

# Sort imports with isort (integrated in Ruff)
ruff check --fix src/ tests/

# Lint with Ruff
ruff check src/ tests/
```

### Type Hints
All public functions must include type hints:

```python
from typing import Dict, List, Optional, Union
import numpy as np

def quantum_algorithm(
    data: np.ndarray, 
    backend: str = "simulator",
    shots: int = 1024,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Union[np.ndarray, float]]:
    """Execute a quantum algorithm with specified parameters."""
    if options is None:
        options = {}
    # Implementation here
    return {"result": data, "shots_used": shots}
```

### Docstring Standards
We follow the Google docstring style with comprehensive examples.

## 🧪 Testing Guidelines

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=superquantx

# Run only fast tests
pytest -m "not slow"
```

## 📖 Documentation

Documentation is built with MkDocs and auto-generated from docstrings.

```bash
# Build and serve docs locally
mkdocs serve
```

## 🔄 Submitting Changes

1. **Run full test suite**: `pytest`
2. **Check formatting**: `black --check src/ tests/`
3. **Run linting**: `ruff check src/ tests/`
4. **Type check**: `mypy src/`
5. **Update documentation** if needed
6. **Create Pull Request** with clear description

## 🔬 Research Contributions

For novel quantum algorithms:
1. Include literature references
2. Document quantum advantage
3. Provide benchmarks vs classical methods
4. Follow our algorithm interface patterns

## 🆘 Getting Help

- **Discussions**: [GitHub Discussions](https://github.com/SuperagenticAI/superquantx/discussions)
- **Issues**: [Bug Reports](https://github.com/SuperagenticAI/superquantx/issues)
- **Email**: research@super-agentic.ai

---

Thank you for contributing to quantum-agentic AI research! 🚀🔬