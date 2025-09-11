# Core API Reference

The core API provides fundamental functionality for SuperQuantX, including backend management, configuration, and system information.

## Package-Level Functions

### Version Information

::: superquantx.get_version
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.get_backend_info
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.print_system_info
    handler: python
    options:
      docstring_style: google
      show_source: true

### Backend Management

::: superquantx.get_backend
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.list_available_backends
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.check_backend_compatibility
    handler: python
    options:
      docstring_style: google
      show_source: true

## Configuration

::: superquantx.config
    handler: python
    options:
      docstring_style: google
      show_source: true

::: superquantx.configure
    handler: python
    options:
      docstring_style: google
      show_source: true

## Backend Classes

### Base Backend

::: superquantx.backends.BaseBackend
    handler: python
    options:
      docstring_style: google
      show_source: true
      show_root_heading: true
      show_root_toc_entry: true

### Auto Backend

::: superquantx.AutoBackend
    handler: python
    options:
      docstring_style: google
      show_source: true

## Quick Start Examples

### Basic Usage

```python
import superquantx as sqx

# Get version information
print(f"SuperQuantX version: {sqx.get_version()}")

# Check available backends
backends = sqx.list_available_backends()
print(f"Available backends: {backends}")

# Get system information
sqx.print_system_info()
```

### Backend Selection

```python
import superquantx as sqx

# Automatic backend selection
backend = sqx.get_backend('auto')

# Specific backend with configuration
qiskit_backend = sqx.get_backend('qiskit', 
                                device='ibmq_qasm_simulator',
                                shots=1000)

# Check backend compatibility
is_compatible = sqx.check_backend_compatibility('qiskit')
print(f"Qiskit compatible: {is_compatible}")
```

### Configuration Management

```python
import superquantx as sqx

# View current configuration
print("Current config:", sqx.config)

# Update configuration
sqx.configure({
    'default_backend': 'simulator',
    'default_shots': 1024,
    'optimization_level': 2,
    'visualization': {
        'style': 'dark',
        'dpi': 300
    }
})
```

## Available Backends

SuperQuantX supports multiple quantum computing backends:

### Hardware Backends
- **IBM Quantum** (`qiskit`): Access to IBM Quantum devices
- **Google Quantum AI** (`cirq`): Integration with Google's quantum hardware
- **Amazon Braket** (`braket`): AWS quantum computing service
- **Quantinuum** (`tket`): H-Series trapped-ion quantum computers
- **D-Wave** (`ocean`): Quantum annealing systems

### Simulator Backends
- **Python Simulator** (`simulator`): Pure Python quantum simulation
- **Local Simulators**: High-performance local quantum simulators
- **Classical Emulation**: Classical algorithms for quantum-inspired computations

### Example Backend Usage

```python
import superquantx as sqx

# Initialize different backends
backends = {}

# Simulator for development
backends['sim'] = sqx.get_backend('simulator', max_qubits=10)

# IBM Quantum for real hardware
backends['ibm'] = sqx.get_backend('qiskit', 
                                 device='ibmq_manila',
                                 token='your-ibm-token')

# Amazon Braket for cloud quantum computing
backends['aws'] = sqx.get_backend('braket',
                                 device_arn='arn:aws:braket:::device/quantum-simulator/amazon/sv1',
                                 s3_bucket='my-braket-bucket')

# D-Wave for optimization problems
backends['dwave'] = sqx.get_backend('ocean',
                                   device='advantage',
                                   token='your-dwave-token')

# Use backends
for name, backend in backends.items():
    if backend.is_available():
        info = backend.get_backend_info()
        print(f"{name}: {info['provider']} - {info['device']}")
```

## Error Handling

SuperQuantX provides comprehensive error handling for backend operations:

```python
import superquantx as sqx
from superquantx.exceptions import QuantumBackendError

try:
    # Attempt to initialize backend
    backend = sqx.get_backend('qiskit', device='fake_device')
    
except QuantumBackendError as e:
    print(f"Backend error: {e}")
    # Fallback to simulator
    backend = sqx.get_backend('simulator')

except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install superquantx[qiskit]")

# Check backend availability
if backend.is_available():
    print("Backend ready for use")
else:
    print("Backend not available, check configuration")
```

## Advanced Configuration

### Environment Variables

SuperQuantX respects environment variables for configuration:

```bash
# Default backend selection
export SUPERQUANTX_BACKEND=qiskit

# IBM Quantum credentials
export QISKIT_IBMQ_TOKEN=your-token-here

# AWS credentials for Braket
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key

# D-Wave credentials
export DWAVE_API_TOKEN=your-dwave-token

# Quantinuum credentials
export QUANTINUUM_API_KEY=your-quantinuum-key
```

### Configuration Files

Create `~/.superquantx/config.yaml` for persistent configuration:

```yaml
default_backend: simulator
default_shots: 1024
optimization_level: 2

backends:
  qiskit:
    token: your-ibm-token
    hub: ibm-q
    group: open
    project: main
    
  braket:
    s3_bucket: my-quantum-bucket
    s3_prefix: experiments
    
  ocean:
    token: your-dwave-token
    solver: Advantage_system4.1

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
visualization:
  backend: matplotlib
  style: seaborn
  figsize: [10, 6]
  dpi: 300
```

### Programmatic Configuration

```python
import superquantx as sqx

# Advanced configuration
config = {
    'default_backend': 'qiskit',
    'default_shots': 2048,
    'optimization': {
        'level': 3,
        'passes': ['cx_cancellation', 'commutative_cancellation'],
        'coupling_map': 'heavy_hex',
    },
    'error_mitigation': {
        'enabled': True,
        'method': 'zero_noise_extrapolation',
        'noise_factors': [1, 3, 5],
    },
    'caching': {
        'enabled': True,
        'backend': 'file',
        'ttl': 3600,  # 1 hour
    }
}

sqx.configure(config)

# Access nested configuration
print(f"Optimization level: {sqx.config.optimization.level}")
print(f"Error mitigation enabled: {sqx.config.error_mitigation.enabled}")
```

## Package Information

### Metadata

```python
import superquantx as sqx

# Package metadata
print(f"Title: {sqx.__title__}")
print(f"Description: {sqx.__description__}")
print(f"Author: {sqx.__author__}")
print(f"License: {sqx.__license__}")
print(f"URL: {sqx.__url__}")
```

### Version Compatibility

SuperQuantX maintains compatibility with major quantum computing frameworks:

| Framework | Supported Versions | Backend Class |
|-----------|-------------------|---------------|
| Qiskit | 0.45+ | `QiskitBackend` |
| PennyLane | 0.32+ | `PennyLaneBackend` |
| Cirq | 1.0+ | `CirqBackend` |
| Amazon Braket | 1.50+ | `BraketBackend` |
| TKET/Pytket | 1.20+ | `TKETBackend` |
| D-Wave Ocean | 6.0+ | `OceanBackend` |

### Checking Dependencies

```python
import superquantx as sqx

# Get detailed backend information
backend_info = sqx.get_backend_info()

for backend_name, version in backend_info.items():
    if version:
        print(f"✓ {backend_name}: {version}")
    else:
        print(f"✗ {backend_name}: Not installed")
        
# Check specific backend availability
backends_to_check = ['qiskit', 'pennylane', 'cirq', 'braket']

for backend in backends_to_check:
    try:
        test_backend = sqx.get_backend(backend)
        print(f"✓ {backend}: Available")
    except Exception as e:
        print(f"✗ {backend}: {e}")
```

## Module Structure

The core SuperQuantX package is organized as follows:

```
superquantx/
├── __init__.py          # Main package initialization
├── algorithms/          # Quantum algorithms and ML models  
├── backends/           # Quantum backend implementations
├── datasets/           # Quantum datasets and data utilities
├── utils/              # Utility functions and tools
├── cli/                # Command-line interface
├── config.py           # Configuration management
├── exceptions.py       # Custom exception classes
├── logging_config.py   # Logging configuration
└── version.py          # Version information
```

## Integration Examples

### With Jupyter Notebooks

```python
# Cell 1: Setup
import superquantx as sqx
import numpy as np
import matplotlib.pyplot as plt

# Configure for notebook use
sqx.configure({
    'visualization': {
        'inline': True,
        'style': 'notebook'
    }
})

# Cell 2: System information  
sqx.print_system_info()

# Cell 3: Backend comparison
backends = ['simulator', 'qiskit']
for backend_name in backends:
    try:
        backend = sqx.get_backend(backend_name)
        print(f"{backend_name}: {backend.get_backend_info()}")
    except Exception as e:
        print(f"{backend_name}: Not available - {e}")
```

### With External Libraries

```python
import superquantx as sqx
import pandas as pd
import plotly.graph_objects as go

# Create backend comparison dashboard
def create_backend_dashboard():
    backend_data = []
    
    for backend_name in ['simulator', 'qiskit', 'cirq', 'braket']:
        try:
            backend = sqx.get_backend(backend_name)
            info = backend.get_backend_info()
            
            backend_data.append({
                'Backend': backend_name,
                'Provider': info.get('provider', 'Unknown'),
                'Available': backend.is_available(),
                'Max Qubits': info.get('capabilities', {}).get('max_qubits', 'N/A')
            })
        except:
            backend_data.append({
                'Backend': backend_name,
                'Provider': 'N/A',
                'Available': False,
                'Max Qubits': 'N/A'
            })
    
    df = pd.DataFrame(backend_data)
    
    # Create interactive table
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns)),
        cells=dict(values=[df[col] for col in df.columns])
    )])
    
    fig.update_layout(title="SuperQuantX Backend Status")
    return fig

# Display dashboard
dashboard = create_backend_dashboard()
dashboard.show()
```

## Best Practices

### Backend Selection

1. **Use Auto Backend for Development**:
   ```python
   backend = sqx.get_backend('auto')  # Automatically selects best available
   ```

2. **Specify Backend for Production**:
   ```python
   backend = sqx.get_backend('qiskit', device='ibmq_manila')
   ```

3. **Check Availability Before Use**:
   ```python
   if backend.is_available():
       result = run_quantum_algorithm(backend)
   else:
       logger.warning("Backend not available, using fallback")
       backend = sqx.get_backend('simulator')
   ```

### Configuration Management

1. **Use Environment Variables for Credentials**:
   ```bash
   export QISKIT_IBMQ_TOKEN=your-token
   ```

2. **Separate Development and Production Configs**:
   ```python
   if os.getenv('ENVIRONMENT') == 'production':
       sqx.configure(production_config)
   else:
       sqx.configure(development_config)
   ```

3. **Validate Configuration**:
   ```python
   def validate_config():
       required_backends = ['qiskit', 'simulator']
       for backend in required_backends:
           if not sqx.check_backend_compatibility(backend):
               raise ValueError(f"Required backend {backend} not available")
   
   validate_config()
   ```

---

For more detailed information about specific modules, see the dedicated API reference sections:
- [Algorithms API](algorithms.md)
- [Backends API](backends.md) 
- [Circuits API](circuits.md)
- [Utilities API](utilities.md)