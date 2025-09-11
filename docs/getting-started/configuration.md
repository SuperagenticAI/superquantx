# Configuration Guide

This comprehensive guide covers how to configure SuperQuantX for your specific needs, from basic settings to advanced customizations for research and production-like environments.

## 🎯 Overview

SuperQuantX can be configured through multiple methods:

- **Configuration Files**: Persistent settings stored in JSON/YAML files
- **Environment Variables**: System-level configuration
- **Runtime Configuration**: Programmatic configuration during execution
- **Backend-Specific Settings**: Per-backend customization

## 📁 Configuration Files

### Default Configuration Locations

SuperQuantX looks for configuration files in these locations (in order of priority):

1. `./superquantx.json` (current directory)
2. `~/.superquantx/config.json` (user home directory)
3. `/etc/superquantx/config.json` (system-wide, Linux/macOS)
4. Built-in defaults

### Basic Configuration File

Create a basic configuration file:

=== "JSON Format"

    ```json
    {
      "default_backend": "simulator",
      "logging": {
        "level": "INFO",
        "file": "superquantx.log",
        "console": true
      },
      "simulation": {
        "max_qubits": 20,
        "default_shots": 1000,
        "seed": null
      },
      "backends": {
        "simulator": {
          "enabled": true,
          "device": "CPU"
        },
        "pennylane": {
          "enabled": true,
          "device": "default.qubit"
        },
        "qiskit": {
          "enabled": false,
          "provider": "local"
        }
      }
    }
    ```

=== "YAML Format"

    ```yaml
    default_backend: simulator
    
    logging:
      level: INFO
      file: superquantx.log
      console: true
    
    simulation:
      max_qubits: 20
      default_shots: 1000
      seed: null
    
    backends:
      simulator:
        enabled: true
        device: CPU
      
      pennylane:
        enabled: true
        device: default.qubit
      
      qiskit:
        enabled: false
        provider: local
    ```

### Creating Configuration Files

#### Method 1: Generate Default Config

```python
import superquantx as sqx

# Generate default configuration file
sqx.create_default_config(
    path="./my_config.json",
    format="json"  # or "yaml"
)

print("✅ Default configuration created!")
```

#### Method 2: Interactive Configuration

```python
import superquantx as sqx

# Interactive configuration wizard
config = sqx.configure_interactive()

# Save the configuration
config.save("./superquantx.json")
```

#### Method 3: Manual Creation

Create the file manually using your preferred text editor:

```bash
# Create configuration directory
mkdir -p ~/.superquantx

# Create configuration file
cat > ~/.superquantx/config.json << EOF
{
  "default_backend": "pennylane",
  "logging": {"level": "DEBUG"},
  "simulation": {"max_qubits": 15}
}
EOF
```

## 🔧 Runtime Configuration

### Loading Configuration

```python
import superquantx as sqx

# Load configuration from default location
sqx.load_config()

# Load from specific file
sqx.load_config("./my_config.json")

# Load from multiple files (later ones override earlier ones)
sqx.load_config(["./base_config.json", "./override_config.json"])
```

### Programmatic Configuration

```python
import superquantx as sqx

# Configure at runtime
sqx.configure(
    default_backend="pennylane",
    max_qubits=25,
    logging_level="DEBUG",
    seed=42
)

# Configure specific aspects
sqx.configure_logging(
    level="INFO",
    file="quantum_experiments.log",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

sqx.configure_simulation(
    max_qubits=30,
    default_shots=5000,
    memory_limit="8GB"
)
```

### Configuration Context Manager

```python
import superquantx as sqx

# Temporary configuration
with sqx.config_context(backend="qiskit", shots=10000):
    # This code runs with the specified configuration
    backend = sqx.get_backend("qiskit")  # Gets qiskit backend
    # Create a simple test circuit
    circuit = backend.create_circuit(n_qubits=2)
    circuit = backend.add_gate(circuit, 'H', 0)
    circuit = backend.add_measurement(circuit)
    result = backend.execute_circuit(circuit, shots=10000)  # Uses 10000 shots

# Configuration reverts after context
```

## 🌍 Environment Variables

SuperQuantX respects these environment variables:

### General Settings

```bash
# Set default backend
export SUPERQUANTX_DEFAULT_BACKEND=pennylane

# Set logging level
export SUPERQUANTX_LOG_LEVEL=DEBUG

# Set configuration file path
export SUPERQUANTX_CONFIG_PATH=/path/to/config.json

# Set seed for reproducibility
export SUPERQUANTX_SEED=42

# Set maximum number of qubits
export SUPERQUANTX_MAX_QUBITS=20
```

### Backend-Specific Variables

```bash
# PennyLane settings
export PENNYLANE_DEVICE=lightning.qubit
export PENNYLANE_SHOTS=1000

# Qiskit settings
export QISKIT_PROVIDER=IBMQ
export IBM_QUANTUM_TOKEN=your_token_here

# AWS Braket settings
export AWS_BRAKET_S3_BUCKET=your-bucket
export AWS_DEFAULT_REGION=us-east-1

# Performance settings
export SUPERQUANTX_PARALLEL_BACKENDS=4
export OMP_NUM_THREADS=8  # For OpenMP-based simulators
```

### Example Environment Setup

Create a `.env` file for your project:

```bash
# .env file
SUPERQUANTX_DEFAULT_BACKEND=pennylane
SUPERQUANTX_LOG_LEVEL=INFO
SUPERQUANTX_MAX_QUBITS=25
SUPERQUANTX_SEED=12345
PENNYLANE_DEVICE=lightning.qubit
```

Load it in your Python script:

```python
import os
from dotenv import load_dotenv
import superquantx as sqx

# Load environment variables from .env file
load_dotenv()

# SuperQuantX will automatically use these settings
print(f"Default backend: {sqx.get_default_backend()}")
print(f"Max qubits: {sqx.get_config('max_qubits')}")
```

## ⚙️ Backend Configuration

### Simulator Backend

```json
{
  "backends": {
    "simulator": {
      "enabled": true,
      "device": "CPU",          // or "GPU"
      "precision": "double",    // or "single"
      "optimization_level": 2,  // 0-3
      "memory_limit": "4GB"
    }
  }
}
```

### PennyLane Backend

```json
{
  "backends": {
    "pennylane": {
      "enabled": true,
      "device": "default.qubit",
      "shots": 1000,
      "wires": "auto",
      "diff_method": "auto",
      "lightning": {
        "enabled": true,
        "device": "lightning.qubit"
      }
    }
  }
}
```

### Qiskit Backend

```json
{
  "backends": {
    "qiskit": {
      "enabled": true,
      "provider": "local",      // "local", "IBMQ", "runtime"
      "backend_name": "qasm_simulator",
      "optimization_level": 1,
      "resilience_level": 1,
      "ibm_quantum": {
        "token": "${IBM_QUANTUM_TOKEN}",
        "hub": "ibm-q",
        "group": "open",
        "project": "main"
      }
    }
  }
}
```

### Cirq Backend

```json
{
  "backends": {
    "cirq": {
      "enabled": true,
      "simulator": "Simulator",  // "Simulator", "SparseSimulator"
      "noise_model": null,
      "device": null
    }
  }
}
```

### Amazon Braket Backend

```json
{
  "backends": {
    "braket": {
      "enabled": false,
      "device": "LocalSimulator",
      "s3_bucket": "${AWS_BRAKET_S3_BUCKET}",
      "s3_prefix": "superquantx-results",
      "aws_region": "${AWS_DEFAULT_REGION}",
      "polling_timeout": 3600
    }
  }
}
```

## 🔐 Security Configuration

### API Keys and Tokens

**Never store sensitive information in configuration files!** Use environment variables or secure key management:

```json
{
  "backends": {
    "qiskit": {
      "ibm_quantum": {
        "token": "${IBM_QUANTUM_TOKEN}",
        "token_file": "~/.qiskit/qiskitrc"
      }
    },
    "braket": {
      "aws_credentials_file": "~/.aws/credentials",
      "aws_profile": "quantum-research"
    }
  }
}
```

### Secure Key Management

```python
import superquantx as sqx
from keyring import get_password

# Store token securely
import keyring
keyring.set_password("superquantx", "ibm_quantum", "your_token_here")

# Use secure token
def get_secure_config():
    token = get_password("superquantx", "ibm_quantum")
    return {
        "backends": {
            "qiskit": {
                "ibm_quantum": {
                    "token": token
                }
            }
        }
    }

# Load secure configuration
sqx.configure(**get_secure_config())
```

## 📊 Logging Configuration

### Basic Logging Setup

```python
import superquantx as sqx

sqx.configure_logging(
    level="INFO",                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
    file="quantum_experiments.log", # Log file path
    console=True,                    # Also log to console
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    max_file_size="10MB",           # Rotate when file gets this large
    backup_count=5                   # Keep 5 old log files
)
```

### Advanced Logging Configuration

```json
{
  "logging": {
    "version": 1,
    "disable_existing_loggers": false,
    "formatters": {
      "detailed": {
        "format": "%(asctime)s [%(levelname)8s] %(name)s: %(message)s"
      },
      "simple": {
        "format": "%(levelname)s: %(message)s"
      }
    },
    "handlers": {
      "console": {
        "class": "logging.StreamHandler",
        "level": "INFO",
        "formatter": "simple",
        "stream": "ext://sys.stdout"
      },
      "file": {
        "class": "logging.handlers.RotatingFileHandler",
        "level": "DEBUG",
        "formatter": "detailed",
        "filename": "superquantx_debug.log",
        "maxBytes": 10485760,
        "backupCount": 5
      }
    },
    "loggers": {
      "superquantx": {
        "level": "DEBUG",
        "handlers": ["console", "file"],
        "propagate": false
      },
      "pennylane": {
        "level": "WARNING"
      },
      "qiskit": {
        "level": "WARNING"
      }
    }
  }
}
```

## 🚀 Performance Configuration

### Simulation Performance

```python
import superquantx as sqx

# Configure for high-performance simulation
sqx.configure_performance(
    parallel_backends=4,      # Number of backends to run in parallel
    thread_count=8,           # Threads per backend
    memory_limit="16GB",      # Memory limit for simulations
    gpu_enabled=True,         # Enable GPU acceleration if available
    optimization_level=3      # Maximum optimization
)
```

### Memory Management

```json
{
  "performance": {
    "memory_limit": "8GB",
    "swap_threshold": "6GB",
    "garbage_collection": "aggressive",
    "cache_size": "1GB",
    "lazy_loading": true
  }
}
```

### GPU Configuration

```json
{
  "gpu": {
    "enabled": true,
    "device_id": 0,           // GPU device ID
    "memory_fraction": 0.8,   // Use 80% of GPU memory
    "backends": ["pennylane", "qiskit"]  // Which backends can use GPU
  }
}
```

## 📈 Research Configuration

### Experiment Tracking

```python
import superquantx as sqx

# Configure experiment tracking
sqx.configure_experiments(
    experiment_dir="./experiments",
    auto_save_results=True,
    save_circuits=True,
    save_metadata=True,
    version_control=True
)

# Run experiment with tracking
with sqx.experiment("quantum_svm_comparison") as exp:
    exp.log_parameter("backend", "pennylane")
    exp.log_parameter("shots", 1000)
    
    # Your experiment code here
    qsvm = sqx.QuantumSVM(backend='simulator')
    # Mock data and training for demonstration
    import numpy as np
    X = np.random.rand(10, 2)  # Mock training data
    y = np.random.choice([0, 1], 10)  # Mock labels
    qsvm.fit(X, y)
    accuracy = 0.85  # Mock accuracy result
    
    exp.log_metric("accuracy", accuracy)
    exp.log_artifact("model.pkl", qsvm)
```

### Reproducibility Settings

```json
{
  "reproducibility": {
    "seed": 42,
    "deterministic_backends": true,
    "version_tracking": true,
    "environment_capture": true,
    "hash_verification": true
  }
}
```

## 🔄 Profile Management

### Creating Profiles

```python
import superquantx as sqx

# Create development profile
sqx.create_profile("development", {
    "default_backend": "simulator",
    "logging": {"level": "DEBUG"},
    "simulation": {"max_qubits": 15}
})

# Create production profile  
sqx.create_profile("production", {
    "default_backend": "pennylane",
    "logging": {"level": "WARNING"},
    "simulation": {"max_qubits": 30}
})

# Create research profile
sqx.create_profile("research", {
    "default_backend": "qiskit",
    "logging": {"level": "INFO"},
    "backends": {
        "qiskit": {"provider": "IBMQ"}
    }
})
```

### Using Profiles

```python
# Activate profile
sqx.activate_profile("research")

# Or use profile temporarily
with sqx.profile("development"):
    # Code here uses development settings
    pass
```

### Profile Commands

```bash
# List available profiles
python -m superquantx profiles list

# Show profile configuration
python -m superquantx profiles show research

# Set default profile
python -m superquantx profiles set-default production

# Copy profile
python -m superquantx profiles copy research research-backup
```

## 🔧 Configuration Validation

### Validate Configuration

```python
import superquantx as sqx

# Validate current configuration
validation_result = sqx.validate_config()

if validation_result.is_valid:
    print("✅ Configuration is valid")
else:
    print("❌ Configuration errors found:")
    for error in validation_result.errors:
        print(f"  - {error}")
```

### Schema Validation

```python
# Validate against schema
try:
    sqx.validate_config_schema("./my_config.json")
    print("✅ Configuration schema is valid")
except sqx.ConfigValidationError as e:
    print(f"❌ Schema validation failed: {e}")
```

## 🐛 Troubleshooting Configuration

### Common Issues and Solutions

#### Issue: Backend Not Found

```python
# Problem: Backend 'pennylane' not available
# Solution: Check backend configuration and installation

# Debug backend availability
available_backends = sqx.list_available_backends()
print(f"Available backends: {available_backends}")

# Check specific backend
try:
    backend = sqx.get_backend('pennylane')
    print("✅ PennyLane backend is available")
except ImportError as e:
    print(f"❌ PennyLane not installed: {e}")
```

#### Issue: Configuration Not Loading

```python
# Debug configuration loading
config_paths = sqx.get_config_search_paths()
print(f"Configuration search paths: {config_paths}")

# Check which config file is loaded
active_config_path = sqx.get_active_config_path()
print(f"Active configuration: {active_config_path}")

# Show current configuration
current_config = sqx.get_config()
print(f"Current settings: {current_config}")
```

#### Issue: Environment Variables Not Working

```bash
# Check environment variables
env | grep SUPERQUANTX

# Test specific variable
python -c "import os; print(os.getenv('SUPERQUANTX_DEFAULT_BACKEND'))"
```

### Configuration Debugging

```python
import superquantx as sqx

# Enable configuration debugging
sqx.configure(debug_config=True)

# This will print detailed configuration loading information
sqx.load_config()
```

## 📋 Configuration Reference

### Complete Configuration Schema

```json
{
  "default_backend": "simulator",
  "seed": null,
  "max_qubits": 25,
  "default_shots": 1000,
  
  "logging": {
    "level": "INFO",
    "file": null,
    "console": true,
    "format": null,
    "max_file_size": "10MB",
    "backup_count": 3
  },
  
  "simulation": {
    "precision": "double",
    "optimization_level": 2,
    "memory_limit": "4GB",
    "parallel": true,
    "thread_count": null
  },
  
  "performance": {
    "gpu_enabled": false,
    "cache_enabled": true,
    "cache_size": "1GB",
    "lazy_loading": true
  },
  
  "backends": {
    "simulator": {
      "enabled": true,
      "device": "CPU"
    },
    "pennylane": {
      "enabled": true,
      "device": "default.qubit",
      "shots": null,
      "diff_method": "auto"
    },
    "qiskit": {
      "enabled": false,
      "provider": "local",
      "backend_name": null,
      "optimization_level": 1
    },
    "cirq": {
      "enabled": false,
      "simulator": "Simulator"
    },
    "braket": {
      "enabled": false,
      "device": "LocalSimulator",
      "s3_bucket": null
    }
  },
  
  "security": {
    "token_storage": "environment",
    "secure_communication": true,
    "audit_logging": false
  },
  
  "experimental": {
    "features": [],
    "beta_backends": false
  }
}
```

## 🚀 Next Steps

Now that you've configured SuperQuantX:

1. **[Test Your Setup](quickstart.md#verify-installation)**: Verify everything works
2. **[Build Your First Program](first-program.md)**: Create quantum circuits
3. **[Explore Backends](../user-guide/backends.md)**: Learn about different quantum frameworks
4. **[Try Algorithms](../user-guide/algorithms.md)**: Use built-in quantum algorithms

## 📞 Getting Help

If you need help with configuration:

- **[FAQ](../help/faq.md)**: Common configuration questions
- **[Troubleshooting](../help/troubleshooting.md)**: Solve configuration issues
- **[GitHub Issues](https://github.com/SuperagenticAI/superquantx/issues)**: Report configuration bugs
- **Email**: [research@super-agentic.ai](mailto:research@super-agentic.ai)

---

!!! tip "Configuration Best Practices"
    - Use profiles for different environments (dev, test, prod)
    - Store secrets in environment variables, not config files
    - Start with simple configurations and add complexity as needed
    - Validate your configuration before deploying
    - Keep configuration files in version control (except secrets!)

!!! warning "Security Reminder"
    Never commit API tokens, passwords, or other sensitive information to version control. Use environment variables or secure key management systems instead.