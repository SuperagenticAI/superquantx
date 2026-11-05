# TKET Integration

TKET (Quantum Toolkit) is a quantum SDK developed by Cambridge Quantum Computing (now part of Quantinuum) that provides powerful quantum circuit compilation, optimization, and execution capabilities. SuperQuantX provides comprehensive integration with TKET, enabling access to advanced quantum compilation techniques and Quantinuum's quantum hardware.

## Overview

The TKET backend in SuperQuantX offers:

- **Advanced Circuit Compilation**: Industry-leading quantum circuit optimization
- **Quantinuum Hardware Access**: Direct integration with H-Series quantum computers
- **Rich Gate Set**: Support for extensive quantum gate operations
- **Variational Algorithms**: Built-in support for parameterized circuits
- **Multi-Backend Support**: Local simulators and remote hardware execution
- **Compiler Passes**: Customizable optimization strategies

## Installation

### Basic TKET Installation

```bash
# Install pytket
pip install pytket

# Install SuperQuantX with TKET support
uv add "superquantx[tket]==0.3.0"
```

### Extended TKET Installations

```bash
# For Quantinuum hardware access
pip install pytket-quantinuum

# For additional TKET extensions
pip install pytket-qiskit pytket-cirq

# For visualization and analysis
pip install pytket-qujax pytket-cutensornet
```

### Verify Installation

```python
import superquantx as sqx

# Check TKET availability
backend = sqx.get_backend('tket')
print(backend.get_version_info())
```

## Quick Start

### Basic Circuit Execution

```python
import superquantx as sqx

# Initialize TKET backend with simulator
backend = sqx.get_backend('tket', device='aer_simulator')

# Create a simple quantum circuit
circuit = backend.create_circuit(2)

# Add quantum gates
circuit = backend.add_gate(circuit, 'h', 0)
circuit = backend.add_gate(circuit, 'cx', [0, 1])

# Add measurements
circuit = backend.add_measurement(circuit, [0, 1])

# Execute circuit
result = backend.execute_circuit(circuit, shots=1000)
print("Results:", result['counts'])
```

### Circuit Optimization

```python
# Create a more complex circuit
circuit = backend.create_circuit(3)
circuit = backend.add_gate(circuit, 'h', 0)
circuit = backend.add_gate(circuit, 'cx', [0, 1])
circuit = backend.add_gate(circuit, 'cx', [1, 2])
circuit = backend.add_gate(circuit, 'h', 0)  # Redundant gate
circuit = backend.add_gate(circuit, 'h', 0)  # Another redundant gate

print(f"Original circuit: {circuit.n_gates} gates")

# Optimize the circuit using TKET compiler passes
optimized_circuit = backend.optimize_circuit(circuit, optimization_level=2)
print(f"Optimized circuit: {optimized_circuit.n_gates} gates")
```

## Configuration

### Backend Initialization Options

```python
# Local simulator (default, needs pytket-qiskit)
backend = sqx.TKETBackend(device='aer_simulator', shots=1024)

```

The `cirq_simulator` device needs `pytket-cirq`, which is not part of the `tket` extra (`pip install pytket-cirq`):

<!-- docs-check: skip -->
```python
cirq_backend = sqx.TKETBackend(device='cirq_simulator')
```

Quantinuum devices (requires credentials and `pytket-quantinuum`, not run here):

<!-- docs-check: skip -->
```python
backend = sqx.TKETBackend(device='H1-1E', shots=100)

# 'machine' overrides the device name sent to Quantinuum
backend = sqx.TKETBackend(device='H1-2E', machine='H1-2E', shots=1000)
```

### Environment Configuration

```bash
# Set up Quantinuum credentials
export QUANTINUUM_API_KEY="your-api-key-here"

# Configure TKET settings
export TKET_LOG_LEVEL="INFO"
export TKET_CACHE_DIR="~/.tket_cache"
```

## Hardware Integration

### Quantinuum H-Series Systems

TKET provides native access to Quantinuum's trapped-ion quantum computers:

!!! note "Requires credentials"
    These need `pytket-quantinuum` and a Quantinuum account; checked against the 0.3.0 constructor, not run. The device name is passed to `QuantinuumBackend(device_name=...)`. `api_key` is stored on the backend, but sign-in is handled by pytket-quantinuum's own login.

<!-- docs-check: skip -->
```python
backend_h1_1e = sqx.TKETBackend(device='H1-1E', shots=100)
backend_h1_2e = sqx.TKETBackend(device='H1-2E', shots=100)
backend_h2_1 = sqx.TKETBackend(device='H2-1', shots=100)
```

### Hardware-Specific Compilation

```python
# Get hardware device information
device_info = backend.get_backend_info()
print("Max qubits:", device_info['capabilities']['max_qubits'])
print("Native gates:", device_info['capabilities']['native_gates'])

# Compile circuit specifically for hardware
circuit = backend.create_circuit(4)
circuit = backend.add_gate(circuit, 'h', 0)
circuit = backend.add_gate(circuit, 'cx', [0, 1])

# Hardware-optimized compilation (level 3)
optimized = backend.optimize_circuit(circuit, optimization_level=3)
```

### Cost Management

```python
# Cost-effective execution strategies
def cost_efficient_execution(backend, circuit, target_precision=0.01):
    """Execute circuit with minimal cost while meeting precision target."""
    
    # Start with small shot count
    shots = 50
    results = []
    
    while shots <= 1000:
        result = backend.execute_circuit(circuit, shots=shots)
        results.append(result)
        
        # Estimate statistical precision
        if len(results) >= 2:
            # Simple convergence check
            prev_counts = results[-2]['counts']
            curr_counts = results[-1]['counts']
            
            # Check if results are converging
            if _results_converged(prev_counts, curr_counts, target_precision):
                break
                
        shots = min(shots * 2, 1000)
    
    return results[-1]

def _results_converged(counts1, counts2, threshold=0.01):
    """Check if two count dictionaries have converged."""
    # Simple implementation - extend as needed
    if not counts1 or not counts2:
        return False
        
    total1 = sum(counts1.values())
    total2 = sum(counts2.values())
    
    for key in set(counts1.keys()) | set(counts2.keys()):
        prob1 = counts1.get(key, 0) / total1
        prob2 = counts2.get(key, 0) / total2
        if abs(prob1 - prob2) > threshold:
            return False
    
    return True
```

## Advanced Features

### Parameterized Circuits

```python
# Variational circuits: rebuild the circuit with numeric angles for each
# parameter vector (there is no symbolic parameter binding in 0.3.0)
import numpy as np


def build_ansatz(backend, theta):
    circuit = backend.create_circuit(2)
    for i in range(2):
        circuit = backend.add_gate(circuit, 'ry', i, [theta[i]])
    circuit = backend.add_gate(circuit, 'cx', [0, 1])
    for i in range(2):
        circuit = backend.add_gate(circuit, 'ry', i, [theta[i + 2]])
    return circuit


param_values = np.array([0.5, 1.2, -0.3, 0.8])
bound_circuit = build_ansatz(backend, param_values)

# Or use the built-in ansatz library
print(backend.ansatz_param_count('RealAmplitudes', 2, reps=1))   # 4
ansatz = backend.create_ansatz('RealAmplitudes', 2, param_values, reps=1)

measured = backend.add_measurement(bound_circuit.copy())
print(backend.execute_circuit(measured, shots=1000)['counts'])
```

### Custom Compiler Passes

```python
from pytket.passes import (
    CliffordSimp,
    DecomposeBoxes,
    FullPeepholeOptimise,
    PauliSimp,
    RemoveRedundancies,
)
from pytket.circuit import OpType


def create_custom_optimization_pass():
    """A custom pass sequence, applied in stages.

    Each pass has gate-set preconditions (PauliSimp, for example, does not
    accept the TK1 gates CliffordSimp produces), so order matters.
    """
    return [
        DecomposeBoxes(),        # expand any boxes
        PauliSimp(),             # resynthesise via Pauli gadgets
        CliffordSimp(),          # simplify Clifford subcircuits
        FullPeepholeOptimise(),  # local peephole rewrites
        RemoveRedundancies(),    # drop gates that cancel
    ]


circuit = backend.create_circuit(5)
for i in range(5):
    circuit = backend.add_gate(circuit, 'h', i)
# Two rounds of ZZ rotations; TKET merges them
for angle in (0.3, 0.2):
    for i in range(4):
        circuit = backend.add_gate(circuit, 'cx', [i, i + 1])
        circuit = backend.add_gate(circuit, 'rz', i + 1, [angle])
        circuit = backend.add_gate(circuit, 'cx', [i, i + 1])

optimized_circuit = circuit.copy()
for compiler_pass in create_custom_optimization_pass():
    compiler_pass.apply(optimized_circuit)

print(f"Gates: {circuit.n_gates} -> {optimized_circuit.n_gates}")
print(f"CX:    {circuit.n_gates_of_type(OpType.CX)} -> {optimized_circuit.n_gates_of_type(OpType.CX)}")
```

### Expectation Value Calculations

```python
import numpy as np

# H2 Hamiltonian (STO-3G, 0.74 Angstrom), qubit 0 is the leftmost label
h2_hamiltonian = {
    'II': -1.0523732,
    'ZI': 0.39793742,
    'IZ': -0.39793742,
    'ZZ': -0.01128010,
    'XX': 0.18093119,
}

# Exact <psi|H|psi> from the TKET statevector (dicts, Hamiltonian objects or matrices)
energy = backend.compute_expectation(bound_circuit, h2_hamiltonian)
print(f"Energy of this ansatz state: {energy:.6f}")


def shot_based_energy(backend, ansatz, hamiltonian, shots=4000):
    """The same quantity from measurements: rotate into each Pauli basis and count."""
    total = 0.0
    for pauli, coeff in hamiltonian.items():
        if set(pauli) == {'I'}:
            total += coeff
            continue
        circuit = ansatz.copy()
        for qubit, p in enumerate(pauli):
            if p == 'X':
                circuit = backend.add_gate(circuit, 'h', qubit)
            elif p == 'Y':
                circuit = backend.add_gate(circuit, 'rx', qubit, [np.pi / 2])
        circuit = backend.add_measurement(circuit)
        counts = backend.counts_big_endian(backend.execute_circuit(circuit, shots=shots)['counts'])
        parity = sum(
            c * (-1) ** sum(int(bits[q]) for q, p in enumerate(pauli) if p != 'I')
            for bits, c in counts.items()
        )
        total += coeff * parity / shots
    return total


print(f"Shot-based estimate:        {shot_based_energy(backend, bound_circuit, h2_hamiltonian):.6f}")
```

### Mid-Circuit Measurements

```python
from pytket.circuit import Bit, Circuit


def create_adaptive_circuit(n_qubits=3):
    """Mid-circuit measurement with a classically controlled correction.

    Classical control is written with pytket directly; the backend
    executes any pytket Circuit.
    """
    circuit = Circuit(n_qubits)
    mid = Bit("mid", 0)
    circuit.add_bit(mid)
    circuit.add_c_register("final", n_qubits)

    for i in range(n_qubits):
        circuit.H(i)
    circuit.Measure(circuit.qubits[0], mid)

    # Flip qubits 1 and 2 only when the mid-circuit result was 0
    circuit.X(1, condition_bits=[mid], condition_value=0)
    circuit.X(2, condition_bits=[mid], condition_value=0)

    for i in range(n_qubits):
        circuit.Measure(circuit.qubits[i], Bit("final", i))
    return circuit


adaptive_circuit = create_adaptive_circuit()
result = backend.execute_circuit(adaptive_circuit, shots=500)
print(result['counts'])
```

## Performance Optimization

### Compilation Strategies

```python
from pytket.circuit import OpType


def compare_optimization_levels(backend, circuit):
    """Compare the TKET optimisation levels used by optimize_circuit (0 to 3)."""
    results = {}
    for level in range(4):
        optimized = backend.optimize_circuit(circuit, optimization_level=level)
        results[f"level_{level}"] = {
            'gates': optimized.n_gates,
            'depth': optimized.depth(),
            'cx_count': optimized.n_gates_of_type(OpType.CX),
        }
    return results


test_circuit = backend.create_circuit(6)
for i in range(6):
    test_circuit = backend.add_gate(test_circuit, 'h', i)
for i in range(5):
    test_circuit = backend.add_gate(test_circuit, 'cx', [i, i + 1])
for i in range(6):
    test_circuit = backend.add_gate(test_circuit, 'rz', i, [0.5])

for level, metrics in compare_optimization_levels(backend, test_circuit).items():
    print(f"{level}: {metrics}")
```

### Circuit Batching

```python
def batch_circuit_execution(backend, circuits, batch_size=10):
    """Execute multiple circuits efficiently."""
    
    results = []
    
    for i in range(0, len(circuits), batch_size):
        batch = circuits[i:i+batch_size]
        
        # Compile all circuits in batch
        compiled_batch = []
        for circuit in batch:
            if backend._backend:
                compiled = backend._backend.get_compiled_circuit(circuit)
                compiled_batch.append(compiled)
            else:
                compiled_batch.append(circuit)
        
        # Execute batch
        batch_results = []
        for compiled_circuit in compiled_batch:
            result = backend.execute_circuit(compiled_circuit)
            batch_results.append(result)
        
        results.extend(batch_results)
    
    return results
```

### Memory Management

```python
def memory_efficient_vqe(backend, ansatz_builder, hamiltonian, max_iterations=100):
    """Memory-efficient VQE implementation."""
    
    # Use generator for parameter optimization to save memory
    def parameter_generator():
        params = np.random.random(6) * 2 * np.pi
        
        for iteration in range(max_iterations):
            # Build circuit for current parameters
            circuit, param_names = backend.create_parameterized_circuit(4, 6)
            circuit = ansatz_builder(circuit, param_names)
            
            param_dict = {name: val for name, val in zip(param_names, params)}
            bound_circuit = backend.bind_parameters(circuit, param_dict)
            
            # Calculate energy
            energy = calculate_energy_expectation(backend, bound_circuit, hamiltonian)
            
            yield params, energy
            
            # Simple parameter update (replace with actual optimizer)
            gradient_step = 0.01
            params += np.random.normal(0, gradient_step, len(params))
    
    # Run optimization
    best_params = None
    best_energy = float('inf')
    
    for params, energy in parameter_generator():
        if energy < best_energy:
            best_energy = energy
            best_params = params.copy()
        
        if abs(energy - best_energy) < 1e-6:
            break
    
    return best_params, best_energy
```

## Error Handling and Debugging

### Comprehensive Error Management

```python
import logging

def setup_tket_logging():
    """Configure logging for TKET operations."""
    logging.basicConfig(level=logging.INFO)
    
    # Create TKET-specific logger
    tket_logger = logging.getLogger('tket_backend')
    tket_logger.setLevel(logging.DEBUG)
    
    # Add handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - TKET - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    tket_logger.addHandler(handler)
    
    return tket_logger

def robust_circuit_execution(backend, circuit, max_retries=3):
    """Execute circuit with error handling and retries."""
    
    logger = setup_tket_logging()
    
    for attempt in range(max_retries):
        try:
            # Validate circuit before execution
            if circuit.n_qubits > backend._get_max_qubits():
                raise ValueError(f"Circuit has {circuit.n_qubits} qubits, but backend supports max {backend._get_max_qubits()}")
            
            # Attempt execution
            result = backend.execute_circuit(circuit)
            
            if result['success']:
                logger.info(f"Circuit execution successful on attempt {attempt + 1}")
                return result
            else:
                logger.warning(f"Circuit execution failed on attempt {attempt + 1}: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Exception on attempt {attempt + 1}: {e}")
            
            if attempt == max_retries - 1:
                # Last attempt failed
                logger.error("All retry attempts exhausted")
                return {
                    'counts': {},
                    'shots': backend.shots,
                    'success': False,
                    'error': str(e),
                    'attempts': max_retries
                }
    
    return None
```

### Circuit Validation

```python
def validate_tket_circuit(circuit, backend):
    """Validate TKET circuit before execution."""
    
    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Check qubit count
    max_qubits = backend._get_max_qubits()
    if circuit.n_qubits > max_qubits:
        validation_results['errors'].append(
            f"Circuit uses {circuit.n_qubits} qubits, but backend supports max {max_qubits}"
        )
        validation_results['valid'] = False
    
    # Check gate support
    supported_gates = set(backend._get_native_gates())
    
    for command in circuit:
        gate_name = command.op.type.name if hasattr(command.op, 'type') else str(command.op)
        
        if gate_name not in supported_gates:
            # Check if gate can be decomposed
            validation_results['warnings'].append(
                f"Gate '{gate_name}' not native, will be decomposed"
            )
    
    # Check for measurement
    has_measurements = any(
        hasattr(command.op, 'type') and command.op.type.name == 'Measure' 
        for command in circuit
    )
    
    if not has_measurements:
        validation_results['warnings'].append(
            "Circuit has no measurements, result will be empty"
        )
    
    return validation_results
```

## Integration Examples

### VQE with TKET Optimization

```python
import numpy as np
from scipy.optimize import minimize


def tket_optimized_vqe(seed=0):
    """VQE where every ansatz circuit is optimised by TKET before evaluation."""
    backend = sqx.get_backend('tket', device='aer_simulator')

    def create_ansatz(params):
        circuit = backend.create_circuit(2)
        circuit = backend.add_gate(circuit, 'x', 0)            # |10> reference state
        circuit = backend.add_gate(circuit, 'ry', 0, [params[0]])
        circuit = backend.add_gate(circuit, 'ry', 1, [params[1]])
        circuit = backend.add_gate(circuit, 'cx', [0, 1])
        circuit = backend.add_gate(circuit, 'ry', 0, [params[2]])
        circuit = backend.add_gate(circuit, 'ry', 1, [params[3]])
        return circuit

    def energy_function(params):
        optimized = backend.optimize_circuit(create_ansatz(params), optimization_level=2)
        return backend.compute_expectation(optimized, h2_hamiltonian)

    rng = np.random.default_rng(seed)
    result = minimize(energy_function, rng.random(4) * 2 * np.pi, method='COBYLA')
    exact = sqx.Hamiltonian.from_dict(h2_hamiltonian).ground_state_energy()

    print(f"Optimized energy: {result.fun:.6f}  (exact {exact:.6f})")
    return result


vqe_result = tket_optimized_vqe()
```

### QAOA with Hardware Backend

```python
import numpy as np
from scipy.optimize import minimize


def tket_qaoa_max_cut(graph_edges, p_layers=2, seed=0):
    """QAOA for Max-Cut, with TKET optimisation before every execution."""
    n_nodes = max(max(edge) for edge in graph_edges) + 1
    backend = sqx.TKETBackend(device='aer_simulator', shots=1000)
    # For hardware use sqx.TKETBackend(device='H1-1E') (requires credentials)

    def create_qaoa_circuit(gamma, beta):
        circuit = backend.create_circuit(n_nodes)
        for i in range(n_nodes):
            circuit = backend.add_gate(circuit, 'h', i)
        for layer in range(p_layers):
            for i, j in graph_edges:                      # problem layer (ZZ terms)
                circuit = backend.add_gate(circuit, 'cx', [i, j])
                circuit = backend.add_gate(circuit, 'rz', j, [2 * gamma[layer]])
                circuit = backend.add_gate(circuit, 'cx', [i, j])
            for i in range(n_nodes):                      # mixer layer
                circuit = backend.add_gate(circuit, 'rx', i, [2 * beta[layer]])
        return circuit

    def cost_function(params):
        circuit = create_qaoa_circuit(params[:p_layers], params[p_layers:])
        optimized = backend.optimize_circuit(circuit, optimization_level=2)
        optimized = backend.add_measurement(optimized, list(range(n_nodes)))
        counts = backend.counts_big_endian(backend.execute_circuit(optimized)['counts'])

        total = sum(counts.values())
        expectation = sum(
            count / total * sum(bits[i] != bits[j] for i, j in graph_edges)
            for bits, count in counts.items()
        )
        return -expectation  # maximise the cut

    rng = np.random.default_rng(seed)
    result = minimize(cost_function, rng.uniform(0, np.pi, 2 * p_layers),
                      method='COBYLA', options={'maxiter': 60})
    print(f"Expected cut: {-result.fun:.3f} (best possible: 3)")
    return result


edges = [(0, 1), (1, 2), (2, 0), (0, 3)]
qaoa_result = tket_qaoa_max_cut(edges)
```

## Best Practices

### Circuit Design

1. **Use Native Gates**: Prefer gates from `backend._get_native_gates()`
2. **Minimize Circuit Depth**: Use TKET optimization to reduce depth
3. **Parameterize Efficiently**: Group related parameters together
4. **Validate Early**: Check circuit constraints before compilation

### Optimization Strategy

1. **Progressive Optimization**: Start with level 1, increase as needed
2. **Hardware-Specific Passes**: Use level 3 for actual hardware
3. **Cache Compiled Circuits**: Reuse compiled circuits when possible
4. **Monitor Gate Counts**: Track optimization effectiveness

### Hardware Usage

1. **Cost Awareness**: Monitor shot usage on hardware
2. **Batch Jobs**: Group multiple circuits for efficiency
3. **Error Mitigation**: Use TKET's built-in error mitigation
4. **Queue Management**: Check hardware availability before submission

## Troubleshooting

### Common Issues

**ImportError: No module named 'pytket'**
```bash
pip install pytket
# Or for all extensions
pip install pytket[all]
```

**Quantinuum API Authentication Failed**
!!! note "Requires credentials"
    Needs `pytket-quantinuum` and a Quantinuum account. Checked against the 0.3.0 constructor, not run.

<!-- docs-check: skip -->
```python
# Is the Quantinuum extension installed?
import importlib.util
print("pytket-quantinuum:", importlib.util.find_spec("pytket.extensions.quantinuum") is not None)

# Then try the device; authentication is handled by pytket-quantinuum
backend = sqx.TKETBackend(device='H1-1E')
print(backend.get_backend_info())
```

**Circuit Compilation Failed**
```python
# Check circuit validity
validation = validate_tket_circuit(circuit, backend)
print("Validation:", validation)

# Try different optimization level
try:
    optimized = backend.optimize_circuit(circuit, optimization_level=1)
except Exception as e:
    print(f"Optimization failed: {e}")
```

**Gate Not Supported**
```python
# Check native gates
native_gates = backend._get_native_gates()
print("Supported gates:", native_gates)

# Use gate decomposition
from pytket.passes import DecomposeBoxes
pass_obj = DecomposeBoxes()
pass_obj.apply(circuit)
```

### Performance Issues

**Slow Circuit Compilation**
- Reduce optimization level for development
- Cache compiled circuits
- Use smaller test circuits first

**Memory Usage High**  
- Process circuits in batches
- Clear intermediate results
- Use generators for parameter sweeps

**Hardware Queue Times**
- Check queue status before submission
- Use smaller shot counts for testing
- Prefer off-peak hours for large jobs

## Advanced Configuration

### Custom Backend Setup

```python
class CustomTKETBackend(sqx.TKETBackend):
    """Custom TKET backend with specialized features."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.custom_passes = self._setup_custom_passes()
    
    def _setup_custom_passes(self):
        """Setup custom compilation passes."""
        from pytket.passes import SequencePass, CustomPass
        
        return {
            'noise_aware': SequencePass([
                # Add noise-aware optimization passes
            ]),
            'depth_optimized': SequencePass([
                # Add depth-focused passes
            ])
        }
    
    def optimize_for_noise(self, circuit):
        """Optimize circuit specifically for noisy hardware."""
        optimized = circuit.copy()
        self.custom_passes['noise_aware'].apply(optimized)
        return optimized
```

### Integration with Other Frameworks

```python
def tket_qiskit_bridge():
    """Bridge between TKET and Qiskit circuits."""
    from qiskit import QuantumCircuit
    from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
    
    # Create Qiskit circuit
    qiskit_circuit = QuantumCircuit(2)
    qiskit_circuit.h(0)
    qiskit_circuit.cx(0, 1)
    
    # Convert to TKET
    tket_circuit = qiskit_to_tk(qiskit_circuit)
    
    # Optimize with TKET
    backend = sqx.get_backend('tket')
    optimized = backend.optimize_circuit(tket_circuit)
    
    # Convert back to Qiskit
    final_qiskit = tk_to_qiskit(optimized)
    
    return final_qiskit

def tket_cirq_integration():
    """Integration with Cirq circuits."""
    import cirq
    from pytket.extensions.cirq import cirq_to_tk, tk_to_cirq
    
    # Create Cirq circuit
    qubits = cirq.GridQubit.rect(1, 2)
    cirq_circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1])
    )
    
    # Convert and optimize
    tket_circuit = cirq_to_tk(cirq_circuit)
    backend = sqx.get_backend('tket')
    optimized = backend.optimize_circuit(tket_circuit)
    
    # Convert back
    final_cirq = tk_to_cirq(optimized)
    
    return final_cirq
```

## API Reference

The TKET backend provides these key methods:

- `create_circuit(n_qubits)`: Create quantum circuit
- `add_gate(circuit, gate, qubits, params)`: Add quantum gate
- `add_measurement(circuit, qubits)`: Add measurements  
- `execute_circuit(circuit, shots)`: Execute circuit
- `optimize_circuit(circuit, level)`: Optimize with compiler passes
- `create_parameterized_circuit(n_qubits, n_params)`: Create parameterized circuit
- `bind_parameters(circuit, param_values)`: Bind parameter values
- `expectation_value(circuit, observable)`: Calculate expectation values
- `get_statevector(circuit)`: Get quantum statevector
- `get_backend_info()`: Backend information
- `get_circuit_info()`: Circuit capabilities

For complete API documentation, see the [API Reference](../api/backends.md) section.

---

*For more information about TKET, visit the [official documentation](https://cqcl.github.io/tket/pytket/api/).*