# Amazon Braket Backend Integration

Amazon Braket is AWS's fully managed quantum computing service that provides access to quantum computers from multiple hardware providers. SuperQuantX integrates with Braket to enable cloud-based quantum computing with enterprise-grade security and scalability.

## Overview

The Amazon Braket backend in SuperQuantX provides:

- **Multi-Vendor Hardware Access**: IonQ, Rigetti, Oxford Quantum Computing, and more
- **Cloud-Scale Simulation**: High-performance quantum simulators
- **Enterprise Integration**: AWS IAM, VPC, and compliance features
- **Hybrid Algorithms**: Classical-quantum workload orchestration
- **Cost Management**: Pay-per-use pricing with detailed billing

## Installation

```bash
# Install SuperQuantX with Braket support
uv add "superquantx[braket]==0.3.0"

# Or install Amazon Braket SDK directly
pip install amazon-braket-sdk boto3
```

## AWS Setup

### Configure AWS Credentials

```bash
# Option 1: AWS CLI
aws configure
# Enter your AWS Access Key ID, Secret Access Key, and region

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_DEFAULT_REGION=us-west-2
```

### IAM Permissions

Create IAM policy with required Braket permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "braket:*",
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket",
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "*"
        }
    ]
}
```

## Quick Start

### Basic Usage

```python
import superquantx as sqx

# Get the Braket backend (local simulator by default)
backend = sqx.get_backend("braket")

# Bell state through the unified circuit API
circuit = backend.create_circuit(2)
circuit = backend.add_gate(circuit, "H", 0)
circuit = backend.add_gate(circuit, "CNOT", [0, 1])
circuit = backend.add_measurement(circuit)

result = backend.execute_circuit(circuit, shots=1000)
print(result["counts"])  # about 500 each of '00' and '11'
```

### Cloud Simulation

!!! note "Requires credentials"
    Needs AWS credentials and an S3 bucket; checked against the 0.3.0 constructor, not run.

<!-- docs-check: skip -->
```python
# AWS managed state-vector simulator
backend = sqx.BraketBackend(
    device='arn:aws:braket:::device/quantum-simulator/amazon/sv1',
    s3_folder=('my-braket-bucket', 'my-experiments'),
    shots=1000,
)

result = backend.execute_circuit(circuit, shots=1000)
print(f"Cloud simulation results: {result['counts']}")
```

### Quantum Hardware Access

!!! note "Requires credentials"
    Needs AWS credentials, Braket access in the device's region and an S3 bucket; checked against the 0.3.0 constructor, not run.

<!-- docs-check: skip -->
```python
# Access an IonQ quantum computer
ionq_backend = sqx.BraketBackend(
    device='arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1',
    s3_folder=('my-braket-bucket', 'ionq-experiments'),
    shots=100,  # fewer shots to control cost
)
result = ionq_backend.execute_circuit(circuit, shots=100)
print(f"IonQ hardware results: {result['counts']} (task {result['task_arn']})")
```

## Device Selection

### Available Devices

!!! note "Requires credentials"
    Lists devices through `AwsDevice.get_devices()`, which needs AWS credentials; not run.

<!-- docs-check: skip -->
```python
def list_braket_devices():
    """List Amazon Braket devices grouped by provider."""
    from braket.aws import AwsDevice

    providers = {}
    for device in AwsDevice.get_devices():
        providers.setdefault(device.provider_name, []).append(device)

    for provider, device_list in providers.items():
        print(f"\n{provider}:")
        for device in device_list:
            status = "ONLINE" if device.is_available else "OFFLINE"
            print(f"  {device.name} ({device.type.value}): {status}")
            print(f"    ARN: {device.arn}")


list_braket_devices()
```

### Device-Specific Configuration

```python
# Device strings understood by BraketBackend(device=...)
DEVICES = {
    'local_sv': 'local:braket/braket_sv',        # free local state-vector simulator
    'local_dm': 'local:braket/braket_dm',        # free local density-matrix simulator
    'sv1': 'arn:aws:braket:::device/quantum-simulator/amazon/sv1',
    'tn1': 'arn:aws:braket:::device/quantum-simulator/amazon/tn1',
    'dm1': 'arn:aws:braket:::device/quantum-simulator/amazon/dm1',
    'ionq': 'arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1',
}


def configure_for_device(name='local_sv', shots=1000, s3_folder=None):
    """Managed simulators and QPUs (ARNs) need AWS credentials and an S3 folder."""
    return sqx.BraketBackend(device=DEVICES[name], shots=shots, s3_folder=s3_folder)


for name in ['local_sv', 'local_dm']:
    backend = configure_for_device(name)
    print(f"{name}: configured, max qubits {backend.capabilities['max_qubits']}")
```

Cloud devices are configured the same way with an ARN and `s3_folder=('bucket', 'prefix')` (requires credentials; not run here). Options such as `max_parallel_threads` or `poll_timeout_seconds` are not available on `BraketBackend` in 0.3.0.

## Working with Circuits

### Native Braket Gates

```python
import numpy as np


def demonstrate_braket_gates():
    """Portable gate names on the Braket backend."""
    backend = sqx.get_backend('braket')
    circuit = backend.create_circuit(4)          # a braket.circuits.Circuit

    for gate, qubit in [('H', 0), ('X', 1), ('Y', 2), ('Z', 3), ('S', 0), ('T', 1)]:
        circuit = backend.add_gate(circuit, gate, qubit)
    circuit = backend.add_gate(circuit, 'RX', 0, [np.pi / 4])
    circuit = backend.add_gate(circuit, 'RY', 1, [np.pi / 3])
    circuit = backend.add_gate(circuit, 'RZ', 2, [np.pi / 6])
    circuit = backend.add_gate(circuit, 'CNOT', [0, 1])
    circuit = backend.add_gate(circuit, 'CZ', [1, 2])
    circuit = backend.add_gate(circuit, 'SWAP', [2, 3])

    # Native Braket methods work too (target first, then angle)
    circuit.phaseshift(0, np.pi / 4)

    result = backend.execute_circuit(circuit, shots=1000)
    print(f"Counts (qubit 0 is the leftmost bit): {result['counts']}")
    return circuit


demo_circuit = demonstrate_braket_gates()
```

### Advanced Circuit Features

```python
def braket_advanced_circuits(seed=0):
    """Native Braket circuit features: parameterised ansatz, noise and result types."""
    import numpy as np
    from braket.circuits import Circuit, Noise, observables
    from braket.devices import LocalSimulator

    # 1. Parameterised ansatz written directly in Braket
    def create_parameterized_ansatz(params, n_qubits=4, depth=2):
        circuit = Circuit()
        idx = 0
        for i in range(n_qubits):
            circuit.ry(i, params[idx])
            idx += 1
        for _ in range(depth):
            for i in range(n_qubits - 1):
                circuit.cnot(i, i + 1)
            for i in range(n_qubits):
                circuit.ry(i, params[idx]).rz(i, params[idx + 1])
                idx += 2
        return circuit

    n_params = 4 + 2 * 2 * 4
    ansatz = create_parameterized_ansatz(np.random.default_rng(seed).random(n_params) * 2 * np.pi)
    print(f"Ansatz with {n_params} parameters, depth {ansatz.depth}")

    # 2. Noise: apply depolarizing noise to every gate, run on the density-matrix simulator
    bell = Circuit().h(0).cnot(0, 1)
    noisy = bell.copy().apply_gate_noise(Noise.Depolarizing(probability=0.02))
    counts = LocalSimulator('braket_dm').run(noisy, shots=2000).result().measurement_counts
    print(f"Noisy Bell counts: {dict(counts)}")

    # 3. Result types (exact on the state-vector simulator with shots=0)
    rt = Circuit().h(0).cnot(0, 1)
    rt.probability(target=[0, 1])
    rt.expectation(observable=observables.Z(), target=[0])
    rt.variance(observable=observables.X(), target=[0])
    values = LocalSimulator().run(rt, shots=0).result().values
    print(f"Probabilities {values[0]}, <Z0> = {values[1]:.3f}, Var(X0) = {values[2]:.3f}")

    # Any of these Braket circuits can also go through the SuperQuantX backend
    print(sqx.BraketBackend().execute_circuit(bell, shots=100)['counts'])


braket_advanced_circuits()
```

## Cost Management

### Cost Estimation

```python
def estimate_braket_costs():
    """Estimate costs for different Braket services."""
    
    print("Amazon Braket Cost Estimation")
    print("=" * 32)
    
    # Current pricing (as of 2024 - check AWS for latest)
    pricing = {
        'sv1': {
            'per_task': 0.075,  # $0.075 per task
            'per_shot': 0.00019  # $0.00019 per shot
        },
        'tn1': {
            'per_task': 0.275,  # $0.275 per task  
            'per_shot': 0.00019  # $0.00019 per shot
        },
        'dm1': {
            'per_task': 0.075,
            'per_shot': 0.00019
        },
        'ionq': {
            'per_task': 0.00,   # Included in shot pricing
            'per_shot': 0.01    # $0.01 per shot
        },
        'rigetti': {
            'per_task': 0.00375,  # $0.00375 per task
            'per_shot': 0.00035   # $0.00035 per shot
        }
    }
    
    # Example workloads
    workloads = {
        'Development': {'tasks': 100, 'shots_per_task': 1000},
        'Research': {'tasks': 50, 'shots_per_task': 5000},
        'Production': {'tasks': 20, 'shots_per_task': 10000}
    }
    
    for workload_name, workload in workloads.items():
        print(f"\n{workload_name} Workload:")
        print(f"  Tasks: {workload['tasks']}")
        print(f"  Shots per task: {workload['shots_per_task']}")
        print(f"  Total shots: {workload['tasks'] * workload['shots_per_task']:,}")
        
        print(f"\n  Cost estimates:")
        
        for device, prices in pricing.items():
            task_cost = workload['tasks'] * prices['per_task']
            shot_cost = workload['tasks'] * workload['shots_per_task'] * prices['per_shot']
            total_cost = task_cost + shot_cost
            
            print(f"    {device:8}: ${total_cost:8.2f} "
                  f"(${task_cost:.2f} tasks + ${shot_cost:.2f} shots)")

estimate_braket_costs()
```

### Cost Optimization

```python
def optimize_braket_usage():
    """Demonstrate cost optimization strategies."""
    
    print("Braket Cost Optimization Strategies")
    print("=" * 38)
    
    # Strategy 1: Use local simulator for development
    def development_workflow():
        """Use free local simulator for development."""
        local_backend = sqx.BraketBackend(
            device='local:braket/braket_sv',
            shots=10000  # No cost for shots
        )
        
        print("✅ Strategy 1: Use local simulator for development")
        print("   - Free execution")
        print("   - Fast iteration")
        print("   - Good for circuit testing")
        
        return local_backend
    
    # Strategy 2: Optimize shot counts
    def optimize_shots(circuit, target_accuracy=0.01):
        """Determine optimal shot count for accuracy."""
        # Rough estimate: std_error = 1/sqrt(shots)
        # For 1% accuracy: 1/sqrt(shots) = 0.01 -> shots = 10,000
        
        optimal_shots = int(1 / (target_accuracy ** 2))
        
        print(f"✅ Strategy 2: Optimize shot counts")
        print(f"   - Target accuracy: {target_accuracy:.1%}")
        print(f"   - Recommended shots: {optimal_shots:,}")
        
        return min(optimal_shots, 100000)  # Cap at 100k for cost control
    
    # Strategy 3: Batch multiple circuits
    def batch_circuits(circuits):
        """Batch multiple circuits to reduce task overhead."""
        print(f"✅ Strategy 3: Batch circuits")
        print(f"   - {len(circuits)} circuits in one task")
        print(f"   - Reduces task-based charges")
        print(f"   - Better resource utilization")
        
        # In practice, you would submit all circuits together
        return circuits
    
    # Strategy 4: Choose appropriate simulators
    def choose_simulator(n_qubits, circuit_depth):
        """Choose most cost-effective simulator."""
        
        recommendations = []
        
        if n_qubits <= 25 and circuit_depth <= 100:
            recommendations.append(("SV1", "Fast and cost-effective"))
        
        if n_qubits <= 50:
            recommendations.append(("TN1", "Handles larger circuits"))
        
        if n_qubits <= 17:  # Density matrix limit
            recommendations.append(("DM1", "Supports noise modeling"))
        
        print(f"✅ Strategy 4: Choose appropriate simulator")
        print(f"   Circuit: {n_qubits} qubits, depth {circuit_depth}")
        
        for sim, desc in recommendations:
            print(f"   - {sim}: {desc}")
        
        return recommendations[0][0] if recommendations else "Local"
    
    # Strategy 5: Monitor costs
    def setup_cost_monitoring():
        """Set up cost monitoring and alerts."""
        print(f"✅ Strategy 5: Cost monitoring")
        print(f"   - Use AWS Cost Explorer")
        print(f"   - Set up billing alerts")
        print(f"   - Tag resources for tracking")
        print(f"   - Regular cost reviews")
    
    # Example usage
    development_workflow()
    print()
    
    shots = optimize_shots(None, target_accuracy=0.02)  # 2% accuracy
    print()
    
    test_circuits = [f"circuit_{i}" for i in range(5)]
    batch_circuits(test_circuits)
    print()
    
    best_sim = choose_simulator(n_qubits=15, circuit_depth=50)
    print()
    
    setup_cost_monitoring()

optimize_braket_usage()
```

## Hybrid Algorithms

### Classical-Quantum Integration

```python
def braket_hybrid_algorithm_example(seed=0):
    """VQE written by hand on the Braket backend, with shot-based Pauli measurements."""
    import numpy as np
    from scipy.optimize import minimize

    backend = sqx.BraketBackend(device='local:braket/braket_sv', shots=2000)

    # H2 Hamiltonian (STO-3G, 0.74 Angstrom) as (coefficient, Pauli string)
    h2_hamiltonian = [
        (-1.0523732, 'II'), (0.39793742, 'ZI'), (-0.39793742, 'IZ'),
        (-0.01128010, 'ZZ'), (0.18093119, 'XX'),
    ]
    n_qubits = 2

    def create_ansatz(params):
        circuit = backend.create_circuit(n_qubits)
        for i in range(n_qubits):
            circuit = backend.add_gate(circuit, 'RY', i, [params[i]])
        circuit = backend.add_gate(circuit, 'CNOT', [0, 1])
        for i in range(n_qubits):
            circuit = backend.add_gate(circuit, 'RY', i, [params[n_qubits + i]])
        return circuit

    def pauli_expectation(params, pauli):
        """Rotate into the Pauli basis, sample, and average the parity."""
        circuit = create_ansatz(params)
        for i, p in enumerate(pauli):
            if p == 'X':
                circuit = backend.add_gate(circuit, 'H', i)
            elif p == 'Y':
                circuit = backend.add_gate(circuit, 'RX', i, [np.pi / 2])
        counts = backend.execute_circuit(circuit)['counts']   # qubit 0 is the leftmost bit
        total = sum(counts.values())
        return sum(
            c * (-1) ** sum(int(bits[i]) for i, p in enumerate(pauli) if p != 'I')
            for bits, c in counts.items()
        ) / total

    def energy(params):
        return sum(
            coeff if pauli == 'II' else coeff * pauli_expectation(params, pauli)
            for coeff, pauli in h2_hamiltonian
        )

    rng = np.random.default_rng(seed)
    result = minimize(energy, rng.uniform(0, np.pi, 2 * n_qubits),
                      method='COBYLA', options={'maxiter': 60})

    exact = sqx.Hamiltonian.from_dict({p: c for c, p in h2_hamiltonian}).ground_state_energy()
    print(f"VQE energy (shots): {result.fun:.4f}   exact: {exact:.4f}")

    # The built-in VQE does the same with exact expectation values
    vqe = sqx.VQE(hamiltonian=sqx.Hamiltonian.from_dict({p: c for c, p in h2_hamiltonian}),
                  backend=backend, seed=seed)
    print(f"sqx.VQE on Braket:  {vqe.find_ground_state():.4f}")


braket_hybrid_algorithm_example()
```

## Error Handling and Monitoring

### Robust Job Management

```python
def braket_job_management():
    """Demonstrate robust Braket job management."""
    
    import time
    from braket.aws import AwsDevice
    from braket.circuits import Circuit
    
    print("Braket Job Management")
    print("=" * 22)
    
    class RobustBraketExecutor:
        """Robust executor with error handling and retries."""
        
        def __init__(self, device_arn, s3_bucket, max_retries=3):
            self.device_arn = device_arn
            self.s3_bucket = s3_bucket
            self.max_retries = max_retries
            self.job_history = []
        
        def submit_with_retry(self, circuit, shots=1000, job_name=None):
            """Submit job with automatic retry logic."""
            
            for attempt in range(self.max_retries):
                try:
                    print(f"Attempt {attempt + 1}: Submitting job...")
                    
                    backend = sqx.BraketBackend(
                        device=self.device_arn,
                        s3_folder=(self.s3_bucket, f'attempt_{attempt}'),
                        shots=shots
                    )
                    
                    start_time = time.time()
                    result = backend.execute_circuit(circuit, shots=shots)
                    execution_time = time.time() - start_time
                    
                    # Record successful job
                    job_info = {
                        'attempt': attempt + 1,
                        'status': 'success',
                        'execution_time': execution_time,
                        'shots': shots,
                        'timestamp': time.time()
                    }
                    self.job_history.append(job_info)
                    
                    print(f"✅ Job completed successfully in {execution_time:.2f}s")
                    return result
                    
                except Exception as e:
                    error_info = {
                        'attempt': attempt + 1,
                        'status': 'failed',
                        'error': str(e),
                        'timestamp': time.time()
                    }
                    self.job_history.append(error_info)
                    
                    print(f"❌ Attempt {attempt + 1} failed: {e}")
                    
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        print(f"   Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        print(f"   Max retries exceeded. Job failed.")
                        raise e
        
        def get_job_statistics(self):
            """Get job execution statistics."""
            if not self.job_history:
                return {"message": "No jobs executed yet"}
            
            successful_jobs = [j for j in self.job_history if j['status'] == 'success']
            failed_jobs = [j for j in self.job_history if j['status'] == 'failed']
            
            stats = {
                'total_jobs': len(self.job_history),
                'successful_jobs': len(successful_jobs),
                'failed_jobs': len(failed_jobs),
                'success_rate': len(successful_jobs) / len(self.job_history) * 100,
            }
            
            if successful_jobs:
                execution_times = [j['execution_time'] for j in successful_jobs]
                stats['avg_execution_time'] = np.mean(execution_times)
                stats['total_execution_time'] = np.sum(execution_times)
            
            return stats
    
    # Example usage
    executor = RobustBraketExecutor(
        device_arn='local:braket/braket_sv',   # an ARN plus a real bucket for AWS
        s3_bucket='my-braket-experiments'
    )
    
    # Create test circuit
    test_circuit = Circuit()
    test_circuit.h(0).cnot(0, 1)
    
    # Submit job with retry logic
    try:
        result = executor.submit_with_retry(test_circuit, shots=1000)
        print(f"Job result: {result['counts']}")
    except Exception as e:
        print(f"Final job failure: {e}")
    
    # Get statistics
    stats = executor.get_job_statistics()
    print(f"\nJob Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

braket_job_management()
```

## Best Practices

### Security and Compliance

```python
def braket_security_best_practices():
    """Demonstrate Braket security best practices."""
    
    print("Braket Security Best Practices")
    print("=" * 32)
    
    practices = [
        {
            "category": "IAM and Access Control",
            "practices": [
                "Use least-privilege IAM policies",
                "Rotate access keys regularly", 
                "Use IAM roles for service-to-service access",
                "Enable CloudTrail for API logging",
                "Use VPC endpoints for private access"
            ]
        },
        {
            "category": "Data Protection",
            "practices": [
                "Encrypt S3 buckets with KMS",
                "Use secure S3 bucket policies",
                "Enable S3 versioning for data recovery",
                "Clean up intermediate results",
                "Avoid storing sensitive data in circuits"
            ]
        },
        {
            "category": "Cost Control",
            "practices": [
                "Set up billing alerts",
                "Use resource tags for cost tracking",
                "Implement circuit validation before submission",
                "Monitor queue times and costs",
                "Use simulators for development"
            ]
        },
        {
            "category": "Development Workflow",
            "practices": [
                "Test circuits locally first",
                "Use version control for quantum code",
                "Implement proper error handling",
                "Monitor job success rates",
                "Document quantum algorithms and parameters"
            ]
        }
    ]
    
    for category_info in practices:
        print(f"\n{category_info['category']}:")
        for i, practice in enumerate(category_info['practices'], 1):
            print(f"  {i}. {practice}")
    
    # Example secure configuration
    print(f"\nExample Secure Configuration:")
    print(f"```python")
    print(f"backend = sqx.BraketBackend(")
    print(f"    device='arn:aws:braket:::device/quantum-simulator/amazon/sv1',")
    print(f"    s3_folder=('my-secure-braket-bucket-{{region}}-{{account}}', 'quantum-experiments/{{project}}'),")
    print(f"    aws_session=session,  # a braket.aws.AwsSession built from a least-privilege role")
    print(f"    shots=1000,")
    print(f")")
    print(f"# Encryption (KMS) and tags are set on the S3 bucket and IAM side, not on the backend")
    print(f"```")

braket_security_best_practices()
```

### Performance Optimization

```python
def braket_performance_optimization():
    """Optimize performance for Braket workloads."""
    
    print("Braket Performance Optimization")
    print("=" * 33)
    
    # 1. Circuit optimization
    def optimize_circuit_for_braket(circuit):
        """Optimize circuit for Braket execution."""
        
        optimizations = [
            "Reduce circuit depth where possible",
            "Use native gate sets when available", 
            "Minimize measurements",
            "Batch similar circuits together",
            "Pre-validate circuits locally"
        ]
        
        print("Circuit Optimization:")
        for i, opt in enumerate(optimizations, 1):
            print(f"  {i}. {opt}")
        
        return circuit
    
    # 2. Device selection
    def select_optimal_device(circuit_qubits, circuit_depth, budget='medium'):
        """Select optimal device based on circuit characteristics."""
        
        recommendations = []
        
        if circuit_qubits <= 12 and budget == 'low':
            recommendations.append(("Local Simulator", "Free, fast for development"))
        
        if circuit_qubits <= 34:
            recommendations.append(("SV1", "Cost-effective, exact simulation"))
        
        if circuit_qubits <= 50:
            recommendations.append(("TN1", "Large-scale simulation"))
        
        if circuit_qubits <= 32 and budget == 'high':
            recommendations.append(("IonQ", "Real quantum hardware"))
        
        print(f"Device Selection for {circuit_qubits} qubits, depth {circuit_depth}:")
        for device, desc in recommendations:
            print(f"  ✅ {device}: {desc}")
        
        return recommendations[0][0] if recommendations else "Local Simulator"
    
    # 3. Execution strategies
    def execution_strategies():
        """Different execution strategies for various use cases."""
        
        strategies = {
            "Development": {
                "device": "Local Simulator",
                "shots": 10000,
                "validation": "Extensive local testing"
            },
            "Research": {
                "device": "SV1 or TN1",
                "shots": 5000,
                "validation": "Cloud simulation validation"
            },
            "Production": {
                "device": "Quantum Hardware",
                "shots": 1000,
                "validation": "Full pipeline testing"
            }
        }
        
        print("Execution Strategies:")
        for strategy, config in strategies.items():
            print(f"  {strategy}:")
            for key, value in config.items():
                print(f"    {key}: {value}")
            print()
    
    # Example optimization workflow
    optimize_circuit_for_braket(None)
    print()
    
    best_device = select_optimal_device(circuit_qubits=16, circuit_depth=20)
    print()
    
    execution_strategies()

braket_performance_optimization()
```

## Resources and Support

### Documentation and Learning

```python
def braket_resources():
    """Comprehensive Braket resources and support."""
    
    resources = {
        "Official Documentation": [
            "AWS Braket Developer Guide",
            "Braket Python SDK Documentation", 
            "Quantum Computing with Amazon Braket",
            "Device-specific documentation"
        ],
        "Learning Materials": [
            "Braket Tutorials and Examples",
            "Quantum Computing Fundamentals",
            "Hybrid Algorithm Patterns",
            "Cost optimization guides"
        ],
        "Community and Support": [
            "AWS Quantum Computing Blog",
            "Braket GitHub Repository",
            "AWS Support (for Enterprise customers)",
            "Quantum computing forums"
        ],
        "Hardware Partner Resources": [
            "IonQ Documentation and API",
            "Rigetti Documentation", 
            "Oxford Quantum Computing",
            "Hardware-specific calibration data"
        ]
    }
    
    print("Amazon Braket Resources")
    print("=" * 25)
    
    for category, items in resources.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")
    
    print(f"\nKey Links:")
    print(f"  📖 Docs: https://docs.aws.amazon.com/braket/")
    print(f"  🐍 SDK:  https://github.com/amazon-braket/amazon-braket-sdk-python")
    print(f"  💡 Examples: https://github.com/amazon-braket/amazon-braket-examples")
    print(f"  💰 Pricing: https://aws.amazon.com/braket/pricing/")

braket_resources()
```

---

*Amazon Braket's integration with SuperQuantX provides enterprise-grade quantum computing capabilities with cloud scalability, multi-vendor hardware access, and robust security features. It's ideal for organizations looking to integrate quantum computing into their existing AWS infrastructure.*

!!! tip "Cost Control"
    Always test circuits on local simulators first, then use cloud simulators, and finally quantum hardware. Use shot optimization and batching to minimize costs.

!!! warning "Hardware Queues"
    Quantum hardware devices may have significant queue times and limited availability. Plan accordingly and have fallback options.

!!! info "Enterprise Features"
    Braket provides enterprise features like VPC support, encryption, compliance certifications, and integration with other AWS services for production quantum computing workflows.