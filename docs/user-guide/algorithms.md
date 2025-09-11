# Quantum Algorithms

SuperQuantX provides a comprehensive library of quantum algorithms, from fundamental quantum computing primitives to cutting-edge quantum machine learning methods. This guide covers how to use, customize, and combine these algorithms in your research.

## 🎯 Algorithm Categories

SuperQuantX organizes quantum algorithms into several categories:

- **🔬 Fundamental Algorithms**: Basic quantum computing building blocks
- **🧠 Quantum Machine Learning**: ML algorithms leveraging quantum computation
- **⚡ Optimization Algorithms**: Quantum approaches to optimization problems
- **🔍 Search Algorithms**: Quantum search and database query methods
- **🔐 Cryptographic Algorithms**: Quantum cryptography and security
- **🧮 Simulation Algorithms**: Quantum system simulation methods

## 🔬 Fundamental Quantum Algorithms

### Quantum Fourier Transform (QFT)

The QFT is fundamental to many quantum algorithms:

```python
import superquantx as sqx
import numpy as np

# Create QFT algorithm
qft = sqx.QuantumFourierTransform(n_qubits=4, backend='pennylane')

# Apply to a quantum state
backend = sqx.get_backend('pennylane')
circuit = backend.create_circuit(4)

# Prepare initial state (can be any superposition)
circuit.h(0)
circuit.h(1)

# Apply QFT
qft_circuit = qft.create_circuit()
combined_circuit = circuit + qft_circuit

# Execute
result = backend.run(combined_circuit, shots=1000)
print(f"QFT result: {result.get_counts()}")

# Visualize the frequency domain
qft.plot_frequency_spectrum(result)
```

### Quantum Phase Estimation (QPE)

Estimate eigenvalues of unitary operators:

```python
# Create a unitary operator (e.g., rotation)
import numpy as np

def create_rotation_unitary(theta):
    """Create a rotation unitary matrix."""
    return np.array([
        [np.cos(theta/2), -1j*np.sin(theta/2)],
        [-1j*np.sin(theta/2), np.cos(theta/2)]
    ])

# Phase estimation algorithm
theta = np.pi/4  # True phase
unitary = create_rotation_unitary(theta)

qpe = sqx.QuantumPhaseEstimation(
    unitary_operator=unitary,
    precision_qubits=4,  # Number of qubits for precision
    backend='qiskit'
)

estimated_phase = qpe.estimate()
print(f"True phase: {theta:.6f}")
print(f"Estimated phase: {estimated_phase:.6f}")
print(f"Error: {abs(theta - estimated_phase):.6f}")
```

### Quantum Teleportation

Demonstrate quantum information transfer:

```python
# Quantum teleportation protocol
teleportation = sqx.QuantumTeleportation(backend='simulator')

# Prepare arbitrary quantum state to teleport
state_to_teleport = [0.6, 0.8j]  # α|0⟩ + β|1⟩

# Perform teleportation
teleported_state = teleportation.teleport_state(state_to_teleport)

print(f"Original state: {state_to_teleport}")
print(f"Teleported state: {teleported_state}")

# Verify fidelity
fidelity = teleportation.calculate_fidelity(state_to_teleport, teleported_state)
print(f"Teleportation fidelity: {fidelity:.6f}")
```

## 🧠 Quantum Machine Learning

### Quantum Support Vector Machine (QSVM)

Quantum-enhanced classification:

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create dataset
X, y = make_classification(
    n_samples=200,
    n_features=4,
    n_classes=2,
    n_redundant=0,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Quantum SVM with different feature maps
feature_maps = ['ZFeatureMap', 'ZZFeatureMap', 'PauliFeatureMap']
results = {}

for feature_map in feature_maps:
    qsvm = sqx.QuantumSVM(
        backend='pennylane',
        feature_map=feature_map,
        num_features=4,
        depth=2
    )
    
    # Train the model
    qsvm.fit(X_train, y_train)
    
    # Evaluate
    accuracy = qsvm.score(X_test, y_test)
    results[feature_map] = accuracy
    
    print(f"QSVM with {feature_map}: {accuracy:.3f}")

# Compare with classical SVM
from sklearn.svm import SVC
classical_svm = SVC(kernel='rbf')
classical_svm.fit(X_train, y_train)
classical_accuracy = classical_svm.score(X_test, y_test)

print(f"Classical SVM: {classical_accuracy:.3f}")
```

### Variational Quantum Classifier (VQC)

Trainable quantum classifier:

```python
# Variational Quantum Classifier
vqc = sqx.VariationalQuantumClassifier(
    n_qubits=4,
    n_layers=3,
    backend='pennylane',
    optimizer='adam'
)

# Configure training
vqc.compile(
    optimizer='adam',
    learning_rate=0.01,
    loss='binary_crossentropy'
)

# Train the classifier
history = vqc.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_test, y_test),
    verbose=1
)

# Plot training history
vqc.plot_training_history(history)

# Make predictions
predictions = vqc.predict(X_test)
probabilities = vqc.predict_proba(X_test)

print(f"VQC Accuracy: {vqc.score(X_test, y_test):.3f}")
```

### Quantum Neural Network (QNN)

Deep quantum neural networks:

```python
# Build quantum neural network
qnn = sqx.QuantumNeuralNetwork(
    architecture=[
        sqx.layers.QuantumConv2D(filters=4, kernel_size=2),
        sqx.layers.QuantumPooling2D(pool_size=2),
        sqx.layers.QuantumDense(units=16, activation='relu'),
        sqx.layers.QuantumDense(units=2, activation='softmax')
    ],
    backend='pennylane'
)

# Compile model
qnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train on quantum data
qnn.fit(X_train, y_train, epochs=50, batch_size=32)

# Evaluate
test_loss, test_accuracy = qnn.evaluate(X_test, y_test)
print(f"QNN Test Accuracy: {test_accuracy:.3f}")
```

### Quantum Generative Adversarial Network (QGAN)

Generate quantum data:

```python
# Quantum GAN for data generation
qgan = sqx.QuantumGAN(
    generator_layers=3,
    discriminator_layers=2,
    latent_dim=4,
    backend='pennylane'
)

# Train QGAN
qgan.fit(
    X_train,
    epochs=100,
    batch_size=16
)

# Generate new samples
generated_samples = qgan.generate(n_samples=100)

# Visualize real vs generated data
qgan.plot_data_comparison(X_train, generated_samples)
```

## ⚡ Optimization Algorithms

### Variational Quantum Eigensolver (VQE)

Find ground state energies:

```python
# Define Hamiltonian (e.g., H2 molecule)
def create_h2_hamiltonian(bond_distance=0.74):
    """Create H2 molecule Hamiltonian."""
    # Simplified H2 Hamiltonian coefficients
    coeffs = [
        -1.0523732,  # Identity
        0.39793742,  # Z0
        -0.39793742, # Z1
        -0.01128010, # Z0Z1
        0.18093119   # X0X1
    ]
    
    paulis = [
        sqx.PauliString('II'),
        sqx.PauliString('ZI'), 
        sqx.PauliString('IZ'),
        sqx.PauliString('ZZ'),
        sqx.PauliString('XX')
    ]
    
    return sqx.Hamiltonian(coeffs, paulis)

# Create VQE algorithm
h2_hamiltonian = create_h2_hamiltonian()

vqe = sqx.VQE(
    hamiltonian=h2_hamiltonian,
    ansatz='UCCSD',  # Unitary Coupled Cluster
    optimizer='COBYLA',
    backend='pennylane'
)

# Find ground state
ground_energy = vqe.find_ground_state()
ground_state = vqe.get_ground_state_vector()

print(f"Ground state energy: {ground_energy:.6f} Ha")

# Compare with exact diagonalization
exact_energy = h2_hamiltonian.ground_state_energy()
print(f"Exact energy: {exact_energy:.6f} Ha")
print(f"Error: {abs(ground_energy - exact_energy):.6f} Ha")
```

### Quantum Approximate Optimization Algorithm (QAOA)

Solve combinatorial optimization:

```python
# Max-Cut problem example
def create_maxcut_problem(graph_edges, weights=None):
    """Create Max-Cut problem instance."""
    if weights is None:
        weights = [1.0] * len(graph_edges)
    
    problem = sqx.MaxCutProblem(edges=graph_edges, weights=weights)
    return problem

# Define graph
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
maxcut_problem = create_maxcut_problem(edges)

# QAOA solver
qaoa = sqx.QAOA(
    problem=maxcut_problem,
    layers=3,  # p parameter
    optimizer='SPSA',
    backend='qiskit'
)

# Solve optimization problem
optimal_solution = qaoa.solve()
optimal_value = qaoa.get_optimal_value()

print(f"Optimal cut value: {optimal_value}")
print(f"Optimal solution: {optimal_solution}")

# Visualize optimization landscape
qaoa.plot_optimization_landscape()
```

### Quantum Annealing

Solve optimization via quantum annealing:

```python
# Quantum annealing for QUBO problems
qubo_matrix = np.array([
    [-1, 2, 2],
    [0, -1, 2], 
    [0, 0, -1]
])

annealer = sqx.QuantumAnnealer(
    backend='ocean',  # D-Wave Ocean backend
    annealing_time=20,  # microseconds
    num_reads=1000
)

# Solve QUBO
solution = annealer.solve_qubo(qubo_matrix)

print(f"Best solution: {solution.best_solution}")
print(f"Best energy: {solution.best_energy}")
print(f"Solution frequency: {solution.solution_counts}")
```

## 🔍 Search Algorithms

### Grover's Algorithm

Quantum database search:

```python
# Grover's algorithm for unstructured search
def oracle_function(x):
    """Oracle that marks target items."""
    # Mark items 3 and 7 as targets
    return x in [3, 7]

grover = sqx.GroversAlgorithm(
    search_space_size=16,  # 2^4 items
    oracle=oracle_function,
    backend='cirq'
)

# Run search
search_results = grover.search()

print(f"Target items found: {search_results.targets}")
print(f"Success probability: {search_results.success_probability:.3f}")
print(f"Number of iterations: {search_results.iterations}")

# Analyze amplitude amplification
grover.plot_amplitude_evolution()
```

### Quantum Walk Algorithm

Quantum random walk for graph traversal:

```python
# Quantum walk on a graph
import networkx as nx

# Create graph
G = nx.cycle_graph(8)  # 8-node cycle graph

quantum_walk = sqx.QuantumWalk(
    graph=G,
    steps=10,
    backend='simulator'
)

# Run quantum walk starting from node 0
probability_distribution = quantum_walk.run(start_node=0)

print(f"Final probability distribution: {probability_distribution}")

# Compare with classical random walk
classical_walk = sqx.ClassicalWalk(graph=G, steps=10)
classical_distribution = classical_walk.run(start_node=0)

# Plot comparison
quantum_walk.plot_distribution_comparison(
    probability_distribution, 
    classical_distribution
)
```

## 🔐 Cryptographic Algorithms

### Quantum Key Distribution (QKD)

Secure key exchange:

```python
# BB84 Quantum Key Distribution protocol
qkd = sqx.BB84Protocol(
    key_length=128,
    noise_level=0.05,  # 5% channel noise
    backend='simulator'
)

# Alice and Bob generate shared key
alice_key, bob_key, security_params = qkd.generate_shared_key()

print(f"Alice's key: {alice_key[:20]}...")  # Show first 20 bits
print(f"Bob's key: {bob_key[:20]}...")
print(f"Key agreement: {alice_key == bob_key}")
print(f"Security parameters: {security_params}")

# Detect eavesdropping
eve_present = qkd.detect_eavesdropping()
print(f"Eavesdropping detected: {eve_present}")
```

### Shor's Algorithm

Factor large integers:

```python
# Shor's algorithm for integer factorization
def shors_algorithm(N, backend='qiskit'):
    """Factor integer N using Shor's algorithm."""
    
    shor = sqx.ShorsAlgorithm(
        N=N,
        backend=backend,
        optimization_level=2
    )
    
    factors = shor.factor()
    return factors

# Example: Factor 15 (should give factors 3 and 5)
N = 15
factors = shors_algorithm(N)

print(f"Factors of {N}: {factors}")
print(f"Verification: {factors[0] * factors[1] == N}")

# For larger numbers (requires more qubits)
# N = 21
# factors = shors_algorithm(N, backend='simulator')
```

## 🧮 Simulation Algorithms

### Quantum System Simulation

Simulate quantum many-body systems:

```python
# Simulate Ising model
def create_ising_hamiltonian(n_spins, coupling_strength=1.0, field_strength=0.5):
    """Create transverse field Ising model Hamiltonian."""
    
    terms = []
    coefficients = []
    
    # Coupling terms: -J * Z_i * Z_{i+1}
    for i in range(n_spins - 1):
        pauli_string = ['I'] * n_spins
        pauli_string[i] = 'Z'
        pauli_string[i + 1] = 'Z'
        terms.append(sqx.PauliString(''.join(pauli_string)))
        coefficients.append(-coupling_strength)
    
    # Transverse field terms: -h * X_i
    for i in range(n_spins):
        pauli_string = ['I'] * n_spins
        pauli_string[i] = 'X'
        terms.append(sqx.PauliString(''.join(pauli_string)))
        coefficients.append(-field_strength)
    
    return sqx.Hamiltonian(coefficients, terms)

# Create and simulate Ising model
n_spins = 4
ising_hamiltonian = create_ising_hamiltonian(n_spins)

simulator = sqx.QuantumSimulator(
    hamiltonian=ising_hamiltonian,
    backend='pennylane'
)

# Time evolution
initial_state = sqx.create_plus_state(n_spins)  # All spins in |+⟩
evolution_times = np.linspace(0, 2*np.pi, 50)

evolved_states = simulator.time_evolution(
    initial_state=initial_state,
    times=evolution_times
)

# Analyze dynamics
magnetization = simulator.measure_observable(
    evolved_states,
    observable='magnetization'
)

simulator.plot_dynamics(evolution_times, magnetization)
```

### Variational Quantum Simulator (VQS)

Simulate ground states of complex systems:

```python
# Variational quantum simulation
vqs = sqx.VariationalQuantumSimulator(
    hamiltonian=ising_hamiltonian,
    ansatz='hardware_efficient',
    layers=6,
    backend='pennylane'
)

# Find ground state
ground_state_energy, ground_state_params = vqs.find_ground_state()

print(f"VQS ground state energy: {ground_state_energy:.6f}")

# Compare with exact diagonalization
exact_ground_energy = ising_hamiltonian.ground_state_energy()
print(f"Exact ground state energy: {exact_ground_energy:.6f}")
print(f"Error: {abs(ground_state_energy - exact_ground_energy):.6f}")

# Analyze excited states
excited_states = vqs.find_excited_states(num_states=3)
print(f"Excited state energies: {excited_states}")
```

## 🔧 Custom Algorithm Development

### Building Custom Algorithms

```python
from superquantx.algorithms import BaseQuantumAlgorithm

class MyQuantumAlgorithm(BaseQuantumAlgorithm):
    """Template for custom quantum algorithms."""
    
    def __init__(self, backend='simulator', **kwargs):
        super().__init__(backend=backend)
        self.parameters = kwargs
    
    def create_ansatz(self, params):
        """Create the quantum circuit ansatz."""
        circuit = self.backend.create_circuit(self.n_qubits)
        
        # Your custom circuit construction logic
        for i, param in enumerate(params):
            circuit.ry(param, i % self.n_qubits)
        
        # Add entangling gates
        for i in range(self.n_qubits - 1):
            circuit.cx(i, i + 1)
        
        return circuit
    
    def cost_function(self, params):
        """Define the cost function to optimize."""
        circuit = self.create_ansatz(params)
        
        # Add measurement
        circuit.measure_all()
        
        # Execute and compute cost
        result = self.backend.run(circuit, shots=1000)
        counts = result.get_counts()
        
        # Example cost: maximize |00...0⟩ state probability
        ground_state_prob = counts.get('0' * self.n_qubits, 0) / 1000
        return -ground_state_prob  # Minimize negative probability
    
    def optimize(self, initial_params=None):
        """Run the optimization."""
        if initial_params is None:
            initial_params = np.random.random(self.n_qubits) * 2 * np.pi
        
        from scipy.optimize import minimize
        
        result = minimize(
            self.cost_function,
            initial_params,
            method='COBYLA'
        )
        
        return result.x, result.fun

# Use custom algorithm
custom_algo = MyQuantumAlgorithm(n_qubits=4, backend='pennylane')
optimal_params, optimal_cost = custom_algo.optimize()

print(f"Optimal parameters: {optimal_params}")
print(f"Optimal cost: {optimal_cost}")
```

### Algorithm Composition

```python
# Combine multiple algorithms
class HybridQuantumAlgorithm:
    """Combine quantum algorithms for complex problems."""
    
    def __init__(self, backend='pennylane'):
        self.backend = backend
        self.vqe = sqx.VQE(backend=backend)
        self.qaoa = sqx.QAOA(backend=backend)
        self.grover = sqx.GroversAlgorithm(backend=backend)
    
    def solve_optimization_with_search(self, problem):
        """Use QAOA + Grover for optimization."""
        
        # First, get approximate solution with QAOA
        qaoa_solution = self.qaoa.solve(problem)
        
        # Use QAOA solution to define search space for Grover
        search_space = self.create_search_space(qaoa_solution)
        
        # Refine with Grover's algorithm
        refined_solution = self.grover.search(search_space)
        
        return refined_solution
    
    def quantum_ml_with_simulation(self, hamiltonian, training_data):
        """Combine VQE and ML for quantum system learning."""
        
        # Use VQE to understand ground state
        ground_energy = self.vqe.find_ground_state(hamiltonian)
        
        # Use ground state information in ML model
        qnn = sqx.QuantumNeuralNetwork(
            n_qubits=hamiltonian.n_qubits,
            backend=self.backend
        )
        
        # Initialize with VQE solution
        initial_params = self.vqe.optimal_parameters
        qnn.set_initial_parameters(initial_params)
        
        # Train ML model
        qnn.fit(training_data)
        
        return qnn

# Example usage
hybrid_algo = HybridQuantumAlgorithm(backend='pennylane')
# ml_model = hybrid_algo.quantum_ml_with_simulation(my_hamiltonian, training_data)
```

## 📊 Algorithm Performance Analysis

### Benchmarking Framework

```python
class QuantumAlgorithmBenchmark:
    """Benchmark quantum algorithms across different metrics."""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_algorithm(self, algorithm_class, problem_instances, backends):
        """Benchmark algorithm across problems and backends."""
        
        results = []
        
        for problem in problem_instances:
            for backend_name in backends:
                try:
                    # Initialize algorithm
                    algo = algorithm_class(backend=backend_name)
                    
                    # Measure performance
                    start_time = time.time()
                    solution = algo.solve(problem)
                    end_time = time.time()
                    
                    # Collect metrics
                    metrics = {
                        'backend': backend_name,
                        'problem_size': getattr(problem, 'size', 'unknown'),
                        'execution_time': end_time - start_time,
                        'solution_quality': self.evaluate_solution(solution, problem),
                        'convergence_steps': getattr(algo, 'num_iterations', None),
                        'success': True
                    }
                    
                except Exception as e:
                    metrics = {
                        'backend': backend_name,
                        'problem_size': getattr(problem, 'size', 'unknown'),
                        'error': str(e),
                        'success': False
                    }
                
                results.append(metrics)
        
        return results
    
    def create_benchmark_report(self, results):
        """Generate comprehensive benchmark report."""
        import pandas as pd
        
        df = pd.DataFrame(results)
        
        # Performance summary
        successful_results = df[df['success'] == True]
        
        report = {
            'success_rate': len(successful_results) / len(df),
            'average_execution_time': successful_results['execution_time'].mean(),
            'best_backend': successful_results.loc[
                successful_results['solution_quality'].idxmax(), 'backend'
            ],
            'fastest_backend': successful_results.loc[
                successful_results['execution_time'].idxmin(), 'backend'
            ]
        }
        
        return report, df

# Example benchmarking
benchmark = QuantumAlgorithmBenchmark()

# Create test problems
problems = [
    sqx.create_random_maxcut_problem(n_nodes=4),
    sqx.create_random_maxcut_problem(n_nodes=6),
    sqx.create_random_maxcut_problem(n_nodes=8)
]

backends = ['simulator', 'pennylane', 'qiskit']

# Run benchmark
results = benchmark.benchmark_algorithm(sqx.QAOA, problems, backends)
report, results_df = benchmark.create_benchmark_report(results)

print("📊 Benchmark Report:")
for key, value in report.items():
    print(f"{key}: {value}")
```

## 🚀 Advanced Algorithm Usage

### Parameter Optimization Strategies

```python
# Advanced parameter optimization for variational algorithms
class AdvancedOptimizer:
    """Advanced optimization strategies for quantum algorithms."""
    
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.optimization_history = []
    
    def adaptive_optimization(self, initial_params):
        """Adaptive optimization with multiple strategies."""
        
        strategies = [
            {'method': 'COBYLA', 'maxiter': 100},
            {'method': 'SPSA', 'maxiter': 200},
            {'method': 'L-BFGS-B', 'maxiter': 100}
        ]
        
        best_result = None
        best_cost = float('inf')
        
        for strategy in strategies:
            try:
                result = self.algorithm.optimize(
                    initial_params=initial_params,
                    **strategy
                )
                
                if result.fun < best_cost:
                    best_cost = result.fun
                    best_result = result
                    
                self.optimization_history.append({
                    'strategy': strategy['method'],
                    'cost': result.fun,
                    'success': result.success,
                    'iterations': result.nit
                })
                
            except Exception as e:
                print(f"Strategy {strategy['method']} failed: {e}")
        
        return best_result
    
    def plot_optimization_comparison(self):
        """Plot comparison of optimization strategies."""
        import matplotlib.pyplot as plt
        
        strategies = [h['strategy'] for h in self.optimization_history]
        costs = [h['cost'] for h in self.optimization_history]
        
        plt.figure(figsize=(10, 6))
        plt.bar(strategies, costs)
        plt.title('Optimization Strategy Comparison')
        plt.ylabel('Final Cost')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Example usage with VQE
vqe = sqx.VQE(hamiltonian=my_hamiltonian, backend='pennylane')
optimizer = AdvancedOptimizer(vqe)

best_params = optimizer.adaptive_optimization(
    initial_params=np.random.random(vqe.num_parameters)
)
optimizer.plot_optimization_comparison()
```

### Error Mitigation

```python
# Error mitigation techniques for NISQ algorithms
class ErrorMitigatedAlgorithm:
    """Wrapper to add error mitigation to quantum algorithms."""
    
    def __init__(self, base_algorithm, mitigation_methods=None):
        self.base_algorithm = base_algorithm
        if mitigation_methods is None:
            mitigation_methods = ['zero_noise_extrapolation', 'readout_correction']
        self.mitigation_methods = mitigation_methods
    
    def run_with_mitigation(self, circuit, shots=1000):
        """Run circuit with error mitigation."""
        
        results = {}
        
        # Raw result (no mitigation)
        raw_result = self.base_algorithm.backend.run(circuit, shots=shots)
        results['raw'] = raw_result.get_counts()
        
        # Apply mitigation methods
        for method in self.mitigation_methods:
            try:
                if method == 'zero_noise_extrapolation':
                    mitigated_result = self.zero_noise_extrapolation(circuit, shots)
                elif method == 'readout_correction':
                    mitigated_result = self.readout_correction(circuit, shots)
                
                results[method] = mitigated_result
                
            except Exception as e:
                print(f"Mitigation method {method} failed: {e}")
        
        return results
    
    def zero_noise_extrapolation(self, circuit, shots):
        """Zero-noise extrapolation error mitigation."""
        
        noise_factors = [1, 2, 3]  # Noise amplification factors
        results = []
        
        for factor in noise_factors:
            # Amplify noise by repeating gates
            noisy_circuit = self.amplify_noise(circuit, factor)
            result = self.base_algorithm.backend.run(noisy_circuit, shots=shots)
            results.append(result.get_counts())
        
        # Extrapolate to zero noise
        extrapolated_result = self.extrapolate_to_zero_noise(results, noise_factors)
        return extrapolated_result
    
    def readout_correction(self, circuit, shots):
        """Readout error correction."""
        
        # Calibrate readout errors
        calibration_matrix = self.calibrate_readout_errors()
        
        # Run original circuit
        raw_result = self.base_algorithm.backend.run(circuit, shots=shots)
        raw_counts = raw_result.get_counts()
        
        # Apply correction
        corrected_counts = self.apply_readout_correction(
            raw_counts, 
            calibration_matrix
        )
        
        return corrected_counts

# Example usage
base_vqe = sqx.VQE(hamiltonian=my_hamiltonian, backend='qiskit')
mitigated_vqe = ErrorMitigatedAlgorithm(base_vqe)

# Run with error mitigation
results = mitigated_vqe.run_with_mitigation(vqe_circuit)
print("Results with error mitigation:", results)
```

## 📚 Algorithm Selection Guide

### Decision Tree

```python
def recommend_algorithm(problem_type, problem_size, available_backends, objectives):
    """Recommend best algorithm based on problem characteristics."""
    
    recommendations = []
    
    if problem_type == 'classification':
        if problem_size < 1000:
            recommendations.append(('QuantumSVM', 'pennylane', 'Good for small datasets'))
        else:
            recommendations.append(('VariationalQuantumClassifier', 'pennylane', 'Scalable approach'))
    
    elif problem_type == 'optimization':
        if problem_size < 50:
            recommendations.append(('QAOA', 'qiskit', 'Combinatorial optimization'))
        else:
            recommendations.append(('QuantumAnnealing', 'ocean', 'Large-scale problems'))
    
    elif problem_type == 'simulation':
        if 'quantum_system' in objectives:
            recommendations.append(('VQE', 'pennylane', 'Ground state finding'))
        if 'dynamics' in objectives:
            recommendations.append(('QuantumSimulator', 'cirq', 'Time evolution'))
    
    elif problem_type == 'search':
        if problem_size <= 2**16:
            recommendations.append(('GroversAlgorithm', 'simulator', 'Unstructured search'))
        else:
            recommendations.append(('QuantumWalk', 'cirq', 'Large search spaces'))
    
    # Filter by available backends
    filtered_recommendations = [
        (algo, backend, desc) for algo, backend, desc in recommendations
        if backend in available_backends
    ]
    
    return filtered_recommendations

# Example usage
available_backends = sqx.list_backends()
recommendations = recommend_algorithm(
    problem_type='optimization',
    problem_size=20,
    available_backends=available_backends,
    objectives=['combinatorial']
)

print("🎯 Algorithm Recommendations:")
for algo, backend, description in recommendations:
    print(f"  {algo} on {backend}: {description}")
```

## 📞 Getting Help

For algorithm-specific help:

- **[Tutorials](../tutorials/quantum-ml.md)**: Step-by-step algorithm tutorials
- **[API Reference](../api/algorithms.md)**: Detailed function documentation
- **[Examples](../tutorials/advanced-examples.md)**: Real-world algorithm applications
- **[GitHub Issues](https://github.com/SuperagenticAI/superquantx/issues)**: Report algorithm bugs

---

!!! tip "Algorithm Selection"
    Start with fundamental algorithms to understand quantum computing principles, then move to specialized ML or optimization algorithms based on your research needs.

!!! warning "NISQ Limitations"
    Current quantum hardware has limitations (noise, limited qubits, short coherence times). Consider these when selecting algorithms for real hardware execution.

!!! info "Performance Optimization"
    For best performance, choose backends that are optimized for your specific algorithm type (e.g., PennyLane for ML, Qiskit for IBM hardware access).