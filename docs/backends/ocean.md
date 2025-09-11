# D-Wave Ocean Integration

D-Wave Ocean is a suite of tools for solving optimization problems on D-Wave quantum annealing systems. Unlike gate-model quantum computers, D-Wave systems are quantum annealers designed specifically for solving optimization problems formulated as Quadratic Unconstrained Binary Optimization (QUBO) or Ising models.

## Overview

The Ocean backend in SuperQuantX provides:

- **Quantum Annealing**: Access to D-Wave's quantum annealing processors
- **QUBO/Ising Solving**: Direct support for optimization problem formulations
- **Hybrid Algorithms**: Classical-quantum hybrid problem solving
- **Simulated Annealing**: Local testing without quantum hardware
- **Optimization Problems**: Built-in solvers for common combinatorial problems
- **Embedding**: Automatic minor-embedding for hardware constraints

!!! note "Quantum Annealing vs Gate Model"
    D-Wave systems are quantum annealers, not gate-model quantum computers. They excel at optimization problems but don't run traditional quantum circuits.

## Installation

### Basic Ocean Installation

```bash
# Install D-Wave Ocean SDK
pip install dwave-ocean-sdk

# Install SuperQuantX with Ocean support
pip install superquantx[ocean]
```

### Extended Ocean Tools

```bash
# Additional D-Wave tools
pip install dwave-inspector      # Problem visualization
pip install dwave-preprocessing  # Problem preprocessing
pip install dwave-hybrid         # Hybrid algorithms
pip install minorminer          # Advanced embedding

# For graph problems
pip install networkx
```

### Verify Installation

```python
import superquantx as sqx

# Check Ocean availability
backend = sqx.get_backend('ocean')
print(backend.get_version_info())
```

## Quick Start

### Basic QUBO Problem

```python
import superquantx as sqx

# Initialize Ocean backend (simulator)
backend = sqx.get_backend('ocean', device='simulator')

# Define a simple QUBO problem
# Minimize: x₀ - 2x₁ + x₀x₁
Q = {
    (0, 0): 1,    # x₀ term
    (1, 1): -2,   # x₁ term  
    (0, 1): 1     # x₀x₁ term
}

# Solve the problem
result = backend.solve_qubo(Q, num_reads=1000)

print("Best solution:")
print("Variables:", result['samples'][0])
print("Energy:", result['energies'][0])
print("Occurrences:", result['num_occurrences'][0])
```

### Basic Ising Model

```python
# Define Ising model: H = ∑h_i*s_i + ∑J_ij*s_i*s_j
h = {0: -1, 1: -1}  # Linear terms (bias)
J = {(0, 1): -1}    # Quadratic terms (coupling)

result = backend.solve_ising(h, J, num_reads=1000)

print("Ising solution:")
for i, sample in enumerate(result['samples'][:3]):
    print(f"Solution {i+1}: {sample}, Energy: {result['energies'][i]}")
```

## Configuration

### Backend Initialization Options

```python
# Local simulator (no hardware required)
backend = sqx.OceanBackend(device='simulator', shots=1000)

# D-Wave Advantage hardware  
backend = sqx.OceanBackend(
    device='advantage',
    token='your-dwave-leap-token',
    shots=100  # Called "num_reads" in D-Wave
)

# D-Wave Hybrid solver (classical-quantum hybrid)
backend = sqx.OceanBackend(
    device='hybrid',
    token='your-token',
    shots=1
)

# Specific solver selection
backend = sqx.OceanBackend(
    device='advantage',
    solver='Advantage_system4.1',
    token='your-token'
)
```

### Environment Configuration

```bash
# Set up D-Wave Leap credentials
export DWAVE_API_TOKEN="your-api-token-here"

# Configure default endpoint
export DWAVE_API_ENDPOINT="https://cloud.dwavesys.com/sapi"

# Set solver preferences
export DWAVE_DEFAULT_SOLVER="Advantage_system4.1"
```

### Configuration File

```python
# ~/.dwave/config
[defaults]
endpoint = https://cloud.dwavesys.com/sapi
token = your-api-token-here
solver = Advantage_system4.1
```

## Hardware Integration

### D-Wave Quantum Annealers

D-Wave offers several quantum annealing systems:

```python
# Advantage Systems (5000+ qubits)
backend_advantage = sqx.OceanBackend(
    device='advantage',
    solver='Advantage_system4.1',  # or Advantage_system6.1
    token='your-token'
)

# Check available solvers
from dwave.system import DWaveSampler
sampler = DWaveSampler(token='your-token')
available_solvers = sampler.client.get_solvers()

for solver in available_solvers:
    print(f"Solver: {solver['id']}")
    print(f"Qubits: {len(solver['properties']['qubits'])}")
    print(f"Topology: {solver['properties']['topology']['type']}")
    print("---")
```

### Hardware Topologies

D-Wave systems use specific qubit connectivity patterns:

```python
def analyze_hardware_topology(backend):
    """Analyze D-Wave hardware topology."""
    
    if hasattr(backend._sampler, 'properties'):
        props = backend._sampler.properties
        
        # Get qubit connectivity
        qubits = props['qubits']
        couplers = props['couplers']
        
        print(f"Active qubits: {len(qubits)}")
        print(f"Active couplers: {len(couplers)}")
        
        # Analyze topology
        if 'topology' in props:
            topology = props['topology']
            print(f"Topology: {topology['type']}")
            if topology['type'] == 'pegasus':
                print(f"Pegasus P{topology['shape'][0]} topology")
            elif topology['type'] == 'zephyr':
                print(f"Zephyr Z{topology['shape'][0]} topology")
    
    return props

# Analyze current hardware
if backend.device_name != 'simulator':
    topology_info = analyze_hardware_topology(backend)
```

### Cost Management for Hardware

```python
def cost_aware_solving(backend, problem, budget_reads=1000):
    """Solve problem with cost awareness."""
    
    # Start with small sample for testing
    test_reads = 10
    test_result = backend.solve_qubo(problem, num_reads=test_reads)
    
    if not test_result['success']:
        print("Test run failed, aborting")
        return test_result
    
    # Analyze solution quality
    energies = test_result['energies']
    best_energy = min(energies)
    energy_spread = max(energies) - min(energies)
    
    print(f"Test run: Best energy = {best_energy}, Spread = {energy_spread}")
    
    # Determine optimal read count based on test
    if energy_spread < 0.1:
        # Problem seems easy, fewer reads needed
        optimal_reads = min(100, budget_reads)
    else:
        # Problem seems hard, use more reads
        optimal_reads = budget_reads
    
    print(f"Running {optimal_reads} reads on hardware")
    final_result = backend.solve_qubo(problem, num_reads=optimal_reads)
    
    return final_result
```

## Problem Formulations

### QUBO Problems

QUBO (Quadratic Unconstrained Binary Optimization) problems have the form:
minimize x^T Q x where x ∈ {0,1}^n

```python
def formulate_number_partitioning(numbers):
    """Formulate number partitioning as QUBO."""
    n = len(numbers)
    S = sum(numbers)
    
    # QUBO formulation for minimizing |∑x_i*numbers[i] - S/2|²
    Q = {}
    
    for i in range(n):
        for j in range(n):
            if i == j:
                Q[(i, i)] = (numbers[i] - S/2) ** 2
            else:
                Q[(i, j)] = Q.get((i, j), 0) + numbers[i] * numbers[j]
    
    return Q

# Example: Partition {3, 1, 1, 2, 2, 1}
numbers = [3, 1, 1, 2, 2, 1]
Q = formulate_number_partitioning(numbers)
result = backend.solve_qubo(Q)

print("Partition result:")
partition_0 = [numbers[i] for i, x in result['samples'][0].items() if x == 0]
partition_1 = [numbers[i] for i, x in result['samples'][0].items() if x == 1]
print(f"Set 0: {partition_0} (sum: {sum(partition_0)})")
print(f"Set 1: {partition_1} (sum: {sum(partition_1)})")
```

### Ising Model Problems

Ising models have the form: E = ∑h_i*s_i + ∑J_ij*s_i*s_j where s_i ∈ {-1,+1}

```python
def create_frustrated_ising_model(size=3):
    """Create a frustrated triangular Ising model."""
    h = {}  # No external fields
    J = {}  # Antiferromagnetic interactions
    
    # Triangle with antiferromagnetic couplings
    for i in range(size):
        for j in range(i+1, size):
            J[(i, j)] = 1  # Positive coupling = antiferromagnetic
    
    return h, J

# Solve frustrated system
h, J = create_frustrated_ising_model(4)
result = backend.solve_ising(h, J)

print("Frustrated Ising solutions:")
for i, sample in enumerate(result['samples'][:5]):
    spins = [sample[j] for j in sorted(sample.keys())]
    print(f"Solution {i+1}: {spins}, Energy: {result['energies'][i]}")
```

### Constraint Handling

```python
def add_equality_constraint(Q, variables, target_sum, penalty=10.0):
    """Add equality constraint to QUBO: ∑x_i = target_sum."""
    
    # Constraint: (∑x_i - target_sum)² = penalty
    n = len(variables)
    
    # Expand (∑x_i - target_sum)²
    for i in variables:
        Q[(i, i)] = Q.get((i, i), 0) + penalty
        
    for i in variables:
        for j in variables:
            if i != j:
                Q[(i, j)] = Q.get((i, j), 0) + penalty
    
    # Linear terms from -2*target_sum*∑x_i
    for i in variables:
        Q[(i, i)] = Q.get((i, i), 0) - 2 * penalty * target_sum
    
    # Constant term (target_sum)² is not needed in QUBO dict
    
    return Q

# Example: Constrained knapsack problem
def constrained_knapsack(values, weights, capacity):
    """Knapsack with capacity constraint."""
    n = len(values)
    
    # Maximize value (minimize negative value)
    Q = {}
    for i in range(n):
        Q[(i, i)] = -values[i]
    
    # Add capacity constraint with penalty
    penalty = max(values) * 2  # Strong penalty
    
    # Constraint: ∑weights[i]*x_i ≤ capacity
    # Convert to equality by adding slack variables
    # This is a simplified version - full implementation needs slack variables
    
    for i in range(n):
        Q[(i, i)] = Q.get((i, i), 0) + penalty * weights[i] ** 2
        
    for i in range(n):
        for j in range(i+1, n):
            Q[(i, j)] = penalty * weights[i] * weights[j]
    
    return Q

# Solve constrained knapsack
values = [10, 40, 30, 50]
weights = [5, 4, 6, 3]
capacity = 10

Q = constrained_knapsack(values, weights, capacity)
result = backend.solve_qubo(Q)
print("Knapsack solution:", result['samples'][0])
```

## Advanced Features

### Hybrid Quantum-Classical Algorithms

```python
def hybrid_decomposition(backend, large_problem, subproblem_size=100):
    """Decompose large problem for hybrid solving."""
    
    if backend.device_name != 'hybrid':
        print("Switching to hybrid sampler for large problem")
        backend = sqx.OceanBackend(device='hybrid', token=backend.token)
    
    # For very large problems, use D-Wave's hybrid solvers
    result = backend.solve_qubo(large_problem, time_limit=60)
    
    return result

# Example: Large random QUBO
def generate_random_qubo(n_vars, density=0.1):
    """Generate random QUBO problem."""
    import random
    
    Q = {}
    num_terms = int(n_vars * (n_vars + 1) * density / 2)
    
    for _ in range(num_terms):
        i = random.randint(0, n_vars - 1)
        j = random.randint(i, n_vars - 1)
        Q[(i, j)] = random.uniform(-1, 1)
    
    return Q

# Solve large problem with hybrid approach
large_Q = generate_random_qubo(1000, density=0.05)
hybrid_result = hybrid_decomposition(backend, large_Q)
```

### Problem Preprocessing

```python
from dwave.preprocessing import ScaleComposite, FixVariablesComposite

def preprocess_problem(backend, Q, preprocessing_steps=None):
    """Apply preprocessing to improve problem solving."""
    
    preprocessing_steps = preprocessing_steps or ['scale', 'roof_dual']
    
    # Wrap sampler with preprocessing
    sampler = backend._sampler
    
    if 'scale' in preprocessing_steps:
        # Scale problem for better numerical properties
        sampler = ScaleComposite(sampler)
    
    if 'fix_variables' in preprocessing_steps:
        # Fix variables using preprocessing heuristics  
        sampler = FixVariablesComposite(sampler)
    
    # Create BQM and solve
    import dimod
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    sampleset = sampler.sample(bqm, num_reads=1000)
    
    return backend._convert_sampleset(sampleset, 'QUBO')

# Example with preprocessing
Q = generate_random_qubo(50, density=0.2)
preprocessed_result = preprocess_problem(backend, Q)
```

### Minor Embedding

```python
def analyze_embedding(backend, problem):
    """Analyze embedding quality for D-Wave hardware."""
    
    if backend.device_name == 'simulator':
        print("Embedding not relevant for simulator")
        return
    
    import dimod
    from minorminer import find_embedding
    
    # Get hardware topology
    if hasattr(backend._sampler, 'target_graph'):
        target_graph = backend._sampler.target_graph
    else:
        print("Cannot access target graph")
        return
    
    # Create problem graph
    bqm = dimod.BinaryQuadraticModel.from_qubo(problem)
    source_graph = bqm.binary.to_networkx_graph()
    
    # Find embedding
    embedding = find_embedding(source_graph, target_graph)
    
    if embedding:
        # Analyze embedding quality
        chain_lengths = [len(chain) for chain in embedding.values()]
        avg_chain_length = sum(chain_lengths) / len(chain_lengths)
        max_chain_length = max(chain_lengths)
        
        print(f"Embedding found:")
        print(f"  Average chain length: {avg_chain_length:.2f}")
        print(f"  Maximum chain length: {max_chain_length}")
        print(f"  Total physical qubits used: {sum(chain_lengths)}")
        
        return embedding
    else:
        print("No embedding found - problem may be too large or dense")
        return None

# Test embedding for medium problem
medium_Q = generate_random_qubo(20, density=0.3)
embedding = analyze_embedding(backend, medium_Q)
```

## Common Optimization Problems

### Maximum Cut Problem

```python
def solve_max_cut_problem(backend, graph_edges, visualize=False):
    """Solve Maximum Cut problem."""
    
    # Use built-in max cut solver
    result = backend.solve_max_cut(graph_edges)
    
    if result['success']:
        best_cut = result['samples'][0]
        
        # Calculate cut value
        cut_value = 0
        for u, v in graph_edges:
            if best_cut.get(u, 0) != best_cut.get(v, 0):
                cut_value += 1
        
        print(f"Maximum cut value: {cut_value} out of {len(graph_edges)} edges")
        
        # Show partition
        set_0 = [node for node, val in best_cut.items() if val == 0]
        set_1 = [node for node, val in best_cut.items() if val == 1]
        
        print(f"Partition 0: {set_0}")
        print(f"Partition 1: {set_1}")
        
        if visualize:
            try:
                import matplotlib.pyplot as plt
                import networkx as nx
                
                # Create and visualize graph
                G = nx.Graph(graph_edges)
                pos = nx.spring_layout(G)
                
                # Color nodes by partition
                colors = ['red' if best_cut.get(node, 0) == 0 else 'blue' 
                         for node in G.nodes()]
                
                plt.figure(figsize=(10, 6))
                nx.draw(G, pos, node_color=colors, with_labels=True)
                plt.title(f"Maximum Cut Solution (Cut Value: {cut_value})")
                plt.show()
                
            except ImportError:
                print("Visualization requires matplotlib and networkx")
    
    return result

# Example: Solve Max Cut on a small graph
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
max_cut_result = solve_max_cut_problem(backend, edges, visualize=True)
```

### Traveling Salesman Problem

```python
def solve_tsp_simplified(backend, cities, distance_func=None):
    """Solve simplified TSP formulation."""
    
    n = len(cities)
    
    # Calculate distance matrix
    if distance_func is None:
        # Euclidean distance for 2D coordinates
        distance_matrix = np.zeros((n, n))
        for i, city_i in enumerate(cities):
            for j, city_j in enumerate(cities):
                if i != j:
                    dx = city_i[0] - city_j[0]
                    dy = city_i[1] - city_j[1]
                    distance_matrix[i, j] = np.sqrt(dx*dx + dy*dy)
    else:
        distance_matrix = np.array([[distance_func(cities[i], cities[j]) 
                                   for j in range(n)] for i in range(n)])
    
    # Use built-in TSP solver (simplified)
    result = backend.solve_tsp(distance_matrix)
    
    if result['success']:
        # Interpret solution (this is problem-specific)
        print("TSP solution found (simplified formulation)")
        print("Note: Full TSP requires more sophisticated constraint handling")
    
    return result

# Example cities (2D coordinates)
cities = [(0, 0), (1, 2), (3, 1), (2, 3), (4, 0)]
tsp_result = solve_tsp_simplified(backend, cities)
```

### Graph Coloring

```python
def solve_graph_coloring(backend, edges, num_colors=3):
    """Solve graph coloring problem as QUBO."""
    
    # Determine number of nodes
    nodes = set()
    for u, v in edges:
        nodes.add(u)
        nodes.add(v)
    nodes = sorted(list(nodes))
    n_nodes = len(nodes)
    
    # Variables: x_{i,c} = 1 if node i has color c
    # Total variables: n_nodes * num_colors
    
    Q = {}
    penalty = 10  # Constraint penalty
    
    # Constraint 1: Each node must have exactly one color
    for node in nodes:
        for c1 in range(num_colors):
            for c2 in range(num_colors):
                var1 = node * num_colors + c1
                var2 = node * num_colors + c2
                
                if c1 == c2:
                    Q[(var1, var1)] = Q.get((var1, var1), 0) + penalty
                else:
                    Q[(var1, var2)] = Q.get((var1, var2), 0) + penalty
                    
        # Linear term from (∑x_{i,c} - 1)²
        for c in range(num_colors):
            var = node * num_colors + c
            Q[(var, var)] = Q.get((var, var), 0) - 2 * penalty
    
    # Constraint 2: Adjacent nodes cannot have the same color
    for u, v in edges:
        for c in range(num_colors):
            var_u = u * num_colors + c
            var_v = v * num_colors + c
            Q[(var_u, var_v)] = Q.get((var_u, var_v), 0) + penalty
    
    # Solve QUBO
    result = backend.solve_qubo(Q)
    
    if result['success']:
        # Decode solution
        best_sample = result['samples'][0]
        coloring = {}
        
        for node in nodes:
            for c in range(num_colors):
                var = node * num_colors + c
                if best_sample.get(var, 0) == 1:
                    coloring[node] = c
                    break
        
        print(f"Graph coloring with {num_colors} colors:")
        for node, color in coloring.items():
            print(f"Node {node}: Color {color}")
        
        # Verify solution
        valid = True
        for u, v in edges:
            if coloring.get(u) == coloring.get(v):
                print(f"Constraint violation: nodes {u} and {v} have same color")
                valid = False
        
        if valid:
            print("Valid coloring found!")
    
    return result

# Example: Color a small graph
edges = [(0, 1), (1, 2), (2, 0), (1, 3), (2, 3)]
coloring_result = solve_graph_coloring(backend, edges, num_colors=3)
```

## Performance Optimization

### Annealing Parameters

```python
def optimize_annealing_schedule(backend, problem, test_schedules=None):
    """Optimize annealing schedule for better results."""
    
    if backend.device_name == 'simulator':
        print("Annealing schedule optimization not applicable for simulator")
        return
    
    test_schedules = test_schedules or [
        [[0, 1], [0.5, 0.5], [1, 0]],        # Linear schedule
        [[0, 1], [0.2, 0.8], [0.8, 0.2], [1, 0]],  # Pause schedule
        [[0, 1], [0.1, 0.9], [0.9, 0.1], [1, 0]]   # Fast start
    ]
    
    results = []
    
    for i, schedule in enumerate(test_schedules):
        print(f"Testing schedule {i+1}: {schedule}")
        
        # Run with custom annealing schedule
        result = backend.solve_qubo(
            problem,
            anneal_schedule=schedule,
            num_reads=100
        )
        
        if result['success']:
            best_energy = min(result['energies'])
            avg_energy = np.mean(result['energies'])
            results.append({
                'schedule': schedule,
                'best_energy': best_energy,
                'avg_energy': avg_energy
            })
    
    # Find best schedule
    if results:
        best_schedule = min(results, key=lambda x: x['best_energy'])
        print(f"Best schedule: {best_schedule['schedule']}")
        print(f"Best energy: {best_schedule['best_energy']}")
        return best_schedule['schedule']
    
    return None
```

### Chain Strength Optimization

```python
def optimize_chain_strength(backend, problem, strength_range=(0.1, 2.0), steps=10):
    """Optimize chain strength for embedding."""
    
    if backend.device_name == 'simulator':
        print("Chain strength not applicable for simulator")
        return 1.0
    
    strengths = np.linspace(strength_range[0], strength_range[1], steps)
    results = []
    
    for strength in strengths:
        print(f"Testing chain strength: {strength:.2f}")
        
        result = backend.solve_qubo(
            problem,
            chain_strength=strength,
            num_reads=100
        )
        
        if result['success']:
            best_energy = min(result['energies'])
            # Check for broken chains (simplified)
            num_solutions = len(set(str(s) for s in result['samples']))
            
            results.append({
                'strength': strength,
                'best_energy': best_energy,
                'diversity': num_solutions
            })
    
    if results:
        # Balance between energy and solution diversity
        best_result = min(results, key=lambda x: x['best_energy'] - 0.1 * x['diversity'])
        optimal_strength = best_result['strength']
        print(f"Optimal chain strength: {optimal_strength:.2f}")
        return optimal_strength
    
    return 1.0
```

### Parallel Problem Solving

```python
import concurrent.futures

def solve_problems_parallel(backend, problems, max_workers=4):
    """Solve multiple problems in parallel."""
    
    def solve_single_problem(prob_id, problem):
        print(f"Solving problem {prob_id}")
        result = backend.solve_qubo(problem)
        result['problem_id'] = prob_id
        return result
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(solve_single_problem, i, prob) 
                  for i, prob in enumerate(problems)]
        
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Problem solving failed: {e}")
    
    return results

# Example: Solve multiple random problems
problems = [generate_random_qubo(20, density=0.1) for _ in range(5)]
parallel_results = solve_problems_parallel(backend, problems)

for result in parallel_results:
    if result['success']:
        print(f"Problem {result['problem_id']}: Best energy = {min(result['energies'])}")
```

## Error Handling and Debugging

### Comprehensive Error Management

```python
import logging

def robust_ocean_solving(backend, problem, problem_type='qubo', max_retries=3):
    """Robust problem solving with error handling."""
    
    logger = logging.getLogger('ocean_backend')
    
    for attempt in range(max_retries):
        try:
            # Validate problem size
            if problem_type == 'qubo':
                num_vars = len(set(var[0] for var in problem.keys()) | 
                              set(var[1] for var in problem.keys()))
            else:
                num_vars = len(problem[0]) + len(problem[1])  # h, J
            
            max_vars = backend._get_max_variables()
            if num_vars > max_vars:
                logger.error(f"Problem has {num_vars} variables, backend supports {max_vars}")
                return {'success': False, 'error': 'Problem too large'}
            
            # Attempt solving
            if problem_type == 'qubo':
                result = backend.solve_qubo(problem)
            elif problem_type == 'ising':
                h, J = problem
                result = backend.solve_ising(h, J)
            else:
                result = backend.solve_optimization_problem(problem, problem_type)
            
            if result['success']:
                logger.info(f"Problem solved successfully on attempt {attempt + 1}")
                return result
            else:
                logger.warning(f"Solving failed on attempt {attempt + 1}: {result.get('error', 'Unknown')}")
        
        except Exception as e:
            logger.error(f"Exception on attempt {attempt + 1}: {e}")
            
            if 'timeout' in str(e).lower():
                # Increase timeout for retry
                logger.info("Increasing timeout for retry")
            elif 'token' in str(e).lower():
                logger.error("Authentication error - check D-Wave token")
                break
    
    return {'success': False, 'error': 'All retry attempts failed'}

# Test robust solving
test_problem = generate_random_qubo(30, density=0.15)
robust_result = robust_ocean_solving(backend, test_problem)
```

### Problem Validation

```python
def validate_ocean_problem(problem, problem_type='qubo'):
    """Validate problem formulation before solving."""
    
    validation = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    if problem_type == 'qubo':
        # Validate QUBO format
        if not isinstance(problem, dict):
            validation['errors'].append("QUBO must be a dictionary")
            validation['valid'] = False
            return validation
        
        # Check key format
        for key in problem.keys():
            if not isinstance(key, tuple) or len(key) != 2:
                validation['errors'].append(f"Invalid QUBO key format: {key}")
                validation['valid'] = False
            elif not all(isinstance(i, int) for i in key):
                validation['errors'].append(f"QUBO keys must be integer pairs: {key}")
                validation['valid'] = False
        
        # Check for diagonal dominance (good for annealing)
        diagonal_terms = {i: problem.get((i, i), 0) for i in 
                         set(k[0] for k in problem.keys()) | set(k[1] for k in problem.keys())}
        
        off_diagonal_sum = {}
        for (i, j), val in problem.items():
            if i != j:
                off_diagonal_sum[i] = off_diagonal_sum.get(i, 0) + abs(val)
                off_diagonal_sum[j] = off_diagonal_sum.get(j, 0) + abs(val)
        
        for var in diagonal_terms:
            diagonal = abs(diagonal_terms[var])
            off_diag = off_diagonal_sum.get(var, 0)
            
            if diagonal < off_diag:
                validation['warnings'].append(
                    f"Variable {var}: weak diagonal dominance may affect convergence"
                )
    
    elif problem_type == 'ising':
        h, J = problem
        
        # Validate Ising format
        if not isinstance(h, dict) or not isinstance(J, dict):
            validation['errors'].append("Ising model requires h (dict) and J (dict)")
            validation['valid'] = False
        
        # Check variable consistency
        h_vars = set(h.keys())
        j_vars = set()
        for (i, j) in J.keys():
            j_vars.add(i)
            j_vars.add(j)
        
        if h_vars != j_vars:
            validation['warnings'].append("Inconsistent variables between h and J")
    
    return validation

# Validate problems before solving
qubo_validation = validate_ocean_problem(test_problem, 'qubo')
print("QUBO validation:", qubo_validation)

if qubo_validation['valid']:
    result = backend.solve_qubo(test_problem)
else:
    print("Problem validation failed, not solving")
```

## Best Practices

### Problem Design

1. **Start Small**: Test with small problems before scaling up
2. **Use Constraints Carefully**: High penalty terms can dominate the objective
3. **Balance Problem Terms**: Avoid extreme coefficient ranges
4. **Consider Problem Structure**: Sparse problems embed better

### Annealing Strategy

1. **Tune Chain Strength**: Balance embedding integrity with problem solving
2. **Use Multiple Reads**: Quantum annealing is probabilistic
3. **Analyze Energy Histograms**: Look for clear energy gaps
4. **Consider Hybrid Approaches**: For large or hard problems

### Cost Optimization

1. **Test Locally**: Use simulator for development
2. **Batch Problems**: Solve multiple variants together  
3. **Monitor Usage**: Track D-Wave Leap usage and credits
4. **Use Appropriate Solvers**: Match solver to problem size/type

## Troubleshooting

### Common Issues

**ImportError: No module named 'dwave'**
```bash
pip install dwave-ocean-sdk
```

**Authentication Error**
```python
# Check token configuration
import os
print("Token:", os.getenv('DWAVE_API_TOKEN'))

# Test connection
from dwave.system import DWaveSampler
try:
    sampler = DWaveSampler(token='your-token')
    print("Connection successful")
except Exception as e:
    print(f"Connection failed: {e}")
```

**No Embedding Found**
```python
# Reduce problem size or density
smaller_problem = {k: v for i, (k, v) in enumerate(problem.items()) if i < 10}

# Or use hybrid solver
hybrid_backend = sqx.OceanBackend(device='hybrid', token='your-token')
result = hybrid_backend.solve_qubo(problem)
```

**Poor Solution Quality**
```python
# Increase number of reads
result = backend.solve_qubo(problem, num_reads=5000)

# Tune chain strength
result = backend.solve_qubo(problem, chain_strength=2.0)

# Use custom annealing schedule
schedule = [[0, 1], [0.2, 0.8], [0.8, 0.2], [1, 0]]
result = backend.solve_qubo(problem, anneal_schedule=schedule)
```

### Performance Issues

**Slow Embedding**
- Reduce problem connectivity
- Use preprocessing to simplify problem
- Consider problem decomposition

**High Error Rates**
- Increase chain strength
- Reduce annealing time
- Use error correction techniques

**Inconsistent Results**
- Increase number of reads
- Check for broken chains
- Validate problem formulation

## Integration Examples

### Hybrid VQE-Annealing Algorithm

```python
def hybrid_vqe_annealing(backend_gate, backend_ocean, hamiltonian, ansatz_params):
    """Combine VQE with annealing for parameter optimization."""
    
    def evaluate_energy(params):
        # Use gate-model backend for VQE energy evaluation
        circuit = create_vqe_circuit(backend_gate, params)
        energy = calculate_expectation_value(circuit, hamiltonian)
        return energy
    
    # Create QUBO for parameter optimization using annealing
    # This is a simplified example - real implementation more complex
    
    param_space = np.linspace(0, 2*np.pi, 10)  # Discretize parameters
    n_params = len(ansatz_params)
    
    # QUBO terms for parameter optimization
    Q = {}
    for i in range(n_params):
        for p1, param_val1 in enumerate(param_space):
            for p2, param_val2 in enumerate(param_space):
                if p1 <= p2:
                    # Evaluate energy for these parameter values
                    test_params = ansatz_params.copy()
                    test_params[i] = param_val1
                    if i != len(test_params) - 1:
                        test_params[i+1] = param_val2
                    
                    energy = evaluate_energy(test_params)
                    var1 = i * len(param_space) + p1
                    var2 = i * len(param_space) + p2
                    
                    Q[(var1, var2)] = energy
    
    # Solve parameter optimization with annealing
    result = backend_ocean.solve_qubo(Q)
    
    return result

# This would require both gate and annealing backends
# gate_backend = sqx.get_backend('qiskit')
# ocean_backend = sqx.get_backend('ocean')
# hybrid_result = hybrid_vqe_annealing(gate_backend, ocean_backend, h2_hamiltonian, [0.5, 1.0])
```

### Portfolio Optimization

```python
def portfolio_optimization_ocean(returns, risks, budget=1.0, risk_tolerance=0.1):
    """Portfolio optimization using quantum annealing."""
    
    n_assets = len(returns)
    
    # Discretize investment levels (e.g., 0%, 25%, 50%, 75%, 100%)
    levels = [0, 0.25, 0.5, 0.75, 1.0]
    n_levels = len(levels)
    
    # Variables: x_{i,l} = 1 if asset i gets investment level l
    Q = {}
    
    # Objective: maximize return, minimize risk
    for asset in range(n_assets):
        for level in range(n_levels):
            var = asset * n_levels + level
            
            # Return term (negative for maximization)
            return_coeff = -returns[asset] * levels[level]
            Q[(var, var)] = Q.get((var, var), 0) + return_coeff
            
            # Risk term (quadratic in investment levels)
            risk_coeff = risk_tolerance * risks[asset] * (levels[level] ** 2)
            Q[(var, var)] = Q.get((var, var), 0) + risk_coeff
    
    # Constraint: Each asset gets exactly one investment level
    penalty = 10.0
    for asset in range(n_assets):
        # (∑_{l} x_{asset,l} - 1)² = penalty
        for l1 in range(n_levels):
            for l2 in range(n_levels):
                var1 = asset * n_levels + l1
                var2 = asset * n_levels + l2
                
                if l1 == l2:
                    Q[(var1, var1)] = Q.get((var1, var1), 0) + penalty
                else:
                    Q[(var1, var2)] = Q.get((var1, var2), 0) + penalty
        
        # Linear terms
        for level in range(n_levels):
            var = asset * n_levels + level
            Q[(var, var)] = Q.get((var, var), 0) - 2 * penalty
    
    # Budget constraint (simplified)
    budget_penalty = 10.0
    for a1 in range(n_assets):
        for l1 in range(n_levels):
            for a2 in range(n_assets):
                for l2 in range(n_levels):
                    if a1 <= a2:
                        var1 = a1 * n_levels + l1
                        var2 = a2 * n_levels + l2
                        
                        coeff = budget_penalty * levels[l1] * levels[l2]
                        Q[(var1, var2)] = Q.get((var1, var2), 0) + coeff
    
    # Solve portfolio optimization
    result = backend.solve_qubo(Q)
    
    if result['success']:
        # Decode portfolio allocation
        best_sample = result['samples'][0]
        portfolio = {}
        
        for asset in range(n_assets):
            for level in range(n_levels):
                var = asset * n_levels + level
                if best_sample.get(var, 0) == 1:
                    portfolio[asset] = levels[level]
                    break
        
        print("Optimal portfolio allocation:")
        total_investment = 0
        expected_return = 0
        expected_risk = 0
        
        for asset, allocation in portfolio.items():
            print(f"Asset {asset}: {allocation*100:.1f}%")
            total_investment += allocation
            expected_return += allocation * returns[asset]
            expected_risk += allocation * risks[asset]
        
        print(f"Total investment: {total_investment*100:.1f}%")
        print(f"Expected return: {expected_return:.4f}")
        print(f"Expected risk: {expected_risk:.4f}")
        
        return portfolio
    
    return None

# Example portfolio optimization
returns = [0.1, 0.15, 0.08, 0.12, 0.20]  # Expected returns
risks = [0.05, 0.10, 0.03, 0.08, 0.15]   # Risk levels

optimal_portfolio = portfolio_optimization_ocean(returns, risks)
```

## API Reference

The Ocean backend provides these key methods:

- `solve_qubo(Q, **kwargs)`: Solve QUBO optimization problems
- `solve_ising(h, J, **kwargs)`: Solve Ising model problems
- `solve_optimization_problem(problem, type)`: Generic problem solver
- `solve_max_cut(graph, **kwargs)`: Maximum cut problem solver
- `solve_tsp(distance_matrix, **kwargs)`: Traveling salesman problem
- `get_backend_info()`: Backend and hardware information
- `get_version_info()`: Ocean SDK version information

For complete API documentation, see the [API Reference](../api/backends.md#ocean-backend) section.

---

*For more information about D-Wave Ocean, visit the [Ocean Documentation](https://docs.ocean.dwavesys.com/en/stable/).*