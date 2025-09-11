# Quantum Machine Learning Tutorial

Learn how to apply quantum computing to machine learning problems using SuperQuantX. This tutorial covers the fundamentals of quantum machine learning (QML) and provides hands-on examples you can run today.

## 🎯 Learning Objectives

By the end of this tutorial, you'll know how to:

- **Apply quantum algorithms to classical ML problems**
- **Build and train quantum neural networks**
- **Use quantum support vector machines**
- **Implement quantum feature maps and kernels**
- **Compare quantum vs classical ML performance**
- **Handle real datasets with quantum algorithms**

## 🚀 Prerequisites

- Completion of [Basic Quantum Computing Tutorial](basic-quantum.md)
- Basic machine learning knowledge (classification, regression, neural networks)
- Python ML libraries (scikit-learn, numpy)
- SuperQuantX installed with ML extras: `pip install superquantx[ml]`

## 📚 Chapter 1: Why Quantum Machine Learning?

### Potential Quantum Advantages

Quantum machine learning promises several advantages:

1. **Exponential state space**: n qubits can represent 2^n states simultaneously
2. **Quantum parallelism**: Process multiple data points in superposition  
3. **Quantum interference**: Enhance correct answers, suppress wrong ones
4. **Quantum entanglement**: Capture complex correlations in data

### Current Reality Check

Let's be honest about current limitations:

```python
import superquantx as sqx
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def quantum_vs_classical_reality_check():
    """Compare current quantum vs classical ML performance."""
    
    print("Quantum vs Classical ML: Reality Check")
    print("=" * 45)
    
    # Generate a simple dataset
    X, y = make_classification(
        n_samples=100, 
        n_features=4, 
        n_classes=2, 
        n_redundant=0, 
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features, 2 classes")
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Classical ML
    from sklearn.svm import SVC
    classical_svm = SVC(kernel='rbf', random_state=42)
    classical_svm.fit(X_train, y_train)
    classical_pred = classical_svm.predict(X_test)
    classical_accuracy = accuracy_score(y_test, classical_pred)
    
    # Quantum ML
    qsvm = sqx.QuantumSVM(backend='simulator', feature_map='ZFeatureMap')
    qsvm.fit(X_train, y_train)
    quantum_pred = qsvm.predict(X_test)
    quantum_accuracy = accuracy_score(y_test, quantum_pred)
    
    print(f"\nResults:")
    print(f"Classical SVM Accuracy: {classical_accuracy:.3f}")
    print(f"Quantum SVM Accuracy:   {quantum_accuracy:.3f}")
    
    print(f"\nCurrent State of QML:")
    print("✅ Quantum algorithms work and can learn")
    print("❗ Classical methods often still outperform quantum (for now)")
    print("🔬 QML is actively researched with rapid progress")
    print("🎯 Focus areas: quantum advantage, NISQ algorithms, hybrid methods")

quantum_vs_classical_reality_check()
```

## 📚 Chapter 2: Quantum Feature Maps

Feature maps encode classical data into quantum states. This is crucial for quantum ML.

### Understanding Feature Maps

```python
def explore_feature_maps():
    """Understand different quantum feature maps."""
    
    print("Quantum Feature Maps Exploration")
    print("=" * 35)
    
    # Sample data point
    x = np.array([0.5, -0.3, 0.8, 0.1])
    print(f"Classical data point: {x}")
    
    feature_maps = ['ZFeatureMap', 'ZZFeatureMap', 'PauliFeatureMap']
    
    for fm_name in feature_maps:
        print(f"\n--- {fm_name} ---")
        
        # Create quantum feature map circuit
        backend = sqx.get_backend('simulator')
        circuit = backend.create_circuit(4)  # 4 qubits for 4 features
        
        if fm_name == 'ZFeatureMap':
            # Z-rotation encoding: RZ(xi) on each qubit
            for i, xi in enumerate(x):
                circuit.rz(xi, i)
                
        elif fm_name == 'ZZFeatureMap':
            # ZZ feature map: RZ rotations + ZZ entangling
            for i, xi in enumerate(x):
                circuit.h(i)  # Superposition
                circuit.rz(xi, i)  # Encode feature
            
            # Entangling gates
            for i in range(len(x) - 1):
                circuit.cx(i, i + 1)
                circuit.rz(x[i] * x[i + 1], i + 1)  # ZZ interaction
                circuit.cx(i, i + 1)
                
        elif fm_name == 'PauliFeatureMap':
            # Full Pauli feature map
            for i, xi in enumerate(x):
                circuit.h(i)
                circuit.rz(xi, i)
            
            # All-to-all ZZ interactions
            for i in range(len(x)):
                for j in range(i + 1, len(x)):
                    circuit.cx(i, j)
                    circuit.rz(x[i] * x[j], j)
                    circuit.cx(i, j)
        
        # Measure expectation values
        circuit.measure_all()
        result = backend.run(circuit, shots=1000)
        counts = result.get_counts()
        
        print(f"  Measurement distribution: {dict(list(counts.items())[:3])}...")
        
        # Calculate some basic statistics
        total_shots = sum(counts.values())
        prob_all_zeros = counts.get('0000', 0) / total_shots
        print(f"  P(|0000⟩): {prob_all_zeros:.3f}")

explore_feature_maps()
```

### Custom Feature Map

```python
def create_custom_feature_map():
    """Create and test a custom quantum feature map."""
    
    class CustomFeatureMap:
        def __init__(self, n_features, backend='simulator'):
            self.n_features = n_features
            self.backend = sqx.get_backend(backend)
        
        def encode(self, x):
            """Encode classical data x into quantum state."""
            circuit = self.backend.create_circuit(self.n_features)
            
            # Layer 1: Individual rotations
            for i, xi in enumerate(x):
                circuit.ry(xi * np.pi, i)  # Scale to [0, π]
            
            # Layer 2: Entangling interactions
            for i in range(self.n_features - 1):
                circuit.cx(i, i + 1)
                # Data-dependent rotation
                circuit.rz(x[i] * x[i + 1] * np.pi, i + 1)
                circuit.cx(i, i + 1)
            
            # Layer 3: Final individual rotations
            for i, xi in enumerate(x):
                circuit.rx(xi * np.pi / 2, i)
            
            return circuit
        
        def kernel(self, x1, x2):
            """Compute quantum kernel between two data points."""
            # Create circuit for x1
            circuit1 = self.encode(x1)
            
            # Create circuit for x2 and take adjoint
            circuit2 = self.encode(x2)
            # In practice, we'd implement the adjoint properly
            
            # For demonstration, we'll use a simplified approach
            combined_circuit = self.backend.create_circuit(self.n_features)
            
            # Encode x1
            for i, xi in enumerate(x1):
                combined_circuit.ry(xi * np.pi, i)
            
            # "Unencode" x2 (simplified adjoint)
            for i, xi in enumerate(x2):
                combined_circuit.ry(-xi * np.pi, i)
            
            # Measure overlap
            combined_circuit.measure_all()
            result = self.backend.run(combined_circuit, shots=1000)
            counts = result.get_counts()
            
            # Kernel value is probability of measuring |0...0⟩
            kernel_value = counts.get('0' * self.n_features, 0) / 1000
            return kernel_value
    
    # Test custom feature map
    print("Custom Quantum Feature Map")
    print("=" * 30)
    
    fm = CustomFeatureMap(n_features=4)
    
    # Test data points
    x1 = np.array([0.1, 0.5, -0.2, 0.8])
    x2 = np.array([0.2, 0.4, -0.1, 0.7])  # Similar to x1
    x3 = np.array([-0.8, -0.5, 0.9, -0.3])  # Different from x1
    
    print(f"Data points:")
    print(f"x1 = {x1}")
    print(f"x2 = {x2} (similar to x1)")
    print(f"x3 = {x3} (different from x1)")
    
    # Compute kernels
    kernel_11 = fm.kernel(x1, x1)
    kernel_12 = fm.kernel(x1, x2)
    kernel_13 = fm.kernel(x1, x3)
    
    print(f"\nQuantum Kernel Values:")
    print(f"K(x1, x1) = {kernel_11:.3f} (should be high)")
    print(f"K(x1, x2) = {kernel_12:.3f} (should be medium-high)")
    print(f"K(x1, x3) = {kernel_13:.3f} (should be lower)")
    
    if kernel_11 > kernel_12 > kernel_13:
        print("✅ Kernel behaves as expected!")
    else:
        print("⚠️  Kernel needs tuning")

create_custom_feature_map()
```

## 📚 Chapter 3: Quantum Support Vector Machines

QSVMs use quantum computers to find optimal separating hyperplanes.

### Basic QSVM Implementation

```python
def qsvm_detailed_example():
    """Detailed QSVM example with visualization."""
    
    print("Quantum Support Vector Machine")
    print("=" * 35)
    
    # Generate a more interesting dataset
    from sklearn.datasets import make_moons
    
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Dataset: Two moons (non-linearly separable)")
    print(f"Training: {len(X_train)} samples")
    print(f"Testing: {len(X_test)} samples")
    
    # Compare different feature maps
    feature_maps = ['ZFeatureMap', 'ZZFeatureMap', 'PauliFeatureMap']
    results = {}
    
    for fm in feature_maps:
        print(f"\n--- Testing {fm} ---")
        
        # Create and train QSVM
        qsvm = sqx.QuantumSVM(
            backend='simulator',
            feature_map=fm,
            num_features=2,
            depth=2
        )
        
        # Train
        print("Training QSVM...")
        qsvm.fit(X_train, y_train)
        
        # Test
        y_pred = qsvm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get prediction probabilities
        y_prob = qsvm.predict_proba(X_test)
        
        results[fm] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_prob
        }
        
        print(f"Accuracy: {accuracy:.3f}")
    
    # Compare with classical SVM
    from sklearn.svm import SVC
    classical_svm = SVC(kernel='rbf', probability=True, random_state=42)
    classical_svm.fit(X_train, y_train)
    classical_pred = classical_svm.predict(X_test)
    classical_accuracy = accuracy_score(y_test, classical_pred)
    
    print(f"\n--- Classical SVM Comparison ---")
    print(f"Classical SVM Accuracy: {classical_accuracy:.3f}")
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Plot original data
    plt.subplot(2, 3, 1)
    colors = ['red', 'blue']
    for i, color in enumerate(colors):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], c=color, label=f'Class {i}', alpha=0.6)
    plt.title('Original Data')
    plt.legend()
    plt.grid(True)
    
    # Plot results for each feature map
    for idx, (fm, result) in enumerate(results.items(), 2):
        plt.subplot(2, 3, idx)
        
        # Plot test points colored by prediction
        for i in range(len(X_test)):
            color = colors[result['predictions'][i]]
            marker = 'o' if result['predictions'][i] == y_test[i] else 'x'
            plt.scatter(X_test[i, 0], X_test[i, 1], c=color, marker=marker, s=50)
        
        plt.title(f'{fm}\nAccuracy: {result["accuracy"]:.3f}')
        plt.grid(True)
    
    # Classical SVM results
    plt.subplot(2, 3, 5)
    for i in range(len(X_test)):
        color = colors[classical_pred[i]]
        marker = 'o' if classical_pred[i] == y_test[i] else 'x'
        plt.scatter(X_test[i, 0], X_test[i, 1], c=color, marker=marker, s=50)
    
    plt.title(f'Classical SVM\nAccuracy: {classical_accuracy:.3f}')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Best quantum method
    best_qm = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_acc = results[best_qm]['accuracy']
    
    print(f"\n📊 Results Summary:")
    print(f"Best Quantum Method: {best_qm} ({best_acc:.3f})")
    print(f"Classical SVM: {classical_accuracy:.3f}")
    
    if best_acc > classical_accuracy:
        print("🎉 Quantum advantage achieved!")
    else:
        print("🤔 Classical still ahead, but quantum is competitive!")

qsvm_detailed_example()
```

### Quantum Kernel Analysis

```python
def analyze_quantum_kernels():
    """Analyze how quantum kernels work."""
    
    print("Quantum Kernel Analysis")
    print("=" * 25)
    
    # Generate simple 2D dataset
    np.random.seed(42)
    X = np.random.randn(20, 2)
    
    def compute_quantum_kernel_matrix(X, feature_map='ZFeatureMap'):
        """Compute full quantum kernel matrix."""
        n_samples = len(X)
        kernel_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i, n_samples):  # Symmetric matrix
                # Simplified quantum kernel computation
                diff = X[i] - X[j]
                # Gaussian-like quantum kernel
                kernel_value = np.exp(-0.5 * np.linalg.norm(diff)**2)
                
                kernel_matrix[i, j] = kernel_value
                kernel_matrix[j, i] = kernel_value  # Symmetry
        
        return kernel_matrix
    
    # Compute different kernel matrices
    kernels = {}
    for fm in ['ZFeatureMap', 'ZZFeatureMap', 'PauliFeatureMap']:
        kernels[fm] = compute_quantum_kernel_matrix(X, fm)
    
    # Classical kernel for comparison
    from sklearn.metrics.pairwise import rbf_kernel
    classical_kernel = rbf_kernel(X, gamma=0.5)
    
    # Visualize kernel matrices
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    kernel_list = list(kernels.items()) + [('Classical RBF', classical_kernel)]
    
    for idx, (name, kernel) in enumerate(kernel_list):
        row, col = idx // 2, idx % 2
        im = axes[row, col].imshow(kernel, cmap='viridis', vmin=0, vmax=1)
        axes[row, col].set_title(f'{name} Kernel')
        axes[row, col].set_xlabel('Sample Index')
        axes[row, col].set_ylabel('Sample Index')
        plt.colorbar(im, ax=axes[row, col])
    
    plt.tight_layout()
    plt.show()
    
    # Analyze kernel properties
    print("\n📊 Kernel Analysis:")
    for name, kernel in kernels.items():
        diagonal_mean = np.mean(np.diag(kernel))
        off_diagonal_mean = np.mean(kernel - np.diag(np.diag(kernel)))
        
        print(f"{name}:")
        print(f"  Diagonal mean: {diagonal_mean:.3f}")
        print(f"  Off-diagonal mean: {off_diagonal_mean:.3f}")
        print(f"  Matrix rank: {np.linalg.matrix_rank(kernel)}")

analyze_quantum_kernels()
```

## 📚 Chapter 4: Variational Quantum Classifiers

Variational quantum classifiers use parameterized quantum circuits that can be trained.

### Building a VQC from Scratch

```python
def build_vqc_from_scratch():
    """Build and train a variational quantum classifier step by step."""
    
    print("Variational Quantum Classifier (VQC)")
    print("=" * 40)
    
    class SimpleVQC:
        def __init__(self, n_qubits, n_layers, backend='simulator'):
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            self.backend = sqx.get_backend(backend)
            
            # Initialize random parameters
            self.n_params = n_qubits * n_layers * 3  # 3 rotations per qubit per layer
            self.params = np.random.random(self.n_params) * 2 * np.pi
            
            print(f"VQC initialized: {n_qubits} qubits, {n_layers} layers")
            print(f"Total parameters: {self.n_params}")
        
        def create_ansatz(self, params):
            """Create the variational ansatz circuit."""
            circuit = self.backend.create_circuit(self.n_qubits)
            
            param_idx = 0
            
            for layer in range(self.n_layers):
                # Parameterized single-qubit rotations
                for qubit in range(self.n_qubits):
                    circuit.rx(params[param_idx], qubit)
                    param_idx += 1
                    circuit.ry(params[param_idx], qubit)  
                    param_idx += 1
                    circuit.rz(params[param_idx], qubit)
                    param_idx += 1
                
                # Entangling gates
                if layer < self.n_layers - 1:  # No entangling on last layer
                    for qubit in range(self.n_qubits - 1):
                        circuit.cx(qubit, qubit + 1)
            
            return circuit
        
        def encode_data(self, circuit, x):
            """Encode classical data into quantum circuit."""
            # Simple amplitude encoding (for demonstration)
            for i, xi in enumerate(x[:self.n_qubits]):
                circuit.ry(xi * np.pi, i)
        
        def measure_expectation(self, circuit):
            """Measure expectation value for classification."""
            # Measure Z expectation on first qubit
            circuit_copy = circuit.copy() if hasattr(circuit, 'copy') else circuit
            
            # Add measurement
            circuit.measure_all()
            result = self.backend.run(circuit, shots=1000)
            counts = result.get_counts()
            
            # Calculate expectation value of Z on first qubit
            exp_val = 0
            total_shots = sum(counts.values())
            
            for bitstring, count in counts.items():
                first_bit = int(bitstring[0])  # First qubit
                sign = 1 if first_bit == 0 else -1
                exp_val += sign * count / total_shots
            
            return exp_val
        
        def predict_single(self, x, params=None):
            """Predict single sample."""
            if params is None:
                params = self.params
            
            # Create circuit
            circuit = self.create_ansatz(params)
            self.encode_data(circuit, x)
            
            # Get expectation value
            exp_val = self.measure_expectation(circuit)
            
            # Convert to class prediction
            return 1 if exp_val > 0 else 0
        
        def cost_function(self, params, X, y):
            """Cost function for training."""
            predictions = []
            
            for xi in X:
                pred = self.predict_single(xi, params)
                predictions.append(pred)
            
            # Simple accuracy-based cost (minimize error rate)
            accuracy = np.mean(np.array(predictions) == y)
            cost = 1 - accuracy  # Minimize error rate
            
            return cost
        
        def train(self, X, y, epochs=50):
            """Train the VQC using classical optimization."""
            from scipy.optimize import minimize
            
            print(f"Training VQC for {epochs} iterations...")
            
            # Store training history
            cost_history = []
            
            def callback(params):
                cost = self.cost_function(params, X, y)
                cost_history.append(cost)
                if len(cost_history) % 10 == 0:
                    print(f"  Iteration {len(cost_history)}: Cost = {cost:.4f}")
            
            # Optimize parameters
            result = minimize(
                fun=lambda p: self.cost_function(p, X, y),
                x0=self.params,
                method='COBYLA',
                options={'maxiter': epochs},
                callback=callback
            )
            
            self.params = result.x
            
            return cost_history
        
        def predict(self, X):
            """Predict multiple samples."""
            predictions = []
            for xi in X:
                pred = self.predict_single(xi)
                predictions.append(pred)
            return np.array(predictions)
    
    # Test the VQC
    print("\n🧪 Testing VQC on iris dataset:")
    
    from sklearn.datasets import load_iris
    iris = load_iris()
    
    # Use only two classes and two features for simplicity
    mask = iris.target != 2  # Remove class 2
    X = iris.data[mask][:, :2]  # Use only first 2 features
    y = iris.target[mask]
    
    # Normalize features
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Dataset: {len(X)} samples, 2 features, 2 classes")
    print(f"Training: {len(X_train)}, Testing: {len(X_test)}")
    
    # Create and train VQC
    vqc = SimpleVQC(n_qubits=2, n_layers=2)
    
    # Train
    cost_history = vqc.train(X_train, y_train, epochs=30)
    
    # Test
    y_pred = vqc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nVQC Test Accuracy: {accuracy:.3f}")
    
    # Plot training curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(cost_history, 'b-', linewidth=2)
    plt.title('VQC Training Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Cost (Error Rate)')
    plt.grid(True)
    
    # Plot data and decision boundary (simplified)
    plt.subplot(1, 2, 2)
    
    colors = ['red', 'blue']
    for i, color in enumerate(colors):
        mask = y_test == i
        correct_mask = (y_pred == y_test) & mask
        incorrect_mask = (y_pred != y_test) & mask
        
        # Correct predictions: circles
        plt.scatter(X_test[correct_mask, 0], X_test[correct_mask, 1], 
                   c=color, marker='o', s=50, alpha=0.8, label=f'Class {i} ✓')
        
        # Incorrect predictions: X marks
        plt.scatter(X_test[incorrect_mask, 0], X_test[incorrect_mask, 1], 
                   c=color, marker='x', s=100, linewidth=3, label=f'Class {i} ✗')
    
    plt.title(f'VQC Predictions (Accuracy: {accuracy:.3f})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

build_vqc_from_scratch()
```

## 📚 Chapter 5: Quantum Neural Networks

QNNs extend the concept of neural networks to the quantum realm.

### Multi-layer QNN

```python
def build_quantum_neural_network():
    """Build a multi-layer quantum neural network."""
    
    print("Quantum Neural Network (QNN)")
    print("=" * 32)
    
    class QuantumLayer:
        """A single quantum layer with parameterized gates."""
        
        def __init__(self, n_qubits, name="QuantumLayer"):
            self.n_qubits = n_qubits
            self.name = name
            self.n_params = n_qubits * 2  # RY and RZ per qubit
            self.params = np.random.random(self.n_params) * 2 * np.pi
        
        def forward(self, circuit, params=None):
            """Apply quantum layer to circuit."""
            if params is None:
                params = self.params
            
            param_idx = 0
            
            # Parameterized rotations
            for qubit in range(self.n_qubits):
                circuit.ry(params[param_idx], qubit)
                param_idx += 1
                circuit.rz(params[param_idx], qubit)
                param_idx += 1
            
            # Entangling gates
            for qubit in range(self.n_qubits - 1):
                circuit.cx(qubit, qubit + 1)
            
            return circuit
    
    class QuantumNeuralNetwork:
        """Multi-layer quantum neural network."""
        
        def __init__(self, n_qubits, layer_config, backend='simulator'):
            self.n_qubits = n_qubits
            self.backend = sqx.get_backend(backend)
            self.layers = []
            
            # Create layers
            for i, layer_type in enumerate(layer_config):
                if layer_type == 'quantum':
                    layer = QuantumLayer(n_qubits, f"QLayer_{i}")
                    self.layers.append(layer)
            
            # Total parameters
            self.n_params = sum(layer.n_params for layer in self.layers)
            print(f"QNN created: {len(self.layers)} layers, {self.n_params} parameters")
        
        def encode_input(self, circuit, x):
            """Encode input data into quantum state."""
            # Amplitude encoding (simplified)
            for i, xi in enumerate(x[:self.n_qubits]):
                # Scale to [0, π]
                angle = (xi + 1) * np.pi / 2  # Assuming x in [-1, 1]
                circuit.ry(angle, i)
        
        def forward(self, x, params=None):
            """Forward pass through QNN."""
            if params is None:
                params = np.concatenate([layer.params for layer in self.layers])
            
            # Create circuit
            circuit = self.backend.create_circuit(self.n_qubits)
            
            # Encode input
            self.encode_input(circuit, x)
            
            # Apply layers
            param_idx = 0
            for layer in self.layers:
                layer_params = params[param_idx:param_idx + layer.n_params]
                circuit = layer.forward(circuit, layer_params)
                param_idx += layer.n_params
            
            return circuit
        
        def measure_output(self, circuit):
            """Measure quantum state to get classical output."""
            # Measure expectation values
            circuit_copy = self.backend.create_circuit(self.n_qubits)
            
            # Recreate circuit (simplified for demo)
            circuit.measure_all()
            result = self.backend.run(circuit, shots=1000)
            counts = result.get_counts()
            
            # Convert to output value
            total_shots = sum(counts.values())
            expectation = 0
            
            for bitstring, count in counts.items():
                # Use parity of measurement
                parity = sum(int(bit) for bit in bitstring) % 2
                sign = 1 if parity == 0 else -1
                expectation += sign * count / total_shots
            
            return expectation
        
        def predict(self, x, params=None):
            """Make prediction for input x."""
            circuit = self.forward(x, params)
            output = self.measure_output(circuit)
            
            # Convert to binary classification
            return 1 if output > 0 else 0
        
        def cost_function(self, params, X, y):
            """Cost function for training."""
            total_cost = 0
            
            for xi, yi in zip(X, y):
                prediction = self.predict(xi, params)
                # Binary cross-entropy-like cost
                cost = (prediction - yi) ** 2
                total_cost += cost
            
            return total_cost / len(X)
        
        def train(self, X, y, epochs=50, learning_rate=0.1):
            """Train QNN using parameter-shift rule."""
            print(f"Training QNN for {epochs} epochs...")
            
            # Get initial parameters
            params = np.concatenate([layer.params for layer in self.layers])
            cost_history = []
            
            for epoch in range(epochs):
                # Compute gradients using parameter-shift rule
                gradients = np.zeros_like(params)
                
                for i in range(len(params)):
                    # Forward shift
                    params_plus = params.copy()
                    params_plus[i] += np.pi / 2
                    cost_plus = self.cost_function(params_plus, X, y)
                    
                    # Backward shift  
                    params_minus = params.copy()
                    params_minus[i] -= np.pi / 2
                    cost_minus = self.cost_function(params_minus, X, y)
                    
                    # Gradient
                    gradients[i] = (cost_plus - cost_minus) / 2
                
                # Update parameters
                params = params - learning_rate * gradients
                
                # Record cost
                current_cost = self.cost_function(params, X, y)
                cost_history.append(current_cost)
                
                if epoch % 10 == 0:
                    print(f"  Epoch {epoch}: Cost = {current_cost:.4f}")
            
            # Update layer parameters
            param_idx = 0
            for layer in self.layers:
                layer.params = params[param_idx:param_idx + layer.n_params]
                param_idx += layer.n_params
            
            return cost_history
    
    # Test QNN on a dataset
    print("\n🧪 Testing QNN on a classification problem:")
    
    # Generate XOR-like dataset (not linearly separable)
    np.random.seed(42)
    X = np.array([
        [-1, -1], [-1, 1], [1, -1], [1, 1],
        [-0.8, -0.8], [-0.9, 0.9], [0.8, -0.9], [0.9, 0.8]
    ])
    y = np.array([0, 1, 1, 0, 0, 1, 1, 0])  # XOR pattern
    
    print(f"Dataset: XOR-like pattern with {len(X)} samples")
    
    # Create QNN
    qnn = QuantumNeuralNetwork(
        n_qubits=2,
        layer_config=['quantum', 'quantum'],  # 2 quantum layers
        backend='simulator'
    )
    
    # Train QNN
    cost_history = qnn.train(X, y, epochs=30, learning_rate=0.3)
    
    # Test predictions
    print(f"\nTesting QNN predictions:")
    correct = 0
    for xi, yi in zip(X, y):
        pred = qnn.predict(xi)
        correct += (pred == yi)
        print(f"  Input: {xi} → True: {yi}, Pred: {pred} {'✓' if pred == yi else '✗'}")
    
    accuracy = correct / len(X)
    print(f"\nQNN Accuracy: {accuracy:.3f}")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    
    # Training curve
    plt.subplot(1, 2, 1)
    plt.plot(cost_history, 'b-', linewidth=2)
    plt.title('QNN Training Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.grid(True)
    
    # Data visualization
    plt.subplot(1, 2, 2)
    colors = ['red', 'blue']
    
    for xi, yi in zip(X, y):
        pred = qnn.predict(xi)
        color = colors[yi]
        marker = 'o' if pred == yi else 'x'
        size = 100 if pred == yi else 150
        
        plt.scatter(xi[0], xi[1], c=color, marker=marker, s=size, alpha=0.8)
    
    plt.title(f'QNN Results (Acc: {accuracy:.2f})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Class 0 ✓'),
        Line2D([0], [0], marker='x', color='red', markersize=10, label='Class 0 ✗'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Class 1 ✓'),
        Line2D([0], [0], marker='x', color='blue', markersize=10, label='Class 1 ✗')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.show()

build_quantum_neural_network()
```

## 📚 Chapter 6: Real-World Application

Let's apply QML to a real dataset and compare with classical methods comprehensively.

### Complete QML Pipeline

```python
def complete_qml_pipeline():
    """Complete quantum machine learning pipeline on real data."""
    
    print("Complete Quantum ML Pipeline")
    print("=" * 35)
    
    # Load a real dataset
    from sklearn.datasets import load_wine
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    wine = load_wine()
    X, y = wine.data, wine.target
    
    print(f"Dataset: Wine classification")
    print(f"Original shape: {X.shape}")
    print(f"Classes: {len(np.unique(y))} ({np.unique(y)})")
    
    # Preprocessing
    # 1. Use only 2 classes for binary classification
    binary_mask = y != 2
    X_binary = X[binary_mask]
    y_binary = y[binary_mask]
    
    # 2. Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_binary)
    
    # 3. Dimensionality reduction (quantum circuits work better with fewer features)
    pca = PCA(n_components=4)
    X_reduced = pca.fit_transform(X_scaled)
    
    print(f"After preprocessing: {X_reduced.shape}")
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y_binary, test_size=0.3, random_state=42, stratify=y_binary
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Model comparison
    models = {}
    
    # 1. Classical models
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    classical_models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    print(f"\n🔬 Training and evaluating models...")
    
    for name, model in classical_models.items():
        print(f"\n--- {name} ---")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        models[name] = {
            'accuracy': accuracy,
            'type': 'classical',
            'predictions': y_pred
        }
        
        print(f"Accuracy: {accuracy:.3f}")
    
    # 2. Quantum models
    quantum_models = {
        'QSVM (ZFeatureMap)': sqx.QuantumSVM(
            backend='simulator', 
            feature_map='ZFeatureMap'
        ),
        'QSVM (ZZFeatureMap)': sqx.QuantumSVM(
            backend='simulator', 
            feature_map='ZZFeatureMap'
        ),
        'VQC': sqx.VariationalQuantumClassifier(
            n_qubits=4, 
            n_layers=2, 
            backend='simulator'
        )
    }
    
    for name, model in quantum_models.items():
        print(f"\n--- {name} ---")
        
        try:
            # Train
            if 'VQC' in name:
                model.compile(optimizer='adam', learning_rate=0.1)
                model.fit(X_train, y_train, epochs=20, verbose=0)
            else:
                model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            models[name] = {
                'accuracy': accuracy,
                'type': 'quantum',
                'predictions': y_pred
            }
            
            print(f"Accuracy: {accuracy:.3f}")
            
        except Exception as e:
            print(f"Error training {name}: {e}")
            models[name] = {
                'accuracy': 0.0,
                'type': 'quantum',
                'predictions': np.zeros_like(y_test),
                'error': str(e)
            }
    
    # Results analysis
    print(f"\n📊 Model Comparison:")
    print("=" * 50)
    
    # Sort by accuracy
    sorted_models = sorted(models.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for i, (name, result) in enumerate(sorted_models, 1):
        status = "🏆" if i == 1 else f"{i:2d}."
        accuracy = result['accuracy']
        model_type = result['type']
        
        print(f"{status} {name:<20} ({model_type:>8}): {accuracy:.3f}")
    
    # Statistical significance test
    from scipy import stats
    
    classical_accuracies = [v['accuracy'] for k, v in models.items() if v['type'] == 'classical']
    quantum_accuracies = [v['accuracy'] for k, v in models.items() if v['type'] == 'quantum' and 'error' not in v]
    
    if len(quantum_accuracies) > 0:
        print(f"\n📈 Statistical Analysis:")
        print(f"Classical methods - Mean: {np.mean(classical_accuracies):.3f}, Std: {np.std(classical_accuracies):.3f}")
        print(f"Quantum methods   - Mean: {np.mean(quantum_accuracies):.3f}, Std: {np.std(quantum_accuracies):.3f}")
        
        if len(classical_accuracies) > 1 and len(quantum_accuracies) > 1:
            t_stat, p_value = stats.ttest_ind(classical_accuracies, quantum_accuracies)
            print(f"T-test p-value: {p_value:.4f}")
            
            if p_value < 0.05:
                print("📊 Significant difference between classical and quantum methods")
            else:
                print("📊 No significant difference between classical and quantum methods")
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Accuracy comparison
    plt.subplot(2, 2, 1)
    names = [name for name, _ in sorted_models]
    accuracies = [result['accuracy'] for _, result in sorted_models]
    colors = ['blue' if models[name]['type'] == 'classical' else 'red' for name in names]
    
    bars = plt.bar(range(len(names)), accuracies, color=colors, alpha=0.7)
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Classical'),
                      Patch(facecolor='red', alpha=0.7, label='Quantum')]
    plt.legend(handles=legend_elements)
    
    # Feature importance (PCA components)
    plt.subplot(2, 2, 2)
    plt.bar(range(len(pca.components_[0])), np.abs(pca.components_[0]), alpha=0.7)
    plt.xlabel('Original Features')
    plt.ylabel('|PCA Component 1|')
    plt.title('Feature Importance (PCA)')
    plt.grid(True, alpha=0.3)
    
    # Data visualization (first 2 PCA components)
    plt.subplot(2, 2, 3)
    colors_data = ['red', 'blue']
    for i, color in enumerate(colors_data):
        mask = y_test == i
        plt.scatter(X_test[mask, 0], X_test[mask, 1], 
                   c=color, label=f'Class {i}', alpha=0.6)
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Test Data Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Confusion matrix for best model
    best_model_name = sorted_models[0][0]
    best_predictions = models[best_model_name]['predictions']
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, best_predictions)
    
    plt.subplot(2, 2, 4)
    im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.colorbar(im)
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.tight_layout()
    plt.show()
    
    # Final recommendations
    print(f"\n🎯 Recommendations:")
    
    best_classical = max(classical_models.keys(), key=lambda x: models[x]['accuracy'])
    best_quantum = max([k for k in models.keys() if models[k]['type'] == 'quantum' and 'error' not in models[k]], 
                      key=lambda x: models[x]['accuracy'], default=None)
    
    if best_quantum:
        classical_acc = models[best_classical]['accuracy']
        quantum_acc = models[best_quantum]['accuracy']
        
        print(f"🏆 Best Classical: {best_classical} ({classical_acc:.3f})")
        print(f"🔬 Best Quantum: {best_quantum} ({quantum_acc:.3f})")
        
        if quantum_acc > classical_acc:
            print("🎉 Quantum methods show promise for this dataset!")
        elif quantum_acc > classical_acc - 0.05:
            print("🤔 Quantum methods are competitive but need optimization")
        else:
            print("📈 Classical methods currently outperform quantum (normal for NISQ era)")
    
    print(f"\n💡 Next Steps:")
    print("1. Try different quantum feature maps")
    print("2. Increase quantum circuit depth (if computationally feasible)")
    print("3. Use error mitigation techniques")
    print("4. Consider hybrid quantum-classical algorithms")
    print("5. Test on quantum hardware when available")

complete_qml_pipeline()
```

## 🎯 What's Next?

You've now mastered the fundamentals of quantum machine learning! Here's what you've learned:

✅ **Quantum feature maps and data encoding**  
✅ **Quantum Support Vector Machines (QSVMs)**  
✅ **Variational Quantum Classifiers (VQCs)**  
✅ **Quantum Neural Networks (QNNs)**  
✅ **Real-world QML pipeline development**  
✅ **Classical vs quantum performance comparison**

### Advanced QML Topics

1. **[Quantum Neural Networks Tutorial](quantum-nn.md)** - Deep dive into advanced QNN architectures
2. **Quantum Generative Models** - QGANs, QVAEs, and quantum generative modeling
3. **Quantum Reinforcement Learning** - QL algorithms and quantum environments
4. **Quantum Natural Language Processing** - Applying QML to text and language tasks

### Research Directions

- **NISQ-era QML algorithms** - Algorithms designed for noisy quantum devices
- **Quantum advantage** - Finding problems where quantum provides clear benefits
- **Hybrid approaches** - Combining quantum and classical components optimally
- **Error mitigation** - Techniques to improve QML performance on real hardware

### Practical Applications

- **Drug discovery** - Molecular simulation and property prediction
- **Financial modeling** - Portfolio optimization and risk analysis
- **Materials science** - Quantum chemistry and materials design
- **Optimization** - Solving complex combinatorial problems

## 🤝 Need Help?

- **[Advanced Examples](advanced-examples.md)** - More complex QML implementations
- **[API Reference](../api/algorithms.md)** - Detailed documentation  
- **[GitHub Issues](https://github.com/SuperagenticAI/superquantx/issues)** - Ask questions
- **QML Community** - Join quantum machine learning forums and discussions

---

*Congratulations on completing the Quantum Machine Learning tutorial! You're now ready to explore the exciting frontier where quantum computing meets artificial intelligence.* 🚀

!!! tip "Keep Learning"
    QML is a rapidly evolving field. Stay updated with the latest research papers, try new algorithms, and experiment with different datasets. The quantum advantage in machine learning is still being discovered!

!!! warning "Hardware Considerations"  
    When moving to real quantum hardware, consider noise, limited connectivity, and shorter coherence times. Start with small problems and gradually scale up.

!!! info "Ethical AI"
    As you develop QML applications, consider the ethical implications of AI systems, data privacy, and the potential societal impact of quantum-enhanced intelligence.