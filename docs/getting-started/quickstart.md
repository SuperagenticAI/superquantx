# Quick Start Guide

Get up and running with SuperQuantX in just 5 minutes! This guide will take you from installation to running your first quantum program.

## 📦 Installation

SuperQuantX requires Python 3.10 or higher. Install it using pip:

=== "Basic Installation"

    ```bash
    pip install superquantx
    ```

    This installs SuperQuantX with the built-in simulator backend.

=== "With Quantum Backend"

    ```bash
    # For PennyLane integration
    pip install superquantx[pennylane]

    # For Qiskit integration
    pip install superquantx[qiskit]

    # For multiple backends
    pip install superquantx[pennylane,qiskit]
    ```

=== "Development Setup"

    ```bash
    # Clone the repository
    git clone https://github.com/SuperagenticAI/superquantx.git
    cd superquantx

    # Install with development dependencies
    pip install -e .[dev]
    ```

## 🔍 Verify Installation

Let's make sure everything is working:

```python
import superquantx as sqx
print(f"SuperQuantX version: {sqx.__version__}")

# List available backends
backends = sqx.list_backends()
print(f"Available backends: {backends}")
```

You should see something like:
```
SuperQuantX version: 0.1.0
Available backends: ['simulator', 'pennylane', 'qiskit']
```

## 🎯 Your First Quantum Program

Let's create a simple quantum circuit that demonstrates quantum superposition:

```python
import superquantx as sqx

# Step 1: Get a quantum backend
backend = sqx.get_backend('simulator')
print(f"Using backend: {backend.name}")

# Step 2: Create a quantum circuit with 2 qubits
circuit = backend.create_circuit(num_qubits=2)

# Step 3: Add quantum gates
circuit.h(0)      # Put qubit 0 in superposition
circuit.cx(0, 1)  # Entangle qubits 0 and 1

# Step 4: Measure the qubits
circuit.measure_all()

# Step 5: Run the circuit
result = backend.run(circuit, shots=1000)

# Step 6: Get the results
counts = result.get_counts()
print(f"Measurement results: {counts}")
```

Expected output:
```
Using backend: simulator
Measurement results: {'00': 523, '11': 477}
```

🎉 Congratulations! You've just created your first **quantum entangled state**. Notice how we only get results `00` and `11`, never `01` or `10` - this is quantum entanglement in action!

## 🧠 Your First Quantum Machine Learning Program

Now let's try something more exciting - quantum machine learning:

```python
import superquantx as sqx
import numpy as np
from sklearn.datasets import make_classification

# Step 1: Create a simple binary classification dataset
X, y = make_classification(n_samples=100, n_features=4, n_classes=2, 
                          n_redundant=0, random_state=42)

# Step 2: Split the data
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# Step 3: Create a Quantum SVM
qsvm = sqx.QuantumSVM(backend='simulator', feature_map='ZFeatureMap')

# Step 4: Train the model
print("Training Quantum SVM...")
qsvm.fit(X_train, y_train)

# Step 5: Make predictions
predictions = qsvm.predict(X_test)
accuracy = qsvm.score(X_test, y_test)

print(f"Quantum SVM Accuracy: {accuracy:.2f}")
print(f"First 5 predictions: {predictions[:5]}")
```

## 🎨 Visualizing Your Quantum Circuit

SuperQuantX makes it easy to visualize what your quantum circuits look like:

```python
import superquantx as sqx

# Create a more complex circuit
backend = sqx.get_backend('simulator')
circuit = backend.create_circuit(3)

# Build a quantum circuit
circuit.h(0)           # Hadamard on qubit 0
circuit.cx(0, 1)       # CNOT from qubit 0 to 1
circuit.ry(0.5, 2)     # Y-rotation on qubit 2
circuit.cz(1, 2)       # Controlled-Z gate

# Visualize the circuit
circuit.draw()         # Text representation
circuit.draw('mpl')    # Matplotlib visualization (if available)
```

## 🔄 Switching Between Quantum Backends

One of SuperQuantX's superpowers is easy backend switching. Let's compare the same algorithm across different frameworks:

```python
import superquantx as sqx
import numpy as np

# Create sample data
X = np.random.rand(50, 2)
y = np.random.randint(0, 2, 50)

# List of backends to compare
backends = ['simulator', 'pennylane', 'qiskit']

results = {}

for backend_name in backends:
    try:
        print(f"\n🔍 Testing with {backend_name}...")
        
        # Create and train model
        qsvm = sqx.QuantumSVM(backend=backend_name)
        qsvm.fit(X[:40], y[:40])  # Train on first 40 samples
        
        # Test accuracy
        accuracy = qsvm.score(X[40:], y[40:])  # Test on last 10 samples
        results[backend_name] = accuracy
        
        print(f"✅ {backend_name} accuracy: {accuracy:.3f}")
        
    except ImportError as e:
        print(f"❌ {backend_name} not available: {e}")

print(f"\n📊 Results Summary: {results}")
```

## 📊 Built-in Quantum Algorithms

SuperQuantX comes with several quantum algorithms ready to use:

=== "Quantum SVM"

    ```python
    import superquantx as sqx
    from sklearn.datasets import make_blobs

    # Generate data
    X, y = make_blobs(n_samples=100, centers=2, random_state=42)

    # Create and train Quantum SVM
    qsvm = sqx.QuantumSVM(backend='simulator')
    qsvm.fit(X, y)
    
    # Evaluate
    accuracy = qsvm.score(X, y)
    print(f"Training accuracy: {accuracy:.3f}")
    ```

=== "Variational Quantum Eigensolver"

    ```python
    import superquantx as sqx
    import numpy as np

    # Create a simple Hamiltonian (matrix)
    hamiltonian = np.array([[1, 0], [0, -1]])

    # Find ground state energy
    vqe = sqx.VQE(backend='simulator', hamiltonian=hamiltonian)
    ground_energy = vqe.find_ground_state()
    
    print(f"Ground state energy: {ground_energy:.6f}")
    ```

=== "Quantum Neural Network"

    ```python
    import superquantx as sqx
    import numpy as np

    # Create training data
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)

    # Build Quantum Neural Network
    qnn = sqx.QuantumNeuralNetwork(
        n_qubits=4,
        n_layers=2,
        backend='simulator'
    )

    # Train the network
    qnn.fit(X, y, epochs=10)
    
    # Make predictions
    predictions = qnn.predict(X[:5])
    print(f"Predictions: {predictions}")
    ```

## 🎯 What's Next?

Great job! You've successfully:

✅ Installed SuperQuantX  
✅ Created your first quantum circuit  
✅ Ran a quantum machine learning algorithm  
✅ Visualized quantum circuits  
✅ Compared multiple backends  

### Continue Your Journey:

1. **Learn the Fundamentals**: [Basic Quantum Computing Tutorial](../tutorials/basic-quantum.md)
2. **Explore More Algorithms**: [Quantum Algorithms Guide](../user-guide/algorithms.md)
3. **Deep Dive into Backends**: [Backend Integration Guide](../user-guide/backends.md)
4. **Try Advanced Examples**: [Advanced Tutorials](../tutorials/advanced-examples.md)

### Need Help?

- 📚 **[User Guide](../user-guide/overview.md)**: Comprehensive documentation
- 🤔 **[FAQ](../help/faq.md)**: Common questions and answers
- 🐛 **[Troubleshooting](../help/troubleshooting.md)**: Solve common issues
- 💬 **[GitHub Issues](https://github.com/SuperagenticAI/superquantx/issues)**: Report bugs or ask questions

---

!!! tip "Pro Tip"
    Start a Jupyter notebook and experiment with the examples above. Quantum computing is best learned by doing!

!!! info "Remember"
    SuperQuantX is research software. Use it for learning, experimentation, and research - not for production applications.