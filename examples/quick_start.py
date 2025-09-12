import superquantx as sqx
print(f"SuperQuantX version: {sqx.__version__}")

# List available backends
backends = sqx.list_available_backends()
print(f"Available backends: {backends}")


import superquantx as sqx

# Step 1: Get a quantum backend
backend = sqx.get_backend('simulator')
print(f"Using backend: {backend.device}")

# Step 2: Create a quantum circuit with 2 qubits
circuit = backend.create_circuit(n_qubits=2)

# Step 3: Add quantum gates
circuit = backend.add_gate(circuit, 'H', 0)      # Put qubit 0 in superposition
circuit = backend.add_gate(circuit, 'CNOT', [0, 1])  # Entangle qubits 0 and 1

# Step 4: Measure the qubits
circuit = backend.add_measurement(circuit)

# Step 5: Run the circuit
result = backend.execute_circuit(circuit, shots=1000)

# Step 6: Get the results
counts = result['counts']
print(f"Measurement results: {counts}")


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

import superquantx as sqx

# Create a more complex circuit
backend = sqx.get_backend('simulator')
circuit = backend.create_circuit(n_qubits=3)

# Build a quantum circuit
circuit = backend.add_gate(circuit, 'H', 0)           # Hadamard on qubit 0
circuit = backend.add_gate(circuit, 'CNOT', [0, 1])   # CNOT from qubit 0 to 1
circuit = backend.add_gate(circuit, 'RY', 2, [0.5])   # Y-rotation on qubit 2
circuit = backend.add_gate(circuit, 'CZ', [1, 2])     # Controlled-Z gate

# Note: Circuit visualization methods would need to be implemented
# For now, you can examine the circuit structure programmatically
print(f"Circuit has {circuit.n_qubits} qubits")
print(f"Current state vector shape: {circuit.state.shape}")


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
