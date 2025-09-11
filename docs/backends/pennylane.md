# PennyLane Backend Integration

PennyLane is a cross-platform quantum machine learning library that focuses on variational quantum circuits and quantum machine learning. SuperQuantX provides seamless integration with PennyLane for quantum ML applications.

## 📋 Overview

PennyLane's strengths make it ideal for:
- **Quantum Machine Learning**: Variational circuits and differentiable programming
- **Hybrid Algorithms**: Seamless quantum-classical optimization
- **Multiple Hardware Backends**: Support for various quantum devices
- **Automatic Differentiation**: Built-in gradient computation

## 🚀 Quick Start

### Installation

```bash
# Install SuperQuantX with PennyLane support
pip install superquantx[pennylane]

# Or install PennyLane directly
pip install pennylane pennylane-qiskit pennylane-cirq
```

### Basic Usage

```python
import superquantx as sqx

# Get PennyLane backend
backend = sqx.get_backend('pennylane')
print(f"Backend: {backend}")

# Create a simple circuit
circuit = backend.create_circuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

# Execute circuit
result = backend.run(circuit, shots=1000)
print(f"Results: {result.get_counts()}")
```

## 🔧 Configuration

### Device Selection

PennyLane supports multiple devices. Configure your preferred device:

```python
import superquantx as sqx

# Different PennyLane devices
devices = {
    'default.qubit': 'CPU simulator (default)',
    'lightning.qubit': 'Fast C++ simulator',
    'qiskit.aer': 'Qiskit Aer simulator',
    'qiskit.ibmq': 'IBM quantum hardware',
    'cirq.simulator': 'Cirq simulator',
    'strawberryfields.fock': 'Photonic simulator'
}

# Create backend with specific device
backend = sqx.PennyLaneBackend(device='lightning.qubit', shots=1024)

# Or use device-specific parameters
backend = sqx.PennyLaneBackend(
    device='qiskit.ibmq',
    shots=1024,
    backend_name='ibmq_qasm_simulator',
    hub='ibm-q',
    group='open',
    project='main'
)
```

### Advanced Configuration

```python
import pennylane as qml

# Custom PennyLane device configuration
device_config = {
    'name': 'default.qubit',
    'wires': 4,
    'shots': 2048,
    'analytic': False  # Use sampling instead of exact computation
}

# Create backend with custom config
backend = sqx.PennyLaneBackend(**device_config)

# Access underlying PennyLane device
pennylane_device = backend.device
print(f"Device capabilities: {pennylane_device.capabilities()}")
```

## 📊 Quantum Machine Learning

PennyLane excels at quantum ML. Here are comprehensive examples:

### Variational Quantum Classifier

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def pennylane_vqc_example():
    """Complete VQC example using PennyLane backend."""
    
    print("PennyLane VQC Example")
    print("=" * 25)
    
    # Generate dataset
    X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create VQC with PennyLane backend
    vqc = sqx.VariationalQuantumClassifier(
        n_qubits=2,
        n_layers=3,
        backend='pennylane',
        optimizer='adam'
    )
    
    # Configure and train
    vqc.compile(
        optimizer='adam',
        learning_rate=0.01,
        loss='binary_crossentropy'
    )
    
    print("Training VQC with PennyLane...")
    history = vqc.fit(
        X_train, y_train,
        epochs=50,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate
    train_accuracy = vqc.score(X_train, y_train)
    test_accuracy = vqc.score(X_test, y_test)
    
    print(f"\nResults:")
    print(f"Training Accuracy: {train_accuracy:.3f}")
    print(f"Test Accuracy: {test_accuracy:.3f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Visualize decision boundary
    plt.subplot(1, 2, 2)
    
    # Create a mesh for plotting
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    try:
        Z = vqc.predict_proba(mesh_points)[:, 1]
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdBu')
        plt.colorbar(label='Probability')
    except:
        print("Skipping decision boundary plot - requires probability prediction")
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='black')
    plt.title(f'VQC Decision Boundary (Acc: {test_accuracy:.2f})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter, label='Class')
    
    plt.tight_layout()
    plt.show()

pennylane_vqc_example()
```

### Quantum Neural Network with PennyLane

```python
def pennylane_qnn_example():
    """Advanced QNN example showcasing PennyLane's capabilities."""
    
    print("PennyLane Quantum Neural Network")
    print("=" * 35)
    
    # Create custom QNN architecture
    class PennyLaneQNN:
        def __init__(self, n_qubits, n_layers, backend='pennylane'):
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            self.backend = sqx.get_backend(backend)
            
            # Initialize parameters
            self.n_params = n_qubits * n_layers * 3  # 3 rotations per layer
            self.params = np.random.random(self.n_params) * 2 * np.pi
            
        def create_variational_circuit(self, inputs, params):
            """Create variational quantum circuit."""
            circuit = self.backend.create_circuit(self.n_qubits)
            
            # Data encoding
            for i, inp in enumerate(inputs):
                if i < self.n_qubits:
                    circuit.ry(np.pi * inp, i)
            
            # Variational layers
            param_idx = 0
            for layer in range(self.n_layers):
                # Rotation gates
                for qubit in range(self.n_qubits):
                    circuit.rx(params[param_idx], qubit)
                    param_idx += 1
                    circuit.ry(params[param_idx], qubit)
                    param_idx += 1
                    circuit.rz(params[param_idx], qubit)
                    param_idx += 1
                
                # Entangling gates
                for qubit in range(self.n_qubits - 1):
                    circuit.cx(qubit, qubit + 1)
                if self.n_qubits > 2:
                    circuit.cx(self.n_qubits - 1, 0)  # Ring connectivity
            
            return circuit
        
        def forward(self, inputs, params=None):
            """Forward pass through QNN."""
            if params is None:
                params = self.params
            
            # Create and run circuit
            circuit = self.create_variational_circuit(inputs, params)
            
            # Measure expectation values
            result = self.measure_expectation(circuit)
            return result
        
        def measure_expectation(self, circuit):
            """Measure expectation value for classification."""
            circuit.measure_all()
            result = self.backend.run(circuit, shots=1000)
            counts = result.get_counts()
            
            # Calculate expectation of first qubit
            expectation = 0
            total_shots = sum(counts.values())
            
            for bitstring, count in counts.items():
                first_bit = int(bitstring[0])
                sign = 1 if first_bit == 0 else -1
                expectation += sign * count / total_shots
            
            return expectation
        
        def cost_function(self, params, X, y):
            """Cost function for training."""
            predictions = []
            for xi in X:
                output = self.forward(xi, params)
                pred = 1 if output > 0 else 0
                predictions.append(pred)
            
            # Binary cross-entropy loss
            predictions = np.array(predictions)
            y = np.array(y)
            
            # Avoid log(0) by adding small epsilon
            epsilon = 1e-15
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            
            loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            return loss
        
        def train(self, X, y, epochs=50, learning_rate=0.1):
            """Train QNN using parameter-shift rule."""
            print(f"Training QNN with {epochs} epochs...")
            
            params = self.params.copy()
            cost_history = []
            
            for epoch in range(epochs):
                # Compute gradients using parameter-shift rule
                gradients = np.zeros_like(params)
                
                for i in range(len(params)):
                    # Forward and backward shifts
                    params_plus = params.copy()
                    params_plus[i] += np.pi / 2
                    cost_plus = self.cost_function(params_plus, X, y)
                    
                    params_minus = params.copy()
                    params_minus[i] -= np.pi / 2
                    cost_minus = self.cost_function(params_minus, X, y)
                    
                    # Parameter-shift rule
                    gradients[i] = (cost_plus - cost_minus) / 2
                
                # Update parameters
                params = params - learning_rate * gradients
                
                # Record cost
                current_cost = self.cost_function(params, X, y)
                cost_history.append(current_cost)
                
                if epoch % 10 == 0:
                    print(f"  Epoch {epoch}: Cost = {current_cost:.4f}")
            
            self.params = params
            return cost_history
        
        def predict(self, X):
            """Make predictions."""
            predictions = []
            for xi in X:
                output = self.forward(xi)
                pred = 1 if output > 0 else 0
                predictions.append(pred)
            return np.array(predictions)
    
    # Test the QNN
    from sklearn.datasets import make_classification
    
    # Generate dataset
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, 
                               n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize features to [-1, 1]
    X_train = 2 * (X_train - X_train.min()) / (X_train.max() - X_train.min()) - 1
    X_test = 2 * (X_test - X_test.min()) / (X_test.max() - X_test.min()) - 1
    
    # Create and train QNN
    qnn = PennyLaneQNN(n_qubits=4, n_layers=2)
    
    cost_history = qnn.train(X_train, y_train, epochs=30, learning_rate=0.3)
    
    # Evaluate
    train_preds = qnn.predict(X_train)
    test_preds = qnn.predict(X_test)
    
    train_acc = np.mean(train_preds == y_train)
    test_acc = np.mean(test_preds == y_test)
    
    print(f"\nQNN Results:")
    print(f"Training Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy: {test_acc:.3f}")
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(cost_history, 'b-', linewidth=2)
    plt.title('QNN Training Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, test_preds)
    
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.show()

pennylane_qnn_example()
```

### Quantum Transfer Learning

```python
def pennylane_transfer_learning():
    """Quantum transfer learning with PennyLane."""
    
    print("Quantum Transfer Learning")
    print("=" * 28)
    
    class QuantumTransferLearning:
        def __init__(self, pretrained_params=None):
            self.backend = sqx.get_backend('pennylane')
            self.n_qubits = 4
            self.n_layers = 2
            
            # Use pretrained parameters or initialize randomly
            if pretrained_params is not None:
                self.params = pretrained_params
            else:
                self.params = np.random.random(self.n_qubits * self.n_layers * 2) * 2 * np.pi
            
            # Freeze some parameters for transfer learning
            self.freeze_mask = np.ones_like(self.params, dtype=bool)
            self.freeze_mask[:len(self.params)//2] = False  # Freeze first half
        
        def create_feature_extractor(self, inputs, params):
            """Pretrained feature extraction layers."""
            circuit = self.backend.create_circuit(self.n_qubits)
            
            # Data encoding
            for i, inp in enumerate(inputs[:self.n_qubits]):
                circuit.ry(np.pi * inp, i)
            
            # Pretrained layers (frozen)
            param_idx = 0
            for layer in range(self.n_layers // 2):
                for qubit in range(self.n_qubits):
                    circuit.ry(params[param_idx], qubit)
                    param_idx += 1
                    circuit.rz(params[param_idx], qubit)
                    param_idx += 1
                
                # Entangling
                for qubit in range(self.n_qubits - 1):
                    circuit.cx(qubit, qubit + 1)
            
            return circuit, param_idx
        
        def create_classifier_head(self, circuit, params, param_idx):
            """Trainable classifier head."""
            # Additional trainable layers
            for layer in range(self.n_layers // 2):
                for qubit in range(self.n_qubits):
                    circuit.ry(params[param_idx], qubit)
                    param_idx += 1
                    circuit.rz(params[param_idx], qubit)
                    param_idx += 1
                
                # Entangling
                for qubit in range(self.n_qubits - 1):
                    circuit.cx(qubit, qubit + 1)
            
            return circuit
        
        def forward(self, inputs, params=None):
            if params is None:
                params = self.params
            
            # Feature extraction
            circuit, param_idx = self.create_feature_extractor(inputs, params)
            
            # Classification head
            circuit = self.create_classifier_head(circuit, params, param_idx)
            
            # Measure
            circuit.measure_all()
            result = self.backend.run(circuit, shots=1000)
            counts = result.get_counts()
            
            # Expectation value
            expectation = 0
            total = sum(counts.values())
            for bitstring, count in counts.items():
                parity = sum(int(bit) for bit in bitstring) % 2
                sign = 1 if parity == 0 else -1
                expectation += sign * count / total
            
            return expectation
        
        def fine_tune(self, X, y, epochs=20, learning_rate=0.1):
            """Fine-tune only unfrozen parameters."""
            print(f"Fine-tuning with {np.sum(self.freeze_mask)} trainable parameters...")
            
            params = self.params.copy()
            cost_history = []
            
            for epoch in range(epochs):
                gradients = np.zeros_like(params)
                
                # Only compute gradients for unfrozen parameters
                for i in range(len(params)):
                    if self.freeze_mask[i]:  # Only update unfrozen params
                        params_plus = params.copy()
                        params_plus[i] += np.pi / 2
                        
                        params_minus = params.copy()
                        params_minus[i] -= np.pi / 2
                        
                        # Compute cost difference
                        cost_plus = self.compute_cost(X, y, params_plus)
                        cost_minus = self.compute_cost(X, y, params_minus)
                        
                        gradients[i] = (cost_plus - cost_minus) / 2
                
                # Update only unfrozen parameters
                params[self.freeze_mask] -= learning_rate * gradients[self.freeze_mask]
                
                current_cost = self.compute_cost(X, y, params)
                cost_history.append(current_cost)
                
                if epoch % 5 == 0:
                    print(f"  Epoch {epoch}: Cost = {current_cost:.4f}")
            
            self.params = params
            return cost_history
        
        def compute_cost(self, X, y, params):
            """Compute classification cost."""
            total_cost = 0
            for xi, yi in zip(X, y):
                output = self.forward(xi, params)
                pred = 1 if output > 0 else 0
                cost = (pred - yi) ** 2  # Squared error
                total_cost += cost
            return total_cost / len(X)
        
        def predict(self, X):
            predictions = []
            for xi in X:
                output = self.forward(xi)
                pred = 1 if output > 0 else 0
                predictions.append(pred)
            return np.array(predictions)
    
    # Simulate pretrained model
    print("Step 1: Creating pretrained model...")
    pretrained_params = np.random.random(16) * 2 * np.pi
    print("Pretrained model created (simulated)")
    
    # Create transfer learning model
    transfer_model = QuantumTransferLearning(pretrained_params)
    
    # Generate new task data
    print("\nStep 2: Generating new task data...")
    X_new, y_new = make_classification(n_samples=50, n_features=4, n_classes=2, random_state=123)
    X_new = 2 * (X_new - X_new.min()) / (X_new.max() - X_new.min()) - 1
    
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
        X_new, y_new, test_size=0.3, random_state=42
    )
    
    # Fine-tune on new task
    print("Step 3: Fine-tuning on new task...")
    fine_tune_history = transfer_model.fine_tune(X_train_new, y_train_new, epochs=25)
    
    # Evaluate
    train_preds = transfer_model.predict(X_train_new)
    test_preds = transfer_model.predict(X_test_new)
    
    train_acc = np.mean(train_preds == y_train_new)
    test_acc = np.mean(test_preds == y_test_new)
    
    print(f"\nTransfer Learning Results:")
    print(f"Training Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy: {test_acc:.3f}")
    
    # Compare with training from scratch
    print("\nStep 4: Comparing with training from scratch...")
    scratch_model = QuantumTransferLearning()  # No pretrained params
    scratch_model.freeze_mask[:] = True  # Train all parameters
    
    scratch_history = scratch_model.fine_tune(X_train_new, y_train_new, epochs=25)
    scratch_preds = scratch_model.predict(X_test_new)
    scratch_acc = np.mean(scratch_preds == y_test_new)
    
    print(f"From Scratch Accuracy: {scratch_acc:.3f}")
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fine_tune_history, 'b-', label='Transfer Learning', linewidth=2)
    plt.plot(scratch_history, 'r--', label='From Scratch', linewidth=2)
    plt.title('Training Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    methods = ['Transfer Learning', 'From Scratch']
    accuracies = [test_acc, scratch_acc]
    colors = ['blue', 'red']
    
    bars = plt.bar(methods, accuracies, color=colors, alpha=0.7)
    plt.title('Test Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

pennylane_transfer_learning()
```

## ⚡ Hardware Integration

### IBM Quantum via PennyLane

```python
def pennylane_ibm_integration():
    """Use IBM Quantum hardware through PennyLane."""
    
    print("PennyLane IBM Quantum Integration")
    print("=" * 37)
    
    try:
        # Note: Requires IBM Quantum account setup
        backend = sqx.PennyLaneBackend(
            device='qiskit.ibmq',
            backend_name='ibmq_qasm_simulator',  # Use simulator for demo
            shots=1024
        )
        
        print(f"Connected to: {backend.device}")
        
        # Create a simple Bell state circuit
        circuit = backend.create_circuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        
        # Execute on IBM backend
        print("Executing on IBM Quantum backend...")
        result = backend.run(circuit, shots=1024)
        
        counts = result.get_counts()
        print(f"Bell state results: {counts}")
        
        # Verify entanglement
        prob_00 = counts.get('00', 0) / 1024
        prob_11 = counts.get('11', 0) / 1024
        prob_entangled = prob_00 + prob_11
        
        print(f"Entanglement probability: {prob_entangled:.3f}")
        
        # Visualize results
        plt.figure(figsize=(8, 6))
        states = list(counts.keys())
        probs = [counts[state] / 1024 for state in states]
        
        plt.bar(states, probs, color=['blue', 'red', 'green', 'orange'][:len(states)])
        plt.title('Bell State Results on IBM Quantum')
        plt.xlabel('Measurement Outcome')
        plt.ylabel('Probability')
        plt.ylim(0, 0.6)
        
        # Add theoretical expectation
        plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, 
                   label='Theoretical (0.5 each)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    except Exception as e:
        print(f"IBM Quantum integration requires setup: {e}")
        print("Please configure IBM Quantum credentials to use real hardware")

# Note: Uncomment to run with proper IBM Quantum setup
# pennylane_ibm_integration()
```

### Google Cirq via PennyLane

```python
def pennylane_cirq_integration():
    """Use Google Cirq simulator through PennyLane."""
    
    print("PennyLane Cirq Integration")
    print("=" * 29)
    
    try:
        # Create Cirq backend through PennyLane
        backend = sqx.PennyLaneBackend(
            device='cirq.simulator',
            shots=1000
        )
        
        print(f"Backend: {backend.device}")
        
        # Create quantum random walk circuit
        n_qubits = 4
        circuit = backend.create_circuit(n_qubits)
        
        # Initialize superposition
        for i in range(n_qubits):
            circuit.h(i)
        
        # Add controlled rotations (quantum walk steps)
        for step in range(3):
            for i in range(n_qubits - 1):
                angle = np.pi / (step + 2)
                circuit.cry(angle, i, i + 1)
        
        circuit.measure_all()
        
        # Execute quantum walk
        print("Executing quantum walk on Cirq...")
        result = backend.run(circuit, shots=1000)
        
        counts = result.get_counts()
        print(f"Quantum walk distribution: {dict(list(counts.items())[:5])}...")
        
        # Analyze distribution
        bitstrings = list(counts.keys())
        probabilities = [counts[bs] / 1000 for bs in bitstrings]
        
        # Convert to position probabilities
        positions = [int(bs, 2) for bs in bitstrings]
        position_probs = {}
        
        for pos, prob in zip(positions, probabilities):
            if pos not in position_probs:
                position_probs[pos] = 0
            position_probs[pos] += prob
        
        # Plot quantum walk distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sorted_positions = sorted(position_probs.keys())
        probs = [position_probs[pos] for pos in sorted_positions]
        
        plt.bar(sorted_positions, probs, alpha=0.7, color='green')
        plt.title('Quantum Walk Position Distribution')
        plt.xlabel('Position')
        plt.ylabel('Probability')
        plt.grid(True, alpha=0.3)
        
        # Compare with classical random walk
        plt.subplot(1, 2, 2)
        # Classical random walk would be more uniform
        classical_prob = 1 / len(position_probs)
        classical_probs = [classical_prob] * len(sorted_positions)
        
        width = 0.35
        x = np.arange(len(sorted_positions))
        
        plt.bar(x - width/2, probs, width, label='Quantum Walk', alpha=0.7, color='green')
        plt.bar(x + width/2, classical_probs, width, label='Classical Random', alpha=0.7, color='orange')
        
        plt.title('Quantum vs Classical Walk')
        plt.xlabel('Position')
        plt.ylabel('Probability')
        plt.xticks(x, sorted_positions)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Cirq integration error: {e}")
        print("Make sure pennylane-cirq is installed: pip install pennylane-cirq")

pennylane_cirq_integration()
```

## 🔍 Advanced Features

### Custom PennyLane Operations

```python
def custom_pennylane_operations():
    """Create custom quantum operations with PennyLane."""
    
    print("Custom PennyLane Operations")
    print("=" * 31)
    
    import pennylane as qml
    from pennylane import numpy as pnp
    
    # Custom parametric gate
    class CustomRotation(qml.operation.Operation):
        """Custom rotation gate with multiple parameters."""
        
        num_wires = 1
        num_params = 3
        par_domain = "R"
        
        @staticmethod
        def compute_matrix(theta, phi, lam):
            """Compute the matrix representation."""
            c = pnp.cos(theta / 2)
            s = pnp.sin(theta / 2)
            
            return pnp.array([
                [c, -s * pnp.exp(1j * lam)],
                [s * pnp.exp(1j * phi), c * pnp.exp(1j * (phi + lam))]
            ])
        
        @staticmethod
        def compute_decomposition(theta, phi, lam, wires):
            """Decompose into basic gates."""
            return [
                qml.RZ(phi, wires=wires),
                qml.RY(theta, wires=wires),
                qml.RZ(lam, wires=wires)
            ]
    
    # Use custom operation in circuit
    dev = qml.device('default.qubit', wires=2, shots=1000)
    
    @qml.qnode(dev)
    def custom_circuit(params):
        # Use custom rotation
        CustomRotation(params[0], params[1], params[2], wires=0)
        
        # Standard gates
        qml.Hadamard(wires=1)
        qml.CNOT(wires=[0, 1])
        
        # Custom two-qubit operation
        qml.ctrl(CustomRotation, control=1)(params[3], params[4], params[5], wires=0)
        
        return qml.probs(wires=[0, 1])
    
    # Test custom circuit
    params = pnp.random.random(6) * 2 * pnp.pi
    probabilities = custom_circuit(params)
    
    print("Custom Circuit Probabilities:")
    states = ['00', '01', '10', '11']
    for state, prob in zip(states, probabilities):
        print(f"  |{state}⟩: {prob:.3f}")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(states, probabilities, color=['red', 'blue', 'green', 'orange'])
    plt.title('Custom Circuit Output')
    plt.xlabel('Quantum State')
    plt.ylabel('Probability')
    plt.ylim(0, max(probabilities) * 1.1)
    
    # Show parameter sensitivity
    plt.subplot(1, 2, 2)
    param_range = np.linspace(0, 2*np.pi, 20)
    prob_evolution = []
    
    for theta in param_range:
        test_params = params.copy()
        test_params[0] = theta  # Vary first parameter
        probs = custom_circuit(test_params)
        prob_evolution.append(probs[0])  # Track |00⟩ probability
    
    plt.plot(param_range, prob_evolution, 'b-', linewidth=2)
    plt.title('Parameter Sensitivity')
    plt.xlabel('Parameter Value (θ)')
    plt.ylabel('P(|00⟩)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

custom_pennylane_operations()
```

### Gradient-Based Optimization

```python
def pennylane_gradient_optimization():
    """Advanced gradient-based optimization with PennyLane."""
    
    print("PennyLane Gradient Optimization")
    print("=" * 35)
    
    import pennylane as qml
    from pennylane import numpy as pnp
    
    # Create device
    dev = qml.device('default.qubit', wires=3, shots=1000)
    
    @qml.qnode(dev, diff_method="parameter-shift")
    def variational_circuit(params, inputs=None):
        """Variational quantum circuit with gradients."""
        
        # Data encoding layer
        if inputs is not None:
            for i, inp in enumerate(inputs):
                if i < 3:
                    qml.RY(pnp.pi * inp, wires=i)
        
        # Variational layers
        layers = 3
        for layer in range(layers):
            # Rotation layer
            for wire in range(3):
                qml.RY(params[layer * 6 + wire * 2], wires=wire)
                qml.RZ(params[layer * 6 + wire * 2 + 1], wires=wire)
            
            # Entangling layer
            if layer < layers - 1:
                for wire in range(2):
                    qml.CNOT(wires=[wire, wire + 1])
                qml.CNOT(wires=[2, 0])
        
        # Measurement
        return qml.expval(qml.PauliZ(0))
    
    # Cost function for optimization
    def cost_function(params, X, y):
        """Variational quantum classifier cost."""
        predictions = []
        for xi in X:
            output = variational_circuit(params, xi)
            pred = 1 if output > 0 else 0
            predictions.append(pred)
        
        predictions = pnp.array(predictions)
        
        # Mean squared error
        return pnp.mean((predictions - y) ** 2)
    
    # Generate training data
    pnp.random.seed(42)
    n_samples = 50
    X_train = pnp.random.uniform(-1, 1, (n_samples, 3))
    y_train = pnp.array([1 if sum(x) > 0 else 0 for x in X_train])
    
    print(f"Training data: {n_samples} samples, 3 features")
    print(f"Class distribution: {pnp.sum(y_train)} positive, {len(y_train) - pnp.sum(y_train)} negative")
    
    # Initialize parameters
    n_params = 3 * 6  # 3 layers * 6 params per layer
    params = pnp.random.uniform(0, 2*pnp.pi, n_params, requires_grad=True)
    
    # Optimization with different optimizers
    optimizers = {
        'GradientDescent': qml.GradientDescentOptimizer(stepsize=0.1),
        'Adam': qml.AdamOptimizer(stepsize=0.1),
        'RMSprop': qml.RMSPropOptimizer(stepsize=0.1)
    }
    
    results = {}
    
    for opt_name, optimizer in optimizers.items():
        print(f"\nOptimizing with {opt_name}...")
        
        current_params = params.copy()
        cost_history = []
        
        for epoch in range(50):
            # Compute cost and gradients
            cost = cost_function(current_params, X_train, y_train)
            cost_history.append(cost)
            
            # Update parameters
            current_params = optimizer.step(
                lambda p: cost_function(p, X_train, y_train), 
                current_params
            )
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Cost = {cost:.4f}")
        
        # Final evaluation
        final_cost = cost_function(current_params, X_train, y_train)
        
        # Test accuracy
        test_predictions = []
        for xi in X_train:
            output = variational_circuit(current_params, xi)
            pred = 1 if output > 0 else 0
            test_predictions.append(pred)
        
        accuracy = pnp.mean(pnp.array(test_predictions) == y_train)
        
        results[opt_name] = {
            'cost_history': cost_history,
            'final_cost': final_cost,
            'accuracy': accuracy,
            'params': current_params
        }
        
        print(f"  Final cost: {final_cost:.4f}")
        print(f"  Accuracy: {accuracy:.3f}")
    
    # Visualize optimization comparison
    plt.figure(figsize=(15, 10))
    
    # Cost evolution
    plt.subplot(2, 3, 1)
    for opt_name, result in results.items():
        plt.plot(result['cost_history'], label=opt_name, linewidth=2)
    
    plt.title('Cost Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # Final performance comparison
    plt.subplot(2, 3, 2)
    opt_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in opt_names]
    
    bars = plt.bar(opt_names, accuracies, color=['blue', 'red', 'green'])
    plt.title('Final Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Parameter visualization for best optimizer
    best_optimizer = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_params = results[best_optimizer]['params']
    
    plt.subplot(2, 3, 3)
    plt.hist(best_params, bins=15, alpha=0.7, color='green')
    plt.title(f'Best Parameters ({best_optimizer})')
    plt.xlabel('Parameter Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Gradient analysis
    plt.subplot(2, 3, 4)
    
    # Compute gradients at final parameters
    grad_fn = qml.grad(lambda p: cost_function(p, X_train, y_train))
    final_gradients = grad_fn(best_params)
    
    plt.plot(final_gradients, 'o-', color='red', linewidth=2)
    plt.title('Final Gradient Magnitudes')
    plt.xlabel('Parameter Index')
    plt.ylabel('Gradient')
    plt.grid(True)
    
    # Learning curve comparison
    plt.subplot(2, 3, 5)
    for opt_name, result in results.items():
        smoothed_cost = pnp.convolve(result['cost_history'], pnp.ones(5)/5, mode='valid')
        plt.plot(smoothed_cost, label=f'{opt_name} (smoothed)', linewidth=2)
    
    plt.title('Smoothed Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    
    # Parameter evolution (for best optimizer)
    plt.subplot(2, 3, 6)
    
    # Track a few parameters during optimization
    param_evolution = []
    test_params = params.copy()
    optimizer = optimizers[best_optimizer]
    
    for epoch in range(30):  # Shorter for visualization
        param_evolution.append([test_params[0], test_params[5], test_params[10]])
        test_params = optimizer.step(
            lambda p: cost_function(p, X_train, y_train), 
            test_params
        )
    
    param_evolution = pnp.array(param_evolution)
    
    for i, label in enumerate(['Param 0', 'Param 5', 'Param 10']):
        plt.plot(param_evolution[:, i], label=label, linewidth=2)
    
    plt.title(f'Parameter Evolution ({best_optimizer})')
    plt.xlabel('Epoch')
    plt.ylabel('Parameter Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nBest optimizer: {best_optimizer}")
    print(f"Best accuracy: {results[best_optimizer]['accuracy']:.3f}")

pennylane_gradient_optimization()
```

## 📈 Performance Tips

### Optimization Strategies

```python
def pennylane_performance_tips():
    """Performance optimization strategies for PennyLane."""
    
    print("PennyLane Performance Optimization")
    print("=" * 38)
    
    import time
    
    # 1. Device Selection Impact
    devices = [
        ('default.qubit', 'CPU simulator'),
        ('lightning.qubit', 'C++ simulator'),
        # ('qiskit.aer', 'Qiskit Aer'),  # If available
    ]
    
    n_qubits = 8
    n_layers = 3
    shots = 1000
    
    performance_results = {}
    
    for device_name, description in devices:
        try:
            print(f"\nTesting {device_name} ({description})...")
            
            # Create device
            if device_name == 'lightning.qubit':
                try:
                    backend = sqx.PennyLaneBackend(device=device_name, shots=shots)
                except:
                    print(f"  {device_name} not available, skipping...")
                    continue
            else:
                backend = sqx.PennyLaneBackend(device=device_name, shots=shots)
            
            # Create test circuit
            circuit = backend.create_circuit(n_qubits)
            
            # Add parameterized layers
            params = np.random.random(n_qubits * n_layers * 2)
            param_idx = 0
            
            for layer in range(n_layers):
                for qubit in range(n_qubits):
                    circuit.ry(params[param_idx], qubit)
                    param_idx += 1
                    circuit.rz(params[param_idx], qubit)
                    param_idx += 1
                
                # Entangling layer
                for qubit in range(n_qubits - 1):
                    circuit.cx(qubit, qubit + 1)
            
            circuit.measure_all()
            
            # Time execution
            start_time = time.time()
            result = backend.run(circuit, shots=shots)
            execution_time = time.time() - start_time
            
            performance_results[device_name] = {
                'execution_time': execution_time,
                'description': description,
                'counts': result.get_counts()
            }
            
            print(f"  Execution time: {execution_time:.3f}s")
            
        except Exception as e:
            print(f"  Error with {device_name}: {e}")
    
    # 2. Shot Count vs Accuracy Trade-off
    print(f"\n--- Shot Count Analysis ---")
    
    backend = sqx.PennyLaneBackend(device='default.qubit')
    
    # Simple Bell state circuit
    bell_circuit = backend.create_circuit(2)
    bell_circuit.h(0)
    bell_circuit.cx(0, 1)
    bell_circuit.measure_all()
    
    shot_counts = [100, 500, 1000, 2000, 5000]
    shot_results = {}
    
    # Theoretical expectation: 50% |00⟩, 50% |11⟩
    theoretical_prob = 0.5
    
    for shots in shot_counts:
        start_time = time.time()
        result = backend.run(bell_circuit, shots=shots)
        exec_time = time.time() - start_time
        
        counts = result.get_counts()
        prob_00 = counts.get('00', 0) / shots
        error = abs(prob_00 - theoretical_prob)
        
        shot_results[shots] = {
            'execution_time': exec_time,
            'prob_00': prob_00,
            'error': error
        }
        
        print(f"  {shots:4d} shots: P(00)={prob_00:.3f}, Error={error:.3f}, Time={exec_time:.3f}s")
    
    # 3. Circuit Depth Impact
    print(f"\n--- Circuit Depth Analysis ---")
    
    depths = [1, 3, 5, 10, 15]
    depth_results = {}
    
    for depth in depths:
        circuit = backend.create_circuit(4)
        
        # Create circuit with specified depth
        params = np.random.random(4 * depth * 2)
        param_idx = 0
        
        for layer in range(depth):
            for qubit in range(4):
                circuit.ry(params[param_idx], qubit)
                param_idx += 1
                circuit.rz(params[param_idx], qubit)
                param_idx += 1
            
            # Entangling
            for qubit in range(3):
                circuit.cx(qubit, qubit + 1)
        
        circuit.measure_all()
        
        start_time = time.time()
        result = backend.run(circuit, shots=1000)
        exec_time = time.time() - start_time
        
        depth_results[depth] = {
            'execution_time': exec_time,
            'n_params': len(params)
        }
        
        print(f"  Depth {depth:2d}: {len(params):2d} params, Time={exec_time:.3f}s")
    
    # Visualize performance results
    plt.figure(figsize=(15, 10))
    
    # Device performance comparison
    if len(performance_results) > 1:
        plt.subplot(2, 3, 1)
        devices = list(performance_results.keys())
        times = [performance_results[d]['execution_time'] for d in devices]
        
        bars = plt.bar(devices, times, color=['blue', 'red', 'green'][:len(devices)])
        plt.title('Device Performance Comparison')
        plt.ylabel('Execution Time (s)')
        plt.xticks(rotation=45)
        
        for bar, time_val in zip(bars, times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{time_val:.3f}s', ha='center', va='bottom')
    
    # Shot count vs accuracy
    plt.subplot(2, 3, 2)
    shots_list = list(shot_results.keys())
    errors = [shot_results[s]['error'] for s in shots_list]
    
    plt.loglog(shots_list, errors, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Number of Shots')
    plt.ylabel('Error')
    plt.title('Accuracy vs Shot Count')
    plt.grid(True)
    
    # Expected 1/sqrt(N) scaling
    theoretical_errors = [0.5 / np.sqrt(s) for s in shots_list]
    plt.loglog(shots_list, theoretical_errors, '--', alpha=0.7, 
              label='1/√N scaling', color='red')
    plt.legend()
    
    # Shot count vs time
    plt.subplot(2, 3, 3)
    times = [shot_results[s]['execution_time'] for s in shots_list]
    
    plt.plot(shots_list, times, 'o-', linewidth=2, color='green')
    plt.xlabel('Number of Shots')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time vs Shots')
    plt.grid(True)
    
    # Circuit depth impact
    plt.subplot(2, 3, 4)
    depth_list = list(depth_results.keys())
    depth_times = [depth_results[d]['execution_time'] for d in depth_list]
    
    plt.semilogy(depth_list, depth_times, 'o-', linewidth=2, color='purple')
    plt.xlabel('Circuit Depth')
    plt.ylabel('Execution Time (s)')
    plt.title('Depth vs Performance')
    plt.grid(True)
    
    # Parameter count impact
    plt.subplot(2, 3, 5)
    param_counts = [depth_results[d]['n_params'] for d in depth_list]
    
    plt.plot(param_counts, depth_times, 'o-', linewidth=2, color='orange')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Execution Time (s)')
    plt.title('Parameters vs Performance')
    plt.grid(True)
    
    # Performance recommendations
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.9, "Performance Recommendations:", fontsize=14, fontweight='bold', 
             transform=plt.gca().transAxes)
    
    recommendations = [
        "1. Use lightning.qubit for large circuits",
        "2. Balance shots vs accuracy needs",
        "3. Minimize circuit depth when possible",
        "4. Consider parameter-shift vs finite-diff",
        "5. Use analytic=True for noiseless sims",
        "6. Batch circuits for efficiency",
        "7. Profile different devices for your use case"
    ]
    
    for i, rec in enumerate(recommendations):
        plt.text(0.1, 0.8 - i*0.1, rec, fontsize=10, 
                transform=plt.gca().transAxes)
    
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Performance summary
    print(f"\n=== Performance Summary ===")
    if len(performance_results) > 0:
        fastest_device = min(performance_results.keys(), 
                           key=lambda x: performance_results[x]['execution_time'])
        print(f"Fastest device: {fastest_device}")
    
    optimal_shots = min(shot_results.keys(), 
                       key=lambda x: shot_results[x]['error'] * shot_results[x]['execution_time'])
    print(f"Optimal shot count (error × time): {optimal_shots}")
    
    efficient_depth = min(depth_results.keys(),
                         key=lambda x: depth_results[x]['execution_time'])
    print(f"Most efficient depth tested: {efficient_depth}")

pennylane_performance_tips()
```

## 🔧 Best Practices

### Code Organization

```python
# pennylane_utils.py
import pennylane as qml
import superquantx as sqx
from pennylane import numpy as pnp

class PennyLaneUtils:
    """Utility functions for PennyLane backend."""
    
    @staticmethod
    def create_optimized_device(n_qubits, shots=1024, device_type='auto'):
        """Create optimized PennyLane device."""
        if device_type == 'auto':
            if n_qubits <= 10:
                device_type = 'lightning.qubit'
            else:
                device_type = 'default.qubit'
        
        try:
            return sqx.PennyLaneBackend(device=device_type, shots=shots)
        except:
            return sqx.PennyLaneBackend(device='default.qubit', shots=shots)
    
    @staticmethod
    def parameter_shift_gradients(circuit_func, params, shift=pnp.pi/2):
        """Compute gradients using parameter-shift rule."""
        gradients = pnp.zeros_like(params)
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += shift
            
            params_minus = params.copy()  
            params_minus[i] -= shift
            
            grad = (circuit_func(params_plus) - circuit_func(params_minus)) / 2
            gradients[i] = grad
            
        return gradients
    
    @staticmethod
    def create_data_encoding_layer(circuit, data, encoding_type='amplitude'):
        """Add data encoding layer to circuit."""
        if encoding_type == 'amplitude':
            for i, val in enumerate(data):
                if i < circuit.num_qubits:
                    circuit.ry(pnp.pi * val, i)
        
        elif encoding_type == 'angle':
            for i, val in enumerate(data):
                if i < circuit.num_qubits:
                    circuit.rx(val, i)
                    circuit.rz(val, i)
        
        return circuit

# Usage example
utils = PennyLaneUtils()
backend = utils.create_optimized_device(n_qubits=4)
```

### Error Handling

```python
def robust_pennylane_execution():
    """Robust PennyLane execution with error handling."""
    
    def safe_circuit_execution(circuit_func, max_retries=3):
        """Execute circuit with retry logic."""
        for attempt in range(max_retries):
            try:
                result = circuit_func()
                return result
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                else:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    # Reduce shots or switch device on retry
                    if hasattr(circuit_func, '__self__'):
                        circuit_func.__self__.shots = max(100, 
                            circuit_func.__self__.shots // 2)
    
    # Example usage
    backend = sqx.get_backend('pennylane')
    circuit = backend.create_circuit(2)
    circuit.h(0)
    circuit.cx(0, 1) 
    circuit.measure_all()
    
    result = safe_circuit_execution(lambda: backend.run(circuit, shots=1000))
    print(f"Robust execution result: {result.get_counts()}")

robust_pennylane_execution()
```

## 📚 Resources

### Official Documentation
- **PennyLane Documentation**: https://pennylane.ai/
- **PennyLane Tutorials**: https://pennylane.ai/qml/
- **API Reference**: https://pennylane.readthedocs.io/

### SuperQuantX Integration
- **Backend Configuration**: See `backends/` documentation
- **Quantum ML Examples**: See `tutorials/quantum-ml.md`
- **API Reference**: See `api/core.md`

### Hardware Integration
- **IBM Quantum**: Requires IBM Quantum account
- **Google Cirq**: Available through PennyLane-Cirq plugin
- **AWS Braket**: Available through PennyLane-Braket plugin

---

*PennyLane's focus on differentiable programming and quantum machine learning makes it an ideal choice for variational quantum algorithms and hybrid quantum-classical optimization. The SuperQuantX integration provides a unified interface while preserving PennyLane's unique capabilities.*

!!! tip "Optimization"
    For best performance, use `lightning.qubit` device for circuits with ≤20 qubits, and consider the parameter-shift rule for gradient computation.

!!! warning "Hardware Limitations"
    When using real quantum hardware through PennyLane, account for device-specific constraints, noise, and queue times.

!!! info "Compatibility"
    PennyLane integrates well with PyTorch, TensorFlow, and JAX for hybrid quantum-classical machine learning workflows.