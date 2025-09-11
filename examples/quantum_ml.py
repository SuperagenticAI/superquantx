"""
Quantum Machine Learning examples using SuperQuantX

This example demonstrates:
- Quantum feature maps and data encoding
- Quantum classifiers and regressors
- Quantum kernels and SVM
- Quantum neural networks
- Performance comparison with classical methods
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification, make_moons, make_regression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from superquantx import SuperQuantXClient
from superquantx.ml import (
    AmplitudeEmbeddingFeatureMap,
    AngleEmbeddingFeatureMap,
    IQPFeatureMap,
    QuantumClassifier,
    QuantumKernel,
    QuantumNeuralNetwork,
    QuantumRegressor,
    QuantumSVM,
)


def generate_quantum_datasets():
    """Generate datasets suitable for quantum machine learning"""
    print("Generating quantum ML datasets...")

    # Dataset 1: Binary classification (moons)
    X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)

    # Dataset 2: Multi-class classification
    X_class, y_class = make_classification(
        n_samples=300, n_features=4, n_redundant=0,
        n_informative=4, n_classes=3, random_state=42
    )

    # Dataset 3: Regression
    X_reg, y_reg = make_regression(
        n_samples=200, n_features=3, noise=0.1, random_state=42
    )

    # Scale features to [0, 2π] for quantum encoding
    scaler_moons = StandardScaler()
    X_moons_scaled = scaler_moons.fit_transform(X_moons) * np.pi + np.pi

    scaler_class = StandardScaler()
    X_class_scaled = scaler_class.fit_transform(X_class) * np.pi + np.pi

    scaler_reg = StandardScaler()
    X_reg_scaled = scaler_reg.fit_transform(X_reg) * np.pi + np.pi
    y_reg_scaled = StandardScaler().fit_transform(y_reg.reshape(-1, 1)).ravel()

    datasets = {
        'binary_classification': {
            'X': X_moons_scaled, 'y': y_moons,
            'name': 'Moons Dataset (Binary Classification)'
        },
        'multi_classification': {
            'X': X_class_scaled, 'y': y_class,
            'name': '4D Dataset (Multi-class Classification)'
        },
        'regression': {
            'X': X_reg_scaled, 'y': y_reg_scaled,
            'name': '3D Dataset (Regression)'
        }
    }

    for key, data in datasets.items():
        print(f"  {data['name']}: {data['X'].shape[0]} samples, {data['X'].shape[1]} features")

    return datasets


def demonstrate_feature_maps():
    """Demonstrate different quantum feature maps"""
    print("\nDemonstrating quantum feature maps...")

    # Sample data point
    x = np.array([0.5, 1.2, 2.1, 0.8])
    num_qubits = 4

    feature_maps = {
        'Angle Embedding': AngleEmbeddingFeatureMap(num_qubits, 4, ['RY']),
        'Amplitude Embedding': AmplitudeEmbeddingFeatureMap(num_qubits, 4),
        'IQP Feature Map': IQPFeatureMap(num_qubits, 4, degree=2)
    }

    print(f"Input data: {x}")
    print(f"Number of qubits: {num_qubits}")

    for name, feature_map in feature_maps.items():
        print(f"\n{name}:")
        circuit = feature_map.map_features(x)
        print(f"  Circuit gates: {len(circuit.gates)}")
        print(f"  Gate types: {set(gate.name for gate in circuit.gates)}")

        # Show first few gates
        for i, gate in enumerate(circuit.gates[:3]):
            params = f"({gate.parameters[0]:.3f})" if gate.parameters else ""
            print(f"    Gate {i+1}: {gate.name}{params} on qubit(s) {gate.qubits}")

        if len(circuit.gates) > 3:
            print(f"    ... and {len(circuit.gates) - 3} more gates")

    return feature_maps


def quantum_classification_example(datasets, client=None):
    """Demonstrate quantum classification"""
    print("\n" + "="*50)
    print("QUANTUM CLASSIFICATION EXAMPLE")
    print("="*50)

    # Use binary classification dataset
    data = datasets['binary_classification']
    X, y = data['X'], data['y']

    print(f"\nDataset: {data['name']}")
    print(f"Samples: {len(X)}, Features: {X.shape[1]}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Test different quantum classifiers
    classifiers = {}

    # 1. Quantum Classifier with Angle Embedding
    print("\n1. Quantum Classifier (Angle Embedding)")
    print("-" * 40)

    try:
        angle_feature_map = AngleEmbeddingFeatureMap(
            num_qubits=min(4, X.shape[1]),
            num_features=X.shape[1],
            rotation_gates=['RY', 'RZ']
        )

        qc_angle = QuantumClassifier(
            feature_map=angle_feature_map,
            ansatz_layers=2,
            client=client,
            max_iter=100
        )

        print("Training quantum classifier...")
        qc_angle.fit(X_train, y_train)

        print("Making predictions...")
        y_pred_qc = qc_angle.predict(X_test)
        accuracy_qc = accuracy_score(y_test, y_pred_qc)

        classifiers['Quantum (Angle)'] = {
            'model': qc_angle,
            'accuracy': accuracy_qc,
            'predictions': y_pred_qc
        }

        print(f"Quantum Classifier Accuracy: {accuracy_qc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred_qc, target_names=['Class 0', 'Class 1']))

    except Exception as e:
        print(f"Error with Quantum Classifier: {e}")
        classifiers['Quantum (Angle)'] = {'accuracy': 0.0, 'error': str(e)}

    # 2. Quantum SVM
    print("\n2. Quantum SVM")
    print("-" * 15)

    try:
        # Use IQP feature map for kernel
        iqp_feature_map = IQPFeatureMap(num_qubits=min(4, X.shape[1]),
                                       num_features=X.shape[1], degree=2)
        quantum_kernel = QuantumKernel(iqp_feature_map, client=client)

        qsvm = QuantumSVM(quantum_kernel=quantum_kernel, C=1.0)

        print("Training quantum SVM...")
        qsvm.fit(X_train, y_train)

        print("Making predictions...")
        y_pred_qsvm = qsvm.predict(X_test)
        accuracy_qsvm = accuracy_score(y_test, y_pred_qsvm)

        classifiers['Quantum SVM'] = {
            'model': qsvm,
            'accuracy': accuracy_qsvm,
            'predictions': y_pred_qsvm
        }

        print(f"Quantum SVM Accuracy: {accuracy_qsvm:.4f}")

    except Exception as e:
        print(f"Error with Quantum SVM: {e}")
        classifiers['Quantum SVM'] = {'accuracy': 0.0, 'error': str(e)}

    # 3. Classical comparison
    print("\n3. Classical Comparison")
    print("-" * 22)

    # Classical SVM
    classical_svm = SVC(kernel='rbf', C=1.0, random_state=42)
    classical_svm.fit(X_train, y_train)
    y_pred_classical = classical_svm.predict(X_test)
    accuracy_classical = accuracy_score(y_test, y_pred_classical)

    classifiers['Classical SVM'] = {
        'model': classical_svm,
        'accuracy': accuracy_classical,
        'predictions': y_pred_classical
    }

    print(f"Classical SVM Accuracy: {accuracy_classical:.4f}")

    # Classical MLP
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

    classifiers['Classical MLP'] = {
        'model': mlp,
        'accuracy': accuracy_mlp,
        'predictions': y_pred_mlp
    }

    print(f"Classical MLP Accuracy: {accuracy_mlp:.4f}")

    # Summary
    print("\n" + "-"*50)
    print("CLASSIFICATION RESULTS SUMMARY")
    print("-"*50)

    for name, results in classifiers.items():
        if 'error' in results:
            print(f"{name:<20}: ERROR - {results['error']}")
        else:
            print(f"{name:<20}: {results['accuracy']:.4f}")

    return classifiers


def quantum_regression_example(datasets, client=None):
    """Demonstrate quantum regression"""
    print("\n" + "="*50)
    print("QUANTUM REGRESSION EXAMPLE")
    print("="*50)

    # Use regression dataset
    data = datasets['regression']
    X, y = data['X'], data['y']

    print(f"\nDataset: {data['name']}")
    print(f"Samples: {len(X)}, Features: {X.shape[1]}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    regressors = {}

    # 1. Quantum Regressor
    print("\n1. Quantum Regressor")
    print("-" * 20)

    try:
        angle_feature_map = AngleEmbeddingFeatureMap(
            num_qubits=min(4, X.shape[1]),
            num_features=X.shape[1]
        )

        qr = QuantumRegressor(
            feature_map=angle_feature_map,
            ansatz_layers=2,
            client=client
        )

        print("Training quantum regressor...")
        qr.fit(X_train, y_train)

        print("Making predictions...")
        y_pred_qr = qr.predict(X_test)
        mse_qr = mean_squared_error(y_test, y_pred_qr)

        regressors['Quantum Regressor'] = {
            'model': qr,
            'mse': mse_qr,
            'predictions': y_pred_qr
        }

        print(f"Quantum Regressor MSE: {mse_qr:.4f}")

    except Exception as e:
        print(f"Error with Quantum Regressor: {e}")
        regressors['Quantum Regressor'] = {'mse': float('inf'), 'error': str(e)}

    # 2. Classical comparison
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    print("\n2. Classical Comparison")
    print("-" * 22)

    # Linear regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)

    regressors['Linear Regression'] = {
        'model': lr,
        'mse': mse_lr,
        'predictions': y_pred_lr
    }

    print(f"Linear Regression MSE: {mse_lr:.4f}")

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)

    regressors['Random Forest'] = {
        'model': rf,
        'mse': mse_rf,
        'predictions': y_pred_rf
    }

    print(f"Random Forest MSE: {mse_rf:.4f}")

    # Summary
    print("\n" + "-"*50)
    print("REGRESSION RESULTS SUMMARY")
    print("-"*50)

    for name, results in regressors.items():
        if 'error' in results:
            print(f"{name:<20}: ERROR - {results['error']}")
        else:
            print(f"{name:<20}: {results['mse']:.4f}")

    return regressors


def quantum_neural_network_example(datasets, client=None):
    """Demonstrate quantum neural networks"""
    print("\n" + "="*50)
    print("QUANTUM NEURAL NETWORK EXAMPLE")
    print("="*50)

    # Use multi-class dataset
    data = datasets['multi_classification']
    X, y = data['X'], data['y']

    print(f"\nDataset: {data['name']}")
    print(f"Samples: {len(X)}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    networks = {}

    # 1. Quantum Neural Network
    print("\n1. Quantum Neural Network")
    print("-" * 27)

    try:
        qnn = QuantumNeuralNetwork(
            num_qubits=4,
            num_layers=3,
            entangling_gates="CNOT",
            client=client,
            optimizer="SLSQP"
        )

        print("Training quantum neural network...")
        qnn.fit(X_train, y_train)

        print("Making predictions...")
        y_pred_qnn = qnn.predict(X_test)
        accuracy_qnn = accuracy_score(y_test, y_pred_qnn)

        networks['Quantum NN'] = {
            'model': qnn,
            'accuracy': accuracy_qnn,
            'predictions': y_pred_qnn
        }

        print(f"Quantum NN Accuracy: {accuracy_qnn:.4f}")

        # Additional QNN analysis
        print(f"QNN Parameters: {qnn.num_parameters}")
        print(f"QNN Layers: {qnn.num_layers}")
        print(f"QNN Qubits: {qnn.num_qubits}")

    except Exception as e:
        print(f"Error with Quantum Neural Network: {e}")
        networks['Quantum NN'] = {'accuracy': 0.0, 'error': str(e)}

    # 2. Classical comparison
    print("\n2. Classical Neural Network")
    print("-" * 27)

    classical_nn = MLPClassifier(
        hidden_layer_sizes=(20, 10),
        max_iter=1000,
        random_state=42,
        alpha=0.01
    )

    classical_nn.fit(X_train, y_train)
    y_pred_classical_nn = classical_nn.predict(X_test)
    accuracy_classical_nn = accuracy_score(y_test, y_pred_classical_nn)

    networks['Classical NN'] = {
        'model': classical_nn,
        'accuracy': accuracy_classical_nn,
        'predictions': y_pred_classical_nn
    }

    print(f"Classical NN Accuracy: {accuracy_classical_nn:.4f}")
    print(f"Classical NN Parameters: {sum(param.size for param in classical_nn.coefs_ + classical_nn.intercepts_)}")

    # Summary
    print("\n" + "-"*50)
    print("NEURAL NETWORK RESULTS SUMMARY")
    print("-"*50)

    for name, results in networks.items():
        if 'error' in results:
            print(f"{name:<15}: ERROR - {results['error']}")
        else:
            print(f"{name:<15}: {results['accuracy']:.4f}")

    return networks


def visualize_results(classifiers, regressors, save_plots=True):
    """Visualize quantum ML results"""
    print("\n" + "="*50)
    print("VISUALIZATION")
    print("="*50)

    try:
        # Classification results
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Accuracy comparison
        names = []
        accuracies = []

        for name, results in classifiers.items():
            if 'error' not in results:
                names.append(name.replace(' ', '\n'))
                accuracies.append(results['accuracy'])

        axes[0].bar(names, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[0].set_title('Classification Accuracy Comparison')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_ylim(0, 1)

        # Add accuracy values on bars
        for i, acc in enumerate(accuracies):
            axes[0].text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')

        # Regression MSE comparison
        reg_names = []
        reg_mses = []

        for name, results in regressors.items():
            if 'error' not in results and results['mse'] != float('inf'):
                reg_names.append(name.replace(' ', '\n'))
                reg_mses.append(results['mse'])

        if reg_mses:  # Only plot if we have data
            axes[1].bar(reg_names, reg_mses, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[1].set_title('Regression MSE Comparison')
            axes[1].set_ylabel('Mean Squared Error')

            # Add MSE values on bars
            for i, mse in enumerate(reg_mses):
                axes[1].text(i, mse + max(reg_mses) * 0.02, f'{mse:.3f}',
                           ha='center', va='bottom')
        else:
            axes[1].text(0.5, 0.5, 'No regression results\navailable',
                        ha='center', va='center', transform=axes[1].transAxes,
                        fontsize=12)
            axes[1].set_title('Regression Results')

        plt.tight_layout()

        if save_plots:
            plt.savefig('quantum_ml_results.png', dpi=150, bbox_inches='tight')
            print("Results plot saved as 'quantum_ml_results.png'")

        plt.show()

        return fig

    except ImportError:
        print("Matplotlib not available, skipping visualization")
        return None


def benchmark_quantum_advantage():
    """Analyze potential quantum advantage scenarios"""
    print("\n" + "="*50)
    print("QUANTUM ADVANTAGE ANALYSIS")
    print("="*50)

    print("Analyzing scenarios where quantum ML might provide advantages:")

    scenarios = {
        "High-dimensional sparse data": {
            "description": "Quantum feature maps can naturally handle high-dimensional spaces",
            "quantum_benefit": "Exponential scaling of Hilbert space",
            "example": "Amplitude embedding for 2^n dimensional data"
        },
        "Kernel methods": {
            "description": "Quantum kernels can be hard to compute classically",
            "quantum_benefit": "Quantum feature maps create rich kernel functions",
            "example": "IQP circuits for complex feature interactions"
        },
        "Structured data": {
            "description": "Data with natural quantum structure (e.g., molecular data)",
            "quantum_benefit": "Native quantum representation",
            "example": "VQE for molecular property prediction"
        },
        "Small training sets": {
            "description": "Quantum models might generalize better with few samples",
            "quantum_benefit": "Quantum inductive bias",
            "example": "Few-shot learning with quantum neural networks"
        }
    }

    for scenario, details in scenarios.items():
        print(f"\n{scenario}:")
        print(f"  Description: {details['description']}")
        print(f"  Quantum benefit: {details['quantum_benefit']}")
        print(f"  Example: {details['example']}")

    print("\nCurrent limitations:")
    print("- NISQ device noise limits circuit depth")
    print("- Classical simulation overhead")
    print("- Limited quantum hardware access")
    print("- Classical preprocessing still needed")

    print("\nFuture prospects:")
    print("- Fault-tolerant quantum computers")
    print("- Better quantum algorithms")
    print("- Hybrid quantum-classical methods")
    print("- Quantum data sets")


def main():
    """Main quantum ML example function"""
    print("SuperQuantX Quantum Machine Learning Examples")
    print("=" * 45)

    # Setup
    api_key = input("Enter your SuperQuantX API key (or press Enter for simulation): ").strip()
    client = SuperQuantXClient(api_key) if api_key else None

    if client:
        print("Using SuperQuantX quantum backend")
    else:
        print("Using local simulation mode")

    # Generate datasets
    datasets = generate_quantum_datasets()

    # Demonstrate feature maps
    feature_maps = demonstrate_feature_maps()

    # Run examples
    classification_results = quantum_classification_example(datasets, client)
    regression_results = quantum_regression_example(datasets, client)
    network_results = quantum_neural_network_example(datasets, client)

    # Visualize results
    visualize_results(classification_results, regression_results)

    # Analyze quantum advantage
    benchmark_quantum_advantage()

    # Final summary
    print("\n" + "="*60)
    print("QUANTUM MACHINE LEARNING SUMMARY")
    print("="*60)

    print("\nKey takeaways:")
    print("1. Quantum feature maps enable rich data representations")
    print("2. Quantum classifiers can match classical performance")
    print("3. Quantum kernels provide unique computational approaches")
    print("4. Current NISQ limitations require careful algorithm design")
    print("5. Hybrid quantum-classical methods show promise")

    print("\nNext steps:")
    print("- Experiment with different feature maps")
    print("- Try larger, more complex datasets")
    print("- Explore quantum advantage scenarios")
    print("- Test on real quantum hardware")

    print("\nQuantum ML examples completed successfully!")


if __name__ == "__main__":
    main()
