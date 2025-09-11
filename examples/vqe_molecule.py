"""
Variational Quantum Eigensolver (VQE) for molecular ground state calculation

This example demonstrates:
- Setting up VQE for molecular systems
- Defining molecular Hamiltonians
- Creating parameterized ansatz circuits
- Running VQE optimization
- Analyzing results and convergence
"""

import matplotlib.pyplot as plt
import numpy as np

from superquantx import VQE, Hamiltonian, PauliString, QuantumCircuit, SuperQuantXClient
from superquantx.algorithms import create_vqe_for_molecule


def hydrogen_molecule_hamiltonian():
    """
    Create Hamiltonian for H2 molecule in STO-3G basis

    This is a simplified 2-qubit Hamiltonian for H2 at equilibrium
    bond distance (0.74 Å) using Jordan-Wigner mapping.
    """
    print("Creating H2 molecule Hamiltonian...")

    # H2 Hamiltonian coefficients (Hartree units)
    pauli_terms = {
        "II": -1.0523732,   # Constant term
        "ZI": -0.39793742,  # Z ⊗ I
        "IZ": -0.39793742,  # I ⊗ Z
        "ZZ": -0.01128010,  # Z ⊗ Z
        "XX": 0.18093119,   # X ⊗ X
        "YY": 0.18093119    # Y ⊗ Y
    }

    pauli_strings = []
    for pauli_ops, coeff in pauli_terms.items():
        pauli_strings.append(PauliString(pauli_ops, coeff))

    hamiltonian = Hamiltonian(pauli_strings)
    print(f"Hamiltonian created with {len(pauli_strings)} terms")
    print(f"Exact ground state energy: {hamiltonian.ground_state_energy():.6f} Hartree")

    return hamiltonian


def create_hardware_efficient_ansatz(num_qubits, num_layers):
    """
    Create hardware-efficient ansatz for VQE

    Args:
        num_qubits: Number of qubits
        num_layers: Number of ansatz layers

    Returns:
        Function that creates ansatz circuit from parameters
    """
    def ansatz(parameters):
        circuit = QuantumCircuit(num_qubits, name=f"HEA-L{num_layers}")

        param_idx = 0

        for layer in range(num_layers):
            # Single-qubit rotations
            for qubit in range(num_qubits):
                circuit.ry(parameters[param_idx], qubit)
                param_idx += 1

            # Entangling layer (linear connectivity)
            for qubit in range(num_qubits - 1):
                circuit.cnot(qubit, qubit + 1)

        # Final layer of single-qubit rotations
        for qubit in range(num_qubits):
            circuit.ry(parameters[param_idx], qubit)
            param_idx += 1

        return circuit

    # Calculate total number of parameters
    total_params = num_qubits * (num_layers + 1)
    print(f"Hardware-efficient ansatz created: {num_layers} layers, {total_params} parameters")

    return ansatz, total_params


def create_uccsd_ansatz(num_qubits):
    """
    Create simplified UCCSD (Unitary Coupled Cluster) ansatz

    This is a simplified version focusing on single and double excitations
    for educational purposes.
    """
    def ansatz(parameters):
        circuit = QuantumCircuit(num_qubits, name="UCCSD")

        # Initialize in Hartree-Fock state (|01⟩ for H2)
        circuit.x(0)  # Occupy first orbital

        # Single excitation: |01⟩ → |10⟩
        theta_single = parameters[0]
        circuit.h(1)
        circuit.cnot(1, 0)
        circuit.rz(theta_single, 0)
        circuit.cnot(1, 0)
        circuit.h(1)

        # Double excitation: |01⟩ → |10⟩ (same as single for 2 qubits)
        if len(parameters) > 1:
            theta_double = parameters[1]
            circuit.h(0)
            circuit.h(1)
            circuit.cnot(0, 1)
            circuit.rz(theta_double, 1)
            circuit.cnot(0, 1)
            circuit.h(0)
            circuit.h(1)

        return circuit

    print("UCCSD ansatz created for H2")
    return ansatz


def run_vqe_simulation(hamiltonian, ansatz, num_params, client=None):
    """
    Run VQE optimization using simulation

    Args:
        hamiltonian: Target Hamiltonian
        ansatz: Parameterized ansatz function
        num_params: Number of parameters
        client: SuperQuantX client (None for local simulation)

    Returns:
        VQE results dictionary
    """
    print(f"Running VQE with {'quantum hardware' if client else 'local simulation'}...")

    # Create VQE instance
    vqe = VQE(
        hamiltonian=hamiltonian,
        ansatz=ansatz,
        client=client,
        optimizer="SLSQP",
        max_iterations=100,
        tolerance=1e-6
    )

    # Initial parameters (small random values)
    np.random.seed(42)  # For reproducibility
    initial_params = np.random.uniform(-0.1, 0.1, num_params)

    print(f"Starting optimization with {num_params} parameters...")
    print(f"Initial parameters: {initial_params}")

    # Run VQE
    results = vqe.run(initial_parameters=initial_params)

    # Display results
    print("\nVQE Results:")
    print(f"Optimal energy: {results['optimal_energy']:.6f} Hartree")
    print(f"Optimal parameters: {results['optimal_parameters']}")
    print(f"Converged: {results['converged']}")
    print(f"Number of iterations: {results['num_iterations']}")

    # Calculate error
    exact_energy = results['ground_state_energy_exact']
    error = abs(results['optimal_energy'] - exact_energy)
    error_mhartree = error * 1000  # Convert to milliHartree

    print(f"Exact ground state energy: {exact_energy:.6f} Hartree")
    print(f"VQE error: {error:.6f} Hartree ({error_mhartree:.3f} mHartree)")

    return results, vqe


def plot_vqe_convergence(vqe_results, title="VQE Convergence"):
    """Plot VQE optimization convergence"""
    history = vqe_results['optimization_history']
    exact_energy = vqe_results['ground_state_energy_exact']

    plt.figure(figsize=(10, 6))

    # Plot energy vs iteration
    plt.subplot(1, 2, 1)
    iterations = range(len(history))
    plt.plot(iterations, history, 'b-', label='VQE Energy')
    plt.axhline(y=exact_energy, color='r', linestyle='--', label='Exact Energy')
    plt.xlabel('Iteration')
    plt.ylabel('Energy (Hartree)')
    plt.title('Energy Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot error vs iteration
    plt.subplot(1, 2, 2)
    errors = [abs(energy - exact_energy) * 1000 for energy in history]  # mHartree
    plt.semilogy(iterations, errors, 'g-', label='Energy Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error (mHartree)')
    plt.title('Error Convergence (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()

    return plt.gcf()


def compare_ansatze(hamiltonian, client=None):
    """Compare different ansätze for VQE"""
    print("\nComparing different ansätze...")

    results_comparison = {}

    # 1. Hardware Efficient Ansatz
    hea_ansatz, hea_params = create_hardware_efficient_ansatz(2, 2)
    hea_results, _ = run_vqe_simulation(hamiltonian, hea_ansatz, hea_params, client)
    results_comparison['HEA (L=2)'] = hea_results

    # 2. UCCSD Ansatz
    uccsd_ansatz = create_uccsd_ansatz(2)
    uccsd_results, _ = run_vqe_simulation(hamiltonian, uccsd_ansatz, 2, client)
    results_comparison['UCCSD'] = uccsd_results

    # 3. Simple Ansatz (1 layer HEA)
    simple_ansatz, simple_params = create_hardware_efficient_ansatz(2, 1)
    simple_results, _ = run_vqe_simulation(hamiltonian, simple_ansatz, simple_params, client)
    results_comparison['HEA (L=1)'] = simple_results

    # Compare results
    print("\nAnsatz Comparison:")
    print("-" * 60)
    print(f"{'Ansatz':<12} {'Energy (Ha)':<12} {'Error (mHa)':<12} {'Params':<8}")
    print("-" * 60)

    exact_energy = hamiltonian.ground_state_energy()

    for name, results in results_comparison.items():
        energy = results['optimal_energy']
        error = abs(energy - exact_energy) * 1000
        num_params = len(results['optimal_parameters'])
        print(f"{name:<12} {energy:<12.6f} {error:<12.3f} {num_params:<8}")

    return results_comparison


def benchmark_vqe_scaling():
    """Benchmark VQE scaling with problem size"""
    print("\nBenchmarking VQE scaling...")

    # Create simple Hamiltonians of different sizes
    scaling_results = {}

    for num_qubits in [2, 3, 4]:
        print(f"\nTesting {num_qubits}-qubit system...")

        # Create simple Ising-like Hamiltonian
        pauli_strings = []
        for i in range(num_qubits):
            # Local Z terms
            z_ops = ['I'] * num_qubits
            z_ops[i] = 'Z'
            pauli_strings.append(PauliString(''.join(z_ops), -0.5))

            # Nearest-neighbor ZZ terms
            if i < num_qubits - 1:
                zz_ops = ['I'] * num_qubits
                zz_ops[i] = 'Z'
                zz_ops[i + 1] = 'Z'
                pauli_strings.append(PauliString(''.join(zz_ops), -0.25))

        hamiltonian = Hamiltonian(pauli_strings)

        # Use hardware-efficient ansatz
        ansatz, num_params = create_hardware_efficient_ansatz(num_qubits, 2)

        # Run VQE (with fewer iterations for speed)
        vqe = VQE(hamiltonian, ansatz, optimizer="SLSQP", max_iterations=50)

        initial_params = np.random.uniform(-0.1, 0.1, num_params)
        results = vqe.run(initial_parameters=initial_params)

        scaling_results[num_qubits] = {
            'energy': results['optimal_energy'],
            'exact': results['ground_state_energy_exact'],
            'params': num_params,
            'iterations': results['num_iterations']
        }

        error = abs(results['optimal_energy'] - results['ground_state_energy_exact'])
        print(f"  Energy: {results['optimal_energy']:.6f} (error: {error*1000:.3f} mHa)")
        print(f"  Parameters: {num_params}, Iterations: {results['num_iterations']}")

    return scaling_results


def demonstrate_vqe_features():
    """Demonstrate advanced VQE features"""
    print("\nDemonstrating advanced VQE features...")

    # Create H2 Hamiltonian
    hamiltonian = hydrogen_molecule_hamiltonian()

    # Test parameter initialization strategies
    print("\nTesting parameter initialization strategies...")

    ansatz, num_params = create_hardware_efficient_ansatz(2, 2)

    strategies = {
        'zeros': np.zeros(num_params),
        'random_small': np.random.uniform(-0.1, 0.1, num_params),
        'random_large': np.random.uniform(-np.pi, np.pi, num_params),
        'heuristic': np.array([0.1, -0.1, 0.05, -0.05, 0.02, 0.02])  # Based on chemistry intuition
    }

    strategy_results = {}

    for strategy_name, initial_params in strategies.items():
        print(f"\nTesting {strategy_name} initialization...")

        vqe = VQE(hamiltonian, ansatz, max_iterations=50, tolerance=1e-6)
        results = vqe.run(initial_parameters=initial_params)

        strategy_results[strategy_name] = results

        exact_energy = results['ground_state_energy_exact']
        error = abs(results['optimal_energy'] - exact_energy) * 1000

        print(f"  Final energy: {results['optimal_energy']:.6f} Hartree")
        print(f"  Error: {error:.3f} mHartree")
        print(f"  Converged: {results['converged']}")
        print(f"  Iterations: {results['num_iterations']}")

    return strategy_results


def main():
    """Main VQE example function"""
    print("SuperQuantX VQE Molecular Ground State Example")
    print("=" * 50)

    # Get API key (optional)
    api_key = input("Enter your SuperQuantX API key (or press Enter for simulation only): ").strip()
    client = SuperQuantXClient(api_key) if api_key else None

    # Example 1: Basic H2 VQE
    print("\n1. Basic H2 VQE Example")
    print("-" * 25)

    # Create H2 Hamiltonian
    h2_hamiltonian = hydrogen_molecule_hamiltonian()

    # Use factory function
    print("Using VQE factory function...")
    vqe_h2 = create_vqe_for_molecule("H2", client=client)
    h2_results = vqe_h2.run()

    print(f"Factory VQE result: {h2_results['optimal_energy']:.6f} Hartree")

    # Example 2: Custom ansatz comparison
    print("\n\n2. Ansatz Comparison")
    print("-" * 20)
    ansatz_results = compare_ansatze(h2_hamiltonian, client)

    # Example 3: VQE scaling
    print("\n\n3. VQE Scaling Analysis")
    print("-" * 23)
    scaling_results = benchmark_vqe_scaling()

    # Example 4: Advanced features
    print("\n\n4. Advanced VQE Features")
    print("-" * 25)
    demonstrate_vqe_features()

    # Example 5: Visualization (if matplotlib available)
    try:
        print("\n\n5. Visualization")
        print("-" * 15)

        # Plot convergence for best ansatz
        best_ansatz = min(ansatz_results.keys(),
                         key=lambda k: abs(ansatz_results[k]['optimal_energy'] -
                                         h2_hamiltonian.ground_state_energy()))

        print(f"Plotting convergence for best ansatz: {best_ansatz}")
        plot_vqe_convergence(ansatz_results[best_ansatz],
                           f"VQE Convergence - {best_ansatz}")

        # Save plot
        plt.savefig('vqe_convergence.png', dpi=150, bbox_inches='tight')
        print("Convergence plot saved as 'vqe_convergence.png'")

    except ImportError:
        print("Matplotlib not available, skipping visualization")

    # Summary
    print("\n\nSummary of Results:")
    print("=" * 30)

    exact_energy = h2_hamiltonian.ground_state_energy()

    print(f"H2 Ground State (Exact): {exact_energy:.6f} Hartree")
    print(f"Factory VQE Result: {h2_results['optimal_energy']:.6f} Hartree")

    print("\nBest ansatz results:")
    for name, results in ansatz_results.items():
        error = abs(results['optimal_energy'] - exact_energy) * 1000
        print(f"  {name}: {results['optimal_energy']:.6f} Ha (error: {error:.3f} mHa)")

    print("\nScaling results:")
    for num_qubits, data in scaling_results.items():
        error = abs(data['energy'] - data['exact']) * 1000
        print(f"  {num_qubits} qubits: {data['energy']:.6f} Ha (error: {error:.3f} mHa)")

    print("\nVQE examples completed successfully!")


if __name__ == "__main__":
    main()
