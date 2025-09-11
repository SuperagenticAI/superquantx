"""Quantum-specific utility functions.

This module provides utility functions for quantum information processing,
including fidelity calculations, quantum distances, and entanglement measures.
"""

import warnings

import numpy as np
from scipy.linalg import sqrtm


def fidelity(
    state1: np.ndarray,
    state2: np.ndarray,
    validate: bool = True
) -> float:
    """Calculate quantum fidelity between two quantum states.

    For pure states |ψ₁⟩ and |ψ₂⟩:
    F(ψ₁, ψ₂) = |⟨ψ₁|ψ₂⟩|²

    For mixed states ρ₁ and ρ₂:
    F(ρ₁, ρ₂) = Tr(√(√ρ₁ ρ₂ √ρ₁))²

    Args:
        state1: First quantum state (vector or density matrix)
        state2: Second quantum state (vector or density matrix)
        validate: Whether to validate inputs

    Returns:
        Fidelity value between 0 and 1

    """
    if validate:
        _validate_quantum_state(state1)
        _validate_quantum_state(state2)

    # Check if states are vectors (pure states) or matrices (mixed states)
    is_pure1 = len(state1.shape) == 1
    is_pure2 = len(state2.shape) == 1

    if is_pure1 and is_pure2:
        # Both pure states
        overlap = np.vdot(state1, state2)
        return abs(overlap) ** 2

    elif is_pure1 and not is_pure2:
        # state1 pure, state2 mixed
        rho2 = state2
        psi1 = state1.reshape(-1, 1)
        return np.real(np.conj(psi1).T @ rho2 @ psi1)[0, 0]

    elif not is_pure1 and is_pure2:
        # state1 mixed, state2 pure
        rho1 = state1
        psi2 = state2.reshape(-1, 1)
        return np.real(np.conj(psi2).T @ rho1 @ psi2)[0, 0]

    else:
        # Both mixed states
        rho1, rho2 = state1, state2

        # F = Tr(√(√ρ₁ ρ₂ √ρ₁))²
        sqrt_rho1 = sqrtm(rho1)
        M = sqrt_rho1 @ rho2 @ sqrt_rho1
        sqrt_M = sqrtm(M)

        fid = np.real(np.trace(sqrt_M)) ** 2

        # Ensure fidelity is in [0, 1] (numerical errors can cause small violations)
        return np.clip(fid, 0, 1)


def trace_distance(
    state1: np.ndarray,
    state2: np.ndarray,
    validate: bool = True
) -> float:
    """Calculate trace distance between two quantum states.

    For quantum states ρ₁ and ρ₂:
    D(ρ₁, ρ₂) = (1/2) * Tr(|ρ₁ - ρ₂|)

    Args:
        state1: First quantum state
        state2: Second quantum state
        validate: Whether to validate inputs

    Returns:
        Trace distance between 0 and 1

    """
    if validate:
        _validate_quantum_state(state1)
        _validate_quantum_state(state2)

    # Convert to density matrices if needed
    rho1 = _to_density_matrix(state1)
    rho2 = _to_density_matrix(state2)

    # Compute difference
    diff = rho1 - rho2

    # Compute eigenvalues and take absolute values
    eigenvals = np.linalg.eigvals(diff)
    abs_eigenvals = np.abs(eigenvals)

    # Trace distance is half the sum of absolute eigenvalues
    return 0.5 * np.sum(abs_eigenvals)


def quantum_mutual_information(
    joint_state: np.ndarray,
    subsystem_dims: tuple[int, int],
    validate: bool = True
) -> float:
    """Calculate quantum mutual information between two subsystems.

    I(A:B) = S(ρₐ) + S(ρᵦ) - S(ρₐᵦ)

    where S(ρ) is the von Neumann entropy.

    Args:
        joint_state: Joint quantum state of both subsystems
        subsystem_dims: Dimensions of subsystems (dim_A, dim_B)
        validate: Whether to validate inputs

    Returns:
        Quantum mutual information

    """
    if validate:
        _validate_quantum_state(joint_state)

    rho_AB = _to_density_matrix(joint_state)
    dim_A, dim_B = subsystem_dims

    if rho_AB.shape[0] != dim_A * dim_B:
        raise ValueError(f"State dimension {rho_AB.shape[0]} doesn't match subsystem dims {dim_A * dim_B}")

    # Partial traces
    rho_A = partial_trace(rho_AB, (dim_A, dim_B), trace_out=1)
    rho_B = partial_trace(rho_AB, (dim_A, dim_B), trace_out=0)

    # Von Neumann entropies
    S_A = von_neumann_entropy(rho_A)
    S_B = von_neumann_entropy(rho_B)
    S_AB = von_neumann_entropy(rho_AB)

    return S_A + S_B - S_AB


def entanglement_measure(
    state: np.ndarray,
    subsystem_dims: tuple[int, int],
    measure: str = 'negativity',
    validate: bool = True
) -> float:
    """Calculate entanglement measure for a bipartite quantum state.

    Args:
        state: Quantum state (pure or mixed)
        subsystem_dims: Dimensions of subsystems (dim_A, dim_B)
        measure: Type of measure ('negativity', 'concurrence', 'entropy')
        validate: Whether to validate inputs

    Returns:
        Entanglement measure value

    """
    if validate:
        _validate_quantum_state(state)

    if measure == 'negativity':
        return negativity(state, subsystem_dims)
    elif measure == 'concurrence':
        return concurrence(state, subsystem_dims)
    elif measure == 'entropy':
        return entanglement_entropy(state, subsystem_dims)
    else:
        raise ValueError(f"Unknown entanglement measure: {measure}")


def negativity(
    state: np.ndarray,
    subsystem_dims: tuple[int, int]
) -> float:
    """Calculate negativity entanglement measure.

    Negativity is defined as:
    N(ρ) = (||ρᵀᴬ||₁ - 1) / 2

    where ρᵀᴬ is the partial transpose with respect to subsystem A.

    Args:
        state: Quantum state
        subsystem_dims: Dimensions of subsystems

    Returns:
        Negativity value

    """
    rho = _to_density_matrix(state)

    # Partial transpose
    rho_TA = partial_transpose(rho, subsystem_dims, transpose_subsystem=0)

    # Calculate trace norm (sum of absolute eigenvalues)
    eigenvals = np.linalg.eigvals(rho_TA)
    trace_norm = np.sum(np.abs(eigenvals))

    return (trace_norm - 1) / 2


def concurrence(
    state: np.ndarray,
    subsystem_dims: tuple[int, int]
) -> float:
    """Calculate concurrence for two-qubit systems.

    Note: This implementation is only valid for 2×2 systems.

    Args:
        state: Two-qubit quantum state
        subsystem_dims: Should be (2, 2) for two qubits

    Returns:
        Concurrence value

    """
    if subsystem_dims != (2, 2):
        raise ValueError("Concurrence is only implemented for two-qubit systems")

    rho = _to_density_matrix(state)

    # Pauli Y matrix
    sigma_y = np.array([[0, -1j], [1j, 0]])

    # Two-qubit Y tensor product
    Y_tensor = np.kron(sigma_y, sigma_y)

    # Spin-flipped state
    rho_tilde = Y_tensor @ np.conj(rho) @ Y_tensor

    # R matrix
    R = rho @ rho_tilde

    # Eigenvalues in descending order
    eigenvals = np.sqrt(np.real(np.linalg.eigvals(R)))
    eigenvals = np.sort(eigenvals)[::-1]

    # Concurrence
    C = max(0, eigenvals[0] - eigenvals[1] - eigenvals[2] - eigenvals[3])

    return C


def entanglement_entropy(
    state: np.ndarray,
    subsystem_dims: tuple[int, int]
) -> float:
    """Calculate entanglement entropy (von Neumann entropy of reduced state).

    Args:
        state: Quantum state
        subsystem_dims: Dimensions of subsystems

    Returns:
        Entanglement entropy

    """
    rho = _to_density_matrix(state)

    # Reduced state of first subsystem
    rho_A = partial_trace(rho, subsystem_dims, trace_out=1)

    return von_neumann_entropy(rho_A)


def von_neumann_entropy(rho: np.ndarray) -> float:
    """Calculate von Neumann entropy of a quantum state.

    S(ρ) = -Tr(ρ log ρ)

    Args:
        rho: Density matrix

    Returns:
        von Neumann entropy

    """
    eigenvals = np.linalg.eigvals(rho)

    # Remove zero eigenvalues to avoid log(0)
    eigenvals = eigenvals[eigenvals > 1e-12]

    if len(eigenvals) == 0:
        return 0.0

    return -np.sum(eigenvals * np.log2(eigenvals))


def partial_trace(
    rho: np.ndarray,
    subsystem_dims: tuple[int, int],
    trace_out: int
) -> np.ndarray:
    """Compute partial trace of a density matrix.

    Args:
        rho: Density matrix
        subsystem_dims: Dimensions of subsystems (dim_A, dim_B)
        trace_out: Which subsystem to trace out (0 for A, 1 for B)

    Returns:
        Reduced density matrix

    """
    dim_A, dim_B = subsystem_dims

    if rho.shape != (dim_A * dim_B, dim_A * dim_B):
        raise ValueError("Density matrix dimensions don't match subsystem dimensions")

    if trace_out == 0:
        # Trace out subsystem A, keep B
        rho_B = np.zeros((dim_B, dim_B), dtype=complex)
        for i in range(dim_A):
            rho_B += rho[i*dim_B:(i+1)*dim_B, i*dim_B:(i+1)*dim_B]
        return rho_B

    elif trace_out == 1:
        # Trace out subsystem B, keep A
        rho_A = np.zeros((dim_A, dim_A), dtype=complex)
        for i in range(dim_A):
            for j in range(dim_A):
                for k in range(dim_B):
                    rho_A[i, j] += rho[i*dim_B + k, j*dim_B + k]
        return rho_A

    else:
        raise ValueError("trace_out must be 0 or 1")


def partial_transpose(
    rho: np.ndarray,
    subsystem_dims: tuple[int, int],
    transpose_subsystem: int
) -> np.ndarray:
    """Compute partial transpose of a density matrix.

    Args:
        rho: Density matrix
        subsystem_dims: Dimensions of subsystems
        transpose_subsystem: Which subsystem to transpose (0 for A, 1 for B)

    Returns:
        Partially transposed density matrix

    """
    dim_A, dim_B = subsystem_dims

    if transpose_subsystem == 0:
        # Transpose subsystem A
        rho_TA = np.zeros_like(rho)
        for i in range(dim_A):
            for j in range(dim_A):
                for k in range(dim_B):
                    for m in range(dim_B):
                        # Transpose indices for subsystem A
                        rho_TA[i*dim_B + k, j*dim_B + m] = rho[j*dim_B + k, i*dim_B + m]
        return rho_TA

    elif transpose_subsystem == 1:
        # Transpose subsystem B
        rho_TB = np.zeros_like(rho)
        for i in range(dim_A):
            for j in range(dim_A):
                for k in range(dim_B):
                    for m in range(dim_B):
                        # Transpose indices for subsystem B
                        rho_TB[i*dim_B + k, j*dim_B + m] = rho[i*dim_B + m, j*dim_B + k]
        return rho_TB

    else:
        raise ValueError("transpose_subsystem must be 0 or 1")


def _validate_quantum_state(state: np.ndarray) -> None:
    """Validate that input represents a valid quantum state."""
    if len(state.shape) == 1:
        # Pure state vector
        if not np.isclose(np.linalg.norm(state), 1.0, atol=1e-10):
            warnings.warn("State vector is not normalized", stacklevel=2)

    elif len(state.shape) == 2:
        # Density matrix
        if state.shape[0] != state.shape[1]:
            raise ValueError("Density matrix must be square")

        # Check if Hermitian
        if not np.allclose(state, np.conj(state.T), atol=1e-10):
            warnings.warn("Density matrix is not Hermitian", stacklevel=2)

        # Check if positive semidefinite
        eigenvals = np.linalg.eigvals(state)
        if np.any(eigenvals < -1e-10):
            warnings.warn("Density matrix is not positive semidefinite", stacklevel=2)

        # Check trace
        if not np.isclose(np.trace(state), 1.0, atol=1e-10):
            warnings.warn("Density matrix trace is not 1", stacklevel=2)

    else:
        raise ValueError("Invalid quantum state format")


def _to_density_matrix(state: np.ndarray) -> np.ndarray:
    """Convert state vector to density matrix if needed."""
    if len(state.shape) == 1:
        # Pure state vector -> density matrix
        psi = state.reshape(-1, 1)
        return psi @ np.conj(psi.T)
    else:
        # Already a density matrix
        return state


def quantum_state_distance(
    state1: np.ndarray,
    state2: np.ndarray,
    metric: str = 'fidelity'
) -> float:
    """Calculate distance between quantum states using various metrics.

    Args:
        state1: First quantum state
        state2: Second quantum state
        metric: Distance metric ('fidelity', 'trace_distance', 'hilbert_schmidt')

    Returns:
        Distance value

    """
    if metric == 'fidelity':
        return 1 - fidelity(state1, state2)
    elif metric == 'trace_distance':
        return trace_distance(state1, state2)
    elif metric == 'hilbert_schmidt':
        rho1 = _to_density_matrix(state1)
        rho2 = _to_density_matrix(state2)
        diff = rho1 - rho2
        return np.sqrt(np.real(np.trace(np.conj(diff.T) @ diff)))
    else:
        raise ValueError(f"Unknown metric: {metric}")
