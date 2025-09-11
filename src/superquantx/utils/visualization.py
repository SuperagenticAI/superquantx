"""Visualization utilities for quantum machine learning.

This module provides functions to visualize quantum circuits, optimization results,
quantum states, and other quantum machine learning concepts.
"""

from typing import Any

import numpy as np


try:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import Normalize
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def visualize_results(
    results: dict[str, Any],
    plot_type: str = 'optimization',
    backend: str = 'matplotlib',
    save_path: str | None = None,
    **kwargs
) -> None:
    """Visualize quantum machine learning results.

    Args:
        results: Results dictionary from algorithm execution
        plot_type: Type of plot ('optimization', 'classification', 'regression')
        backend: Plotting backend ('matplotlib' or 'plotly')
        save_path: Path to save the plot
        **kwargs: Additional plotting arguments

    """
    if backend == 'matplotlib' and not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for matplotlib backend")
    if backend == 'plotly' and not HAS_PLOTLY:
        raise ImportError("plotly is required for plotly backend")

    if plot_type == 'optimization':
        plot_optimization_history(results, backend=backend, save_path=save_path, **kwargs)
    elif plot_type == 'classification':
        plot_classification_results(results, backend=backend, save_path=save_path, **kwargs)
    elif plot_type == 'regression':
        plot_regression_results(results, backend=backend, save_path=save_path, **kwargs)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")


def plot_optimization_history(
    results: dict[str, Any],
    backend: str = 'matplotlib',
    save_path: str | None = None,
    **kwargs
) -> None:
    """Plot optimization history from algorithm results.

    Args:
        results: Results containing 'cost_history' or similar
        backend: Plotting backend
        save_path: Path to save the plot
        **kwargs: Additional plotting arguments

    """
    # Extract cost history
    cost_history = None
    if 'cost_history' in results:
        cost_history = results['cost_history']
    elif 'loss_history' in results:
        cost_history = results['loss_history']
    elif 'objective_history' in results:
        cost_history = results['objective_history']

    if cost_history is None:
        raise ValueError("No optimization history found in results")

    if backend == 'matplotlib':
        _plot_optimization_matplotlib(cost_history, save_path, **kwargs)
    elif backend == 'plotly':
        _plot_optimization_plotly(cost_history, save_path, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def plot_circuit(
    circuit_data: dict[str, Any],
    backend: str = 'matplotlib',
    save_path: str | None = None,
    **kwargs
) -> None:
    """Plot quantum circuit diagram.

    Args:
        circuit_data: Circuit information dictionary
        backend: Plotting backend
        save_path: Path to save the plot
        **kwargs: Additional plotting arguments

    """
    if backend == 'matplotlib':
        _plot_circuit_matplotlib(circuit_data, save_path, **kwargs)
    elif backend == 'plotly':
        _plot_circuit_plotly(circuit_data, save_path, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def plot_quantum_state(
    state_vector: np.ndarray,
    backend: str = 'matplotlib',
    representation: str = 'bar',
    save_path: str | None = None,
    **kwargs
) -> None:
    """Plot quantum state vector.

    Args:
        state_vector: Complex quantum state vector
        backend: Plotting backend
        representation: How to represent state ('bar', 'phase', 'bloch')
        save_path: Path to save the plot
        **kwargs: Additional plotting arguments

    """
    if backend == 'matplotlib':
        _plot_state_matplotlib(state_vector, representation, save_path, **kwargs)
    elif backend == 'plotly':
        _plot_state_plotly(state_vector, representation, save_path, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def plot_bloch_sphere(
    state_vector: np.ndarray,
    backend: str = 'matplotlib',
    save_path: str | None = None,
    **kwargs
) -> None:
    """Plot quantum state on Bloch sphere (for single qubit states).

    Args:
        state_vector: Single qubit state vector [α, β]
        backend: Plotting backend
        save_path: Path to save the plot
        **kwargs: Additional plotting arguments

    """
    if len(state_vector) != 2:
        raise ValueError("Bloch sphere visualization only supports single qubit states")

    # Convert to Bloch vector
    bloch_vector = _state_to_bloch_vector(state_vector)

    if backend == 'matplotlib':
        _plot_bloch_matplotlib(bloch_vector, save_path, **kwargs)
    elif backend == 'plotly':
        _plot_bloch_plotly(bloch_vector, save_path, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def plot_classification_results(
    results: dict[str, Any],
    backend: str = 'matplotlib',
    save_path: str | None = None,
    **kwargs
) -> None:
    """Plot classification results including confusion matrix and metrics."""
    if backend == 'matplotlib':
        _plot_classification_matplotlib(results, save_path, **kwargs)
    elif backend == 'plotly':
        _plot_classification_plotly(results, save_path, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def plot_regression_results(
    results: dict[str, Any],
    backend: str = 'matplotlib',
    save_path: str | None = None,
    **kwargs
) -> None:
    """Plot regression results including predictions vs actual."""
    if backend == 'matplotlib':
        _plot_regression_matplotlib(results, save_path, **kwargs)
    elif backend == 'plotly':
        _plot_regression_plotly(results, save_path, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# Matplotlib implementations
def _plot_optimization_matplotlib(
    cost_history: list[float],
    save_path: str | None = None,
    **kwargs
) -> None:
    """Plot optimization history using matplotlib."""
    plt.figure(figsize=kwargs.get('figsize', (10, 6)))
    plt.plot(cost_history, linewidth=2, color=kwargs.get('color', 'blue'))
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Optimization History')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def _plot_circuit_matplotlib(
    circuit_data: dict[str, Any],
    save_path: str | None = None,
    **kwargs
) -> None:
    """Plot circuit diagram using matplotlib."""
    n_qubits = circuit_data.get('n_qubits', 4)
    gates = circuit_data.get('gates', [])

    fig, ax = plt.subplots(figsize=(12, 2 * n_qubits))

    # Draw qubit lines
    for i in range(n_qubits):
        ax.plot([0, 10], [i, i], 'k-', linewidth=2)
        ax.text(-0.5, i, f'q{i}', ha='right', va='center', fontsize=12)

    # Draw gates (simplified representation)
    gate_positions = np.linspace(1, 9, len(gates)) if gates else []

    for pos, gate in zip(gate_positions, gates, strict=False):
        gate_type = gate.get('type', 'X')
        qubit = gate.get('qubit', 0)

        # Draw gate box
        rect = patches.Rectangle(
            (pos - 0.2, qubit - 0.2),
            0.4, 0.4,
            facecolor='lightblue',
            edgecolor='black'
        )
        ax.add_patch(rect)
        ax.text(pos, qubit, gate_type, ha='center', va='center', fontsize=10)

    ax.set_xlim(-1, 11)
    ax.set_ylim(-0.5, n_qubits - 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Quantum Circuit')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def _plot_state_matplotlib(
    state_vector: np.ndarray,
    representation: str,
    save_path: str | None = None,
    **kwargs
) -> None:
    """Plot quantum state using matplotlib."""
    n_states = len(state_vector)

    if representation == 'bar':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Amplitudes
        amplitudes = np.abs(state_vector)
        ax1.bar(range(n_states), amplitudes, color='skyblue')
        ax1.set_xlabel('Basis State')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('State Amplitudes')

        # Phases
        phases = np.angle(state_vector)
        ax2.bar(range(n_states), phases, color='orange')
        ax2.set_xlabel('Basis State')
        ax2.set_ylabel('Phase (radians)')
        ax2.set_title('State Phases')

    elif representation == 'phase':
        # Phase-amplitude plot
        amplitudes = np.abs(state_vector)
        phases = np.angle(state_vector)

        plt.figure(figsize=(10, 8))
        plt.scatter(phases, amplitudes, s=100, alpha=0.7)
        plt.xlabel('Phase (radians)')
        plt.ylabel('Amplitude')
        plt.title('Quantum State (Phase-Amplitude)')

        # Add state labels
        for i, (phase, amp) in enumerate(zip(phases, amplitudes, strict=False)):
            if amp > 0.01:  # Only label significant amplitudes
                plt.annotate(f'|{i}⟩', (phase, amp), xytext=(5, 5),
                           textcoords='offset points')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def _plot_bloch_matplotlib(
    bloch_vector: np.ndarray,
    save_path: str | None = None,
    **kwargs
) -> None:
    """Plot Bloch sphere using matplotlib."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='lightblue')

    # Draw axes
    ax.plot([0, 1], [0, 0], [0, 0], 'k-', linewidth=2)
    ax.plot([0, 0], [0, 1], [0, 0], 'k-', linewidth=2)
    ax.plot([0, 0], [0, 0], [0, 1], 'k-', linewidth=2)

    # Draw state vector
    x, y, z = bloch_vector
    ax.quiver(0, 0, 0, x, y, z, color='red', arrow_length_ratio=0.1, linewidth=3)
    ax.scatter([x], [y], [z], color='red', s=100)

    # Labels
    ax.text(1.1, 0, 0, 'X')
    ax.text(0, 1.1, 0, 'Y')
    ax.text(0, 0, 1.1, 'Z')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Bloch Sphere')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def _plot_classification_matplotlib(
    results: dict[str, Any],
    save_path: str | None = None,
    **kwargs
) -> None:
    """Plot classification results using matplotlib."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Confusion matrix
    if 'confusion_matrix' in results:
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')

    # Training history
    if 'cost_history' in results:
        axes[0, 1].plot(results['cost_history'])
        axes[0, 1].set_title('Training Cost')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Cost')

    # Accuracy metrics
    if 'metrics' in results:
        metrics = results['metrics']
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        axes[1, 0].bar(metric_names, metric_values)
        axes[1, 0].set_title('Performance Metrics')
        axes[1, 0].set_ylabel('Score')
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)

    # Feature importance (if available)
    if 'feature_importance' in results:
        importance = results['feature_importance']
        axes[1, 1].bar(range(len(importance)), importance)
        axes[1, 1].set_title('Feature Importance')
        axes[1, 1].set_xlabel('Feature Index')
        axes[1, 1].set_ylabel('Importance')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def _plot_regression_matplotlib(
    results: dict[str, Any],
    save_path: str | None = None,
    **kwargs
) -> None:
    """Plot regression results using matplotlib."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Predictions vs Actual
    if 'y_pred' in results and 'y_true' in results:
        y_pred = results['y_pred']
        y_true = results['y_true']

        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predictions')
        axes[0, 0].set_title('Predictions vs Actual')

    # Residuals
    if 'y_pred' in results and 'y_true' in results:
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predictions')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')

    # Training history
    if 'cost_history' in results:
        axes[1, 0].plot(results['cost_history'])
        axes[1, 0].set_title('Training Cost')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Cost')

    # Error metrics
    if 'metrics' in results:
        metrics = results['metrics']
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        axes[1, 1].bar(metric_names, metric_values)
        axes[1, 1].set_title('Error Metrics')
        axes[1, 1].set_ylabel('Error')
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


# Plotly implementations (simplified)
def _plot_optimization_plotly(
    cost_history: list[float],
    save_path: str | None = None,
    **kwargs
) -> None:
    """Plot optimization history using plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=cost_history,
        mode='lines',
        name='Cost',
        line={'width': 3}
    ))
    fig.update_layout(
        title='Optimization History',
        xaxis_title='Iteration',
        yaxis_title='Cost'
    )

    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()


def _plot_circuit_plotly(
    circuit_data: dict[str, Any],
    save_path: str | None = None,
    **kwargs
) -> None:
    """Plot circuit using plotly (simplified)."""
    # This would require more complex implementation
    # For now, just create a placeholder
    fig = go.Figure()
    fig.add_annotation(
        text="Circuit visualization with Plotly not fully implemented",
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False
    )

    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()


def _plot_state_plotly(
    state_vector: np.ndarray,
    representation: str,
    save_path: str | None = None,
    **kwargs
) -> None:
    """Plot state using plotly."""
    amplitudes = np.abs(state_vector)
    phases = np.angle(state_vector)

    fig = make_subplots(rows=2, cols=1,
                       subplot_titles=['Amplitudes', 'Phases'])

    fig.add_trace(go.Bar(y=amplitudes, name='Amplitudes'), row=1, col=1)
    fig.add_trace(go.Bar(y=phases, name='Phases'), row=2, col=1)

    fig.update_layout(title='Quantum State Vector')

    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()


def _plot_bloch_plotly(
    bloch_vector: np.ndarray,
    save_path: str | None = None,
    **kwargs
) -> None:
    """Plot Bloch sphere using plotly."""
    # Create sphere
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

    fig = go.Figure()

    # Add sphere surface
    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        opacity=0.1, showscale=False,
        colorscale=[[0, 'lightblue'], [1, 'lightblue']]
    ))

    # Add state vector
    x, y, z = bloch_vector
    fig.add_trace(go.Scatter3d(
        x=[0, x], y=[0, y], z=[0, z],
        mode='lines+markers',
        line={'width': 5, 'color': 'red'},
        marker={'size': 8, 'color': 'red'}
    ))

    fig.update_layout(
        title='Bloch Sphere',
        scene={
            'xaxis_title': 'X',
            'yaxis_title': 'Y',
            'zaxis_title': 'Z'
        }
    )

    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()


def _plot_classification_plotly(
    results: dict[str, Any],
    save_path: str | None = None,
    **kwargs
) -> None:
    """Plot classification results using plotly."""
    # Simplified implementation
    fig = make_subplots(rows=2, cols=2)

    if 'cost_history' in results:
        fig.add_trace(
            go.Scatter(y=results['cost_history'], name='Cost'),
            row=1, col=1
        )

    fig.update_layout(title='Classification Results')

    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()


def _plot_regression_plotly(
    results: dict[str, Any],
    save_path: str | None = None,
    **kwargs
) -> None:
    """Plot regression results using plotly."""
    # Simplified implementation
    fig = make_subplots(rows=2, cols=2)

    if 'cost_history' in results:
        fig.add_trace(
            go.Scatter(y=results['cost_history'], name='Cost'),
            row=1, col=1
        )

    if 'y_pred' in results and 'y_true' in results:
        fig.add_trace(
            go.Scatter(
                x=results['y_true'],
                y=results['y_pred'],
                mode='markers',
                name='Predictions vs Actual'
            ),
            row=1, col=2
        )

    fig.update_layout(title='Regression Results')

    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()


def _state_to_bloch_vector(state_vector: np.ndarray) -> np.ndarray:
    """Convert single qubit state to Bloch vector."""
    # Normalize state
    state = state_vector / np.linalg.norm(state_vector)

    # Extract amplitudes
    alpha, beta = state[0], state[1]

    # Compute Bloch vector components
    x = 2 * np.real(np.conj(alpha) * beta)
    y = 2 * np.imag(np.conj(alpha) * beta)
    z = np.abs(alpha)**2 - np.abs(beta)**2

    return np.array([x, y, z])
