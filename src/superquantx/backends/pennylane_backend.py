"""PennyLane backend implementation for SuperQuantX.

Circuits are recorded as a list of PennyLane operations and executed by
building a QNode on demand, so the unified ``create_circuit`` / ``add_gate`` /
``execute_circuit`` API works the same way as on the other backends.
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from .base_backend import BaseBackend


logger = logging.getLogger(__name__)

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    qml = None


class PennyLaneCircuit:
    """A recorded PennyLane circuit: ``n_qubits`` wires plus a list of operations."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.ops: list[tuple[str, tuple[float, ...], tuple[int, ...]]] = []
        self.measured: list[int] | None = None

    def copy(self) -> 'PennyLaneCircuit':
        new = PennyLaneCircuit(self.n_qubits)
        new.ops = list(self.ops)
        new.measured = None if self.measured is None else list(self.measured)
        return new

    def apply(self) -> None:
        """Queue the recorded operations inside a QNode."""
        for name, params, wires in self.ops:
            getattr(qml, name)(*params, wires=list(wires))

    def __call__(self) -> None:  # allows use as a plain quantum function
        self.apply()

    def __repr__(self) -> str:
        return f"PennyLaneCircuit(n_qubits={self.n_qubits}, ops={len(self.ops)})"


_GATES = {
    'H': 'Hadamard', 'HADAMARD': 'Hadamard', 'X': 'PauliX', 'PAULI_X': 'PauliX',
    'Y': 'PauliY', 'PAULI_Y': 'PauliY', 'Z': 'PauliZ', 'PAULI_Z': 'PauliZ',
    'S': 'S', 'T': 'T', 'I': 'Identity', 'RX': 'RX', 'RY': 'RY', 'RZ': 'RZ',
    'P': 'PhaseShift', 'PHASE': 'PhaseShift', 'CNOT': 'CNOT', 'CX': 'CNOT',
    'CZ': 'CZ', 'SWAP': 'SWAP', 'CRX': 'CRX', 'CRY': 'CRY', 'CRZ': 'CRZ',
    'CCX': 'Toffoli', 'TOFFOLI': 'Toffoli',
}
_PARAMETRIC = {'RX', 'RY', 'RZ', 'PhaseShift', 'CRX', 'CRY', 'CRZ'}


class PennyLaneBackend(BaseBackend):
    """PennyLane backend.

    Args:
        device: PennyLane device name (``'default.qubit'``, ``'lightning.qubit'``, ...).
        wires: Maximum number of wires a circuit may use.
        shots: Default number of shots for sampled execution.
        **kwargs: Extra keyword arguments passed to ``qml.device``.

    """

    def __init__(self, device: str = 'default.qubit', wires: int = 16, shots: int = 1024, **kwargs):
        if not PENNYLANE_AVAILABLE:
            raise ImportError(
                "PennyLane is required for PennyLaneBackend. Install with: pip install 'superquantx[pennylane]'"
            )
        self.wires = wires
        self._devices: dict[int, Any] = {}
        super().__init__(device=device, shots=shots, **kwargs)
        self.capabilities = {
            'supports_gradient': True,
            'supports_parameter_shift': True,
            'supports_finite_diff': True,
            'supports_backprop': device == 'default.qubit',
            'supports_adjoint': device in ('default.qubit', 'lightning.qubit'),
        }

    def _initialize_backend(self) -> None:
        # Validate the device name eagerly so misconfiguration fails fast.
        self.dev = self._device(1)

    def _device(self, n_wires: int) -> Any:
        if n_wires not in self._devices:
            kwargs = {k: v for k, v in self.config.items() if k != 'wires'}
            self._devices[n_wires] = qml.device(self.device, wires=n_wires, **kwargs)
        return self._devices[n_wires]

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    def create_circuit(self, n_qubits: int) -> PennyLaneCircuit:
        if n_qubits > self.wires:
            raise ValueError(f"Requested {n_qubits} qubits, but backend allows {self.wires} wires")
        return PennyLaneCircuit(n_qubits)

    def add_gate(self, circuit: PennyLaneCircuit, gate: str, qubits: int | list[int],
                 params: list[float] | None = None) -> PennyLaneCircuit:
        if isinstance(qubits, int):
            qubits = [qubits]
        name = _GATES.get(gate.upper())
        if name is None:
            raise ValueError(f"Unsupported gate for PennyLaneBackend: {gate}")
        values = tuple(float(p) for p in (params or []))[:1] if name in _PARAMETRIC else ()
        if name in _PARAMETRIC and not values:
            raise ValueError(f"Gate {gate} needs a rotation angle")
        circuit.ops.append((name, values, tuple(int(q) for q in qubits)))
        return circuit

    def add_measurement(self, circuit: PennyLaneCircuit, qubits: list[int] | None = None) -> PennyLaneCircuit:
        circuit.measured = list(range(circuit.n_qubits)) if qubits is None else list(qubits)
        return circuit

    def execute_circuit(self, circuit: PennyLaneCircuit | Callable, shots: int | None = None) -> dict[str, Any]:
        """Sample the circuit and return counts (big-endian: wire 0 is the leftmost bit).

        A plain quantum function (legacy API) is also accepted; it is run as a
        QNode and its return value is passed back under ``'result'``.
        """
        shots = shots or self.shots
        if not isinstance(circuit, PennyLaneCircuit):
            qnode = qml.set_shots(qml.QNode(circuit, self._device(self.wires)), shots=shots)
            return {'result': qnode(), 'shots': shots, 'device': self.device}

        wires = circuit.measured if circuit.measured is not None else list(range(circuit.n_qubits))

        def qfunc():
            circuit.apply()
            return qml.counts(wires=wires)

        qnode = qml.set_shots(qml.QNode(qfunc, self._device(circuit.n_qubits)), shots=shots)
        counts = {str(k): int(v) for k, v in qnode().items()}
        return {'counts': counts, 'shots': shots, 'device': self.device}

    def get_statevector(self, circuit: PennyLaneCircuit | Callable) -> np.ndarray:
        """Exact statevector (big-endian, wire 0 most significant)."""
        n = circuit.n_qubits if isinstance(circuit, PennyLaneCircuit) else self.wires

        def qfunc():
            circuit()
            return qml.state()

        return np.asarray(qml.QNode(qfunc, self._device(n))())

    def _get_n_qubits(self, circuit: Any) -> int:
        return circuit.n_qubits if isinstance(circuit, PennyLaneCircuit) else self.wires

    def get_version_info(self) -> dict[str, Any]:
        info = super().get_version_info()
        info['pennylane_version'] = qml.__version__
        return info
