"""Cirq backend implementation for SuperQuantX.

Local execution uses ``cirq.Simulator``. Google Quantum AI hardware via
``cirq_google`` is wired up but untested (it requires processor access).
"""

import logging
from typing import Any

import numpy as np

from .base_backend import BaseBackend


logger = logging.getLogger(__name__)

try:
    import cirq
    CIRQ_AVAILABLE = True
except ImportError:
    CIRQ_AVAILABLE = False
    cirq = None


class CirqBackend(BaseBackend):
    """Cirq backend.

    Circuits are ``(cirq.Circuit, [LineQubit, ...])`` tuples. Counts and
    statevectors are big-endian (qubit 0 is the leftmost bit).

    Args:
        device: ``'simulator'`` (default).
        processor_id: Google Quantum AI processor id (hardware, untested).
        shots: Number of measurement shots.

    """

    def __init__(self, device: str = 'simulator', processor_id: str | None = None,
                 shots: int = 1024, **kwargs):
        if not CIRQ_AVAILABLE:
            raise ImportError("Cirq is required for CirqBackend. Install with: pip install 'superquantx[cirq]'")
        # These must exist before BaseBackend.__init__ calls _initialize_backend().
        self.processor_id = processor_id
        self.simulator = None
        self.quantum_engine = None
        super().__init__(device=device, shots=shots, **kwargs)
        self.capabilities = {
            'supports_gradient': False,
            'supports_parameter_shift': True,
            'supports_finite_diff': True,
            'supports_hardware': processor_id is not None,
            'supports_noise_models': True,
        }

    def _initialize_backend(self) -> None:
        if self.processor_id:
            import cirq_google  # raises ImportError with a clear message if missing
            self.quantum_engine = cirq_google.get_engine()
        else:
            self.simulator = cirq.Simulator()

    @staticmethod
    def _unpack(circuit_info: Any) -> tuple[Any, list]:
        if isinstance(circuit_info, tuple):
            return circuit_info
        return circuit_info, sorted(circuit_info.all_qubits())

    def create_circuit(self, n_qubits: int) -> tuple[Any, list]:
        return cirq.Circuit(), [cirq.LineQubit(i) for i in range(n_qubits)]

    def add_gate(self, circuit_info: tuple[Any, list], gate: str,
                 qubits: int | list[int], params: list[float] | None = None) -> tuple[Any, list]:
        circuit, qubit_list = circuit_info
        if isinstance(qubits, int):
            qubits = [qubits]
        q = [qubit_list[i] for i in qubits]
        params = list(params) if params is not None else []
        name = gate.upper()
        fixed = {'H': cirq.H, 'HADAMARD': cirq.H, 'X': cirq.X, 'PAULI_X': cirq.X,
                 'Y': cirq.Y, 'PAULI_Y': cirq.Y, 'Z': cirq.Z, 'PAULI_Z': cirq.Z,
                 'S': cirq.S, 'T': cirq.T, 'I': cirq.I, 'CNOT': cirq.CNOT, 'CX': cirq.CNOT,
                 'CZ': cirq.CZ, 'SWAP': cirq.SWAP, 'CCX': cirq.CCX, 'TOFFOLI': cirq.CCX}
        rot = {'RX': cirq.rx, 'RY': cirq.ry, 'RZ': cirq.rz}
        if name in fixed:
            circuit.append(fixed[name](*q))
        elif name in rot:
            circuit.append(rot[name](float(params[0]))(*q))
        elif name in ('P', 'PHASE'):
            circuit.append(cirq.ZPowGate(exponent=float(params[0]) / np.pi)(*q))
        elif name in ('CRX', 'CRY', 'CRZ'):
            circuit.append(rot['R' + name[2]](float(params[0])).controlled()(*q))
        else:
            raise ValueError(f"Unsupported gate for CirqBackend: {gate}")
        return circuit, qubit_list

    def add_measurement(self, circuit_info: tuple[Any, list],
                        qubits: list[int] | None = None) -> tuple[Any, list]:
        circuit, qubit_list = circuit_info
        if qubits is None:
            qubits = list(range(len(qubit_list)))
        circuit.append(cirq.measure(*[qubit_list[i] for i in qubits], key='result'))
        return circuit, qubit_list

    def execute_circuit(self, circuit_info: Any, shots: int | None = None) -> dict[str, Any]:
        """Sample the circuit; unmeasured circuits get all qubits measured on a copy."""
        circuit, qubit_list = self._unpack(circuit_info)
        shots = shots or self.shots
        if not circuit.has_measurements():
            circuit = circuit + cirq.Circuit(cirq.measure(*qubit_list, key='result'))

        if self.quantum_engine is not None:
            processor = self.quantum_engine.get_processor(self.processor_id)
            job = processor.run(circuit, repetitions=shots)
            result = job.results()[0]
            backend_name = f'google_quantum_{self.processor_id}'
        else:
            result = self.simulator.run(circuit, repetitions=shots)
            backend_name = 'cirq_simulator'

        keys = list(result.measurements)
        bits = np.concatenate([result.measurements[k] for k in keys], axis=1)
        values, freq = np.unique(bits, axis=0, return_counts=True)
        counts = {''.join(str(int(b)) for b in row): int(c) for row, c in zip(values, freq, strict=True)}
        return {'counts': counts, 'shots': shots, 'backend': backend_name}

    def get_statevector(self, circuit_info: Any) -> np.ndarray:
        circuit, qubit_list = self._unpack(circuit_info)
        clean = cirq.drop_terminal_measurements(circuit) if circuit.has_measurements() else circuit
        simulator = self.simulator or cirq.Simulator()
        result = simulator.simulate(clean, qubit_order=qubit_list)
        return np.asarray(result.final_state_vector, dtype=complex)

    def _get_n_qubits(self, circuit_info: Any) -> int:
        return len(self._unpack(circuit_info)[1])

    def get_version_info(self) -> dict[str, Any]:
        info = super().get_version_info()
        info['cirq_version'] = cirq.__version__
        return info
