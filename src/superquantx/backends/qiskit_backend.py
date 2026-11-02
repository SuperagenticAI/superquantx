"""Qiskit backend implementation for SuperQuantX.

Local execution uses Qiskit Aer. IBM Quantum hardware is reached through
``qiskit-ibm-runtime`` (``QiskitRuntimeService`` with ``SamplerV2``); the
hardware path is untested in CI because it needs an IBM Quantum account.
"""

import logging
from typing import Any

import numpy as np

from .base_backend import BaseBackend


logger = logging.getLogger(__name__)

try:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QuantumCircuit = None
    AerSimulator = None
    Statevector = None

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler
    IBM_RUNTIME_AVAILABLE = True
except ImportError:
    QiskitRuntimeService = None
    RuntimeSampler = None
    IBM_RUNTIME_AVAILABLE = False

_AER_DEVICES = ('aer_simulator', 'qasm_simulator', 'statevector_simulator', 'simulator')
_AER_METHODS = ('automatic', 'statevector', 'density_matrix', 'stabilizer',
                'matrix_product_state', 'extended_stabilizer', 'unitary', 'superop')


class QiskitBackend(BaseBackend):
    """Qiskit backend (Aer simulator locally, IBM Quantum via qiskit-ibm-runtime).

    Args:
        device: ``'aer_simulator'`` (default), an Aer method name such as
            ``'statevector'``, or an IBM backend name such as ``'ibm_brisbane'``.
        provider: Optional dict of ``QiskitRuntimeService`` keyword arguments
            (``channel``, ``token``, ``instance``) for IBM hardware.
        shots: Number of measurement shots.

    """

    def __init__(self, device: str = 'aer_simulator', provider: dict | None = None,
                 shots: int = 1024, **kwargs):
        if not QISKIT_AVAILABLE:
            raise ImportError(
                "Qiskit is required for QiskitBackend. Install with: pip install 'superquantx[qiskit]'"
            )
        self.provider = provider
        self.backend = None
        self.service = None
        self.is_hardware = False
        super().__init__(device=device, shots=shots, **kwargs)
        self.capabilities = {
            'supports_gradient': False,
            'supports_parameter_shift': True,
            'supports_finite_diff': True,
            'supports_hardware': IBM_RUNTIME_AVAILABLE,
            'supports_noise_models': True,
        }

    def _initialize_backend(self) -> None:
        device = self.device or 'aer_simulator'
        if device in _AER_DEVICES:
            self.backend = AerSimulator()
        elif device in _AER_METHODS:
            self.backend = AerSimulator(method=device)
        elif device.startswith('ibm') or self.provider:
            if not IBM_RUNTIME_AVAILABLE:
                raise ImportError(
                    "qiskit-ibm-runtime is required for IBM Quantum hardware. "
                    "Install with: pip install 'qiskit-ibm-runtime>=0.50'"
                )
            self.service = QiskitRuntimeService(**(self.provider or {}))
            self.backend = self.service.backend(device)
            self.is_hardware = True
            logger.info("Using IBM Quantum backend %s (untested in CI)", device)
        else:
            raise ValueError(
                f"Unknown Qiskit device '{device}'. Use 'aer_simulator', an Aer method "
                "name, or an IBM backend name such as 'ibm_brisbane'."
            )

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    def create_circuit(self, n_qubits: int) -> Any:
        return QuantumCircuit(QuantumRegister(n_qubits, 'q'), ClassicalRegister(n_qubits, 'c'))

    def add_gate(self, circuit: Any, gate: str, qubits: int | list[int],
                 params: list[float] | None = None) -> Any:
        if isinstance(qubits, int):
            qubits = [qubits]
        params = list(params) if params is not None else []
        name = gate.upper()
        single = {'H': 'h', 'HADAMARD': 'h', 'X': 'x', 'PAULI_X': 'x', 'Y': 'y',
                  'PAULI_Y': 'y', 'Z': 'z', 'PAULI_Z': 'z', 'S': 's', 'SDG': 'sdg',
                  'T': 't', 'TDG': 'tdg', 'I': 'id'}
        rot = {'RX': 'rx', 'RY': 'ry', 'RZ': 'rz', 'P': 'p', 'PHASE': 'p',
               'CRX': 'crx', 'CRY': 'cry', 'CRZ': 'crz'}
        multi = {'CNOT': 'cx', 'CX': 'cx', 'CZ': 'cz', 'SWAP': 'swap',
                 'CCX': 'ccx', 'TOFFOLI': 'ccx'}
        if name in single:
            getattr(circuit, single[name])(qubits[0])
        elif name in rot:
            getattr(circuit, rot[name])(float(params[0]), *qubits)
        elif name in multi:
            getattr(circuit, multi[name])(*qubits)
        else:
            raise ValueError(f"Unsupported gate for QiskitBackend: {gate}")
        return circuit

    def add_measurement(self, circuit: Any, qubits: list[int] | None = None) -> Any:
        if qubits is None:
            qubits = list(range(circuit.num_qubits))
        if circuit.num_clbits < len(qubits):
            circuit.add_register(ClassicalRegister(len(qubits) - circuit.num_clbits, 'meas'))
        for i, qubit in enumerate(qubits):
            circuit.measure(qubit, i)
        return circuit

    @staticmethod
    def _has_measurements(circuit: Any) -> bool:
        return any(instr.operation.name == 'measure' for instr in circuit.data)

    def execute_circuit(self, circuit: Any, shots: int | None = None) -> dict[str, Any]:
        """Run the circuit and return counts (little-endian: qubit 0 is the rightmost bit).

        If the circuit has no measurements, all qubits are measured on a copy.
        """
        shots = shots or self.shots
        if not self._has_measurements(circuit):
            circuit = self.add_measurement(circuit.copy())

        if self.is_hardware:
            from qiskit.transpiler import generate_preset_pass_manager
            pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
            isa = pm.run(circuit)
            job = RuntimeSampler(mode=self.backend).run([isa], shots=shots)
            pub = job.result()[0]
            counts: dict[str, int] = {}
            for reg in isa.cregs:
                data = getattr(pub.data, reg.name)
                for k, v in data.get_counts().items():
                    counts[k] = counts.get(k, 0) + v
                break
            return {'counts': counts, 'shots': shots, 'backend': self.device,
                    'job_id': job.job_id()}

        job = self.backend.run(transpile(circuit, self.backend), shots=shots)
        counts = job.result().get_counts()
        return {'counts': dict(counts), 'shots': shots, 'backend': self.device,
                'job_id': job.job_id()}

    def get_statevector(self, circuit: Any) -> np.ndarray:
        """Exact statevector (little-endian), computed with qiskit.quantum_info."""
        clean = circuit.remove_final_measurements(inplace=False)
        return np.asarray(Statevector.from_instruction(clean).data)

    def _get_n_qubits(self, circuit: Any) -> int:
        return circuit.num_qubits

    def get_version_info(self) -> dict[str, Any]:
        info = super().get_version_info()
        from importlib.metadata import PackageNotFoundError, version
        for pkg in ('qiskit', 'qiskit-aer', 'qiskit-ibm-runtime'):
            try:
                info[f"{pkg.replace('-', '_')}_version"] = version(pkg)
            except PackageNotFoundError:
                pass
        info['backend_name'] = getattr(self.backend, 'name', str(self.backend))
        return info
