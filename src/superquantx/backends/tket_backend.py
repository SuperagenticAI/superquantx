"""TKET/Quantinuum backend implementation for SuperQuantX.

This module provides integration with TKET (Quantum Toolkit) and Quantinuum
hardware for advanced quantum circuit compilation and optimization.
"""

import logging
from typing import Any

import numpy as np

from .base_backend import BaseBackend


logger = logging.getLogger(__name__)

# Try to import TKET/pytket
try:
    from pytket import Circuit, OpType
    TKET_AVAILABLE = True
except ImportError:
    TKET_AVAILABLE = False
    Circuit = None
    OpType = None

# Local shot-based simulation goes through pytket-qiskit's AerBackend.
try:
    from pytket.extensions.qiskit import AerBackend
    TKET_EXTENSIONS_AVAILABLE = True
except ImportError:
    AerBackend = None
    TKET_EXTENSIONS_AVAILABLE = False

try:
    from pytket.extensions.quantinuum import QuantinuumBackend
    QUANTINUUM_AVAILABLE = True
except ImportError:
    QuantinuumBackend = None
    QUANTINUUM_AVAILABLE = False

try:
    from pytket.extensions.cirq import CirqStateSampleBackend as TKETCirqBackend
    TKET_CIRQ_AVAILABLE = True
except ImportError:
    TKETCirqBackend = None
    TKET_CIRQ_AVAILABLE = False

class TKETBackend(BaseBackend):
    """TKET (Quantum Toolkit) backend for quantum computing operations.

    This backend provides access to TKET's quantum circuit compilation,
    optimization, and execution capabilities, including Quantinuum hardware.

    Args:
        device: Device name (e.g., 'aer_simulator', 'H1-1E', 'H1-2E', 'simulator')
        shots: Number of measurement shots
        machine: Specific Quantinuum machine for hardware execution
        api_key: Quantinuum API key (required for hardware)
        **kwargs: Additional backend configuration

    Example:
        >>> # Local simulator
        >>> backend = TKETBackend(device='aer_simulator')
        >>>
        >>> # Quantinuum hardware (requires API key)
        >>> backend = TKETBackend(
        ...     device='H1-1E',
        ...     api_key='your-quantinuum-api-key'
        ... )

    """

    def __init__(
        self,
        device: str = 'aer_simulator',
        shots: int = 1024,
        machine: str | None = None,
        api_key: str | None = None,
        **kwargs
    ) -> None:
        if not TKET_AVAILABLE:
            raise ImportError(
                "TKET not available. Install with: pip install pytket"
            )

        self.device_name = device
        self.machine = machine or device
        self.api_key = api_key
        self._backend = None

        super().__init__(device=device, shots=shots, **kwargs)

    def _initialize_backend(self) -> None:
        """Initialize TKET backend and device."""
        if self.device_name in ('aer_simulator', 'simulator'):
            if not TKET_EXTENSIONS_AVAILABLE:
                raise ImportError(
                    "Local TKET simulation needs pytket-qiskit. Install with: pip install pytket-qiskit"
                )
            self._backend = AerBackend()
        elif self.device_name.startswith('H') and '-' in self.device_name:
            # Quantinuum hardware or emulators (untested: needs a Quantinuum account)
            if not QUANTINUUM_AVAILABLE:
                raise ImportError("Quantinuum backend not available. Install with: pip install pytket-quantinuum")
            self._backend = QuantinuumBackend(device_name=self.machine)
        elif self.device_name in ('cirq', 'cirq_simulator'):
            if not TKET_CIRQ_AVAILABLE:
                raise ImportError("TKET Cirq extension not available. Install with: pip install pytket-cirq")
            self._backend = TKETCirqBackend()
        else:
            raise ValueError(
                f"Unknown TKET device '{self.device_name}'. Use 'aer_simulator', "
                "'cirq_simulator' or a Quantinuum device name such as 'H1-1E'."
            )

        self.capabilities = {
            'supports_measurements': True,
            'supports_parameterized_circuits': True,
            'supports_classical_control': True,
            'supports_mid_circuit_measurement': True,
            'max_qubits': self._get_max_qubits(),
            'native_gates': self._get_native_gates(),
            'supports_optimization': True,
        }

    def _get_max_qubits(self) -> int:
        """Get maximum number of qubits for the device."""
        if self._backend and hasattr(self._backend, 'device'):
            try:
                device_info = self._backend.device
                if hasattr(device_info, 'n_nodes'):
                    return device_info.n_nodes
            except (AttributeError, Exception):
                pass

        # Default estimates based on device name
        if 'H1-1' in self.device_name:
            return 20  # H1-1E
        elif 'H1-2' in self.device_name:
            return 56  # H1-2E
        elif 'H2' in self.device_name:
            return 32  # H2-1
        else:
            return 32  # Conservative estimate

    def _get_native_gates(self) -> list[str]:
        """Get native gates supported by TKET."""
        # TKET supports a rich set of gates
        return [
            'H', 'X', 'Y', 'Z', 'S', 'Sdg', 'T', 'Tdg', 'V', 'Vdg',
            'Rx', 'Ry', 'Rz', 'R1', 'U1', 'U2', 'U3',
            'CX', 'CY', 'CZ', 'CH', 'CV', 'CVdg', 'CS', 'CSdg',
            'CRx', 'CRy', 'CRz', 'CU1', 'CU2', 'CU3',
            'CCX', 'SWAP', 'CSWAP', 'ISWAP', 'PHASEDX',
            'ZZMax', 'XXPhase', 'YYPhase', 'ZZPhase'
        ]

    # ========================================================================
    # Core Circuit Operations
    # ========================================================================

    def create_circuit(self, n_qubits: int) -> Circuit:
        """Create a TKET quantum circuit."""
        if n_qubits > self._get_max_qubits():
            logger.warning(f"Requested {n_qubits} qubits exceeds device limit")

        circuit = Circuit(n_qubits)
        return circuit

    _GATES = {
        'i': 'noop', 'x': 'X', 'y': 'Y', 'z': 'Z', 'h': 'H', 'hadamard': 'H',
        's': 'S', 'sdg': 'Sdg', 't': 'T', 'tdg': 'Tdg',
        'rx': 'Rx', 'ry': 'Ry', 'rz': 'Rz', 'p': 'U1', 'phase': 'U1', 'u1': 'U1',
        'cnot': 'CX', 'cx': 'CX', 'cz': 'CZ', 'cy': 'CY', 'ch': 'CH', 'swap': 'SWAP',
        'crx': 'CRx', 'cry': 'CRy', 'crz': 'CRz',
        'ccnot': 'CCX', 'ccx': 'CCX', 'toffoli': 'CCX', 'cswap': 'CSWAP', 'fredkin': 'CSWAP',
    }

    def add_gate(self, circuit: Circuit, gate: str, qubits: int | list[int],
                 params: list[float] | None = None) -> Circuit:
        """Add a gate. Angles are given in radians (converted to TKET half-turns)."""
        if isinstance(qubits, int):
            qubits = [qubits]
        name = self._GATES.get(gate.lower())
        if name is None:
            raise ValueError(f"Gate '{gate}' not supported in TKET backend")
        op_type = getattr(OpType, name)
        tket_qubits = [circuit.qubits[q] for q in qubits]
        if params:
            circuit.add_gate(op_type, [float(p) / np.pi for p in params], tket_qubits)
        else:
            circuit.add_gate(op_type, tket_qubits)
        return circuit

    def add_measurement(self, circuit: Circuit, qubits: int | list[int] | None = None) -> Circuit:
        """Measure ``qubits`` (all qubits by default) into classical bits 0..k-1."""
        if qubits is None:
            qubits = list(range(circuit.n_qubits))
        elif isinstance(qubits, int):
            qubits = [qubits]
        if circuit.n_bits < len(qubits):
            circuit.add_c_register("c", len(qubits) - circuit.n_bits)
        for i, qubit in enumerate(qubits):
            circuit.Measure(circuit.qubits[qubit], circuit.bits[i])
        return circuit

    def execute_circuit(self, circuit: Circuit, shots: int | None = None) -> dict[str, Any]:
        """Compile and run the circuit. Counts are big-endian (bit 0 is the leftmost character).

        Unmeasured circuits get all qubits measured on a copy.
        """
        shots = shots or self.shots
        if circuit.n_gates_of_type(OpType.Measure) == 0:
            circuit = circuit.copy()
            circuit.measure_all()
        compiled_circuit = self._backend.get_compiled_circuit(circuit)
        handle = self._backend.process_circuit(compiled_circuit, n_shots=shots)
        result = self._backend.get_result(handle)
        counts = {''.join(str(int(b)) for b in key): int(v) for key, v in result.get_counts().items()}
        return {
            'counts': counts,
            'shots': shots,
            'success': True,
            'backend': self._backend.__class__.__name__,
            'compiled': True,
            'device': self.device_name,
        }

    # ========================================================================
    # TKET-Specific Features
    # ========================================================================

    def optimize_circuit(self, circuit: Circuit, optimization_level: int = 2) -> Circuit:
        """Optimize quantum circuit using TKET compiler passes."""
        try:
            from pytket.passes import (
                CliffordSimp,
                DecomposeBoxes,
                OptimisePhaseGadgets,
                RemoveRedundancies,
                SequencePass,
            )

            optimized = circuit.copy()

            if optimization_level >= 1:
                # Basic optimization
                basic_pass = SequencePass([
                    DecomposeBoxes(),
                    CliffordSimp(),
                    RemoveRedundancies()
                ])
                basic_pass.apply(optimized)

            if optimization_level >= 2:
                # Advanced optimization
                advanced_pass = SequencePass([
                    OptimisePhaseGadgets(),
                    CliffordSimp(),
                    RemoveRedundancies()
                ])
                advanced_pass.apply(optimized)

            if optimization_level >= 3 and self._backend:
                # Backend-specific optimization
                backend_pass = self._backend.default_compilation_pass()
                backend_pass.apply(optimized)

            logger.info(f"Circuit optimized: {circuit.n_gates} -> {optimized.n_gates} gates")
            return optimized

        except Exception as e:
            logger.error(f"Circuit optimization failed: {e}")
            return circuit

    # ========================================================================
    # Backend Information
    # ========================================================================

    def get_backend_info(self) -> dict[str, Any]:
        """Get information about the TKET backend."""
        info = {
            'backend_name': 'tket',
            'device': self.device_name,
            'provider': 'Quantinuum/Cambridge Quantum Computing',
            'shots': self.shots,
            'capabilities': self.capabilities,
        }

        if self._backend:
            info.update({
                'backend_class': self._backend.__class__.__name__,
                'supports_compilation': True,
                'supports_optimization': True,
            })

            if hasattr(self._backend, 'device'):
                device_info = self._backend.device
                info['device_info'] = str(device_info)

        return info

    def get_version_info(self) -> dict[str, str]:
        """Get version information for TKET dependencies."""
        version_info = {'backend_version': '1.0.0'}

        try:
            import pytket
            version_info['pytket'] = pytket.__version__
        except (ImportError, AttributeError):
            pass

        if QUANTINUUM_AVAILABLE:
            try:
                import pytket.extensions.quantinuum
                version_info['pytket_quantinuum'] = getattr(pytket.extensions.quantinuum, '__version__', 'unknown')
            except (ImportError, AttributeError):
                pass

        return version_info

    def is_available(self) -> bool:
        """Check if the backend is available and properly configured."""
        return TKET_AVAILABLE

    def get_circuit_info(self) -> dict[str, Any]:
        """Get information about circuit execution capabilities."""
        return {
            'max_qubits': self._get_max_qubits(),
            'native_gates': self._get_native_gates(),
            'supports_mid_circuit_measurement': True,
            'supports_reset': True,
            'supports_conditional': True,
            'supports_optimization': True,
            'supports_compilation': True,
        }

    def _get_n_qubits(self, circuit: Circuit) -> int:
        """Get number of qubits in TKET circuit."""
        return circuit.n_qubits

    def get_statevector(self, circuit: Circuit) -> np.ndarray:
        """Exact statevector (big-endian) computed by pytket, ignoring measurements."""
        clean = Circuit(circuit.n_qubits)
        for cmd in circuit.get_commands():
            if cmd.op.type == OpType.Measure:
                continue
            clean.add_gate(cmd.op, cmd.args)
        return np.asarray(clean.get_statevector(), dtype=complex)
