"""AWS Braket backend implementation for SuperQuantX.

This module provides integration with Amazon Braket for quantum computing
on AWS quantum hardware and simulators.
"""

import logging
from typing import Any

import numpy as np

from .base_backend import BaseBackend


logger = logging.getLogger(__name__)

# Try to import AWS Braket
try:
    from braket.circuits import Circuit as BraketCircuit
    from braket.circuits import Instruction, Measure
    from braket.circuits import gates as bg
    from braket.devices import LocalSimulator
    try:
        from braket.aws import AwsDevice
        AWS_AVAILABLE = True
    except ImportError:
        AWS_AVAILABLE = False
    BRAKET_AVAILABLE = True
except ImportError:
    BRAKET_AVAILABLE = False
    AWS_AVAILABLE = False
    BraketCircuit = None
    LocalSimulator = None

class BraketBackend(BaseBackend):
    """AWS Braket backend for quantum computing operations.

    This backend provides access to AWS Braket's quantum devices including
    local simulators, managed simulators, and QPU hardware from IonQ, Rigetti,
    and other providers available on AWS.

    Args:
        device: Device name or ARN (e.g., 'local:braket/braket_sv', 'arn:aws:braket::device/qpu/ionq/ionQdevice')
        shots: Number of measurement shots
        aws_session: Optional AWS session for authentication
        s3_folder: S3 bucket folder for results (required for hardware)
        **kwargs: Additional device configuration

    Example:
        >>> # Local simulator
        >>> backend = BraketBackend(device='local:braket/braket_sv')
        >>>
        >>> # AWS hardware (requires authentication)
        >>> backend = BraketBackend(
        ...     device='arn:aws:braket::device/qpu/ionq/ionQdevice',
        ...     s3_folder=('my-bucket', 'quantum-results')
        ... )

    """

    def __init__(
        self,
        device: str = 'local:braket/braket_sv',
        shots: int = 1024,
        aws_session = None,
        s3_folder: tuple[str, str] | None = None,
        **kwargs
    ) -> None:
        if not BRAKET_AVAILABLE:
            raise ImportError(
                "AWS Braket not available. Install with: pip install amazon-braket-sdk"
            )

        self.device_name = device
        self.aws_session = aws_session
        self.s3_folder = s3_folder
        self._device = None

        super().__init__(device=device, shots=shots, **kwargs)

    def _initialize_backend(self) -> None:
        """Initialize AWS Braket backend and device."""
        try:
            if self.device_name.startswith('local:'):
                # Local simulator
                # Accept 'local:braket_sv' and the documented 'local:braket/braket_sv'
                simulator_name = self.device_name.split('local:', 1)[1]
                if simulator_name.startswith('braket/'):
                    simulator_name = simulator_name[len('braket/'):]
                self._device = LocalSimulator(simulator_name)
                logger.info(f"Initialized Braket local simulator: {simulator_name}")

            elif self.device_name.startswith('arn:aws:braket'):
                # AWS hardware/managed simulators
                if not AWS_AVAILABLE:
                    raise ImportError("AWS Braket cloud access not available")

                self._device = AwsDevice(
                    arn=self.device_name,
                    aws_session=self.aws_session
                )
                logger.info(f"Initialized AWS Braket device: {self.device_name}")

                # Verify S3 folder for hardware
                if 'qpu' in self.device_name.lower() and not self.s3_folder:
                    logger.warning("QPU devices typically require S3 folder for results")

            else:
                raise ValueError(f"Invalid device name: {self.device_name}")

            self.capabilities = {
                'supports_measurements': True,
                'supports_parameterized_circuits': True,
                'supports_classical_control': False,  # Limited support
                'max_qubits': self._get_max_qubits(),
                'native_gates': self._get_native_gates(),
            }

        except Exception as e:
            logger.error(f"Failed to initialize Braket backend: {e}")
            raise

    def _get_max_qubits(self) -> int:
        """Get maximum number of qubits for the device."""
        if hasattr(self._device, 'properties'):
            try:
                return self._device.properties.paradigm.qubit_count
            except (AttributeError, Exception):
                pass
        # Default estimates
        if 'sv' in self.device_name:
            return 34  # SV1 simulator
        elif 'tn1' in self.device_name:
            return 50  # TN1 tensor network
        elif 'dm1' in self.device_name:
            return 17  # DM1 density matrix
        else:
            return 30  # Conservative estimate for hardware

    def _get_native_gates(self) -> list[str]:
        """Get native gates supported by the device."""
        # Common Braket gates
        return [
            'i', 'x', 'y', 'z', 'h', 's', 'si', 't', 'ti',
            'rx', 'ry', 'rz', 'cnot', 'cz', 'cy', 'swap',
            'iswap', 'pswap', 'phaseshift', 'cphaseshift',
            'ccnot', 'cswap'
        ]

    # ========================================================================
    # Core Circuit Operations
    # ========================================================================

    def create_circuit(self, n_qubits: int) -> BraketCircuit:
        """Create a Braket circuit on ``n_qubits`` qubits.

        Braket circuits only know about qubits that carry an instruction, so an
        identity is placed on every qubit to fix the register size.
        """
        if n_qubits > self._get_max_qubits():
            raise ValueError(
                f"Requested {n_qubits} qubits exceeds the device limit of {self._get_max_qubits()}"
            )
        circuit = BraketCircuit()
        for q in range(n_qubits):
            circuit.add_instruction(Instruction(bg.I(), q))
        return circuit

    _GATES = {
        'i': 'I', 'x': 'X', 'y': 'Y', 'z': 'Z', 'h': 'H', 'hadamard': 'H',
        's': 'S', 'sdg': 'Si', 't': 'T', 'tdg': 'Ti',
        'rx': 'Rx', 'ry': 'Ry', 'rz': 'Rz', 'p': 'PhaseShift', 'phase': 'PhaseShift',
        'phaseshift': 'PhaseShift', 'cphaseshift': 'CPhaseShift',
        'cnot': 'CNot', 'cx': 'CNot', 'cz': 'CZ', 'cy': 'CY',
        'swap': 'Swap', 'iswap': 'ISwap',
        'ccnot': 'CCNot', 'ccx': 'CCNot', 'toffoli': 'CCNot',
        'cswap': 'CSwap', 'fredkin': 'CSwap',
    }
    _PARAMETRIC = {'Rx', 'Ry', 'Rz', 'PhaseShift', 'CPhaseShift'}

    def add_gate(self, circuit: BraketCircuit, gate: str, qubits: int | list[int],
                 params: list[float] | None = None) -> BraketCircuit:
        """Add a quantum gate to the circuit."""
        if isinstance(qubits, int):
            qubits = [qubits]
        name = self._GATES.get(gate.lower())
        if name is None:
            raise ValueError(f"Gate '{gate}' not supported in Braket backend")
        gate_class = getattr(bg, name)
        if name in self._PARAMETRIC:
            if not params:
                raise ValueError(f"Gate '{gate}' needs a rotation angle")
            op = gate_class(float(params[0]))
        else:
            op = gate_class()
        target = qubits[0] if len(qubits) == 1 else list(qubits)
        circuit.add_instruction(Instruction(op, target))
        return circuit

    def add_measurement(self, circuit: BraketCircuit, qubits: int | list[int] | None = None) -> BraketCircuit:
        """Record which qubits to measure (all qubits when ``qubits`` is None)."""
        if qubits is None:
            qubits = list(range(circuit.qubit_count))
        elif isinstance(qubits, int):
            qubits = [qubits]
        if not hasattr(circuit, '_measurement_qubits'):
            circuit._measurement_qubits = set()
        circuit._measurement_qubits.update(qubits)
        return circuit

    def execute_circuit(self, circuit: BraketCircuit, shots: int | None = None) -> dict[str, Any]:
        """Sample the circuit. Counts are big-endian (qubit 0 is the leftmost bit)."""
        shots = shots or self.shots
        measure_qubits = sorted(getattr(circuit, '_measurement_qubits', range(circuit.qubit_count)))

        measured_circuit = circuit.copy()
        if measure_qubits != list(range(circuit.qubit_count)):
            measured_circuit.measure(measure_qubits)

        if self.s3_folder and not self.device_name.startswith('local:'):
            task = self._device.run(measured_circuit, shots=shots, s3_destination_folder=self.s3_folder)
        else:
            task = self._device.run(measured_circuit, shots=shots)
        result = task.result()
        counts = {str(k): int(v) for k, v in result.measurement_counts.items()}

        return {
            'counts': counts,
            'shots': shots,
            'success': True,
            'measurement_qubits': measure_qubits,
            'task_arn': getattr(task, 'id', 'local'),
            'device': self.device_name,
        }

    # ========================================================================
    # Algorithm-Specific Operations
    # ========================================================================

    def create_parameterized_circuit(self, n_qubits: int, n_params: int) -> tuple[BraketCircuit, list[str]]:
        """Create a parameterized quantum circuit for variational algorithms."""
        circuit = self.create_circuit(n_qubits)

        # Create parameter names
        param_names = [f"theta_{i}" for i in range(n_params)]

        # Store parameter info
        circuit._param_names = param_names
        circuit._n_params = n_params

        return circuit, param_names

    # ========================================================================
    # Backend Information
    # ========================================================================

    def get_backend_info(self) -> dict[str, Any]:
        """Get information about the Braket backend."""
        info = {
            'backend_name': 'braket',
            'device': self.device_name,
            'provider': 'AWS Braket',
            'shots': self.shots,
            'capabilities': self.capabilities,
            'local_simulator': self.device_name.startswith('local:'),
        }

        if hasattr(self._device, 'properties'):
            try:
                props = self._device.properties
                info.update({
                    'device_type': getattr(props, 'deviceType', 'unknown'),
                    'provider_name': getattr(props, 'providerName', 'AWS'),
                    'max_qubits': getattr(props.paradigm, 'qubit_count', 'unknown') if hasattr(props, 'paradigm') else 'unknown',
                })
            except (AttributeError, Exception):
                pass

        return info

    def get_version_info(self) -> dict[str, str]:
        """Get version information for Braket dependencies."""
        from importlib.metadata import PackageNotFoundError, version
        try:
            sdk = version('amazon-braket-sdk')
        except PackageNotFoundError:
            sdk = 'unknown'
        return {'braket_sdk': sdk, 'backend_version': '1.0.0'}

    def is_available(self) -> bool:
        """Check if the backend is available and properly configured."""
        return BRAKET_AVAILABLE and self._device is not None

    def get_circuit_info(self) -> dict[str, Any]:
        """Get information about circuit execution capabilities."""
        return {
            'max_qubits': self._get_max_qubits(),
            'native_gates': self._get_native_gates(),
            'supports_mid_circuit_measurement': False,
            'supports_reset': False,
            'supports_conditional': False,
        }

    def _get_n_qubits(self, circuit: BraketCircuit) -> int:
        """Get number of qubits in Braket circuit."""
        return circuit.qubit_count

    def get_statevector(self, circuit: BraketCircuit) -> np.ndarray:
        """Exact statevector (big-endian) from Braket's local state-vector simulator."""
        sv_circuit = BraketCircuit()
        for instr in circuit.instructions:
            if not isinstance(instr.operator, Measure):
                sv_circuit.add_instruction(instr)
        sv_circuit.state_vector()
        result = LocalSimulator('braket_sv').run(sv_circuit, shots=0).result()
        return np.asarray(result.values[0], dtype=complex)
