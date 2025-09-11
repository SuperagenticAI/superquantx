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
    from pytket.backends import Backend
    from pytket.backends.simulator import AerBackend
    from pytket.circuit import Bit, Qubit

    # Try to import Quantinuum backend
    try:
        from pytket.extensions.quantinuum import QuantinuumBackend
        QUANTINUUM_AVAILABLE = True
    except ImportError:
        QUANTINUUM_AVAILABLE = False

    # Try to import other TKET backends
    try:
        from pytket.extensions.cirq import CirqBackend as TKETCirqBackend
        from pytket.extensions.qiskit import AerStateBackend
        TKET_EXTENSIONS_AVAILABLE = True
    except ImportError:
        TKET_EXTENSIONS_AVAILABLE = False

    TKET_AVAILABLE = True

except ImportError:
    TKET_AVAILABLE = False
    QUANTINUUM_AVAILABLE = False
    TKET_EXTENSIONS_AVAILABLE = False
    Circuit = None
    OpType = None

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
        try:
            if self.device_name in ['aer_simulator', 'simulator']:
                # Use TKET Aer simulator
                if TKET_EXTENSIONS_AVAILABLE:
                    from pytket.extensions.qiskit import AerBackend
                    self._backend = AerBackend()
                else:
                    # Fallback to basic simulator
                    logger.warning("TKET extensions not available, using basic simulator")
                    self._backend = None

            elif self.device_name.startswith('H') and '-' in self.device_name:
                # Quantinuum hardware (H1-1E, H1-2E, H2-1, etc.)
                if not QUANTINUUM_AVAILABLE:
                    raise ImportError("Quantinuum backend not available. Install with: pip install pytket-quantinuum")

                if not self.api_key:
                    raise ValueError("API key required for Quantinuum hardware")

                self._backend = QuantinuumBackend(
                    device_name=self.machine,
                    api_key=self.api_key
                )
                logger.info(f"Initialized Quantinuum backend: {self.machine}")

            elif self.device_name in ['cirq', 'cirq_simulator']:
                # TKET Cirq integration
                if TKET_EXTENSIONS_AVAILABLE:
                    self._backend = TKETCirqBackend()
                else:
                    raise ImportError("TKET Cirq extension not available")

            else:
                # Try to create generic backend
                logger.warning(f"Unknown device {self.device_name}, using simulator fallback")
                self._backend = None

            self.capabilities = {
                'supports_measurements': True,
                'supports_parameterized_circuits': True,
                'supports_classical_control': True,
                'supports_mid_circuit_measurement': True,
                'max_qubits': self._get_max_qubits(),
                'native_gates': self._get_native_gates(),
                'supports_optimization': True,  # TKET's key feature
            }

        except Exception as e:
            logger.error(f"Failed to initialize TKET backend: {e}")
            raise

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

    def add_gate(self, circuit: Circuit, gate: str, qubits: int | list[int],
                 params: list[float] | None = None) -> Circuit:
        """Add a quantum gate to the circuit."""
        if isinstance(qubits, int):
            qubits = [qubits]

        params = params or []

        # Map gate names to TKET OpTypes
        gate_mapping = {
            'i': OpType.noop,  # Identity
            'x': OpType.X, 'y': OpType.Y, 'z': OpType.Z, 'h': OpType.H,
            's': OpType.S, 'sdg': OpType.Sdg, 't': OpType.T, 'tdg': OpType.Tdg,
            'rx': OpType.Rx, 'ry': OpType.Ry, 'rz': OpType.Rz,
            'cnot': OpType.CX, 'cx': OpType.CX, 'cz': OpType.CZ, 'cy': OpType.CY,
            'ch': OpType.CH, 'swap': OpType.SWAP,
            'ccnot': OpType.CCX, 'ccx': OpType.CCX, 'toffoli': OpType.CCX,
            'cswap': OpType.CSWAP, 'fredkin': OpType.CSWAP,
            'iswap': OpType.ISWAPMax,
            'u1': OpType.U1, 'u2': OpType.U2, 'u3': OpType.U3,
        }

        gate_upper = gate.upper()
        gate_lower = gate.lower()

        # Try exact match first, then case variations
        op_type = None
        if hasattr(OpType, gate_upper):
            op_type = getattr(OpType, gate_upper)
        elif gate_lower in gate_mapping:
            op_type = gate_mapping[gate_lower]

        if op_type is None:
            raise ValueError(f"Gate '{gate}' not supported in TKET backend")

        try:
            # Convert qubit indices to TKET Qubit objects
            tket_qubits = [circuit.qubits[q] for q in qubits]

            if len(params) == 0:
                # Parameter-less gates
                circuit.add_gate(op_type, tket_qubits)
            else:
                # Parameterized gates
                circuit.add_gate(op_type, tket_qubits, params)

        except Exception as e:
            logger.error(f"Failed to add gate {gate} to TKET circuit: {e}")
            raise

        return circuit

    def add_measurement(self, circuit: Circuit, qubits: int | list[int]) -> Circuit:
        """Add measurement instructions to specified qubits."""
        if isinstance(qubits, int):
            qubits = [qubits]

        # TKET requires explicit bit allocation for measurements
        n_bits = len(qubits)
        if circuit.n_bits < n_bits:
            circuit.add_c_register("c", n_bits - circuit.n_bits)

        for i, qubit in enumerate(qubits):
            circuit.Measure(circuit.qubits[qubit], circuit.bits[i])

        return circuit

    def execute_circuit(self, circuit: Circuit, shots: int | None = None) -> dict[str, Any]:
        """Execute quantum circuit and return results."""
        shots = shots or self.shots

        try:
            if self._backend is None:
                # Simple simulation fallback
                logger.warning("No TKET backend available, using simulation fallback")
                n_qubits = circuit.n_qubits
                n_bits = circuit.n_bits or n_qubits

                # Generate random results for fallback
                counts = {}
                for _ in range(shots):
                    bitstring = ''.join([str(np.random.randint(0, 2)) for _ in range(n_bits)])
                    counts[bitstring] = counts.get(bitstring, 0) + 1

                return {
                    'counts': counts,
                    'shots': shots,
                    'success': True,
                    'backend': 'tket_fallback',
                }

            # Compile circuit for backend
            compiled_circuit = self._backend.get_compiled_circuit(circuit)

            # Execute on backend
            handle = self._backend.process_circuit(compiled_circuit, n_shots=shots)
            result = self._backend.get_result(handle)

            # Extract counts
            counts_dict = {}
            if hasattr(result, 'get_counts'):
                counts_dict = result.get_counts()
            elif hasattr(result, 'counts'):
                counts_dict = result.counts
            else:
                logger.warning("Could not extract counts from TKET result")

            return {
                'counts': counts_dict,
                'shots': shots,
                'success': True,
                'backend': self._backend.__class__.__name__,
                'compiled': True,
                'device': self.device_name,
            }

        except Exception as e:
            logger.error(f"TKET circuit execution failed: {e}")
            return {
                'counts': {},
                'shots': shots,
                'success': False,
                'error': str(e),
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

    def create_parameterized_circuit(self, n_qubits: int, n_params: int) -> tuple[Circuit, list[str]]:
        """Create a parameterized quantum circuit for variational algorithms."""
        from pytket.circuit import fresh_symbol

        circuit = self.create_circuit(n_qubits)

        # Create symbolic parameters
        symbols = [fresh_symbol(f"theta_{i}") for i in range(n_params)]
        param_names = [str(symbol) for symbol in symbols]

        # Store parameter info
        circuit._symbols = symbols
        circuit._param_names = param_names

        return circuit, param_names

    def bind_parameters(self, circuit: Circuit, param_values: dict[str, float]) -> Circuit:
        """Bind parameter values to parameterized circuit."""
        try:
            bound_circuit = circuit.copy()

            # Create symbol substitution map
            if hasattr(circuit, '_symbols'):
                symbol_map = {}
                for symbol, name in zip(circuit._symbols, circuit._param_names, strict=False):
                    if name in param_values:
                        symbol_map[symbol] = param_values[name]

                # Substitute symbols with values
                bound_circuit.symbol_substitution(symbol_map)

            return bound_circuit

        except Exception as e:
            logger.error(f"Parameter binding failed: {e}")
            return circuit

    def expectation_value(self, circuit: Circuit, observable: str | np.ndarray,
                         shots: int | None = None) -> float:
        """Calculate expectation value of observable using TKET."""
        shots = shots or self.shots

        try:
            if isinstance(observable, str) and observable.upper() == 'Z':
                # Measure in Z basis
                measured_circuit = circuit.copy()
                measured_circuit.add_c_register("c", 1)
                measured_circuit.Measure(measured_circuit.qubits[0], measured_circuit.bits[0])

                result = self.execute_circuit(measured_circuit, shots)
                counts = result['counts']

                # Calculate <Z> expectation
                expectation = 0.0
                total_counts = sum(counts.values())

                for bitstring, count in counts.items():
                    prob = count / total_counts
                    if bitstring[0] == '0':
                        expectation += prob
                    else:
                        expectation -= prob

                return expectation
            else:
                logger.warning("Only Z observable currently implemented")
                return 0.0

        except Exception as e:
            logger.error(f"Expectation value calculation failed: {e}")
            return 0.0

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
        """Get statevector from TKET circuit."""
        try:
            if TKET_EXTENSIONS_AVAILABLE:
                from pytket.extensions.qiskit import AerStateBackend
                state_backend = AerStateBackend()
                compiled_circuit = state_backend.get_compiled_circuit(circuit)
                handle = state_backend.process_circuit(compiled_circuit)
                result = state_backend.get_result(handle)

                if hasattr(result, 'get_state'):
                    return result.get_state()

            logger.warning("Statevector not available with current TKET setup")
            return np.zeros(2**circuit.n_qubits, dtype=complex)

        except Exception as e:
            logger.error(f"Failed to get statevector: {e}")
            return np.zeros(2**circuit.n_qubits, dtype=complex)
