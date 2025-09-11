"""D-Wave Ocean backend implementation for SuperQuantX.

This module provides integration with D-Wave Ocean SDK for quantum annealing
and optimization problems using D-Wave quantum annealing processors.

Note: This backend focuses on QUBO/Ising model problems rather than
gate-model quantum circuits.
"""

import logging
from typing import Any

import numpy as np

from .base_backend import BaseBackend


logger = logging.getLogger(__name__)

# Try to import D-Wave Ocean
try:
    import dimod
    import networkx as nx
    from dwave.embedding import embed_ising, unembed_sampleset
    from dwave.samplers import SimulatedAnnealingSampler
    from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
    OCEAN_AVAILABLE = True
except ImportError:
    OCEAN_AVAILABLE = False
    dimod = None
    DWaveSampler = None
    LeapHybridSampler = None
    SimulatedAnnealingSampler = None
    nx = None

class OceanBackend(BaseBackend):
    """D-Wave Ocean backend for quantum annealing operations.

    This backend provides access to D-Wave's quantum annealing systems
    for solving optimization problems formulated as QUBO or Ising models.

    Note: Unlike gate-model quantum computers, D-Wave systems solve optimization
    problems rather than running quantum circuits.

    Args:
        device: Sampler type ('DWave', 'hybrid', 'simulator', 'advantage')
        solver: Specific solver name (optional)
        token: D-Wave API token (required for hardware)
        endpoint: D-Wave API endpoint
        shots: Number of samples (reads)
        **kwargs: Additional sampler configuration

    Example:
        >>> # Simulated annealing (no hardware required)
        >>> backend = OceanBackend(device='simulator')
        >>>
        >>> # D-Wave hardware (requires API token)
        >>> backend = OceanBackend(
        ...     device='advantage',
        ...     token='your-dwave-token'
        ... )

    """

    def __init__(
        self,
        device: str = 'simulator',
        solver: str | None = None,
        token: str | None = None,
        endpoint: str | None = None,
        shots: int = 1000,  # Called "num_reads" in D-Wave
        **kwargs
    ) -> None:
        if not OCEAN_AVAILABLE:
            raise ImportError(
                "D-Wave Ocean not available. Install with: pip install dwave-ocean-sdk"
            )

        self.device_name = device
        self.solver_name = solver
        self.token = token
        self.endpoint = endpoint
        self.num_reads = shots
        self._sampler = None
        self._is_quantum_annealing = True  # This backend is for annealing, not gates

        super().__init__(device=device, shots=shots, **kwargs)

    def _initialize_backend(self) -> None:
        """Initialize D-Wave Ocean backend and sampler."""
        try:
            if self.device_name == 'simulator':
                # Simulated annealing sampler (classical simulation)
                self._sampler = SimulatedAnnealingSampler()
                logger.info("Initialized D-Wave simulated annealing sampler")

            elif self.device_name in ['dwave', 'advantage', 'advantage2']:
                # D-Wave quantum annealer
                if not self.token:
                    logger.warning("No D-Wave token provided, attempting to use environment/config")

                # Create base sampler
                base_sampler = DWaveSampler(
                    solver=self.solver_name,
                    token=self.token,
                    endpoint=self.endpoint
                )

                # Wrap with embedding for automatic minor-embedding
                self._sampler = EmbeddingComposite(base_sampler)
                logger.info(f"Initialized D-Wave hardware sampler: {base_sampler.solver.name}")

            elif self.device_name == 'hybrid':
                # D-Wave Leap Hybrid solver
                self._sampler = LeapHybridSampler(
                    token=self.token,
                    endpoint=self.endpoint
                )
                logger.info("Initialized D-Wave Leap Hybrid sampler")

            else:
                raise ValueError(f"Unknown D-Wave device: {self.device_name}")

            self.capabilities = {
                'supports_measurements': False,  # Not applicable for annealing
                'supports_parameterized_circuits': False,  # Not applicable
                'supports_classical_control': False,
                'supports_optimization': True,  # Main capability
                'supports_qubo': True,
                'supports_ising': True,
                'max_variables': self._get_max_variables(),
                'annealing_backend': True,
            }

        except Exception as e:
            logger.error(f"Failed to initialize D-Wave Ocean backend: {e}")
            raise

    def _get_max_variables(self) -> int:
        """Get maximum number of variables for the sampler."""
        if hasattr(self._sampler, 'properties'):
            try:
                return len(self._sampler.properties['qubits'])
            except (AttributeError, KeyError, Exception):
                pass

        # Default estimates
        if self.device_name == 'simulator':
            return 10000  # Simulated annealer can handle large problems
        elif self.device_name in ['advantage', 'advantage2']:
            return 5000  # Advantage systems have ~5000 qubits
        else:
            return 2000  # Conservative estimate

    # ========================================================================
    # Optimization Problem Interface (replaces circuit operations)
    # ========================================================================

    def solve_qubo(self, Q: dict[tuple[int, int], float], **kwargs) -> dict[str, Any]:
        """Solve a Quadratic Unconstrained Binary Optimization (QUBO) problem.

        Args:
            Q: QUBO dictionary where keys are (i, j) variable pairs and values are coefficients
            **kwargs: Additional solving parameters

        Returns:
            Solution results with energies and samples

        """
        try:
            num_reads = kwargs.get('num_reads', self.num_reads)

            # Create BQM from QUBO
            bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

            # Sample from the BQM
            sampleset = self._sampler.sample(bqm, num_reads=num_reads, **kwargs)

            # Convert results to standard format
            results = {
                'samples': [],
                'energies': [],
                'num_occurrences': [],
                'success': True,
                'num_reads': len(sampleset),
                'timing': getattr(sampleset, 'info', {}).get('timing', {}),
                'problem_type': 'QUBO'
            }

            for sample, energy, num_occur in sampleset.data(['sample', 'energy', 'num_occurrences']):
                results['samples'].append(dict(sample))
                results['energies'].append(energy)
                results['num_occurrences'].append(num_occur)

            return results

        except Exception as e:
            logger.error(f"QUBO solving failed: {e}")
            return {
                'samples': [],
                'energies': [],
                'success': False,
                'error': str(e),
                'problem_type': 'QUBO'
            }

    def solve_ising(self, h: dict[int, float], J: dict[tuple[int, int], float], **kwargs) -> dict[str, Any]:
        """Solve an Ising model problem.

        Args:
            h: Linear coefficients (bias terms) for each variable
            J: Quadratic coefficients (coupling terms) between variables
            **kwargs: Additional solving parameters

        Returns:
            Solution results with energies and samples

        """
        try:
            num_reads = kwargs.get('num_reads', self.num_reads)

            # Create BQM from Ising model
            bqm = dimod.BinaryQuadraticModel.from_ising(h, J)

            # Sample from the BQM
            sampleset = self._sampler.sample(bqm, num_reads=num_reads, **kwargs)

            # Convert results to standard format
            results = {
                'samples': [],
                'energies': [],
                'num_occurrences': [],
                'success': True,
                'num_reads': len(sampleset),
                'timing': getattr(sampleset, 'info', {}).get('timing', {}),
                'problem_type': 'Ising'
            }

            for sample, energy, num_occur in sampleset.data(['sample', 'energy', 'num_occurrences']):
                results['samples'].append(dict(sample))
                results['energies'].append(energy)
                results['num_occurrences'].append(num_occur)

            return results

        except Exception as e:
            logger.error(f"Ising model solving failed: {e}")
            return {
                'samples': [],
                'energies': [],
                'success': False,
                'error': str(e),
                'problem_type': 'Ising'
            }

    def solve_optimization_problem(self, problem: dict | Any, problem_type: str = 'auto') -> dict[str, Any]:
        """Generic optimization problem solver.

        Args:
            problem: Problem specification (QUBO dict, Ising model, or BQM)
            problem_type: Type of problem ('qubo', 'ising', 'bqm', 'auto')

        Returns:
            Solution results

        """
        try:
            if problem_type == 'auto':
                # Try to detect problem type
                if isinstance(problem, dict) and all(len(k) == 2 for k in problem.keys()):
                    problem_type = 'qubo'
                elif hasattr(problem, 'linear') and hasattr(problem, 'quadratic'):
                    problem_type = 'bqm'
                else:
                    raise ValueError("Cannot auto-detect problem type")

            if problem_type == 'qubo':
                return self.solve_qubo(problem)
            elif problem_type == 'ising':
                h, J = problem  # Expecting tuple (h, J)
                return self.solve_ising(h, J)
            elif problem_type == 'bqm':
                # Direct BQM solving
                sampleset = self._sampler.sample(problem, num_reads=self.num_reads)
                return self._convert_sampleset(sampleset, problem_type)
            else:
                raise ValueError(f"Unknown problem type: {problem_type}")

        except Exception as e:
            logger.error(f"Optimization problem solving failed: {e}")
            return {
                'samples': [],
                'energies': [],
                'success': False,
                'error': str(e),
                'problem_type': problem_type
            }

    def _convert_sampleset(self, sampleset, problem_type: str) -> dict[str, Any]:
        """Convert D-Wave sampleset to standard format."""
        results = {
            'samples': [],
            'energies': [],
            'num_occurrences': [],
            'success': True,
            'num_reads': len(sampleset),
            'timing': getattr(sampleset, 'info', {}).get('timing', {}),
            'problem_type': problem_type
        }

        for sample, energy, num_occur in sampleset.data(['sample', 'energy', 'num_occurrences']):
            results['samples'].append(dict(sample))
            results['energies'].append(energy)
            results['num_occurrences'].append(num_occur)

        return results

    # ========================================================================
    # Gate Model Compatibility (Limited Support)
    # ========================================================================

    def create_circuit(self, n_qubits: int) -> dict[str, Any]:
        """Limited circuit support for compatibility.

        Note: D-Wave is not a gate-model quantum computer.
        This returns a placeholder for optimization problems.
        """
        logger.warning("D-Wave Ocean backend does not support gate-model circuits")
        return {
            'type': 'optimization_placeholder',
            'n_variables': n_qubits,
            'backend': 'ocean',
            'problem': None
        }

    def add_gate(self, circuit: dict, gate: str, qubits: int | list[int],
                 params: list[float] | None = None) -> dict:
        """Limited gate support - not applicable for annealing."""
        logger.warning("Gate operations not supported on D-Wave annealing backend")
        return circuit

    def add_measurement(self, circuit: dict, qubits: int | list[int]) -> dict:
        """Limited measurement support - not applicable for annealing."""
        logger.warning("Measurements not applicable for D-Wave annealing backend")
        return circuit

    def execute_circuit(self, circuit: dict, shots: int | None = None) -> dict[str, Any]:
        """Limited circuit execution - redirects to optimization solving."""
        logger.warning("Circuit execution not supported - use solve_qubo() or solve_ising() instead")
        return {
            'counts': {},
            'shots': shots or self.shots,
            'success': False,
            'error': 'Use optimization problem methods instead of circuit execution',
            'backend': 'ocean'
        }

    # ========================================================================
    # Backend Information
    # ========================================================================

    def get_backend_info(self) -> dict[str, Any]:
        """Get information about the D-Wave Ocean backend."""
        info = {
            'backend_name': 'ocean',
            'device': self.device_name,
            'provider': 'D-Wave Systems',
            'num_reads': self.num_reads,
            'capabilities': self.capabilities,
            'quantum_annealing': True,
            'gate_model': False,
        }

        if self._sampler and hasattr(self._sampler, 'properties'):
            try:
                props = self._sampler.properties
                info.update({
                    'solver_name': getattr(self._sampler, 'solver', {}).get('name', 'unknown'),
                    'num_qubits': len(props.get('qubits', [])),
                    'connectivity': 'Chimera/Pegasus/Zephyr',  # D-Wave topologies
                })
            except (AttributeError, Exception):
                pass

        return info

    def get_version_info(self) -> dict[str, str]:
        """Get version information for Ocean dependencies."""
        version_info = {'backend_version': '1.0.0'}

        try:
            version_info['dimod'] = dimod.__version__
        except (AttributeError, ImportError):
            pass

        try:
            import dwave.system
            version_info['dwave_system'] = dwave.system.__version__
        except (ImportError, AttributeError):
            pass

        try:
            import dwave.samplers
            version_info['dwave_samplers'] = dwave.samplers.__version__
        except (ImportError, AttributeError):
            pass

        return version_info

    def is_available(self) -> bool:
        """Check if the backend is available and properly configured."""
        return OCEAN_AVAILABLE and self._sampler is not None

    def get_circuit_info(self) -> dict[str, Any]:
        """Get circuit capabilities (limited for annealing backend)."""
        return {
            'max_qubits': 0,  # Not applicable
            'max_variables': self._get_max_variables(),
            'native_gates': [],  # Not applicable
            'supports_mid_circuit_measurement': False,
            'supports_reset': False,
            'supports_conditional': False,
            'supports_optimization': True,
            'quantum_annealing': True,
        }

    def _get_n_qubits(self, circuit: dict) -> int:
        """Get number of qubits/variables (not applicable for annealing)."""
        if isinstance(circuit, dict):
            return circuit.get('n_variables', 0)
        return 0

    def get_statevector(self, circuit: dict) -> np.ndarray:
        """Get statevector (not applicable for annealing backend)."""
        logger.warning("Statevector not applicable for quantum annealing backend")
        return np.array([1.0 + 0j])  # Dummy statevector

    # ========================================================================
    # Convenience Methods for Common Optimization Problems
    # ========================================================================

    def solve_max_cut(self, graph: Any | list[tuple], **kwargs) -> dict[str, Any]:
        """Solve Maximum Cut problem on a graph."""
        try:
            if not nx:
                raise ImportError("NetworkX not available for graph operations")

            if isinstance(graph, list):
                # Convert edge list to networkx graph
                G = nx.Graph()
                G.add_edges_from(graph)
            else:
                G = graph

            # Convert to QUBO formulation
            Q = {}
            for i, j in G.edges():
                Q[(i, i)] = Q.get((i, i), 0) + 1
                Q[(j, j)] = Q.get((j, j), 0) + 1
                Q[(i, j)] = Q.get((i, j), 0) - 2

            return self.solve_qubo(Q, **kwargs)

        except Exception as e:
            logger.error(f"Max Cut solving failed: {e}")
            return {'success': False, 'error': str(e)}

    def solve_tsp(self, distance_matrix: np.ndarray, **kwargs) -> dict[str, Any]:
        """Solve Traveling Salesman Problem (simplified formulation)."""
        try:
            n_cities = len(distance_matrix)

            # This is a simplified TSP formulation
            # Full TSP requires more complex constraints
            logger.warning("TSP formulation is simplified - may not give valid tours")

            Q = {}
            # Add distance costs
            for i in range(n_cities):
                for j in range(i+1, n_cities):
                    for t in range(n_cities):
                        # Variable x_it means city i is visited at time t
                        var_i = i * n_cities + t
                        var_j = j * n_cities + ((t + 1) % n_cities)
                        Q[(var_i, var_j)] = distance_matrix[i, j]

            return self.solve_qubo(Q, **kwargs)

        except Exception as e:
            logger.error(f"TSP solving failed: {e}")
            return {'success': False, 'error': str(e)}
