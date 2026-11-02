"""SuperQuantX MCP server (stdio transport).

Run with ``superquantx-mcp`` or ``python -m superquantx.mcp``. Requires the
optional extra: ``pip install "superquantx[mcp]"``.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from . import tools


INSTRUCTIONS = """\
SuperQuantX exposes one API over several quantum SDKs (built-in numpy
simulator, Qiskit Aer, PennyLane, and optionally Cirq, Braket, TKET).
Start with list_backends to see which backends work in this environment,
prefer 'simulator', 'qiskit' and 'pennylane'. Circuits are OpenQASM 2/3 strings
or JSON gate lists. Counts are little-endian (qubit 0 is the rightmost bit).
Everything runs on local noiseless simulators unless the operator set
SUPERQUANTX_MCP_ALLOW_HARDWARE=1. Results are toy-scale demonstrations,
not evidence of quantum advantage.
"""


def build_server() -> Any:
    """Create the MCPServer with all SuperQuantX tools registered."""
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            'The SuperQuantX MCP server needs the optional extra: pip install "superquantx[mcp]"'
        ) from exc
    from mcp.types import ToolAnnotations

    server = MCPServer(
        name="superquantx",
        title="SuperQuantX",
        instructions=INSTRUCTIONS,
        version=tools._version(),
        website_url="https://github.com/SuperagenticAI/superquantx",
    )
    read_only = ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, openWorldHint=False
    )

    @server.tool(annotations=read_only)
    def list_backends(smoke_test: bool = True) -> dict[str, Any]:
        """List SuperQuantX backends, whether their SDK is installed, and whether they work.

        With smoke_test=true each gate-model backend runs a 2-qubit Bell circuit
        (128 shots) so 'working' means it produced correct results, not just imported.
        Broken backends are reported with their real error instead of failing the call.
        """
        return tools.list_backends(smoke_test=smoke_test)

    @server.tool(annotations=read_only)
    def backend_versions() -> dict[str, Any]:
        """Versions of SuperQuantX, Python, the MCP SDK, and every installed quantum SDK."""
        return tools.backend_versions()

    @server.tool(annotations=read_only)
    def run_circuit(
        circuit: str | dict[str, Any] | list[dict[str, Any]],
        backend: str = "simulator",
        shots: int = 1024,
        device: str | None = None,
    ) -> dict[str, Any]:
        """Run a quantum circuit and return measurement counts and timing.

        circuit: an OpenQASM 2.0 or 3.0 string (standard gates, qreg/qubit
        declarations, measure statements are accepted and every qubit is measured
        at the end), or a JSON gate list such as
        {"n_qubits": 2, "gates": [{"gate": "h", "qubits": [0]}, {"gate": "cx", "qubits": [0, 1]}]}.
        Supported gates: id x y z h s sdg t tdg sx rx ry rz p u1 u2 u3 u cx cz swap ccx
        (angles in radians, 'pi' allowed in QASM).
        backend: simulator, qiskit, pennylane (cirq, braket, tket if installed and working).
        shots: 1 to the configured maximum (default cap 10000). Qubits are capped (default 10).
        device: only accepted when the server runs with SUPERQUANTX_MCP_ALLOW_HARDWARE=1.
        Counts are little-endian: qubit 0 is the rightmost bit.
        """
        return tools.run_circuit(circuit, backend=backend, shots=shots, device=device)

    @server.tool(annotations=read_only)
    def bell_state_demo(
        backends: list[str] | None = None, shots: int = 1024
    ) -> dict[str, Any]:
        """Prepare the Bell state (|00> + |11>)/sqrt(2) on several SDKs and compare counts.

        backends defaults to ["simulator", "qiskit", "pennylane"]. A correct run only
        ever measures '00' or '11'.
        """
        return tools.bell_state_demo(backends=backends, shots=shots)

    @server.tool(annotations=read_only)
    def qsvm_classify(
        dataset: str | None = "moons",
        backend: str = "simulator",
        X: list[list[float]] | None = None,
        y: list[int] | None = None,
        X_test: list[list[float]] | None = None,
        y_test: list[int] | None = None,
        n_samples: int = 40,
        test_size: float = 0.3,
        feature_map: str = "ZZFeatureMap",
        feature_map_reps: int = 1,
        shots: int = 256,
        seed: int = 0,
        classical_baseline: bool = True,
        kernel: str = "statevector",
        bandwidth: float = 0.5,
    ) -> dict[str, Any]:
        """Train a quantum-kernel SVM (QuantumSVM) and report test accuracy.

        Use a toy dataset (moons, circles, blobs, xor, iris) or pass small arrays X/y
        (optionally X_test/y_test). One qubit per feature, features capped (default 4),
        samples capped (default 80). Also returns a classical RBF-SVM baseline and kernel
        diagnostics that flag a broken quantum kernel. Feature maps: ZZFeatureMap,
        ZFeatureMap. kernel='statevector' (default) builds the fidelity kernel from exact
        statevectors produced by the chosen SDK; kernel='backend' uses the backend's
        built-in kernel routine (shot-based, quality varies by backend). bandwidth scales
        features before encoding; values such as 0.3 to 0.5 often avoid kernel concentration.
        """
        return tools.qsvm_classify(
            dataset=dataset,
            backend=backend,
            X=X,
            y=y,
            X_test=X_test,
            y_test=y_test,
            n_samples=n_samples,
            test_size=test_size,
            feature_map=feature_map,
            feature_map_reps=feature_map_reps,
            shots=shots,
            seed=seed,
            classical_baseline=classical_baseline,
            kernel=kernel,
            bandwidth=bandwidth,
        )

    @server.tool(annotations=read_only)
    def qaoa_maxcut(
        edges: list[list[int | float]],
        n_nodes: int | None = None,
        backend: str = "simulator",
        p: int = 1,
        shots: int = 1024,
        grid_points: int = 8,
        refine_iterations: int = 30,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Solve a small weighted Max-Cut with QAOA and check it against brute force.

        edges: [[u, v], ...] or [[u, v, weight], ...] with integer node ids. Nodes are
        capped (default 10, one qubit each). p is the number of QAOA layers (1 to 3).
        Parameters are found by a grid search then COBYLA refinement on measured counts.
        Returns the best sampled cut, partition, approximation ratio, and the exact
        classical optimum.
        """
        return tools.qaoa_maxcut(
            edges=edges,
            n_nodes=n_nodes,
            backend=backend,
            p=p,
            shots=shots,
            grid_points=grid_points,
            refine_iterations=refine_iterations,
            seed=seed,
        )

    return server


def main(argv: list[str] | None = None) -> None:
    """Console entry point: serve over stdio."""
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in {"-h", "--help"}:
        print(
            "superquantx-mcp: SuperQuantX MCP server over stdio.\n"
            "Configure it in your MCP client (Claude Desktop, Cursor) as a stdio command.\n"
            "Environment: SUPERQUANTX_MCP_MAX_QUBITS, SUPERQUANTX_MCP_MAX_SHOTS,\n"
            "SUPERQUANTX_MCP_MAX_GATES, SUPERQUANTX_MCP_MAX_SAMPLES, SUPERQUANTX_MCP_MAX_FEATURES,\n"
            "SUPERQUANTX_MCP_MAX_QAOA_NODES, SUPERQUANTX_MCP_ALLOW_HARDWARE=1 (off by default).",
            file=sys.stderr,
        )
        return
    # stdout carries the JSON-RPC stream; keep every log line on stderr.
    logging.basicConfig(
        level=logging.WARNING,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        server = build_server()
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
    server.run("stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
