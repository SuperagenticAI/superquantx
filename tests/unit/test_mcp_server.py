"""Tests for the SuperQuantX MCP server (superquantx.mcp).

Tool functions are tested directly (no ``mcp`` package needed). The in-process
MCP client test is skipped when the optional ``mcp`` extra is not installed.
"""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from superquantx.mcp import tools
from superquantx.mcp.circuits import (
    CircuitError,
    feature_map_circuit,
    lower,
    parse_circuit,
    parse_gate_list,
    parse_qasm,
    safe_eval_expr,
)
from superquantx.mcp.config import HARD_MAX_QUBITS, get_limits


HAS_QISKIT = (
    importlib.util.find_spec("qiskit") is not None
    and importlib.util.find_spec("qiskit_aer") is not None
)
HAS_PENNYLANE = importlib.util.find_spec("pennylane") is not None
HAS_MCP = importlib.util.find_spec("mcp") is not None and (
    importlib.util.find_spec("mcp.server.mcpserver") is not None
)

BELL2 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q -> c;'
GHZ3 = (
    'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[3] q;\nbit[3] c;\n'
    "h q[0];\ncx q[0], q[1];\ncx q[1], q[2];\nc = measure q;"
)


@pytest.fixture
def limits():
    return get_limits()


# ---------------------------------------------------------------- parsing


class TestParsing:
    def test_qasm2_bell(self, limits):
        c = parse_qasm(BELL2, limits)
        assert c.n_qubits == 2
        assert [g.name for g in c.gates] == ["h", "cx"]
        assert c.source_format == "openqasm2"

    def test_qasm3_ghz(self, limits):
        c = parse_qasm(GHZ3, limits)
        assert c.n_qubits == 3
        assert c.source_format == "openqasm3"
        assert len(c.gates) == 3

    def test_broadcast_and_multiple_registers(self, limits):
        src = "OPENQASM 2.0; qreg a[2]; qreg b[1]; h a; cx a[1], b[0];"
        c = parse_qasm(src, limits)
        assert c.n_qubits == 3
        assert [g.qubits for g in c.gates] == [(0,), (1,), (1, 2)]

    def test_param_expressions(self, limits):
        c = parse_qasm(
            "OPENQASM 2.0; qreg q[1]; rx(pi/2) q[0]; rz(-3*pi/4 + 0.5) q[0];", limits
        )
        assert math.isclose(c.gates[0].params[0], math.pi / 2)
        assert math.isclose(c.gates[1].params[0], -3 * math.pi / 4 + 0.5)

    def test_safe_eval_rejects_code(self):
        for bad in ["__import__('os')", "a.b", "[1]", "2**100", "open('x')"]:
            with pytest.raises(CircuitError):
                safe_eval_expr(bad)

    @pytest.mark.parametrize(
        "src",
        [
            "qreg q[1]; h q[0];",  # missing header
            "OPENQASM 2.0; qreg q[1]; foo q[0];",  # unknown gate
            "OPENQASM 2.0; qreg q[1]; gate g a { h a; } g q[0];",  # custom gate
            "OPENQASM 2.0; qreg q[1]; reset q[0];",
            "OPENQASM 2.0; qreg q[2]; cx q[0], q[0];",  # repeated qubit
            "OPENQASM 2.0; qreg q[1]; h q[3];",  # out of range
            "OPENQASM 2.0; qreg q[1]; rx q[0];",  # missing param
        ],
    )
    def test_qasm_errors(self, src, limits):
        with pytest.raises(CircuitError):
            parse_qasm(src, limits)

    def test_qubit_cap(self, limits):
        with pytest.raises(CircuitError, match="limit"):
            parse_qasm(
                f"OPENQASM 2.0; qreg q[{limits.max_qubits + 1}]; h q[0];", limits
            )

    def test_json_gate_list(self, limits):
        c = parse_gate_list(
            {
                "n_qubits": 2,
                "gates": [
                    {"gate": "H", "qubits": [0]},
                    {"name": "cnot", "qubits": [0, 1]},
                ],
            },
            limits,
        )
        assert [g.name for g in c.gates] == ["h", "cx"]

    def test_json_param_expressions(self, limits):
        c = parse_gate_list([{"gate": "ry", "qubits": [0], "params": ["pi/3"]}], limits)
        assert math.isclose(c.gates[0].params[0], math.pi / 3)

    def test_json_inferred_qubits_and_text(self, limits):
        c = parse_circuit('[{"gate": "x", "qubits": 2}]', limits)
        assert c.n_qubits == 3

    @pytest.mark.parametrize(
        "spec",
        [
            {"gates": [{"gate": "h"}]},
            {"gates": [{"gate": "rx", "qubits": [0], "params": ["__import__('os')"]}]},
            {"gates": [{"gate": "h", "qubits": [True]}]},
            {"n_qubits": 1, "gates": [{"gate": "cx", "qubits": [0, 1]}]},
            "not a list",
        ],
    )
    def test_json_errors(self, spec, limits):
        with pytest.raises(CircuitError):
            parse_circuit(spec, limits) if not isinstance(
                spec, str
            ) else parse_gate_list(spec, limits)

    def test_gate_cap(self, monkeypatch):
        monkeypatch.setenv("SUPERQUANTX_MCP_MAX_GATES", "3")
        lim = get_limits()
        with pytest.raises(CircuitError, match="limit"):
            parse_gate_list([{"gate": "h", "qubits": [0]}] * 4, lim)

    def test_env_cannot_exceed_hard_ceiling(self, monkeypatch):
        monkeypatch.setenv("SUPERQUANTX_MCP_MAX_QUBITS", "999")
        assert get_limits().max_qubits == HARD_MAX_QUBITS

    def test_lowering_is_backend_gate_set(self, limits):
        c = parse_qasm(
            "OPENQASM 2.0; qreg q[3]; s q[0]; t q[1]; u3(0.1,0.2,0.3) q[2]; ccx q[0],q[1],q[2];",
            limits,
        )
        names = {g.name for g in lower(c, native_ccx=False)}
        assert names <= {"h", "x", "y", "z", "rx", "ry", "rz", "cx", "cz", "swap"}


# ---------------------------------------------------------------- run_circuit


class TestRunCircuit:
    def test_bell_simulator(self):
        r = tools.run_circuit(BELL2, backend="simulator", shots=500)
        assert r["ok"], r
        assert set(r["counts"]) <= {"00", "11"}
        assert r["shots"] == 500
        assert r["timing_ms"]["total"] >= 0

    def test_little_endian(self):
        r = tools.run_circuit(
            [{"gate": "x", "qubits": [0]}], backend="simulator", shots=10
        )
        assert r["counts"] == {"1": 10}
        r = tools.run_circuit(
            {"n_qubits": 2, "gates": [{"gate": "x", "qubits": [0]}]}, shots=10
        )
        assert r["counts"] == {"01": 10}

    def test_toffoli_decomposition_on_simulator(self):
        gates = [
            {"gate": "x", "qubits": [0]},
            {"gate": "x", "qubits": [1]},
            {"gate": "ccx", "qubits": [0, 1, 2]},
        ]
        r = tools.run_circuit(
            {"n_qubits": 3, "gates": gates}, backend="simulator", shots=50
        )
        assert r["counts"] == {"111": 50}

    def test_validation_errors_are_structured(self):
        r = tools.run_circuit("OPENQASM 2.0; qreg q[1]; bogus q[0];")
        assert r["ok"] is False and r["error"]["type"] == "ValidationError"
        r = tools.run_circuit(BELL2, shots=10**9)
        assert r["ok"] is False and "limit" in r["error"]["message"]
        r = tools.run_circuit(BELL2, backend="nope")
        assert r["ok"] is False and "Unknown backend" in r["error"]["message"]
        r = tools.run_circuit(BELL2, backend="ocean")
        assert r["ok"] is False

    def test_device_requires_hardware_opt_in(self, monkeypatch):
        monkeypatch.delenv("SUPERQUANTX_MCP_ALLOW_HARDWARE", raising=False)
        r = tools.run_circuit(BELL2, backend="qiskit", device="ibm_brisbane")
        assert r["ok"] is False
        assert "SUPERQUANTX_MCP_ALLOW_HARDWARE" in r["error"]["message"]

    def test_backend_failure_is_reported_not_raised(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("simulated backend crash")

        monkeypatch.setattr(tools, "_get_backend", boom)
        r = tools.run_circuit(BELL2, backend="simulator")
        assert r["ok"] is False
        assert r["error"]["stage"] == "backend_init"
        assert "simulated backend crash" in r["error"]["message"]

    @pytest.mark.parametrize(
        "backend",
        [
            pytest.param(
                "qiskit",
                marks=pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed"),
            ),
            pytest.param(
                "pennylane",
                marks=pytest.mark.skipif(
                    not HAS_PENNYLANE, reason="pennylane not installed"
                ),
            ),
        ],
    )
    def test_sdks_agree_with_simulator(self, backend):
        circ = "OPENQASM 2.0; qreg q[3]; x q[0]; ry(pi/3) q[1]; cx q[1], q[2]; ccx q[0], q[1], q[2];"
        shots = 4000
        a = tools.run_circuit(circ, backend="simulator", shots=shots)
        b = tools.run_circuit(circ, backend=backend, shots=shots)
        assert a["ok"] and b["ok"], (a, b)
        keys = set(a["counts"]) | set(b["counts"])
        tv = 0.5 * sum(
            abs(a["counts"].get(k, 0) - b["counts"].get(k, 0)) / shots for k in keys
        )
        assert tv < 0.06, (a["counts"], b["counts"])
        # x q[0] is always 1 -> rightmost bit always '1'
        assert all(k.endswith("1") for k in b["counts"])


# ---------------------------------------------------------------- other tools


class TestOtherTools:
    def test_list_backends(self):
        r = tools.list_backends(smoke_test=True)
        assert r["ok"]
        assert r["backends"]["simulator"]["status"] == "working"
        assert "simulator" in r["working_backends"]
        for info in r["backends"].values():
            assert "status" in info
            if info["status"] not in ("working", "initialized"):
                assert "error" in info or "smoke_test" in info

    def test_backend_versions(self):
        r = tools.backend_versions()
        assert r["ok"] and r["superquantx"]
        assert "numpy" in r["packages"]

    def test_bell_state_demo(self):
        r = tools.bell_state_demo(backends=["simulator", "does-not-exist"], shots=200)
        assert r["ok"]
        assert r["results"]["simulator"]["correlated_fraction"] == 1.0
        assert r["results"]["does-not-exist"]["ok"] is False

    def test_qaoa_triangle(self):
        r = tools.qaoa_maxcut(
            [[0, 1], [1, 2], [0, 2]],
            backend="simulator",
            shots=256,
            grid_points=4,
            refine_iterations=5,
        )
        assert r["ok"], r
        assert r["classical_brute_force"]["max_cut"] == 2.0
        assert r["classical_brute_force"]["n_optimal_assignments"] == 6
        assert r["found_optimum"] is True
        assert 0 < r["qaoa"]["approximation_ratio"] <= 1.0

    def test_qaoa_accepts_integral_float_node_ids(self):
        r = tools.qaoa_maxcut(
            [[0.0, 1.0], [1.0, 2.0]], shots=64, grid_points=2, refine_iterations=0
        )
        assert r["ok"], r
        assert r["classical_brute_force"]["max_cut"] == 2.0

    def test_qaoa_validation(self):
        assert tools.qaoa_maxcut([[0, 0]])["ok"] is False
        assert tools.qaoa_maxcut([[0, 1], [1, 0]])["ok"] is False
        assert tools.qaoa_maxcut([[0, 11]])["ok"] is False
        assert tools.qaoa_maxcut([[0, 1]], p=9)["ok"] is False
        assert tools.qaoa_maxcut([[0, 1, -1]])["ok"] is False
        assert tools.qaoa_maxcut([[0.5, 1]])["ok"] is False

    def test_qsvm_toy(self):
        r = tools.qsvm_classify("blobs", backend="simulator", n_samples=20)
        assert r["ok"], r
        assert 0.0 <= r["quantum"]["test_accuracy"] <= 1.0
        assert r["kernel_diagnostics"]["looks_valid"]
        assert math.isclose(r["kernel_diagnostics"]["mean_diagonal"], 1.0, abs_tol=1e-6)
        assert "classical_rbf_baseline" in r

    def test_qsvm_provided_arrays(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(20, 2))
        y = (X[:, 0] > 0).astype(int)
        r = tools.qsvm_classify(
            dataset=None, X=X.tolist(), y=y.tolist(), classical_baseline=False
        )
        assert r["ok"], r
        assert r["n_train"] + r["n_test"] == 20
        assert "classical_rbf_baseline" not in r

    def test_qsvm_backend_kernel_flagged_on_simulator(self):
        r = tools.qsvm_classify(
            "blobs", backend="simulator", n_samples=12, kernel="backend"
        )
        assert r["ok"]
        assert r["kernel_diagnostics"]["looks_valid"] is False
        assert "placeholder" in r["warning"]

    def test_qsvm_validation(self):
        X = np.zeros((20, 8)).tolist()
        y = [0, 1] * 10
        assert tools.qsvm_classify(X=X, y=y)["ok"] is False  # too many features
        assert tools.qsvm_classify("nope")["ok"] is False
        assert tools.qsvm_classify(n_samples=10_000)["ok"] is False
        assert tools.qsvm_classify(feature_map="Custom")["ok"] is False

    def test_feature_map_statevector_fidelity_is_one_on_diagonal(self):
        from superquantx.backends import get_backend
        from superquantx.mcp.circuits import statevector

        b = get_backend("simulator")
        s = statevector(
            feature_map_circuit([0.3, -0.7], "ZZFeatureMap", 2), "simulator", b
        )
        assert math.isclose(float(np.vdot(s, s).real), 1.0, abs_tol=1e-9)


# ---------------------------------------------------------------- MCP protocol


@pytest.mark.skipif(not HAS_MCP, reason="optional 'mcp' extra not installed")
class TestMCPProtocol:
    def test_in_process_client(self):
        import anyio
        from mcp import Client

        from superquantx.mcp import build_server

        async def scenario():
            async with Client(build_server()) as client:
                listed = await client.list_tools()
                names = {t.name for t in listed.tools}
                assert names == {
                    "list_backends",
                    "backend_versions",
                    "run_circuit",
                    "bell_state_demo",
                    "qsvm_classify",
                    "qaoa_maxcut",
                }
                res = await client.call_tool(
                    "run_circuit", {"circuit": BELL2, "shots": 64}
                )
                assert not res.is_error
                assert res.structured_content["ok"] is True
                assert set(res.structured_content["counts"]) <= {"00", "11"}
                qa = await client.call_tool(
                    "qaoa_maxcut",
                    {
                        "edges": [[0, 1], [1, 2], [0, 2]],
                        "grid_points": 3,
                        "refine_iterations": 0,
                    },
                )
                assert qa.structured_content["ok"] is True, qa.structured_content
                assert qa.structured_content["classical_brute_force"]["max_cut"] == 2.0
                bad = await client.call_tool("run_circuit", {"circuit": "nonsense"})
                assert bad.structured_content["ok"] is False

        anyio.run(scenario)
