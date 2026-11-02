"""End to end demo: talk to the SuperQuantX MCP server over stdio.

This is exactly what an AI assistant (Claude Desktop, Cursor, ...) does: it
spawns ``superquantx-mcp`` (or ``python -m superquantx.mcp``) as a subprocess
and speaks the Model Context Protocol over stdin/stdout.

Usage::

    pip install "superquantx[mcp,qiskit,pennylane]"
    python examples/mcp_client_demo.py              # uses python -m superquantx.mcp
    python examples/mcp_client_demo.py --script     # uses the superquantx-mcp script
    python examples/mcp_client_demo.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time

import anyio
from mcp import Client, StdioServerParameters


BELL_QASM3 = """OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[3] c;
h q[0];
cx q[0], q[1];
cx q[1], q[2];
c = measure q;
"""

GATE_LIST = {
    "n_qubits": 2,
    "gates": [
        {"gate": "ry", "qubits": [0], "params": ["pi/3"]},
        {"gate": "cx", "qubits": [0, 1]},
    ],
}

SQUARE_WITH_DIAGONAL = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]]


def calls(backends: list[str]) -> list[tuple[str, dict]]:
    plan: list[tuple[str, dict]] = [
        ("list_backends", {}),
        ("backend_versions", {}),
        ("bell_state_demo", {"shots": 1000}),
    ]
    for b in backends:
        plan.append(
            ("run_circuit", {"circuit": BELL_QASM3, "backend": b, "shots": 1000})
        )
    plan.append(
        ("run_circuit", {"circuit": GATE_LIST, "backend": "simulator", "shots": 1000})
    )
    for b in backends:
        plan.append(
            (
                "qsvm_classify",
                {"dataset": "moons", "backend": b, "n_samples": 40, "seed": 7},
            )
        )
    plan.append(
        (
            "qsvm_classify",
            {"dataset": "iris", "backend": "simulator", "n_samples": 40, "seed": 7},
        )
    )
    for b in backends:
        plan.append(
            (
                "qaoa_maxcut",
                {"edges": SQUARE_WITH_DIAGONAL, "backend": b, "p": 1, "seed": 0},
            )
        )
    plan.append(
        ("run_circuit", {"circuit": "OPENQASM 2.0; qreg q[64]; h q[0];"})
    )  # rejected: too many qubits
    return plan


async def main(args: argparse.Namespace) -> int:
    if args.script:
        exe = shutil.which("superquantx-mcp")
        if exe is None:
            print("superquantx-mcp not found on PATH", file=sys.stderr)
            return 2
        params = StdioServerParameters(command=exe, args=[])
    else:
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "superquantx.mcp"]
        )

    transcript: list[dict] = []
    async with Client(params) as client:
        tools = await client.list_tools()
        names = [t.name for t in tools.tools]
        print(f"connected via {params.command} {' '.join(params.args)}")
        print(f"tools: {names}\n")
        transcript.append({"launch": [params.command, *params.args], "tools": names})

        for name, arguments in calls(args.backends):
            t0 = time.perf_counter()
            result = await client.call_tool(name, arguments)
            wall = round((time.perf_counter() - t0) * 1000, 1)
            payload = result.structured_content
            if payload is None:
                payload = {"raw": [getattr(c, "text", str(c)) for c in result.content]}
            print(f"=== {name}({json.dumps(arguments)[:120]}) [{wall} ms round trip]")
            print(json.dumps(payload, indent=2)[: args.max_chars])
            print()
            transcript.append(
                {
                    "tool": name,
                    "arguments": arguments,
                    "round_trip_ms": wall,
                    "result": payload,
                }
            )

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(transcript, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--script",
        action="store_true",
        help="launch the superquantx-mcp console script",
    )
    p.add_argument(
        "--backends", nargs="+", default=["simulator", "qiskit", "pennylane"]
    )
    p.add_argument("--json", help="write the full transcript to this file")
    p.add_argument("--max-chars", type=int, default=2500)
    sys.exit(anyio.run(main, p.parse_args()))
