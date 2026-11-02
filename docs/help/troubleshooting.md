# Troubleshooting

**A backend is missing.**
Install its extra, for example `pip install "superquantx[qiskit]"`, or all
of them with `uv sync --extra all-backends`. Then run
`superquantx list-backends`, which runs a Bell circuit on each backend and
prints the error for any that fail.

**`braket` fails to start.**
Use the local simulator explicitly:
`sqx.get_backend("braket", device="local:braket_sv")`. Real Braket devices
need AWS credentials.

**`ocean` raises an error for a circuit.**
D-Wave Ocean is an annealer. It solves QUBO/Ising problems and does not run
gate-model circuits. Hardware access needs a D-Wave Leap token.

**An algorithm result has `error` set.**
`BaseQuantumAlgorithm.optimize` records exceptions on the returned
`QuantumResult.error` instead of raising. Check it before using the result.

**Results differ in bit order between backends.**
Use `backend.counts_big_endian(result["counts"])` and
`backend.statevector_big_endian(circuit)`, which normalise every backend to
big-endian (qubit 0 is the leftmost bit).

**Tests fail with "async def functions are not natively supported".**
Install the dev extra, which includes `pytest-asyncio`:
`uv sync --extra dev`.
