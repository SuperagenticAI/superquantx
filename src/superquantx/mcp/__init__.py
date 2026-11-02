"""Model Context Protocol (MCP) server for SuperQuantX.

Lets AI assistants such as Claude Desktop and Cursor drive quantum workloads
across SDKs through the SuperQuantX unified API. Install with
``pip install "superquantx[mcp]"`` and run ``superquantx-mcp``.

The tool implementations in :mod:`superquantx.mcp.tools` are plain Python and
do not need the ``mcp`` package; only :func:`build_server` / :func:`main` do.
"""

from .server import build_server, main


__all__ = ["build_server", "main"]
