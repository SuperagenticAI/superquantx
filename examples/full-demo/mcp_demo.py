"""Local MCP discovery and one Bell-state tool call; no AI model or hardware."""

import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    # The MCP SDK forwards a minimal environment. Preserve only explicit
    # cache/config and headless settings needed by the child process's SDKs.
    server_env = {
        name: os.environ[name]
        for name in (
            "XDG_CONFIG_HOME",
            "XDG_CACHE_HOME",
            "MPLCONFIGDIR",
            "MPLBACKEND",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        )
        if name in os.environ
    }
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "superquantx.mcp"],
        env=server_env,
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = sorted(tool.name for tool in tools.tools)
            expected = sorted(
                [
                    "list_backends",
                    "backend_versions",
                    "run_circuit",
                    "bell_state_demo",
                    "qsvm_classify",
                    "qaoa_maxcut",
                ]
            )
            if names != expected:
                raise RuntimeError(f"Unexpected tool inventory: {names}")
            print("Connected to the local SuperQuantX MCP server over stdio.")
            print("Six discovered tools:")
            for name in names:
                print(f"  {name}")
            print("\nCalling bell_state_demo on the built-in simulator:")
            result = await session.call_tool(
                "bell_state_demo",
                arguments={
                    "backends": ["simulator"],
                    "shots": 128,
                },
            )
            if getattr(result, "is_error", getattr(result, "isError", False)):
                raise RuntimeError(str(result.content))
            data = getattr(
                result, "structured_content", getattr(result, "structuredContent", None)
            )
            if data is None:
                data = json.loads(
                    next(item.text for item in result.content if hasattr(item, "text"))
                )
            print(json.dumps(data, indent=2))
            if not data.get("ok"):
                raise RuntimeError("Tool returned an unsuccessful result")
            print("\nPASS: actual tool discovery and a local MCP tool call completed.")
            print("This exercises MCP tools without an AI model.")


if __name__ == "__main__":
    asyncio.run(main())
