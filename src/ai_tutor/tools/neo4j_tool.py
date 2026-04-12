"""
Neo4j tools via Google GenAI Toolbox (async, framework-agnostic).

Uses toolbox-core directly — the LlamaIndex-native choice.
toolbox-core is async-first and has no framework coupling, unlike the
toolbox-langchain adapter (used in the main branch) which wraps tools
as sync LangChain BaseTool objects and requires asyncio.to_thread.

Prerequisites:
  1. Start the toolbox server: docker compose up toolbox
  2. Set TOOLBOX_URL env var (default: http://toolbox:5001)

The three tools correspond to the proficiency-gated Cypher queries in tools.yaml:
  - get_prerequisites     : LOW proficiency (하)
  - get_current_concept   : MEDIUM proficiency (중)
  - get_advanced_concepts : HIGH proficiency (상)
"""

import asyncio
import os

from toolbox_core import ToolboxClient
from toolbox_core.protocol import Protocol

TOOLBOX_URL = os.getenv("TOOLBOX_URL", "http://toolbox:5001")

_tool_map_cache: dict | None = None
_cache_lock: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    """Lazy-init the lock so it is created inside a running event loop."""
    global _cache_lock
    if _cache_lock is None:
        _cache_lock = asyncio.Lock()
    return _cache_lock


async def _load_neo4j_tools() -> dict:
    """Load and cache async-callable Neo4j tools from the Toolbox server.

    Protected by an asyncio.Lock so concurrent coroutines don't trigger
    multiple round-trips to the Toolbox server on cold start.
    """
    global _tool_map_cache
    async with _get_lock():
        if _tool_map_cache is None:
            client = ToolboxClient(TOOLBOX_URL, protocol=Protocol.MCP_v20251125)
            tools = await client.load_toolset("neo4j-tools")
            _tool_map_cache = {t._name: t for t in tools}
    assert _tool_map_cache is not None
    return _tool_map_cache


async def get_tool(name: str):
    """Retrieve a single async-callable tool by name from the cached toolset."""
    tool_map = await _load_neo4j_tools()
    if name not in tool_map:
        raise KeyError(f"Tool '{name}' not found. Available: {list(tool_map.keys())}")
    return tool_map[name]
