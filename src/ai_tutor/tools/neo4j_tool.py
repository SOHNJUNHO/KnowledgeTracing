"""
Neo4j tools via Google GenAI Toolbox (MCP pattern).

Prerequisites:
  1. Install the toolbox server: https://github.com/googleapis/genai-toolbox
  2. Configure tools.yaml at the project root with your Neo4j credentials.
  3. Start the server: toolbox --tools-file tools.yaml
  4. Set TOOLBOX_URL env var (default: http://localhost:5001)

The three tools correspond to the proficiency-gated Cypher queries in tools.yaml:
  - get_prerequisites     : LOW proficiency (하)
  - get_current_concept   : MEDIUM proficiency (중)
  - get_advanced_concepts : HIGH proficiency (상)
"""

import os

from toolbox_core.protocol import Protocol
from toolbox_langchain import ToolboxClient

TOOLBOX_URL = os.getenv("TOOLBOX_URL", "http://localhost:5001")

_tools_cache: list | None = None
_tool_map_cache: dict | None = None


def _load_neo4j_tools() -> tuple[list, dict]:
    """Load and cache Neo4j Cypher tools from the running GenAI Toolbox server.

    Both the list and the name→tool dict are cached so the server is
    contacted only once per process and get_tool() pays no rebuild cost.
    """
    global _tools_cache, _tool_map_cache
    if _tools_cache is None:
        client = ToolboxClient(TOOLBOX_URL, protocol=Protocol.MCP_v20251125)
        _tools_cache = client.load_toolset("neo4j-tools")
        _tool_map_cache = {t.name: t for t in _tools_cache}
    assert _tools_cache is not None and _tool_map_cache is not None
    return _tools_cache, _tool_map_cache


def get_tool(name: str):
    """Retrieve a single tool by name from the cached toolset."""
    _, tool_map = _load_neo4j_tools()
    if name not in tool_map:
        raise KeyError(f"Tool '{name}' not found. Available: {list(tool_map.keys())}")
    return tool_map[name]
