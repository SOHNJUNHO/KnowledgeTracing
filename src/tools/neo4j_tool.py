"""
Neo4j tools via Google GenAI Toolbox (MCP pattern).

Prerequisites:
  1. Install the toolbox server: https://github.com/googleapis/genai-toolbox
  2. Configure tools.yaml at the project root with your Neo4j credentials.
  3. Start the server: toolbox --tools-file tools.yaml
  4. Set TOOLBOX_URL env var (default: http://localhost:5000)

The three tools exposed here correspond to the proficiency-gated Cypher queries
defined in tools.yaml:
  - get_prerequisites     : for students with LOW proficiency (하)
  - get_current_concept   : for students with MEDIUM proficiency (중)
  - get_advanced_concepts : for students with HIGH proficiency (상)
"""

import os
from toolbox_langchain import ToolboxClient
from toolbox_core.protocol import Protocol

TOOLBOX_URL = os.getenv("TOOLBOX_URL", "http://localhost:5001")

_tools_cache: list | None = None


def _load_neo4j_tools() -> list:
    """Load and cache Neo4j Cypher tools from the running GenAI Toolbox server.

    Cached so the server is only contacted once per process.
    """
    global _tools_cache
    if _tools_cache is None:
        client = ToolboxClient(TOOLBOX_URL, protocol=Protocol.MCP_v20251125)
        _tools_cache = client.load_toolset("neo4j-tools")
    return _tools_cache


def get_tool(name: str):
    """Retrieve a single tool by name from the loaded toolset."""
    tools = _load_neo4j_tools()
    tool_map = {t.name: t for t in tools}
    if name not in tool_map:
        raise KeyError(f"Tool '{name}' not found. Available: {list(tool_map.keys())}")
    return tool_map[name]
