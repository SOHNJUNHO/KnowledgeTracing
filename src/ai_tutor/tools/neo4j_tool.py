"""
Neo4j tools via async direct driver connection.

Cypher queries mirror tools.yaml exactly — that file remains as a human-readable
reference but is no longer executed at runtime.

Environment variables required:
    NEO4J_URI       — e.g. neo4j+s://xxxx.databases.neo4j.io
    NEO4J_USERNAME  — e.g. neo4j
    NEO4J_PASSWORD  — your Aura password
"""

import os
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

_driver: AsyncDriver | None = None


async def _get_driver() -> AsyncDriver:
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        )
    return _driver


async def _run_query(query: str, params: dict) -> list[dict]:
    driver = await _get_driver()
    async with driver.session() as session:
        result = await session.run(query, params)
        records: list[dict] = await result.data()
        return records


# ---------------------------------------------------------------------------
# Cypher queries (mirrored from tools.yaml)
# ---------------------------------------------------------------------------

_GET_PREREQUISITES = """
MATCH (current:KnowledgeComponent {skill_id: $skill_id})
OPTIONAL MATCH (current)-[:REQUIRE]->(prereq:KnowledgeComponent)
RETURN
  current.skill_id    AS skill_id,
  current.name        AS name,
  current.semester    AS semester,
  current.description AS description,
  collect({
    skill_id:    prereq.skill_id,
    name:        prereq.name,
    semester:    prereq.semester,
    description: prereq.description
  }) AS next_skills
"""

_GET_CURRENT_CONCEPT = """
MATCH (current:KnowledgeComponent {skill_id: $skill_id})
RETURN
  current.skill_id    AS skill_id,
  current.name        AS name,
  current.semester    AS semester,
  current.description AS description
"""

_GET_ADVANCED_CONCEPTS = """
MATCH (current:KnowledgeComponent {skill_id: $skill_id})
OPTIONAL MATCH (advanced:KnowledgeComponent)-[:REQUIRE]->(current)
RETURN
  current.skill_id    AS skill_id,
  current.name        AS name,
  current.semester    AS semester,
  current.description AS description,
  collect({
    skill_id:    advanced.skill_id,
    name:        advanced.name,
    semester:    advanced.semester,
    description: advanced.description
  }) AS next_skills
"""

_QUERIES: dict[str, str] = {
    "get_prerequisites":     _GET_PREREQUISITES,
    "get_current_concept":   _GET_CURRENT_CONCEPT,
    "get_advanced_concepts": _GET_ADVANCED_CONCEPTS,
}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class _Tool:
    """Async-callable wrapper so recommendation_node can do: await tool(skill_id=x)."""

    def __init__(self, name: str, query: str) -> None:
        self.name = name
        self._query = query

    async def __call__(self, **kwargs: Any) -> list[dict]:
        return await _run_query(self._query, kwargs)


_tool_cache: dict[str, _Tool] = {}


def get_tool(name: str) -> _Tool:
    """Retrieve an async-callable Neo4j tool by name (sync — pure cache lookup)."""
    if name not in _QUERIES:
        raise KeyError(f"Tool '{name}' not found. Available: {list(_QUERIES.keys())}")
    if name not in _tool_cache:
        _tool_cache[name] = _Tool(name, _QUERIES[name])
    return _tool_cache[name]
