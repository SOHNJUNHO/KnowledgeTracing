import os
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

_NEO4J_URI      = os.environ.get("NEO4J_URI", "")
_NEO4J_USER     = os.environ.get("NEO4J_USERNAME", "")
_NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

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

_TOOL_TO_QUERY: dict[str, str] = {
    "get_prerequisites":     _GET_PREREQUISITES,
    "get_current_concept":   _GET_CURRENT_CONCEPT,
    "get_advanced_concepts": _GET_ADVANCED_CONCEPTS,
}

_driver: AsyncDriver | None = None


def _get_driver() -> AsyncDriver:
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            _NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASSWORD)
        )
    return _driver


async def close_driver() -> None:
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None


class _Tool:
    def __init__(self, name: str, query: str) -> None:
        self._name = name
        self._query = query

    async def __call__(self, **kwargs: Any) -> list[dict]:
        async with _get_driver().session() as session:
            result = await session.run(self._query, skill_id=kwargs["skill_id"])
            records = await result.data()
        return records


_tool_cache: dict[str, _Tool] = {}


def get_tool(name: str) -> _Tool:
    if name not in _tool_cache:
        if name not in _TOOL_TO_QUERY:
            raise KeyError(f"Unknown tool: {name!r}")
        _tool_cache[name] = _Tool(name, _TOOL_TO_QUERY[name])
    return _tool_cache[name]
