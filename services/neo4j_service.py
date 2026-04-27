"""
Neo4j microservice — FastAPI wrapper around the three curriculum graph queries.

Endpoints:
    POST /prerequisites   {"skill_id": int} -> list of concept dicts
    POST /current         {"skill_id": int} -> list of concept dicts
    POST /advanced        {"skill_id": int} -> list of concept dicts
    GET  /health

Environment variables:
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from neo4j import AsyncGraphDatabase, AsyncDriver
from pydantic import BaseModel


_driver: AsyncDriver | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _driver
    _driver = AsyncGraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    yield
    if _driver is not None:
        await _driver.close()
    _driver = None


app = FastAPI(title="Neo4j Service", version="1.0.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    skill_id: int


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


async def _run(query: str, skill_id: int) -> list[dict]:
    if _driver is None:
        raise HTTPException(status_code=503, detail="Neo4j driver not initialised")
    async with _driver.session() as session:
        result = await session.run(query, {"skill_id": skill_id})
        return await result.data()


@app.post("/prerequisites")
async def prerequisites(req: QueryRequest) -> list:
    return await _run(_GET_PREREQUISITES, req.skill_id)


@app.post("/current")
async def current(req: QueryRequest) -> list:
    return await _run(_GET_CURRENT_CONCEPT, req.skill_id)


@app.post("/advanced")
async def advanced(req: QueryRequest) -> list:
    return await _run(_GET_ADVANCED_CONCEPTS, req.skill_id)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "driver_connected": _driver is not None}
