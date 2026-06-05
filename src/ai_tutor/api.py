"""
FastAPI service for the AI Tutor pipeline.

Exposes the full LlamaIndex Workflow pipeline (BKT → diagnose → recommend) as
an HTTP endpoint so any frontend or batch caller can trigger it on demand.

Run locally (requires a running Triton server at TRITON_URL):
    uv run uvicorn ai_tutor.api:app --port 8000 --reload

Environment variables: see .env.example.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from typing import Annotated, Literal
from pydantic import BaseModel, Field

from ai_tutor.tools.neo4j_tool import close_driver

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import os
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
        LlamaIndexInstrumentor().instrument()
    logger.info("startup.complete")
    yield
    from langfuse import get_client
    get_client().flush()
    await close_driver()
    from ai_tutor.workflow.diagnosis import close_triton_client
    await close_triton_client()
    logger.info("shutdown.complete")


app = FastAPI(title="AI Tutor API", version="1.0.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


def _get_run_tutor():
    from ai_tutor.workflow.workflow import run_tutor
    return run_tutor


def serve() -> None:
    """Entry point for `ai-tutor-api` CLI command."""
    import os
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("ai_tutor.api:app", host="0.0.0.0", port=port, reload=False)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class TutorRequest(BaseModel):
    student_id: str
    sequence: list[list[float]]
    """Raw student sequence: list of [skill_id, correct] pairs, shape (T, 2)."""
    skill_id_to_name: dict[str, str]
    """Maps skill_id (as string key) to human-readable name."""

# SkillId = Annotated[int, Field(ge=0)]
# Correct = Annotated[int, Field(ge=0, le=1)]
# StudentStep = tuple[SkillId, Correct]

# class TutorRequest(BaseModel):
#     student_id: Annotated[str, Field(min_length=1)]
#     sequence: Annotated[
#         list[StudentStep],
#         Field(min_length=2, description="Student sequence as [skill_id, correct] pairs."),
#     ]
#     skill_id_to_name: dict[str, str]


# class FeedbackItem(BaseModel):
#     kc_name: str
#     kc_id: int
#     proficiency_level: str
#     reasoning: str
#     feedback: str

class FeedbackItem(BaseModel):
    kc_name: str
    kc_id: int
    proficiency_level: Literal["상", "중", "하"]
    reasoning: Annotated[str, Field(min_length=1)]
    feedback: Annotated[str, Field(min_length=1)]


class TutorResponse(BaseModel):
    student_id: str
    feedback: list[FeedbackItem]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/healthz")
def healthz() -> dict:
    """Liveness probe — is the process alive?"""
    return {"ok": True}


@app.get("/readyz")
async def readyz() -> dict:
    """Readiness probe — are all dependencies reachable?"""
    from ai_tutor.workflow.diagnosis import is_bkt_ready
    from ai_tutor.tools.neo4j_tool import check_connectivity
    from ai_tutor.llm_client import get_llm_client

    if not await is_bkt_ready():
        raise HTTPException(503, detail="bkt not ready")

    try:
        await check_connectivity()
    except Exception as exc:
        raise HTTPException(503, detail=f"neo4j unreachable: {exc}") from exc

    try:
        await get_llm_client().models.list()
    except Exception as exc:
        raise HTTPException(503, detail=f"llm unreachable: {exc}") from exc

    return {"ok": True}


@app.post("/tutor", response_model=TutorResponse)
async def tutor(req: TutorRequest) -> TutorResponse:
    """Run the full AI Tutor pipeline for one student and return feedback."""
    if len(req.sequence) < 2:
        raise HTTPException(status_code=422, detail="sequence must have at least 2 timesteps")

    logger.info("tutor.request", extra={"student_id": req.student_id, "n_steps": len(req.sequence)})

    seq    = req.sequence
    obs    = [seq[:-1]]   # (1, T-1, 2) as nested list
    output = [seq[1:]]    # (1, T-1, 2) as nested list

    skill_id_to_name = {int(k): v for k, v in req.skill_id_to_name.items()}

    state = {
        "student_id":       req.student_id,
        "obs":              obs,
        "output":           output,
        "skill_id_to_name": skill_id_to_name,
    }

    result = await _get_run_tutor()(state)
    feedback = result.get("feedback", [])
    logger.info("tutor.response", extra={"student_id": req.student_id, "n_feedback": len(feedback)})
    return TutorResponse(student_id=req.student_id, feedback=feedback)
