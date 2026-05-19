"""
FastAPI service for the AI Tutor pipeline.

Exposes the full LlamaIndex Workflow pipeline (BKT → diagnose → recommend) as
an HTTP endpoint so any frontend or batch caller can trigger it on demand.

Run locally:
    uvicorn ai_tutor.api:app --port 8000 --reload

Environment variables: same as the CLI (see .env.example).
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ai_tutor.observability import setup_langfuse
from ai_tutor.tools.neo4j_tool import close_driver


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_langfuse()
    yield
    from langfuse import get_client
    get_client().flush()
    await close_driver()


app = FastAPI(title="AI Tutor API", version="1.0.0", lifespan=lifespan)


def _get_run_tutor():
    # Import lazily so helper/unit tests can import this module without the
    # full workflow stack installed.
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


class FeedbackItem(BaseModel):
    kc_name: str
    kc_id: int
    proficiency_level: str
    reasoning: str
    feedback: str


class TutorResponse(BaseModel):
    student_id: str
    feedback: list[FeedbackItem]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/tutor", response_model=TutorResponse)
async def tutor(req: TutorRequest) -> TutorResponse:
    """Run the full AI Tutor pipeline for one student and return feedback."""
    if len(req.sequence) < 2:
        raise HTTPException(status_code=422, detail="sequence must have at least 2 timesteps")

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
    return TutorResponse(student_id=req.student_id, feedback=result.get("feedback", []))
