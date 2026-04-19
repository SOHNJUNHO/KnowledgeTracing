"""
FastAPI service for the AI Tutor pipeline.

Exposes the full LangGraph pipeline (BKT → diagnose → recommend) as an
HTTP endpoint so any frontend or batch caller can trigger it without
running a CLI command.

Run locally:
    uvicorn ai_tutor.api:app --port 8000 --reload

Environment variables: same as the CLI (see .env.example).
"""

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ai_tutor.agents.graph import run_tutor


app = FastAPI(title="AI Tutor API", version="1.0.0")


def serve() -> None:
    """Entry point for `ai-tutor-api` CLI command."""
    import uvicorn
    uvicorn.run("ai_tutor.api:app", host="0.0.0.0", port=8000, reload=False)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class TutorRequest(BaseModel):
    student_id: str
    sequence: list[list[float]]
    """Raw student sequence: list of [skill_id, correct] pairs, shape (T, 2)."""
    skill_id_to_name: dict[str, str]
    """Maps skill_id (as string) to human-readable name."""


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

    t = torch.tensor(req.sequence, dtype=torch.float32)
    obs    = t[:-1].unsqueeze(0)   # (1, T-1, 2)
    output = t[1:].unsqueeze(0)    # (1, T-1, 2)

    skill_id_to_name = {int(k): v for k, v in req.skill_id_to_name.items()}

    state = {
        "student_id":       req.student_id,
        "obs":              obs,
        "output":           output,
        "skill_id_to_name": skill_id_to_name,
        "diagnosis": {},
        "analysis":  [],
        "feedback":  [],
    }

    result = await run_tutor(state)
    return TutorResponse(student_id=req.student_id, feedback=result.get("feedback", []))
