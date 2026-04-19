"""
Pydantic models that define and validate the data contract at each
LangGraph node boundary.

  BKTTimestep   — one row of run_bkt_node output (per timestep)
  AnalysisRecord — one row of diagnose_node output (per unique skill)
  FeedbackRecord — one row of recommend_node output (per unique skill)

Validation is intentionally strict: probability fields are bounded
[0, 1] and proficiency_level is a closed enum so schema drift from
the LLM is caught at the node that produces bad data, not three
nodes later as a KeyError.
"""

from typing import Literal

from pydantic import BaseModel, Field


class BKTTimestep(BaseModel):
    skill_id: int
    skill_name: str
    actual_correct: int = Field(ge=0, le=1)
    prior: float = Field(ge=0.0, le=1.0)
    learning_rate: float = Field(ge=0.0, le=1.0)
    guess: float = Field(ge=0.0, le=1.0)
    slip: float = Field(ge=0.0, le=1.0)
    predicted_correct: float = Field(ge=0.0, le=1.0)


class AnalysisRecord(BaseModel):
    kc_id: int
    kc_name: str
    proficiency_level: Literal["상", "중", "하"]
    reasoning: str = Field(min_length=1)


class FeedbackRecord(BaseModel):
    kc_id: int
    kc_name: str
    proficiency_level: Literal["상", "중", "하"]
    reasoning: str
    feedback: str = Field(min_length=1)
    # Fields kept for Langfuse tracing / RAGAS evaluation
    prompt: str
    input_data: str
    graph_context: str
