"""
LlamaIndex Workflow for the Agentic GraphRAG Tutor.

Workflow flow:
  StartEvent → run_bkt  → BKTDoneEvent
             → diagnose → DiagnosisDoneEvent
             → recommend → StopEvent

  run_bkt   : BKTransformer inference → per-skill BKT parameters
  diagnose  : LLM assigns proficiency level + reasoning per skill
  recommend : Neo4j GraphRAG + LLM generates study feedback
              (async step — all skills are processed concurrently)
"""

from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from langfuse.llama_index import LlamaIndexCallbackHandler
from langfuse.decorators import observe, langfuse_context

# Native Langfuse integration: automatically traces all LlamaIndex operations
# (step execution, event routing) without requiring manual @observe per step.
# The @observe decorators on node functions still add named child spans on top.
_langfuse_handler = LlamaIndexCallbackHandler()
Settings.callback_manager = CallbackManager([_langfuse_handler])

from ai_tutor.agents.state import AgentState, BKTDoneEvent, DiagnosisDoneEvent
from ai_tutor.agents.diagnosis_node import run_bkt_node, diagnose_node
from ai_tutor.agents.recommendation_node import recommend_node


class TutorWorkflow(Workflow):

    @step
    async def run_bkt(self, ev: StartEvent) -> BKTDoneEvent:
        """Call BKT inference service and build per-timestep records."""
        state: AgentState = {
            "student_id":     ev.get("student_id"),
            "obs":            ev.get("obs"),
            "output":         ev.get("output"),
            "skill_id_to_name": ev.get("skill_id_to_name"),
            "diagnosis": {},
            "analysis":  [],
            "feedback":  [],
        }
        result = await run_bkt_node(state)
        return BKTDoneEvent(
            student_id=state["student_id"],
            obs=state["obs"],
            output=state["output"],
            skill_id_to_name=state["skill_id_to_name"],
            diagnosis=result["diagnosis"],
        )

    @step
    async def diagnose(self, ev: BKTDoneEvent) -> DiagnosisDoneEvent:
        """Aggregate BKT data and call LLM for proficiency diagnosis."""
        state: AgentState = {
            "student_id":     ev.student_id,
            "obs":            ev.obs,
            "output":         ev.output,
            "skill_id_to_name": ev.skill_id_to_name,
            "diagnosis":      ev.diagnosis,
            "analysis":  [],
            "feedback":  [],
        }
        result = await diagnose_node(state)
        return DiagnosisDoneEvent(
            student_id=ev.student_id,
            analysis=result["analysis"],
        )

    @step
    async def recommend(self, ev: DiagnosisDoneEvent) -> StopEvent:
        """Fetch graph context and generate per-skill feedback concurrently."""
        result = await recommend_node({"analysis": ev.analysis})  # type: ignore[arg-type]
        return StopEvent(result=result["feedback"])


@observe(name="tutor_pipeline")
async def run_tutor(state: AgentState) -> dict:
    """Top-level entry point. Runs the full pipeline under one Langfuse trace."""
    langfuse_context.update_current_trace(
        user_id=state["student_id"],
        session_id=state["student_id"],
        tags=["production"],
    )
    workflow = TutorWorkflow(timeout=120, verbose=False)
    feedback = await workflow.run(
        student_id=state["student_id"],
        obs=state["obs"],
        output=state["output"],
        skill_id_to_name=state["skill_id_to_name"],
    )
    return {"feedback": feedback}
