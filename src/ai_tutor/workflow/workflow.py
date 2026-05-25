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

from llama_index.core.workflow import Workflow, Context, StartEvent, StopEvent, step
from langfuse import observe, propagate_attributes

from ai_tutor.workflow.events import BKTDoneEvent, DiagnosisDoneEvent
from ai_tutor.workflow.diagnosis import run_bkt_node, diagnose_node
from ai_tutor.workflow.recommendation import recommend_node


class TutorWorkflow(Workflow):

    @step
    async def run_bkt(self, ctx: Context, ev: StartEvent) -> BKTDoneEvent:
        """Call BKT inference service and build per-timestep records."""
        await ctx.store.set("student_id", ev.get("student_id"))
        await ctx.store.set("obs", ev.get("obs"))
        await ctx.store.set("output", ev.get("output"))
        await ctx.store.set("skill_id_to_name", ev.get("skill_id_to_name"))

        result = await run_bkt_node({
            "obs":            await ctx.store.get("obs"),
            "output":         await ctx.store.get("output"),
            "skill_id_to_name": await ctx.store.get("skill_id_to_name"),
        })
        await ctx.store.set("diagnosis", result["diagnosis"])
        return BKTDoneEvent()

    @step
    async def diagnose(self, ctx: Context, ev: BKTDoneEvent) -> DiagnosisDoneEvent:
        """Aggregate BKT data and call LLM for proficiency diagnosis."""
        result = await diagnose_node({
            "student_id":     await ctx.store.get("student_id"),
            "diagnosis":      await ctx.store.get("diagnosis"),
        })
        await ctx.store.set("analysis", result["analysis"])
        return DiagnosisDoneEvent()

    @step
    async def recommend(self, ctx: Context, ev: DiagnosisDoneEvent) -> StopEvent:
        """Fetch graph context and generate per-skill feedback concurrently."""
        result = await recommend_node({"analysis": await ctx.store.get("analysis")})
        return StopEvent(result=result["feedback"])


@observe(name="tutor_pipeline")
async def run_tutor(state: dict) -> dict:
    """Top-level entry point. Runs the full pipeline under one Langfuse trace."""
    with propagate_attributes(
        user_id=state["student_id"],
        session_id=state.get("run_id", state["student_id"]),
        tags=["production"],
    ):
        workflow = TutorWorkflow(timeout=3600, verbose=False)
        feedback = await workflow.run(
            student_id=state["student_id"],
            obs=state["obs"],
            output=state["output"],
            skill_id_to_name=state["skill_id_to_name"],
        )
        return {"feedback": feedback}
