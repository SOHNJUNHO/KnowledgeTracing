"""
LangGraph state machine for the Agentic GraphRAG Tutor.

Graph flow:
  START → run_bkt → diagnose → recommend → END

  run_bkt   : BKTransformer inference → per-skill BKT parameters
  diagnose  : LLM assigns proficiency level + reasoning per skill
  recommend : Neo4j GraphRAG (via MCP Toolbox) + LLM generates study feedback
              (async node — all skills are processed concurrently)
"""

from typing import Any, cast

from langfuse.decorators import langfuse_context, observe
from langgraph.graph import END, START, StateGraph

from ai_tutor.agents.diagnosis_node import diagnose_node, run_bkt_node
from ai_tutor.agents.recommendation_node import recommend_node
from ai_tutor.agents.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("run_bkt",   run_bkt_node)
    graph.add_node("diagnose",  diagnose_node)
    graph.add_node("recommend", recommend_node)

    graph.add_edge(START,       "run_bkt")
    graph.add_edge("run_bkt",   "diagnose")
    graph.add_edge("diagnose",  "recommend")
    graph.add_edge("recommend", END)

    return graph.compile()


tutor_graph = build_graph()


@observe(name="tutor_pipeline")
async def run_tutor(state: AgentState) -> dict:
    """Top-level entry point. Runs the full pipeline under one Langfuse trace."""
    langfuse_context.update_current_trace(
        user_id=state["student_id"],
        session_id=state["student_id"],
        tags=["production"],
    )
    return cast(dict[Any, Any], await tutor_graph.ainvoke(state))
