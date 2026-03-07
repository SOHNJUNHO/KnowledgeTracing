"""
LangGraph state machine for the Agentic GraphRAG Tutor.

Graph flow:
  START → run_bkt → diagnose → recommend → END

  run_bkt   : BKTransformer inference → per-skill BKT parameters
  diagnose  : LLM assigns proficiency level + reasoning per skill
  recommend : Neo4j GraphRAG (via MCP Toolbox) + LLM generates study feedback
"""

from langgraph.graph import StateGraph, START, END
from langfuse.decorators import observe, langfuse_context

from .state import AgentState
from .diagnosis_node import run_bkt_node, diagnose_node
from .recommendation_node import recommend_node


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("run_bkt",   run_bkt_node)
    graph.add_node("diagnose",  diagnose_node)
    graph.add_node("recommend", recommend_node)

    graph.add_edge(START,      "run_bkt")
    graph.add_edge("run_bkt",  "diagnose")
    graph.add_edge("diagnose", "recommend")
    graph.add_edge("recommend", END)

    return graph.compile()


# Compiled graph — call run_tutor() to invoke with Langfuse tracing
tutor_graph = build_graph()


@observe(name="tutor_pipeline")
def run_tutor(state: AgentState) -> dict:
    """Top-level entry point. Runs the full pipeline under one Langfuse trace."""
    langfuse_context.update_current_trace(
        user_id=state["student_id"],
        session_id=state["student_id"],
        tags=["production"],
    )
    return tutor_graph.invoke(state)
