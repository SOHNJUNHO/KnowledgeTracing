from typing import Any, TypedDict

from llama_index.core.workflow import Event


# ---------------------------------------------------------------------------
# LlamaIndex Workflow Events — carry data between pipeline steps
# ---------------------------------------------------------------------------

class BKTDoneEvent(Event):
    """Emitted by run_bkt step. Carries inference results into diagnose step."""
    model_config = {"arbitrary_types_allowed": True}
    student_id: str
    obs: Any               # torch.Tensor (1, T, 2)
    output: Any            # torch.Tensor (1, T, 2)
    skill_id_to_name: dict
    diagnosis: dict        # keyed by "timestep{t}", values are BKTTimestep dicts


class DiagnosisDoneEvent(Event):
    """Emitted by diagnose step. Carries proficiency analysis into recommend step."""
    student_id: str
    analysis: list         # list of AnalysisRecord dicts


# ---------------------------------------------------------------------------
# AgentState TypedDict — kept for backward-compatible type hints in node
# functions (diagnosis_node.py, recommendation_node.py). Not used by the
# workflow engine itself.
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    student_id: str
    obs: Any
    output: Any
    skill_id_to_name: dict
    diagnosis: dict
    analysis: list
    feedback: list
