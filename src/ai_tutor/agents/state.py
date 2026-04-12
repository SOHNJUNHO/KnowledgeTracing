from typing import Any, TypedDict


class AgentState(TypedDict):
    # Input
    student_id: str
    obs: Any               # torch.Tensor (1, T, 2): [skill_id, correct] per timestep
    output: Any            # torch.Tensor (1, T, 2): [skill_id, correct] per timestep (target)
    skill_id_to_name: dict # {skill_id (int): kc_name (str)}

    # After run_bkt node — keyed by "timestep{t}", values are BKTTimestep dicts
    diagnosis: dict

    # After diagnose node — list of AnalysisRecord dicts, one per unique skill
    analysis: list

    # After recommend node — list of FeedbackRecord dicts, one per unique skill
    feedback: list
