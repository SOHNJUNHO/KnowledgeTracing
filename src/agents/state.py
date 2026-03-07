from typing import Any, TypedDict


class AgentState(TypedDict):
    # Input
    student_id: str
    obs: Any               # torch.Tensor (1, T, 2): [skill_id, correct] per timestep (input)
    output: Any            # torch.Tensor (1, T, 2): [skill_id, correct] per timestep (target)
    skill_id_to_name: dict # {skill_id (int): kc_name (str)}

    # After run_bkt node
    # {
    #   "timestep0": {
    #     "skill_id": int, "skill_name": str,
    #     "actual_correct": int,
    #     "prior": float,         # P(knowledge) at this timestep
    #     "learning_rate": float, # BKT learn param
    #     "guess": float,         # BKT guess param
    #     "slip": float,          # BKT slip param
    #     "predicted_correct": float
    #   }, ...
    # }
    diagnosis: dict

    # After diagnose node — one entry per unique skill seen
    # [{"kc_id": int, "kc_name": str, "proficiency_level": "상"|"중"|"하", "reasoning": str}, ...]
    analysis: list

    # After recommend node — same as analysis plus "feedback" field per entry
    # [{"kc_id", "kc_name", "proficiency_level", "reasoning", "feedback"}, ...]
    feedback: list
