"""
Diagnosis pipeline: two LangGraph nodes.

  run_bkt_node   — loads the trained BKTransformer, runs inference on a student's
                   interaction sequence, extracts per-skill BKT parameters.

  diagnose_node  — sends the BKT output to an LLM which assigns a proficiency
                   level (상/중/하) and a natural-language reasoning per skill.
"""

import json
import os
import sys

import torch
from langfuse.openai import OpenAI
from langfuse.decorators import observe, langfuse_context

# Allow pickle to resolve 'model.BKTransformer' from the saved checkpoint
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bkt"))

from .state import AgentState
from bkt.model import BKTransformer

# ---------------------------------------------------------------------------
# Model config must match the checkpoint used at inference time.
# Override via env vars if needed.
# ---------------------------------------------------------------------------
_CHECKPOINT = os.getenv(
    "BKT_CHECKPOINT",
    "src/bkt/checkpoints/upgraded-best-epoch=10-val_auc=0.7965.ckpt"
)
_N_SKILLS = int(os.getenv("N_SKILLS", "138"))  # icecream_8th dataset 137 + 1 for padding

class _BKTConfig:
    n_skills = _N_SKILLS
    n_embd = 256
    n_layer = 3
    n_head = 4
    block_size = 512
    dropout = 0.1

_pytorch_model: BKTransformer | None = None

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def _load_model() -> BKTransformer:
    global _pytorch_model
    if _pytorch_model is None:
        # 1. Initialize the raw PyTorch architecture
        _pytorch_model = BKTransformer(_BKTConfig())

        # 2. Inject a Config stub into __main__ so pickle can deserialize
        #    __main__.Config that was stored when the model was trained via main.py
        import __main__
        if not hasattr(__main__, "Config"):
            class Config:
                def __init__(self, **kwargs): self.__dict__.update(kwargs)
            __main__.Config = Config

        # 3. Load the Lightning checkpoint file to CPU first (MPS requires this)
        checkpoint = torch.load(_CHECKPOINT, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]

        # 4. Strip the 'model.' prefix that Lightning adds during training
        clean_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key.replace("model.", "")
            clean_state_dict[clean_key] = value

        # 5. Load weights then move to target device
        _pytorch_model.load_state_dict(clean_state_dict)
        _pytorch_model = _pytorch_model.to(device)

        # 6. Set to evaluation mode (turns off dropout)
        _pytorch_model.eval()
        
    return _pytorch_model


# ---------------------------------------------------------------------------
# Node 1: run BKT model
# ---------------------------------------------------------------------------

@observe(name="run_bkt")
def run_bkt_node(state: AgentState) -> dict:
    """Run the trained BKTransformer on a single student's sequence.

    Reads:   state["obs"], state["output"], state["skill_id_to_name"]
    Writes:  state["diagnosis"]
    """
    model = _load_model()

    # Ensure batch dimension is present: (1, T, 2)
    obs = state["obs"].to(device)
    output = state["output"].to(device)
    if obs.dim() == 2:
        obs = obs.unsqueeze(0)
        output = output.unsqueeze(0)

    with torch.no_grad():
        correct, latents, params, _ = model(obs, output)
    # correct:  (1, T, 1)
    # latents:  list of T tensors, each (1, n_skills)
    # params:   (1, T, n_skills, 4)  indices: 0=learn, 1=unused, 2=guess, 3=slip

    skill_id_to_name: dict = state["skill_id_to_name"]
    diagnosis: dict = {}
    T = output.shape[1]

    for t in range(T):
        skill_id = int(output[0, t, 0].item())
        if skill_id == -1000:
            continue
        diagnosis[f"timestep{t}"] = {
            "skill_id":          skill_id,
            "skill_name":        skill_id_to_name.get(skill_id, str(skill_id)),
            "actual_correct":    int(output[0, t, 1].item()),
            "prior":             latents[t][0][skill_id].item(),
            "learning_rate":     params[0, t, skill_id, 0].item(),
            "guess":             params[0, t, skill_id, 2].item(),
            "slip":              params[0, t, skill_id, 3].item(),
            "predicted_correct": correct[0, t].item(),
        }

    langfuse_context.update_current_observation(
        output=diagnosis,
        metadata={"n_timesteps": T, "device": str(device)},
    )
    return {"diagnosis": diagnosis}


# ---------------------------------------------------------------------------
# Node 2: LLM diagnosis → proficiency level + reasoning
# ---------------------------------------------------------------------------

def _build_output_template(student_id: str, diagnosis: dict) -> dict:
    template: list = []
    seen: set = set()
    for data in diagnosis.values():
        sid = data["skill_id"]
        if sid not in seen:
            template.append({
                "kc_name":          data["skill_name"],
                "kc_id":            sid,
                "proficiency_level": "?",
                "reasoning":        "?",
            })
            seen.add(sid)
    return {student_id: template}


@observe(name="diagnose")
def diagnose_node(state: AgentState) -> dict:
    """Call the LLM to assign proficiency levels from BKT parameters.

    Reads:   state["student_id"], state["diagnosis"]
    Writes:  state["analysis"]
    """
    student_id = state["student_id"]
    diagnosis = {student_id: state["diagnosis"]}
    template = _build_output_template(student_id, state["diagnosis"])

    prompt = f"""당신은 학습 데이터 분석 전문가입니다.

    ## 입력 데이터
    아래 JSON은 각 학생의 시간 순서별(timestep) BKT 파라미터를 포함합니다.
    각 timestep에는 해당 시점의 실제 정답(actual_correct), prior, learning_rate, guess, slip, predicted_correct, skill_name이 포함되어 있습니다.

    {json.dumps(diagnosis, indent=2, ensure_ascii=False)}

    ## 분석 기준
    - proficiency_level: 마지막 prior 기준으로 '상'(≥0.85), '중'(0.4~0.85), '하'(<0.4)
    - reasoning: 아래 항목들을 모두 반영하여 4~5문장으로 기술합니다.
    1. 정답률: 전체 timestep의 실제 정답 비율(`actual_correct`)을 바탕으로 성취 수준을 요약합니다.
    2. 학습 추세: `prior`와 `learning_rate`의 변화 추이를 분석해 학습 곡선이 상승/정체/하락 중 어느 방향인지 설명합니다.
    3. 오답 원인: `slip`(실수 확률)이 높으면 "알고도 실수", `guess`(추측 확률)가 높으면 "우연한 정답" 가능성을 언급합니다.
    4. 예측 변화: `predicted_correct`의 변화 방향을 근거로 학습 효과나 변동성을 요약합니다.
    5. 종합 결론: 위 정보를 종합하여 proficiency_level과 일관되게 마무리합니다.

    ## 출력 형식
    반드시 아래 구조와 동일한 JSON 형식으로 출력하세요:

    {json.dumps(template, indent=2, ensure_ascii=False)}

    주의사항:
    1. 모든 키 이름을 정확히 유지하세요.
    2. "?" 부분을 실제 분석 결과로 채우세요.
    - proficiency_level: "상", "중", "하" 중 하나
    - reasoning: 실제 분석 내용 (4~5문장)
    3. **순수 JSON만 출력**하세요."""

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 JSON 형식으로 정확하게 응답하는 학습 데이터 분석 전문가입니다."},
            {"role": "user",   "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    result = json.loads(response.choices[0].message.content)
    return {"analysis": result.get(student_id, [])}
