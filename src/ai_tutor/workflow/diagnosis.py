"""
Diagnosis pipeline: two workflow steps.

  run_bkt   — calls BKTransformer via Triton Inference Server (gRPC), builds
              timestep records.

  diagnose  — aggregates per-timestep BKT data into compact per-skill summaries,
              then calls an LLM to assign a proficiency level (상/중/하) and
              natural-language reasoning per skill.
"""

import json
import os
from collections import defaultdict

import numpy as np
import tritonclient.grpc.aio as triton_grpc

from langfuse import get_client, observe
from langfuse.model import TextPromptClient
from openai import AsyncOpenAI as _RawAsyncOpenAI, APIConnectionError, APITimeoutError

from ai_tutor.llm_client import get_llm_client, get_llm_model
from pydantic import ValidationError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from ai_tutor.workflow.schemas import BKTTimestep, AnalysisRecord

# ---------------------------------------------------------------------------
# Proficiency thresholds — calibrate on the validation set
# ---------------------------------------------------------------------------

_LEVEL_HIGH_CUT = 0.90 #0.95 ~ 0.85
_LEVEL_LOW_CUT  = 0.30 #0.3 ~ 0.35


def _level_from_prior(prior: float) -> str:
    """Map a BKT mastery prior in [0, 1] to a proficiency level (상/중/하)."""
    if prior >= _LEVEL_HIGH_CUT:
        return "상"
    if prior >= _LEVEL_LOW_CUT:
        return "중"
    return "하"


# ---------------------------------------------------------------------------
# Triton gRPC client — lazy singleton, one connection shared across requests
# ---------------------------------------------------------------------------

_triton_client: triton_grpc.InferenceServerClient | None = None


def _get_triton_client() -> triton_grpc.InferenceServerClient:
    global _triton_client
    if _triton_client is None:
        url = os.getenv("TRITON_URL", "localhost:8001")
        _triton_client = triton_grpc.InferenceServerClient(url=url)
    return _triton_client


async def close_triton_client() -> None:
    global _triton_client
    if _triton_client is not None:
        await _triton_client.close()
        _triton_client = None


async def is_bkt_ready() -> bool:
    try:
        return await _get_triton_client().is_model_ready("bkt_transformer")
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Step 1: BKT inference via Triton → build timestep records
# ---------------------------------------------------------------------------

async def _infer_triton(
    obs_np: np.ndarray, output_np: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    client = _get_triton_client()

    obs_in    = triton_grpc.InferInput("obs",    list(obs_np.shape),    "FP32")
    output_in = triton_grpc.InferInput("output", list(output_np.shape), "FP32")
    obs_in.set_data_from_numpy(obs_np)
    output_in.set_data_from_numpy(output_np)

    result = await client.infer(
        model_name="bkt_transformer",
        inputs=[obs_in, output_in],
        outputs=[
            triton_grpc.InferRequestedOutput("corrects"),
            triton_grpc.InferRequestedOutput("latents"),
            triton_grpc.InferRequestedOutput("params"),
        ],
    )
    return (
        result.as_numpy("corrects"),
        result.as_numpy("latents"),
        result.as_numpy("params"),
    )


@observe(name="run_bkt")
async def run_bkt_node(state: dict) -> dict:
    """Run BKTransformer inference for a single student via Triton gRPC.

    Reads:   state['obs'], state['output'], state['skill_id_to_name']
    Writes:  state['diagnosis']
    """
    get_client().update_current_span(input=state)
    skill_id_to_name: dict = state["skill_id_to_name"]

    obs_np    = np.array(state["obs"],    dtype=np.float32)
    output_np = np.array(state["output"], dtype=np.float32)
    T = obs_np.shape[1]

    corrects, latents, params = await _infer_triton(obs_np, output_np)

    diagnosis: dict = {}
    for t in range(T):
        skill_id = int(output_np[0, t, 0])
        if skill_id == -1000:
            continue
        entry = BKTTimestep(
            skill_id=skill_id,
            skill_name=skill_id_to_name.get(skill_id, str(skill_id)),
            actual_correct=int(output_np[0, t, 1]),
            prior=float(latents[0, t, skill_id]),
            learning_rate=float(params[0, t, skill_id, 0]),
            guess=float(params[0, t, skill_id, 2]),
            slip=float(params[0, t, skill_id, 3]),
            predicted_correct=float(corrects[0, t, 0]),
        )
        diagnosis[f"timestep{t}"] = entry.model_dump()

    get_client().update_current_span(
        output=diagnosis,
        metadata={"n_timesteps": T},
    )
    return {"diagnosis": diagnosis}


# ---------------------------------------------------------------------------
# BKT aggregation — condenses timestep rows into compact per-skill summaries
# ---------------------------------------------------------------------------

def _aggregate_bkt_by_skill(diagnosis: dict) -> dict[int, dict]:
    """Collapse timestep-level BKT data into per-skill summary statistics.

    The LLM reasons over trajectory summaries (trend, accuracy, final prior),
    not raw timestep rows.  Computing trend/accuracy in Python keeps the
    prompt compact and ensures those statistics are exact rather than inferred
    by the model.  The full timestep data is still stored in state['diagnosis']
    for Langfuse tracing and human review.
    """
    skill_timesteps: dict[int, list[dict]] = defaultdict(list)
    for data in diagnosis.values():
        skill_timesteps[data["skill_id"]].append(data)

    aggregated: dict[int, dict] = {}
    for skill_id, steps in skill_timesteps.items():
        priors = [s["prior"] for s in steps]
        corrects = [s["actual_correct"] for s in steps]

        aggregated[skill_id] = {
            "skill_name":     steps[0]["skill_name"],
            "n_observations": len(steps),
            "accuracy_rate":  round(sum(corrects) / len(corrects), 3),
            "priors":         [round(s["prior"], 4) for s in steps],
            "learning_rates": [round(s["learning_rate"], 4) for s in steps],
            "guesses":        [round(s["guess"], 4) for s in steps],
            "slips":          [round(s["slip"],  4) for s in steps],
        }

    return aggregated


def _build_output_template(
    student_id: str,
    aggregated: dict[int, dict],
    levels: dict[int, str],
) -> dict:
    template = [
        {
            "kc_name":           agg["skill_name"],
            "kc_id":             sid,
            "proficiency_level": levels[sid],
            "reasoning":         "?",
        }
        for sid, agg in aggregated.items()
    ]
    return {student_id: template}


# ---------------------------------------------------------------------------
# Step 2: LLM diagnosis → proficiency level + reasoning
# ---------------------------------------------------------------------------

@retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((APIConnectionError, APITimeoutError)),
    reraise=True,
)
async def _call_diagnose_llm(
    client: _RawAsyncOpenAI,
    messages: list[dict],
    prompt: TextPromptClient,
) -> str:
    response = await client.chat.completions.create(
        model=get_llm_model(),
        messages=messages,
        response_format={"type": "json_object"},
        temperature=prompt.config.get("temperature", 0.2),
        langfuse_prompt=prompt,
    )
    return response.choices[0].message.content or ""


@observe(name="diagnose")
async def diagnose_node(state: dict) -> dict:
    """Call the LLM to assign proficiency levels from BKT summaries.

    Reads:   state['student_id'], state['diagnosis']
    Writes:  state['analysis']
    """
    get_client().update_current_span(input=state)
    student_id = state["student_id"]
    aggregated = _aggregate_bkt_by_skill(state["diagnosis"])
    levels = {sid: _level_from_prior(agg["priors"][-1]) for sid, agg in aggregated.items()}
    template = _build_output_template(student_id, aggregated, levels)

    prompt = get_client().get_prompt("diagnosis_prompt", label="production")
    compiled = prompt.compile(
        aggregated_json=json.dumps({student_id: aggregated}, indent=2, ensure_ascii=False),
        template_json=json.dumps(template, indent=2, ensure_ascii=False),
    )

    client = get_llm_client()
    messages = [
        {"role": "system", "content": "당신은 Bayesian Knowledg Tracing(BKT) 데이터를 해석하여 학생의 지식 수준을 진단하는 학습 데이터 분석 전문가입니다. /no_think"},
        {"role": "user",   "content": compiled},
    ]

    raw = json.loads(await _call_diagnose_llm(client, messages, prompt))
    raw_list = raw.get(student_id, [])

    validated: list[dict] = []
    for record in raw_list:
        kc_id = record.get("kc_id")
        if kc_id in levels:
            record["proficiency_level"] = levels[kc_id]
        try:
            validated.append(AnalysisRecord(**record).model_dump())
        except ValidationError as exc:
            get_client().update_current_span(
                metadata={"validation_error": str(exc), "bad_record": record}
            )

    return {"analysis": validated}
