"""
Diagnosis pipeline: two workflow steps.

  run_bkt   — calls the separate BKTransformer inference service via HTTP,
              receives per-skill BKT parameters, builds timestep records.

  diagnose  — aggregates per-timestep BKT data into compact per-skill summaries,
              then calls an LLM to assign a proficiency level (상/중/하) and
              natural-language reasoning per skill.

Deployment note
---------------
For single-instance / demo deployments the BKTransformer can be loaded
in-process via FastAPI lifespan instead of running a separate service:

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = _load_model()
        yield

The inference service is preferred for production scale so that one GPU-
hosted model is shared across many API workers rather than replicated.
"""

import json
import os
from collections import defaultdict

import httpx
from langfuse import get_client
from openai import AsyncOpenAI as _RawAsyncOpenAI, RateLimitError, APIConnectionError, APITimeoutError

from ai_tutor.llm_client import get_llm_client, get_llm_model
from pydantic import ValidationError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from ai_tutor.workflow.schemas import BKTTimestep, AnalysisRecord

# ---------------------------------------------------------------------------
# BKTransformer inference service URL — override via BKT_SERVICE_URL env var
# ---------------------------------------------------------------------------
_BKT_SERVICE_URL = os.getenv("BKT_SERVICE_URL", "http://localhost:8001")

# ---------------------------------------------------------------------------
# Step 1: Call BKT inference service → build timestep records
# ---------------------------------------------------------------------------

async def run_bkt_node(state: dict) -> dict:
    """Run BKTransformer inference on a single student's sequence.

    Sends the observation and output tensors to the separate inference
    service and reconstructs per-timestep BKT records from the response.

    Reads:   state['obs'], state['output'], state['skill_id_to_name']
    Writes:  state['diagnosis']
    """
    skill_id_to_name: dict = state["skill_id_to_name"]

    payload = {
        "obs": state["obs"],
        "output": state["output"],
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{_BKT_SERVICE_URL}/infer", json=payload)
        resp.raise_for_status()
        result = resp.json()

    corrects = result["corrects"]      # [[...]]
    latents = result["latents"]        # [[...]] — per timestep, per skill
    params = result["params"]          # [[...]] — per timestep, per skill, 4 values

    T = len(latents)
    diagnosis: dict = {}

    for t in range(T):
        skill_id = int(state["output"][0][t][0])
        if skill_id == -1000:
            continue
        entry = BKTTimestep(
            skill_id=skill_id,
            skill_name=skill_id_to_name.get(skill_id, str(skill_id)),
            actual_correct=int(state["output"][0][t][1]),
            prior=float(latents[t][skill_id]),
            learning_rate=float(params[t][skill_id][0]),
            guess=float(params[t][skill_id][2]),
            slip=float(params[t][skill_id][3]),
            predicted_correct=float(corrects[0][t]),
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


def _build_output_template(student_id: str, diagnosis: dict) -> dict:
    template: list = []
    seen: set = set()
    for data in diagnosis.values():
        sid = data["skill_id"]
        if sid not in seen:
            template.append({
                "kc_name":           data["skill_name"],
                "kc_id":             sid,
                "proficiency_level": "?",
                "reasoning":         "?",
            })
            seen.add(sid)
    return {student_id: template}


# ---------------------------------------------------------------------------
# Step 2: LLM diagnosis → proficiency level + reasoning
# ---------------------------------------------------------------------------

@retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
    reraise=True,
)
async def _call_diagnose_llm(client: _RawAsyncOpenAI, messages: list, prompt_obj) -> str:
    response = await client.chat.completions.create(
        model=get_llm_model(),
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.2,
        langfuse_prompt=prompt_obj,
    )
    return response.choices[0].message.content or ""


async def diagnose_node(state: dict) -> dict:
    """Call the LLM to assign proficiency levels from BKT summaries.

    Reads:   state['student_id'], state['diagnosis']
    Writes:  state['analysis']
    """
    student_id = state["student_id"]
    aggregated = _aggregate_bkt_by_skill(state["diagnosis"])
    template = _build_output_template(student_id, state["diagnosis"])

    prompt_obj = get_client().get_prompt("diagnosis_prompt", label="production")
    prompt = prompt_obj.compile(
        aggregated_json=json.dumps({student_id: aggregated}, indent=2, ensure_ascii=False),
        template_json=json.dumps(template, indent=2, ensure_ascii=False),
    )

    client = get_llm_client()
    messages = [
        {"role": "system", "content": "당신은 JSON 형식으로 정확하게 응답하는 학습 데이터 분석 전문가입니다."},
        {"role": "user",   "content": prompt},
    ]

    raw = json.loads(await _call_diagnose_llm(client, messages, prompt_obj))
    raw_list = raw.get(student_id, [])

    validated: list[dict] = []
    for record in raw_list:
        try:
            validated.append(AnalysisRecord(**record).model_dump())
        except ValidationError as exc:
            get_client().update_current_span(
                metadata={"validation_error": str(exc), "bad_record": record}
            )

    return {"analysis": validated}
