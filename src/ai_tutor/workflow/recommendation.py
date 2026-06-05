"""
Recommendation pipeline: one workflow step.

  recommend — for each skill in the diagnosis, deterministically selects
              the appropriate Neo4j Cypher tool based on proficiency level
              (상/중/하), fetches curriculum graph context from Neo4j, and
              generates a natural-language study recommendation via the LLM.

  All per-skill graph + LLM calls are executed concurrently with asyncio.gather,
  so total wall time equals the slowest single skill rather than the sum of all.

Prompt text lives in the Langfuse Prompt Registry ("feedback_prompt", label
"production").  Run scripts/create_prompts.py once to register it.
"""

import asyncio
import json
from typing import Any, cast

from langfuse import get_client, observe
from langfuse.model import TextPromptClient
from openai import AsyncOpenAI as _RawAsyncOpenAI, APIConnectionError, APITimeoutError

from ai_tutor.llm_client import get_llm_client, get_llm_model
from pydantic import ValidationError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from ai_tutor.workflow.schemas import FeedbackRecord
from ai_tutor.tools.neo4j_tool import get_tool

_TOOL_MAP = {
    "하": "get_prerequisites",
    "중": "get_current_concept",
    "상": "get_advanced_concepts",
}

# The knowledge graph is static — same skill_id + tool always returns the same
# result. Cache results in memory so repeat queries pay zero Neo4j latency.
_graph_cache: dict[tuple[str, int], dict | None] = {}

# ---------------------------------------------------------------------------
# Pure helper — conditional next-skills block (no Langfuse dependency)
# Pre-computing this in Python keeps conditional logic out of the template
# and makes it independently testable.
# ---------------------------------------------------------------------------

def _build_next_skills_section(
    level: str,
    graph_context: dict | None,
    analysis_by_id: dict[int, dict] | None = None,
) -> str:
    """Build the prerequisite / advanced-concept block for the feedback prompt.

    Returns an empty string when there is nothing to show (no graph context
    or no named next skills).
    """
    if not graph_context:
        return ""

    next_skills = [s for s in graph_context.get("next_skills", []) if s.get("name")]
    if not next_skills:
        return ""

    if level == "하":
        header = "\n## 먼저 학습해야 할 선행 개념:\n"
    else:
        header = "\n## 다음 학습 추천 개념 (도전할 수 있는 연계 내용):\n"

    lookup = analysis_by_id or {}
    lines = [header]
    for idx, skill in enumerate(next_skills, 1):
        lines.append(f"{idx}. **{skill.get('name', 'N/A')}**")
        lines.append(f"   - 학기: {skill.get('semester', 'N/A')}")
        lines.append(f"   - 대단원-중단원-소단원: {skill.get('chapter', 'N/A')}")
        lines.append(f"   - 성취 기준(소단원 기준): {skill.get('achievement', 'N/A')}")
        lines.append(f"   - 개념에 대한 설명: {skill.get('description', '개념 설명 없음')}")
        diagnosed = lookup.get(skill.get("skill_id"))
        if diagnosed:
            lines.append(f"   - 학생의 숙련도: {diagnosed['proficiency_level']}")
            lines.append(f"   - 진단: {diagnosed['reasoning']}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# LLM call with retry
# ---------------------------------------------------------------------------

@retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((APIConnectionError, APITimeoutError)),
    reraise=True,
)
async def _call_recommend_llm(
    client: _RawAsyncOpenAI,
    messages: list[dict],
    prompt: TextPromptClient,
) -> str:
    response = await client.chat.completions.create(
        model=get_llm_model(),
        messages=messages,
        response_format={"type": "json_object"},
        temperature=prompt.config.get("temperature", 0.3),
        langfuse_prompt=prompt,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Per-skill coroutine — graph fetch + prompt compile + LLM call
# ---------------------------------------------------------------------------

@observe(name="recommend_skill")
async def _process_skill(client: _RawAsyncOpenAI, kc_data: dict, analysis_by_id: dict[int, dict]) -> dict | None:
    """Fetch graph context and generate LLM feedback for a single skill."""
    get_client().update_current_span(input=kc_data)
    level    = kc_data["proficiency_level"]
    skill_id = kc_data["kc_id"]

    tool_name = _TOOL_MAP.get(level, "get_current_concept")
    cache_key = (tool_name, skill_id)

    if cache_key not in _graph_cache:
        tool = get_tool(tool_name)
        raw = await tool(skill_id=skill_id)
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
        else:
            parsed = raw
        if isinstance(parsed, list):
            _graph_cache[cache_key] = parsed[0] if parsed else None
        else:
            _graph_cache[cache_key] = parsed  # dict | None

    graph_context = _graph_cache[cache_key]

    ctx_name        = graph_context.get("name",        "N/A")         if graph_context else "N/A"
    ctx_semester    = graph_context.get("semester",    "N/A")         if graph_context else "N/A"
    _chapter_raw = graph_context.get("chapter", "") if graph_context else ""
    _parts = [p.strip() for p in _chapter_raw.split(">")] if _chapter_raw else []
    ctx_chapter = (
        f"- 대단원: {_parts[0] if len(_parts) > 0 else 'N/A'}\n"
        f"- 중단원: {_parts[1] if len(_parts) > 1 else 'N/A'}\n"
        f"- 소단원: {_parts[2] if len(_parts) > 2 else 'N/A'}"
    ) if _chapter_raw else "N/A"
    ctx_achievement = graph_context.get("achievement", "N/A")         if graph_context else "N/A"
    ctx_desc        = graph_context.get("description", "개념 설명 없음") if graph_context else "없음"

    # Fetch the versioned prompt from Langfuse — linked to the active trace span
    prompt = get_client().get_prompt("feedback_prompt", label="production")
    compiled = prompt.compile(
        kc_name=kc_data["kc_name"],
        level=level,
        reasoning=kc_data.get("reasoning", ""),
        ctx_name=ctx_name,
        ctx_semester=ctx_semester,
        ctx_chapter=ctx_chapter,
        ctx_achievement=ctx_achievement,
        ctx_desc=ctx_desc,
        next_skills_section=_build_next_skills_section(level, graph_context, analysis_by_id),
    )

    messages = [
        {
            "role": "system",
            "content": "당신은 숙련도 진단 결과와, 교육과정에 기반하여 학생에게 맞춤형 피드백을 제공하는 친절한 교육 전문가입니다. /no_think",
        },
        {"role": "user", "content": compiled},
    ]

    content = await _call_recommend_llm(client, messages, prompt)

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {"reasoning": "", "feedback": content}

    try:
        record = FeedbackRecord(
            kc_id=kc_data["kc_id"],
            kc_name=kc_data["kc_name"],
            proficiency_level=kc_data["proficiency_level"],
            reasoning=result.get("reasoning", kc_data.get("reasoning", "")),
            feedback=result.get("feedback", ""),
            prompt=compiled,
            input_data=json.dumps(kc_data, ensure_ascii=False),
            graph_context=json.dumps(graph_context, ensure_ascii=False) if graph_context else "",
        )
        record_dict = cast(dict[Any, Any], record.model_dump())
        get_client().update_current_span(output=record_dict)
        return record_dict
    except ValidationError:
        return None


# ---------------------------------------------------------------------------
# Workflow step implementation
# ---------------------------------------------------------------------------

async def recommend_node(state: dict) -> dict:
    """Retrieve graph context and generate per-skill feedback concurrently.

    Reads:   state['analysis']
    Writes:  state['feedback']
    """
    client = get_llm_client()
    analysis_by_id = {a["kc_id"]: a for a in state["analysis"]}

    # Fan out: all skills are processed concurrently
    results = await asyncio.gather(*[
        _process_skill(client, kc_data, analysis_by_id)
        for kc_data in state["analysis"]
    ])

    feedback_records = [r for r in results if r is not None]

    get_client().update_current_span(
        output=feedback_records,
        metadata={"n_skills": len(state["analysis"])},
    )
    return {"feedback": feedback_records}
