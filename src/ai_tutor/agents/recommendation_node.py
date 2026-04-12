"""
Recommendation pipeline: one LangGraph node.

  recommend_node — for each skill in the diagnosis, deterministically selects
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

from langfuse import Langfuse
from langfuse.openai import AsyncOpenAI
from langfuse.decorators import observe, langfuse_context
from openai import RateLimitError, APIConnectionError, APITimeoutError
from pydantic import ValidationError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from ai_tutor.agents.state import AgentState
from ai_tutor.agents.schemas import FeedbackRecord
from ai_tutor.tools.neo4j_tool import get_tool

_TOOL_MAP = {
    "하": "get_prerequisites",
    "중": "get_current_concept",
    "상": "get_advanced_concepts",
}

# The knowledge graph is static — same skill_id + tool always returns the same
# result. Cache results in memory so repeat queries pay zero Neo4j latency.
_graph_cache: dict[tuple[str, int], dict | None] = {}

_langfuse: Langfuse | None = None


def _get_langfuse() -> Langfuse:
    global _langfuse
    if _langfuse is None:
        _langfuse = Langfuse()
    return _langfuse


# ---------------------------------------------------------------------------
# Pure helper — conditional next-skills block (no Langfuse dependency)
# Pre-computing this in Python keeps conditional logic out of the template
# and makes it independently testable.
# ---------------------------------------------------------------------------

def _build_next_skills_section(level: str, graph_context: dict | None) -> str:
    """Build the prerequisite / advanced-concept block for the feedback prompt.

    Returns an empty string when there is nothing to show (no graph context,
    no named next skills, or medium proficiency which has neither).
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

    lines = [header]
    for idx, skill in enumerate(next_skills, 1):
        lines.append(f"{idx}. **{skill.get('name', 'N/A')}**")
        lines.append(f"   - 학기: {skill.get('semester', 'N/A')}")
        lines.append(f"   - 설명: {skill.get('description', '개념 설명 없음')}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# LLM call with retry
# ---------------------------------------------------------------------------

@retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
    reraise=True,
)
async def _call_recommend_llm(client: AsyncOpenAI, messages: list) -> str:
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Per-skill coroutine — graph fetch + prompt compile + LLM call
# ---------------------------------------------------------------------------

async def _process_skill(client: AsyncOpenAI, kc_data: dict) -> dict | None:
    """Fetch graph context and generate LLM feedback for a single skill."""
    level    = kc_data["proficiency_level"]
    skill_id = kc_data["kc_id"]

    tool_name = _TOOL_MAP.get(level, "get_current_concept")
    cache_key = (tool_name, skill_id)

    if cache_key not in _graph_cache:
        tool = await get_tool(tool_name)
        raw = await tool(skill_id=skill_id)
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
        else:
            parsed = raw
        _graph_cache[cache_key] = parsed[0] if isinstance(parsed, list) and parsed else parsed

    graph_context = _graph_cache[cache_key]

    ctx_name     = graph_context.get("name",        "N/A")        if graph_context else "N/A"
    ctx_semester = graph_context.get("semester",    "N/A")        if graph_context else "N/A"
    ctx_desc     = graph_context.get("description", "개념 설명 없음") if graph_context else "없음"

    # Fetch the versioned prompt from Langfuse — linked to the active trace span
    prompt_obj = _get_langfuse().get_prompt("feedback_prompt", label="production")
    prompt = prompt_obj.compile(
        kc_name=kc_data["kc_name"],
        level=level,
        reasoning=kc_data.get("reasoning", ""),
        ctx_name=ctx_name,
        ctx_semester=ctx_semester,
        ctx_desc=ctx_desc,
        next_skills_section=_build_next_skills_section(level, graph_context),
    )

    messages = [
        {
            "role": "system",
            "content": "당신은 학생의 학습 상태를 분석하고, 교육과정 기반으로 맞춤형 피드백을 제공하는 교육 전문가입니다.",
        },
        {"role": "user", "content": prompt},
    ]

    content = await _call_recommend_llm(client, messages)

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
            prompt=prompt,
            input_data=json.dumps(kc_data, ensure_ascii=False),
            graph_context=json.dumps(graph_context, ensure_ascii=False) if graph_context else "",
        )
        return cast(dict[Any, Any], record.model_dump())
    except ValidationError:
        return None


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

@observe(name="recommend")
async def recommend_node(state: AgentState) -> dict:
    """Retrieve graph context and generate per-skill feedback concurrently.

    Reads:   state['analysis']
    Writes:  state['feedback']
    """
    client = AsyncOpenAI()

    # Fan out: all skills are processed concurrently
    results = await asyncio.gather(*[
        _process_skill(client, kc_data)
        for kc_data in state["analysis"]
    ])

    feedback_records = [r for r in results if r is not None]

    langfuse_context.update_current_observation(
        output=feedback_records,
        metadata={"n_skills": len(state["analysis"])},
    )
    return {"feedback": feedback_records}
