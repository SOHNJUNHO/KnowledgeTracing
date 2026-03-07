"""
Recommendation pipeline: one LangGraph node.

  recommend_node — for each skill in the diagnosis, selects the appropriate
                   Neo4j Cypher tool based on proficiency level (상/중/하),
                   retrieves curriculum graph context, then calls the LLM to
                   generate a natural-language study recommendation.
"""

import json

from langfuse.openai import OpenAI
from langfuse.decorators import observe, langfuse_context

from .state import AgentState
from tools.neo4j_tool import get_tool


# Proficiency level → tool name mapping
_TOOL_MAP = {
    "하": "get_prerequisites",
    "중": "get_current_concept",
    "상": "get_advanced_concepts",
}


def _build_feedback_prompt(kc_data: dict, graph_context: dict | None) -> str:
    level = kc_data["proficiency_level"]

    ctx_name = graph_context.get("name", "N/A") if graph_context else "N/A"
    ctx_semester = graph_context.get("semester", "N/A") if graph_context else "N/A"
    ctx_desc = graph_context.get("description", "개념 설명 없음") if graph_context else "없음"

    text = f"""학생 분석 결과:
- 지식 구성 요소: {kc_data['kc_name']}
- 숙련도 수준: {level}
- 분석 내용: {kc_data.get('reasoning', '')}

교육과정 정보:
- 개념명: {ctx_name}
- 학기: {ctx_semester}
- 설명: {ctx_desc}
"""

    if graph_context:
        next_skills = [s for s in graph_context.get("next_skills", []) if s.get("name")]
        if next_skills:
            if level == "하":
                text += "\n## 먼저 학습해야 할 선행 개념:\n"
            else:  # 상
                text += "\n## 다음 학습 추천 개념 (도전할 수 있는 연계 내용):\n"

            for idx, skill in enumerate(next_skills, 1):
                text += f"""{idx}. **{skill.get('name', 'N/A')}**
   - 학기: {skill.get('semester', 'N/A')}
   - 설명: {skill.get('description', '개념 설명 없음')}
"""

    text += f"""
위 학생 분석 결과와 교육과정 정보를 바탕으로, 학생에게 맞춤형 피드백을 작성하세요.

## 출력 형식
다음 JSON 형식으로만 출력하세요:
{{
  "reasoning": "분석에 기반한 현재 상태 요약 (2~3문장)",
  "feedback": "학생에게 전달할 맞춤형 학습 추천 (2~3문장)"
}}

주의사항:
- proficiency_level '{level}'에 맞는 피드백을 제공하세요.
- 교육과정 정보(학기, 연계 개념)를 구체적으로 언급하세요.
- **순수 JSON만 출력**하세요."""

    return text


@observe(name="recommend")
def recommend_node(state: AgentState) -> dict:
    """Retrieve graph context from Neo4j and generate per-skill feedback.

    Reads:   state["analysis"]
    Writes:  state["feedback"]
    """
    client = OpenAI()
    feedback_records: list = []

    for kc_data in state["analysis"]:
        level = kc_data["proficiency_level"]
        skill_id = kc_data["kc_id"]

        # Deterministic tool selection based on proficiency level
        tool_name = _TOOL_MAP.get(level, "get_current_concept")
        tool = get_tool(tool_name)
        raw = tool.invoke({"skill_id": skill_id})

        # toolbox_langchain returns a JSON string; parse to dict
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
        else:
            parsed = raw
        # Cypher returns a list of records; take the first
        graph_context = parsed[0] if isinstance(parsed, list) and parsed else parsed

        prompt = _build_feedback_prompt(kc_data, graph_context)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 학생의 학습 상태를 분석하고, 교육과정 기반으로 맞춤형 피드백을 제공하는 교육 전문가입니다.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        try:
            result = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            result = {"reasoning": "", "feedback": response.choices[0].message.content}

        feedback_records.append({
            "kc_id":             kc_data["kc_id"],
            "kc_name":           kc_data["kc_name"],
            "proficiency_level": kc_data["proficiency_level"],
            "reasoning":         result.get("reasoning", kc_data.get("reasoning", "")),
            "feedback":          result.get("feedback", ""),
            # Fields kept for Langfuse tracing / RAGAS evaluation
            "prompt":            prompt,
            "input_data":        json.dumps(kc_data, ensure_ascii=False),
            "graph_context":     json.dumps(graph_context, ensure_ascii=False) if graph_context else "",
        })

    langfuse_context.update_current_observation(
        output=feedback_records,
        metadata={"n_skills": len(state["analysis"])},
    )
    return {"feedback": feedback_records}
