"""
One-time script: register prompt templates in the Langfuse Prompt Registry.

Run once per environment (dev / prod) after setting your LANGFUSE_* credentials.
Re-run whenever you want to publish a new prompt version — the previous version
is preserved in the registry and any live trace will show which version it used.

Usage:
    python scripts/create_prompts.py
"""

from dotenv import load_dotenv
load_dotenv()

from langfuse import get_client

lf = get_client()


# ---------------------------------------------------------------------------
# 1. Diagnosis prompt
#
# Variables injected at call time:
#   {{aggregated_json}}  — per-skill BKT summary dict (JSON string)
#   {{template_json}}    — output schema with "?" placeholders (JSON string)
# ---------------------------------------------------------------------------

DIAGNOSIS_PROMPT = """\
당신은 학습 데이터 분석 전문가입니다.

## 입력 데이터
아래 JSON은 각 학생의 지식 구성요소(KC)별 BKT 파라미터 요약입니다.
각 항목에는 accuracy_rate와 시간 순서로 정렬된 네 가지 시계열이 포함됩니다:
- priors  : 각 timestep의 P(knowledge) — 지식 상태 시계열
- learning_rates : 각 timestep의 P(learn) — 학습 속도 시계열
- guesses : 각 timestep의 P(guess)    — 추측 확률 시계열
- slips   : 각 timestep의 P(slip)     — 실수 확률 시계열

{{aggregated_json}}

## 분석 기준
- proficiency_level: priors의 마지막 값 기준으로 '상'(≥0.85), '중'(0.4~0.85), '하'(<0.4)
- reasoning: 아래 항목들을 모두 반영하여 4~5문장으로 기술합니다.
1. 정답률: accuracy_rate를 바탕으로 성취 수준을 요약합니다.
2. 학습 궤적: priors 시계열 전체를 보고 상승, 하락, 안정, 오르내림(oscillating), 회복(dip then rise) 등의 패턴을 구체적으로 설명합니다.
3. 학습 속도: learning_rates 시계열이 높으면 빠르게 습득 중임을, 낮거나 감소하면 추가 학습이 필요함을 언급합니다.
4. 오답 원인: slips 시계열이 높거나 증가하면 "알고도 실수"가 늘고 있음을, guesses 시계열이 높거나 증가하면 "우연한 정답"에 의존하고 있음을 언급합니다.
5. 종합 결론: 위 정보를 종합하여 proficiency_level과 일관되게 마무리합니다.

## 출력 형식
반드시 아래 구조와 동일한 JSON 형식으로 출력하세요:

{{template_json}}

주의사항:
1. 모든 키 이름을 정확히 유지하세요.
2. "?" 부분을 실제 분석 결과로 채우세요.
   - proficiency_level: "상", "중", "하" 중 하나
   - reasoning: 실제 분석 내용 (4~5문장)
3. **순수 JSON만 출력**하세요.\
"""


# ---------------------------------------------------------------------------
# 2. Feedback prompt
#
# Variables injected at call time:
#   {{kc_name}}             — knowledge component name
#   {{level}}               — proficiency level (상 / 중 / 하)
#   {{reasoning}}           — diagnosis reasoning from previous node
#   {{ctx_name}}            — KC name from Neo4j
#   {{ctx_semester}}        — semester from Neo4j
#   {{ctx_desc}}            — description from Neo4j
#   {{next_skills_section}} — pre-built prerequisite / advanced concept block
#                             (empty string when not applicable)
# ---------------------------------------------------------------------------

FEEDBACK_PROMPT = """\
학생 분석 결과:
- 지식 구성 요소: {{kc_name}}
- 숙련도 수준: {{level}}
- 분석 내용: {{reasoning}}

교육과정 정보:
- 개념명: {{ctx_name}}
- 학기: {{ctx_semester}}
- 설명: {{ctx_desc}}
{{next_skills_section}}
위 학생 분석 결과와 교육과정 정보를 바탕으로, 학생에게 맞춤형 피드백을 작성하세요.

## 출력 형식
다음 JSON 형식으로만 출력하세요:
{
  "reasoning": "분석에 기반한 현재 상태 요약 (2~3문장)",
  "feedback": "학생에게 전달할 맞춤형 학습 추천 (2~3문장)"
}

주의사항:
- proficiency_level '{{level}}'에 맞는 피드백을 제공하세요.
- 교육과정 정보(학기, 연계 개념)를 구체적으로 언급하세요.
- **순수 JSON만 출력**하세요.\
"""


def main() -> None:
    lf.create_prompt(
        name="diagnosis_prompt",
        prompt=DIAGNOSIS_PROMPT,
        labels=["production"],
        config={"model": "gpt-4o-mini", "temperature": 0.2},
    )
    print("Created: diagnosis_prompt [production]")

    lf.create_prompt(
        name="feedback_prompt",
        prompt=FEEDBACK_PROMPT,
        labels=["production"],
        config={"model": "gpt-4o-mini", "temperature": 0.3},
    )
    print("Created: feedback_prompt [production]")

    print("\nDone. View your prompts at: https://cloud.langfuse.com → Prompts")


if __name__ == "__main__":
    main()
