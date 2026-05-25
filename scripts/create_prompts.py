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

## 입력 데이터
아래 JSON은 정답율(accuracy_rate)과 개념(KC)별 BKT 파라미터를 시간순으로 나열한 것입니다.
BKT 파라미터의 정의는 다음과 같습니다.
- priors  : P(knowledge) — 알고 있을 확률
- learning_rates : 학습했을 확율
- guesses : 모르는데 찍어서 맞았을 확률
- slips   : 아는데 실수했을 확률

{{aggregated_json}}

## 분석 가이드라인

- reasoning: 아래 항목들을 모두 반영하여 4~5문장으로 기술합니다.
1. 정답률: accuracy_rate를 바탕으로 정답률을 언급하세요.
2. priors 값의 시간에 따른 패턴(상승,하락,안정) 등을 구체적으로 언급하세요.
3. learning_rates 값이 높으면 학습 중이나, 낮거나 감소하면 학습이 일어나지 않는 것입니다. 이미 priors 값이 높다면 학습이 일어나지 않을 수 있습니다. 학습율은 prior값이 높지 않을 때 중요합니다.
4. slips 값은 아는데 실수했을 가능성입니다. priors가 높은 학생의 정답률이 100%가 아니고 해당값이 높게 나타나는 경우에만 언급하세요.
5. guess 값은 모르는데 찍어서 맞았을 가능성입니다. priors가 낮은 학생의 정답률이 0%가 아니고 해당값이 높게 나타나는 경우에만 언급하세요.
6. 종합 결론: 엄격하게 위의 정보만을 종합하여 proficiency_level과 일관되게 마무리합니다.

## 출력 형식
반드시 아래 구조와 동일한 JSON 형식으로 출력하세요:

{{template_json}}

주의사항:
1. 모든 키 이름을 정확히 유지하세요.
2. reasoning의 '?' 부분을 분석 가이드라인을 반영하여 채우세요. 입력받은 정보만을 활용하세요. 마크다운이나 추가 설명은
절대 포함하지 마세요.
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

학생의 숙련도에 대한 진단 결과는 다음과 같습니다.
- 개념명: {{kc_name}}
- 숙련도 레벨: {{level}}
- 분석 내용: {{reasoning}}


교육과정은 대단원>중단원>소단원>개념의 체계를 따릅니다.
현재 개념의 교육과정 정보는 다음과 같습니다.
- 학기: {{ctx_semester}}
{{ctx_chapter}}
- 개념명: {{ctx_name}}
- 성취 기준(소단원 기준): {{ctx_achievement}}
- 개념에 대한 설명: {{ctx_desc}}

위 학생 분석 결과와 교육과정 정보를 바탕으로, 학생에게 맞춤형 피드백을 작성하세요.

## 피드백 작성 전략

숙련도 수준({{level}})에 따라 다음 전략을 적용하세요:
- [상]: 달성한 성취를 칭찬하고, '다음 학습 추천 개념' 항목의 교육 정보를 제시하며 도전을 제안하세요.
- [중]: 조금만 더 힘내자는 응원과 함께 성취 기준과 개념에 대한 자세한 설명을 제시하세요.
- [하]: 좌절하지 않도록 격려하며, 먼저 학습해야 할 선행 개념의 복습을 유도하세요.
- 모든 숙련도 수준에서 교육과정의 정보를 언급하세요.
- 교육과정 정보의 값이 대부분 'N/A'인 경우, 현재 개념에 대한 복습과 칭찬 위주로 작성하세요.


## 출력 형식
반드시 아래 구조의 순수 JSON 형식으로만 출력하세요. 마크다운이나 추가 설명은 절대 포함하지 마세요.

{
  "feedback": "학생에게 직접 전달될 최종 피드백 메시지입니다. 피드백 작성 전략에 따라 따뜻한 대화체로 작성하세요."
}
"""


def main() -> None:
    lf.create_prompt(
        name="diagnosis_prompt",
        prompt=DIAGNOSIS_PROMPT,
        labels=["production"],
        config={"model": "qwen3:8b-q4_K_M", "temperature": 0.2},
    )
    print("Created: diagnosis_prompt [production]")

    lf.create_prompt(
        name="feedback_prompt",
        prompt=FEEDBACK_PROMPT,
        labels=["production"],
        config={"model": "qwen3:8b-q4_K_M", "temperature": 0.1},
    )
    print("Created: feedback_prompt [production]")

    print("\nDone. View your prompts at: https://cloud.langfuse.com → Prompts")


if __name__ == "__main__":
    main()
