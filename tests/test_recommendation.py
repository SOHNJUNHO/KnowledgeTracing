"""
Unit tests for recommendation step helpers.

All LLM, Neo4j toolbox, and Langfuse calls are mocked — these tests verify
orchestration logic, the pure next-skills section builder, Pydantic validation,
and concurrent fan-out behaviour without making any network calls.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ai_tutor.workflow.recommendation import _build_next_skills_section
from ai_tutor.workflow.schemas import FeedbackRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

KC_LOW = {
    "kc_id": 1, "kc_name": "순환소수",
    "proficiency_level": "하",
    "reasoning": "Prior is low and not improving.",
}
KC_HIGH = {
    "kc_id": 2, "kc_name": "유리수",
    "proficiency_level": "상",
    "reasoning": "Prior is consistently high.",
}
GRAPH_CONTEXT = {
    "name": "순환소수", "semester": "1학기",
    "description": "소수점 이하가 반복되는 수",
    "next_skills": [{"name": "유리수", "semester": "1학기", "description": "..."}],
}


# ---------------------------------------------------------------------------
# _build_next_skills_section — pure function, no mocking needed
# ---------------------------------------------------------------------------

def test_next_skills_section_low_has_prerequisite_header():
    section = _build_next_skills_section("하", GRAPH_CONTEXT)
    assert "선행 개념" in section


def test_next_skills_section_high_has_advanced_header():
    section = _build_next_skills_section("상", GRAPH_CONTEXT)
    assert "다음 학습 추천" in section


def test_next_skills_section_mid_with_next_skills_still_shows_advanced():
    """중 proficiency falls into the else branch — shows advanced header."""
    section = _build_next_skills_section("중", GRAPH_CONTEXT)
    assert "다음 학습 추천" in section


def test_next_skills_section_returns_empty_for_none_context():
    assert _build_next_skills_section("하", None) == ""


def test_next_skills_section_returns_empty_when_no_named_skills():
    ctx = {**GRAPH_CONTEXT, "next_skills": [{"name": "", "semester": "1학기"}]}
    assert _build_next_skills_section("하", ctx) == ""


def test_next_skills_section_returns_empty_for_missing_key():
    ctx = {"name": "순환소수", "semester": "1학기", "description": "..."}
    # no "next_skills" key at all
    assert _build_next_skills_section("하", ctx) == ""


# ---------------------------------------------------------------------------
# FeedbackRecord Pydantic validation
# ---------------------------------------------------------------------------

def test_feedback_record_valid():
    record = FeedbackRecord(
        kc_id=1, kc_name="순환소수", proficiency_level="하",
        reasoning="현재 이해도가 낮습니다.",
        feedback="선행 개념부터 복습하세요.",
        prompt="prompt text", input_data="{}", graph_context="{}",
    )
    assert record.kc_id == 1


def test_feedback_record_rejects_empty_feedback():
    with pytest.raises(Exception):
        FeedbackRecord(
            kc_id=1, kc_name="순환소수", proficiency_level="하",
            reasoning="현재 이해도가 낮습니다.",
            feedback="",  # must not be empty
            prompt="p", input_data="{}", graph_context="{}",
        )


def test_feedback_record_rejects_invalid_proficiency():
    with pytest.raises(Exception):
        FeedbackRecord(
            kc_id=1, kc_name="순환소수", proficiency_level="low",  # must be 상/중/하
            reasoning="r", feedback="f",
            prompt="p", input_data="{}", graph_context="{}",
        )


# ---------------------------------------------------------------------------
# recommend_node — concurrent fan-out (all I/O mocked)
# ---------------------------------------------------------------------------

def _mock_langfuse_prompt(mocker, compiled_text: str = "mocked prompt"):
    """Patch _get_langfuse so prompt.compile() returns a predictable string."""
    mock_prompt = MagicMock()
    mock_prompt.compile.return_value = compiled_text
    mock_lf = MagicMock()
    mock_lf.get_prompt.return_value = mock_prompt
    mocker.patch("ai_tutor.workflow.recommendation.get_client", return_value=mock_lf)
    mocker.patch("ai_tutor.workflow.recommendation.get_llm_model", return_value="test-model")


@pytest.mark.asyncio
async def test_recommend_node_returns_feedback_for_all_skills(mocker):
    _mock_langfuse_prompt(mocker)

    mock_tool = AsyncMock(return_value=json.dumps([GRAPH_CONTEXT]))
    mocker.patch("ai_tutor.workflow.recommendation.get_tool", return_value=mock_tool)

    llm_payload = json.dumps({"reasoning": "분석 요약", "feedback": "학습 추천 내용"})
    mock_response = MagicMock()
    mock_response.choices[0].message.content = llm_payload
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response
    mocker.patch("ai_tutor.workflow.recommendation.get_llm_client", return_value=mock_client)

    from ai_tutor.workflow.recommendation import recommend_node
    result = await recommend_node({"analysis": [KC_LOW, KC_HIGH]})

    assert len(result["feedback"]) == 2


@pytest.mark.asyncio
async def test_recommend_node_calls_llm_once_per_skill(mocker):
    _mock_langfuse_prompt(mocker)

    mock_tool = AsyncMock(return_value=json.dumps([GRAPH_CONTEXT]))
    mocker.patch("ai_tutor.workflow.recommendation.get_tool", return_value=mock_tool)

    llm_payload = json.dumps({"reasoning": "r", "feedback": "f"})
    mock_response = MagicMock()
    mock_response.choices[0].message.content = llm_payload
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response
    mocker.patch("ai_tutor.workflow.recommendation.get_llm_client", return_value=mock_client)

    from ai_tutor.workflow.recommendation import recommend_node
    n_skills = 3
    await recommend_node({"analysis": [KC_LOW] * n_skills})

    assert mock_client.chat.completions.create.call_count == n_skills


@pytest.mark.asyncio
async def test_recommend_node_skips_invalid_llm_output(mocker):
    """A ValidationError on one skill must not crash the whole pipeline."""
    _mock_langfuse_prompt(mocker)

    mock_tool = AsyncMock(return_value=json.dumps([GRAPH_CONTEXT]))
    mocker.patch("ai_tutor.workflow.recommendation.get_tool", return_value=mock_tool)

    # Empty feedback string fails FeedbackRecord validation (min_length=1)
    bad_payload = json.dumps({"reasoning": "r", "feedback": ""})
    mock_response = MagicMock()
    mock_response.choices[0].message.content = bad_payload
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response
    mocker.patch("ai_tutor.workflow.recommendation.get_llm_client", return_value=mock_client)

    from ai_tutor.workflow.recommendation import recommend_node
    result = await recommend_node({"analysis": [KC_LOW, KC_HIGH]})

    # Both records invalid — list is empty, not a crash
    assert isinstance(result["feedback"], list)
    assert len(result["feedback"]) == 0
