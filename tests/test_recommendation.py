"""
Unit tests for recommendation step helpers.

These tests cover pure-Python logic only: the next-skills section builder
and Pydantic validation.
"""

import pytest

from ai_tutor.workflow.recommendation import _build_next_skills_section
from ai_tutor.workflow.schemas import FeedbackRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
