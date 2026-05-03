"""
Unit tests for diagnosis step helpers.

These tests cover pure-Python logic only (aggregation, template building,
Pydantic validation).  No model checkpoint or LLM calls are made.
"""

import pytest
from ai_tutor.workflow.diagnosis import _aggregate_bkt_by_skill, _build_output_template
from ai_tutor.workflow.schemas import AnalysisRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_DIAGNOSIS = {
    "timestep0": {
        "skill_id": 1, "skill_name": "Skill A",
        "actual_correct": 0, "prior": 0.30, "learning_rate": 0.10,
        "guess": 0.12, "slip": 0.06, "predicted_correct": 0.32,
    },
    "timestep1": {
        "skill_id": 1, "skill_name": "Skill A",
        "actual_correct": 1, "prior": 0.52, "learning_rate": 0.10,
        "guess": 0.12, "slip": 0.06, "predicted_correct": 0.51,
    },
    "timestep2": {
        "skill_id": 2, "skill_name": "Skill B",
        "actual_correct": 1, "prior": 0.80, "learning_rate": 0.08,
        "guess": 0.10, "slip": 0.04, "predicted_correct": 0.78,
    },
}


# ---------------------------------------------------------------------------
# _aggregate_bkt_by_skill
# ---------------------------------------------------------------------------

def test_aggregate_groups_by_skill_id():
    result = _aggregate_bkt_by_skill(MOCK_DIAGNOSIS)
    assert set(result.keys()) == {1, 2}


def test_aggregate_n_observations():
    result = _aggregate_bkt_by_skill(MOCK_DIAGNOSIS)
    assert result[1]["n_observations"] == 2
    assert result[2]["n_observations"] == 1


def test_aggregate_accuracy_rate():
    result = _aggregate_bkt_by_skill(MOCK_DIAGNOSIS)
    assert result[1]["accuracy_rate"] == pytest.approx(0.5, abs=1e-3)
    assert result[2]["accuracy_rate"] == pytest.approx(1.0, abs=1e-3)


def test_aggregate_priors_sequence_order():
    """priors list must be in timestep order so the LLM reads the trajectory correctly."""
    result = _aggregate_bkt_by_skill(MOCK_DIAGNOSIS)
    assert result[1]["priors"] == [pytest.approx(0.30, abs=1e-4), pytest.approx(0.52, abs=1e-4)]


def test_aggregate_priors_captures_oscillation():
    """The full sequence [0.30, 0.75, 0.28, 0.80, 0.31] is passed intact —
    the LLM can see the oscillation directly without any derived statistic."""
    raw_priors = [0.30, 0.75, 0.28, 0.80, 0.31]
    oscillating = {
        f"t{i}": {**MOCK_DIAGNOSIS["timestep0"], "skill_id": 4, "skill_name": "Skill D", "prior": p}
        for i, p in enumerate(raw_priors)
    }
    result = _aggregate_bkt_by_skill(oscillating)
    assert result[4]["priors"] == [pytest.approx(p, abs=1e-4) for p in raw_priors]


# ---------------------------------------------------------------------------
# _build_output_template
# ---------------------------------------------------------------------------

def test_template_deduplicates_skills():
    template = _build_output_template("s1", MOCK_DIAGNOSIS)
    assert len(template["s1"]) == 2  # skill 1 and skill 2


def test_template_placeholder_values():
    template = _build_output_template("s1", MOCK_DIAGNOSIS)
    for entry in template["s1"]:
        assert entry["proficiency_level"] == "?"
        assert entry["reasoning"] == "?"


# ---------------------------------------------------------------------------
# AnalysisRecord Pydantic validation
# ---------------------------------------------------------------------------

def test_analysis_record_valid():
    record = AnalysisRecord(
        kc_id=1, kc_name="Skill A",
        proficiency_level="중", reasoning="학습 추세가 안정적입니다."
    )
    assert record.proficiency_level == "중"


def test_analysis_record_rejects_invalid_level():
    with pytest.raises(Exception):
        AnalysisRecord(
            kc_id=1, kc_name="Skill A",
            proficiency_level="high",  # must be 상/중/하
            reasoning="valid reasoning"
        )


def test_analysis_record_rejects_empty_reasoning():
    with pytest.raises(Exception):
        AnalysisRecord(
            kc_id=1, kc_name="Skill A",
            proficiency_level="상",
            reasoning=""
        )
