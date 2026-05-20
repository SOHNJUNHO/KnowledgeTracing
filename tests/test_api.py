import pytest
from httpx import ASGITransport, AsyncClient

from ai_tutor.api import app


@pytest.mark.asyncio
async def test_tutor_endpoint_returns_feedback(monkeypatch):
    monkeypatch.setattr("ai_tutor.api._warmup_bkt", lambda: None)

    async def fake_run_tutor(state):
        assert state["student_id"] == "s1"
        assert state["obs"] == [[[1, 1], [2, 0]]]
        assert state["output"] == [[[2, 0], [1, 1]]]
        return {
            "feedback": [
                {
                    "kc_name": "순환소수",
                    "kc_id": 1,
                    "proficiency_level": "중",
                    "reasoning": "학습 추세가 안정적입니다.",
                    "feedback": "현재 개념을 한 번 더 연습하세요.",
                }
            ]
        }

    monkeypatch.setattr("ai_tutor.api._get_run_tutor", lambda: fake_run_tutor)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post(
            "/tutor",
            json={
                "student_id": "s1",
                "sequence": [[1, 1], [2, 0], [1, 1]],
                "skill_id_to_name": {"1": "순환소수", "2": "유리수"},
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["student_id"] == "s1"
    assert payload["feedback"][0]["kc_id"] == 1


@pytest.mark.asyncio
async def test_tutor_endpoint_rejects_short_sequence(monkeypatch):
    monkeypatch.setattr("ai_tutor.api._warmup_bkt", lambda: None)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post(
            "/tutor",
            json={
                "student_id": "s1",
                "sequence": [[1, 1]],
                "skill_id_to_name": {"1": "순환소수"},
            },
        )

    assert resp.status_code == 422
