from unittest.mock import AsyncMock

import pytest

import ai_tutor.tools.neo4j_tool as neo4j_tool


@pytest.mark.asyncio
async def test_close_driver_closes_and_resets_global(monkeypatch):
    fake_driver = AsyncMock()
    monkeypatch.setattr(neo4j_tool, "_driver", fake_driver)

    await neo4j_tool.close_driver()

    fake_driver.close.assert_awaited_once()
    assert neo4j_tool._driver is None


@pytest.mark.asyncio
async def test_close_driver_is_noop_when_uninitialized(monkeypatch):
    monkeypatch.setattr(neo4j_tool, "_driver", None)

    await neo4j_tool.close_driver()

    assert neo4j_tool._driver is None
