"""Integration smoke test against a real (tiny) model.

Marked `requires_model` so it is deselected by default. Run explicitly with:

    pytest -m requires_model tests/integration/

Uses Qwen/Qwen2.5-0.5B-Instruct, which is small enough to fit in 8GB of VRAM
or run on CPU in a few minutes. The refusal direction is NOT loaded here,
so the safety_pressure block returns the uncalibrated stub.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.requires_model


@pytest.fixture(scope="module")
def client():
    # Imported lazily so the module load doesn't drag torch into the default
    # `pytest` run.
    from fastapi.testclient import TestClient

    from esta.api.server import app

    with TestClient(app) as c:
        yield c


def test_health_endpoint(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_basic_completion_returns_epistemic_state(client) -> None:
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "local",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 32,
            "temperature": 0.0,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "epistemic_state" in body
    state = body["epistemic_state"]
    assert state["schema_version"] == "0.1.0"
    assert "confidence" in state
    assert "safety_pressure" in state
    assert "provenance" in state
    # The model wasn't given a refusal direction in this smoke test, so safety
    # pressure should be uncalibrated.
    assert state["safety_pressure"]["calibrated_pressure"] == "uncalibrated"
