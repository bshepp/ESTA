"""Tests for the epistemic_state pydantic schema.

Covers: round-trip serialization, schema-version constant, default population,
required-field enforcement, and PressureLabel literal constraint.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from esta.schema import (
    SCHEMA_VERSION,
    ChatCompletionRequest,
    ChatMessage,
    ConfidenceMetrics,
    EpistemicState,
    ModelInfo,
    Provenance,
    SafetyPressure,
)


def _make_state(**overrides) -> EpistemicState:
    defaults: dict = dict(
        model=ModelInfo(name="Qwen/Qwen2.5-7B-Instruct", quantization="bfloat16"),
        confidence=ConfidenceMetrics(
            mean_entropy=0.5,
            median_entropy=0.4,
            p90_entropy=1.2,
            max_entropy=2.0,
            mean_margin=1.0,
            low_margin_fraction=0.1,
            entropy_spike_count=1,
        ),
        safety_pressure=SafetyPressure(
            refusal_projection_max=0.1,
            refusal_projection_mean=0.05,
            calibrated_pressure="low",
            probe_version="arditi_v1_unrefined",
            layer=14,
        ),
        provenance=Provenance(
            timestamp="2026-05-11T00:00:00Z",
            request_id="esta-test",
            audit_log_path="audit_logs/esta-2026-05-11.jsonl",
        ),
    )
    defaults.update(overrides)
    return EpistemicState(**defaults)


def test_schema_version_constant() -> None:
    assert SCHEMA_VERSION == "0.1.0"


def test_schema_version_default_on_state() -> None:
    state = _make_state()
    assert state.schema_version == SCHEMA_VERSION


def test_roundtrip_dump_validate() -> None:
    original = _make_state()
    dumped = original.model_dump()
    rehydrated = EpistemicState.model_validate(dumped)
    assert rehydrated == original


def test_missing_required_field_raises() -> None:
    with pytest.raises(ValidationError):
        ConfidenceMetrics(
            mean_entropy=0.0,
            # median_entropy missing
            p90_entropy=0.0,
            max_entropy=0.0,
            mean_margin=0.0,
            low_margin_fraction=0.0,
            entropy_spike_count=0,
        )


def test_pressure_label_accepts_known_values() -> None:
    for label in ("low", "moderate", "high", "uncalibrated"):
        SafetyPressure(
            refusal_projection_max=0.0,
            refusal_projection_mean=0.0,
            calibrated_pressure=label,
            probe_version="v",
            layer=14,
        )


def test_pressure_label_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        SafetyPressure(
            refusal_projection_max=0.0,
            refusal_projection_mean=0.0,
            calibrated_pressure="extreme",  # not a valid PressureLabel
            probe_version="v",
            layer=14,
        )


def test_calibrated_confidence_defaults_to_none() -> None:
    metrics = ConfidenceMetrics(
        mean_entropy=0.0,
        median_entropy=0.0,
        p90_entropy=0.0,
        max_entropy=0.0,
        mean_margin=0.0,
        low_margin_fraction=0.0,
        entropy_spike_count=0,
    )
    assert metrics.calibrated_confidence is None


def test_chat_completion_request_defaults() -> None:
    req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
    assert req.model == "local"
    assert req.max_tokens == 512
    assert req.temperature == 0.7
    assert req.top_p == 0.95
    assert req.return_activations is False


def test_chat_completion_request_return_activations_in_body() -> None:
    # Confirms return_activations is a body-level field (not query) per the spec decision.
    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="hi")],
        return_activations=True,
    )
    assert req.return_activations is True


def test_chat_completion_request_rejects_empty_messages() -> None:
    # An empty message list must fail validation (422), not reach generation.
    with pytest.raises(ValidationError):
        ChatCompletionRequest(messages=[])


@pytest.mark.parametrize(
    "field, value",
    [
        ("max_tokens", 0),
        ("max_tokens", -1),
        ("temperature", -0.1),
        ("temperature", 2.1),
        ("top_p", 0.0),
        ("top_p", 1.1),
    ],
)
def test_chat_completion_request_rejects_out_of_bounds_params(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hi")],
            **{field: value},
        )


@pytest.mark.parametrize(
    "field, value",
    [
        ("max_tokens", 1),
        ("max_tokens", 8192),
        ("temperature", 0.0),
        ("temperature", 2.0),
        ("top_p", 1.0),
    ],
)
def test_chat_completion_request_accepts_boundary_params(field: str, value: float) -> None:
    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="hi")],
        **{field: value},
    )
    assert getattr(req, field) == value
