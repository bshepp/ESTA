"""Pydantic models for the epistemic_state metadata returned with every response.

Schema is versioned via SCHEMA_VERSION. Breaking changes bump the minor digit
through Phase 1; Phase 2 (conflict + features) bumps the schema to 0.2.0.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.1.0"

PressureLabel = Literal["low", "moderate", "high", "uncalibrated"]


class ConfidenceMetrics(BaseModel):
    """Aggregate confidence statistics across the generated token sequence."""

    mean_entropy: float = Field(..., description="Mean per-token entropy in nats.")
    median_entropy: float
    p90_entropy: float
    max_entropy: float
    mean_margin: float = Field(..., description="Mean top1-top2 logprob gap.")
    low_margin_fraction: float = Field(
        ..., description="Fraction of tokens where the top1-top2 gap was below threshold."
    )
    entropy_spike_count: int = Field(
        ..., description="Number of tokens whose entropy exceeded the spike threshold."
    )
    calibrated_confidence: float | None = Field(
        default=None,
        description="0-1 calibrated confidence; null until a calibration set has been fit.",
    )


class SafetyPressure(BaseModel):
    """Magnitude of safety-training pressure inferred from refusal-direction projection."""

    refusal_projection_max: float
    refusal_projection_mean: float
    calibrated_pressure: PressureLabel
    probe_version: str
    layer: int


class ModelInfo(BaseModel):
    name: str
    revision: str | None = None
    quantization: str


class Provenance(BaseModel):
    timestamp: str
    request_id: str
    audit_log_path: str


class EpistemicState(BaseModel):
    """The structured-metadata extension returned alongside every chat completion."""

    schema_version: str = SCHEMA_VERSION
    model: ModelInfo
    confidence: ConfidenceMetrics
    safety_pressure: SafetyPressure
    provenance: Provenance
