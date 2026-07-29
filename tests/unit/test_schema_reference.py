"""Guard docs/SCHEMA.md against drifting from the pydantic models.

The JSON contract already has a drift test; prose does not, and prose is what
integrators actually read. A field added to the models without a line in the
reference is a silent documentation gap, so it fails here instead.

This checks coverage and version agreement, not wording — it cannot tell you
the description is *correct*, only that the field is not missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from esta.schema import (
    SCHEMA_VERSION,
    CalibrationInfo,
    ConfidenceMetrics,
    EpistemicState,
    ModelInfo,
    Provenance,
    SafetyPressure,
)

DOCUMENTED_MODELS: tuple[type[BaseModel], ...] = (
    EpistemicState,
    ModelInfo,
    ConfidenceMetrics,
    SafetyPressure,
    CalibrationInfo,
    Provenance,
)

REFERENCE = Path(__file__).resolve().parents[2] / "docs" / "SCHEMA.md"


@pytest.fixture(scope="module")
def reference_text() -> str:
    assert REFERENCE.exists(), f"schema reference missing at {REFERENCE}"
    return REFERENCE.read_text(encoding="utf-8")


def test_every_field_is_documented(reference_text: str) -> None:
    missing = [
        f"{model.__name__}.{field}"
        for model in DOCUMENTED_MODELS
        for field in model.model_fields
        if f"`{field}`" not in reference_text
    ]
    assert not missing, (
        f"fields present in the models but absent from docs/SCHEMA.md: {missing}. "
        "Document them, or the reference silently drifts."
    )


def test_every_model_is_documented(reference_text: str) -> None:
    missing = [m.__name__ for m in DOCUMENTED_MODELS if m.__name__ not in reference_text]
    assert not missing, f"models absent from docs/SCHEMA.md: {missing}"


def test_reference_states_the_current_schema_version(reference_text: str) -> None:
    assert SCHEMA_VERSION in reference_text, (
        f"docs/SCHEMA.md does not mention SCHEMA_VERSION {SCHEMA_VERSION}; "
        "bumping the version requires updating the reference and its changelog table."
    )


def test_reference_documents_every_pressure_label(reference_text: str) -> None:
    """All four labels must be described — 'uncalibrated' especially."""
    for label in ("low", "moderate", "high", "uncalibrated"):
        assert f"`{label}`" in reference_text, label
