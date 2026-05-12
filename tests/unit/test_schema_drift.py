"""Drift test: on-disk JSON Schema must match what pydantic generates.

If this test fails, a pydantic model changed without the canonical JSON
Schema being regenerated. Fix with:

    python -m esta.scripts.dump_schema

and commit the updated `src/esta/schema/epistemic_state.schema.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

from esta.scripts.dump_schema import DEFAULT_OUTPUT, build_schema, render_schema


def test_on_disk_schema_matches_pydantic_model() -> None:
    on_disk = Path(DEFAULT_OUTPUT).read_text(encoding="utf-8")
    generated = render_schema(build_schema())
    assert on_disk == generated, (
        "epistemic_state.schema.json is out of date with the pydantic model. "
        "Run `python -m esta.scripts.dump_schema` and commit the updated file."
    )


def test_schema_has_expected_top_level_fields() -> None:
    schema = build_schema()
    # Top-level required fields on EpistemicState.
    required = set(schema.get("required", []))
    expected = {"model", "confidence", "safety_pressure", "provenance"}
    assert expected.issubset(required), (
        f"Required top-level fields missing: {expected - required}"
    )


def test_schema_includes_referenced_definitions() -> None:
    schema = build_schema()
    defs = schema.get("$defs", {})
    expected = {"ConfidenceMetrics", "SafetyPressure", "ModelInfo", "Provenance"}
    assert expected.issubset(set(defs.keys())), (
        f"Missing definitions in schema: {expected - set(defs.keys())}"
    )


def test_schema_file_is_valid_json() -> None:
    data = json.loads(Path(DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert "$defs" in data or "definitions" in data
