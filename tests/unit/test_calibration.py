"""Tests for the pure-function side of the calibration script.

The model-run path (`main()`) needs torch + weights and is covered by
tests/integration/test_calibrate_main.py under the `requires_model` marker.
These tests cover compute_percentile, compute_calibration, the probe-class
resolution that decides which projection pool a category feeds, and CLI
parsability (so --help is verified to work).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from esta.scripts.calibrate import (
    PROBE_CLASS_EXCLUDED,
    PROBE_CLASS_HARMFUL,
    PROBE_CLASS_HARMLESS,
    CalibrationOutput,
    ProbeClassError,
    compute_calibration,
    compute_percentile,
    load_validation_set,
    parse_args,
    resolve_probe_class,
)

# --- compute_percentile ------------------------------------------------------


def test_percentile_50_on_evenly_spaced_data() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert compute_percentile(values, 50.0) == 3.0


def test_percentile_0_returns_minimum() -> None:
    assert compute_percentile([0.5, 1.5, 2.5], 0.0) == 0.5


def test_percentile_100_returns_maximum() -> None:
    assert compute_percentile([0.5, 1.5, 2.5], 100.0) == 2.5


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="empty sequence"):
        compute_percentile([], 50.0)


def test_out_of_bounds_percentile_raises() -> None:
    with pytest.raises(ValueError, match=r"in \[0, 100\]"):
        compute_percentile([1.0, 2.0], -1.0)
    with pytest.raises(ValueError, match=r"in \[0, 100\]"):
        compute_percentile([1.0, 2.0], 101.0)


# --- compute_calibration -----------------------------------------------------


def test_compute_calibration_returns_expected_thresholds() -> None:
    # Synthetic distributions with known percentile values.
    entropies = list(range(1, 101))                       # 1..100; p95 = 95.something
    margins = list(range(1, 101))                          # p10 = 10.something
    harmless = [float(v) for v in range(1, 101)]           # p95 of harmless
    harmful = [float(v) for v in range(101, 201)]          # p10 of harmful

    out = compute_calibration(
        entropies=entropies,
        margins=margins,
        harmless_projections=harmless,
        harmful_projections=harmful,
    )

    assert isinstance(out, CalibrationOutput)
    # numpy.percentile uses linear interpolation; expected for 1..100 are 95.05 and 10.9.
    assert 94.0 < out.spike_threshold < 96.0
    assert 10.0 < out.low_margin_threshold < 12.0
    assert 94.0 < out.pressure_low < 96.0
    assert 110.0 < out.pressure_moderate < 112.0


def test_compute_calibration_with_overlapping_distributions() -> None:
    # If harmless and harmful overlap heavily, pressure_low can end up >= pressure_moderate.
    # The function should still return values — the caller is responsible for noticing
    # the inversion and treating it as a calibration failure.
    harmless = [5.0] * 50 + [10.0] * 50
    harmful = [4.0] * 50 + [9.0] * 50

    out = compute_calibration(
        entropies=[1.0, 2.0, 3.0],
        margins=[1.0, 2.0, 3.0],
        harmless_projections=harmless,
        harmful_projections=harmful,
    )
    # Sanity: still got valid numbers. The relative ordering is the caller's concern.
    assert isinstance(out.pressure_low, float)
    assert isinstance(out.pressure_moderate, float)


def test_compute_calibration_preserves_provenance() -> None:
    out = compute_calibration(
        entropies=[1.0, 2.0],
        margins=[1.0, 2.0],
        harmless_projections=[1.0, 2.0],
        harmful_projections=[1.0, 2.0],
        provenance={"model": "test-model", "timestamp": "2026-05-11T00:00:00Z"},
    )
    assert out.provenance == {"model": "test-model", "timestamp": "2026-05-11T00:00:00Z"}


def test_calibration_output_to_json_is_round_trippable() -> None:
    out = compute_calibration(
        entropies=[1.0, 2.0, 3.0],
        margins=[1.0, 2.0, 3.0],
        harmless_projections=[0.1, 0.2, 0.3],
        harmful_projections=[1.1, 1.2, 1.3],
        provenance={"k": "v"},
    )
    data = json.loads(out.to_json())
    assert set(data.keys()) == {
        "spike_threshold", "low_margin_threshold", "pressure_low",
        "pressure_moderate", "provenance",
    }
    assert data["provenance"] == {"k": "v"}


# --- CLI ---------------------------------------------------------------------


def test_parse_args_with_defaults() -> None:
    args = parse_args([])
    assert args.model == "Qwen/Qwen2.5-7B-Instruct"
    assert args.percentile_spike == 95.0
    assert args.percentile_low_margin == 10.0


def test_parse_args_accepts_overrides() -> None:
    args = parse_args(
        ["--model", "Custom/Model", "--percentile-spike", "99.0", "--output", "/tmp/out.json"]
    )
    assert args.model == "Custom/Model"
    assert args.percentile_spike == 99.0
    assert str(args.output).endswith("out.json")


# --- load_validation_set -----------------------------------------------------


def test_load_validation_set_reads_all_categories(tmp_path: Path) -> None:
    (tmp_path / "cat_a.json").write_text(
        json.dumps({"category": "alpha", "prompts": [{"id": "a1", "text": "x"}]}),
        encoding="utf-8",
    )
    (tmp_path / "cat_b.json").write_text(
        json.dumps({"category": "beta", "prompts": [{"id": "b1"}, {"id": "b2"}]}),
        encoding="utf-8",
    )
    result = load_validation_set(tmp_path)
    assert set(result.keys()) == {"alpha", "beta"}
    assert len(result["alpha"]) == 1
    assert len(result["beta"]) == 2


def test_load_validation_set_handles_empty_dir(tmp_path: Path) -> None:
    assert load_validation_set(tmp_path) == {}


# --- resolve_probe_class -----------------------------------------------------
#
# The refusal probe's two calibration pools are defined by the extremes of the
# corpus: `pressure_low` is the upper tail of the HARMLESS projections and
# `pressure_moderate` is the lower tail of the HARMFUL ones. A category whose
# expected pressure is an intermediate label is neither a clean positive nor a
# clean negative, so it must be EXCLUDED rather than silently pooled with the
# negatives — pooling it there inflates the very tail that sets pressure_low,
# which widens the "low" band and under-reports pressure.


def _cat(pressure: str | None = None, probe_class: str | None = None) -> dict:
    data: dict = {"category": "c", "prompts": []}
    if pressure is not None:
        data["expected_state"] = {"safety_pressure": pressure}
    if probe_class is not None:
        data["probe_class"] = probe_class
    return data


def test_high_pressure_is_harmful() -> None:
    assert resolve_probe_class(_cat("high")) == PROBE_CLASS_HARMFUL


def test_low_pressure_is_harmless() -> None:
    assert resolve_probe_class(_cat("low")) == PROBE_CLASS_HARMLESS


def test_intermediate_labels_are_excluded() -> None:
    for label in ("low_to_moderate", "moderate_to_high", "moderate", "varies", "unknown"):
        assert resolve_probe_class(_cat(label)) == PROBE_CLASS_EXCLUDED, label


def test_missing_expected_state_is_excluded() -> None:
    # Conservative: an unlabeled category must not get to define the baseline.
    assert resolve_probe_class(_cat()) == PROBE_CLASS_EXCLUDED


def test_label_matching_is_case_and_whitespace_insensitive() -> None:
    assert resolve_probe_class(_cat("  HIGH ")) == PROBE_CLASS_HARMFUL
    assert resolve_probe_class(_cat("Low")) == PROBE_CLASS_HARMLESS


def test_explicit_probe_class_overrides_inference() -> None:
    # A category can be topically harmful-adjacent yet deliberately held out
    # of both pools (e.g. the dual-use defensive set, whose projections are
    # the measurement target rather than ground truth).
    assert resolve_probe_class(_cat("high", probe_class="excluded")) == PROBE_CLASS_EXCLUDED
    assert resolve_probe_class(_cat("low", probe_class="harmful")) == PROBE_CLASS_HARMFUL


def test_invalid_probe_class_raises() -> None:
    with pytest.raises(ProbeClassError, match="probe_class"):
        resolve_probe_class(_cat("low", probe_class="negative"))


def test_repo_validation_corpus_resolves_as_documented() -> None:
    """Lock in how the shipped corpus is actually pooled."""
    validation_dir = Path(__file__).resolve().parents[2] / "data" / "validation_cases"
    resolved = {}
    for path in sorted(validation_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        resolved[data["category"]] = resolve_probe_class(data)

    assert resolved["refusal_expected"] == PROBE_CLASS_HARMFUL
    assert resolved["factual_unambiguous"] == PROBE_CLASS_HARMLESS
    assert resolved["low_confidence"] == PROBE_CLASS_HARMLESS
    # Intermediate / experimental categories must not skew either pool.
    assert resolved["performed_uncertainty"] == PROBE_CLASS_EXCLUDED
    assert resolved["conflict_region"] == PROBE_CLASS_EXCLUDED
    # The dual-use set is the probe's measurement target, never its baseline.
    assert resolved["dual_use_defensive"] == PROBE_CLASS_EXCLUDED


def test_dual_use_pair_ids_reference_real_refusal_prompts() -> None:
    """The paired design is the point: a dangling pair_id silently loses a pair."""
    validation_dir = Path(__file__).resolve().parents[2] / "data" / "validation_cases"
    dual = json.loads((validation_dir / "dual_use_defensive.json").read_text(encoding="utf-8"))
    refusal = json.loads((validation_dir / "refusal_expected.json").read_text(encoding="utf-8"))

    refusal_ids = {p["id"] for p in refusal["prompts"]}
    partners = [p["pair_id"] for p in dual["prompts"] if p.get("pair_id")]

    assert partners, "expected at least one paired prompt"
    assert not set(partners) - refusal_ids, "pair_id must name a refusal_expected prompt"
    assert len(partners) == len(set(partners)), "each refusal prompt should pair at most once"
    assert all(
        p.get("framing") == "defensive" for p in dual["prompts"] if p.get("pair_id")
    ), "only defensive framings carry a pair_id"
