"""Tests for the pure-function side of the calibration script.

The model-run path raises NotImplementedError and is exercised end-to-end
in a follow-up session. These tests cover compute_percentile and
compute_calibration plus the CLI parsability (so --help is verified to work).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from esta.scripts.calibrate import (
    CalibrationOutput,
    compute_calibration,
    compute_percentile,
    load_validation_set,
    parse_args,
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
