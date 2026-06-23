"""Tests for the pure-numpy metric-extraction function.

These tests verify the wiring from per-token log-probs and per-prompt
projections into the ConfidenceMetrics / SafetyPressure schema objects.
They run without torch installed, which is what CI uses.

The torch-dependent wrapper `generate_with_epistemic_state` is covered by
the integration smoke test under tests/integration/.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from esta.calibration import Calibration
from esta.extraction import extract_metrics
from esta.schema import CalibrationInfo


def _peaked_log_probs(vocab_size: int = 100) -> np.ndarray:
    """Single near-deterministic next-token distribution."""
    top = -1e-6
    rest = math.log((1.0 - math.exp(top)) / (vocab_size - 1))
    arr = np.full(vocab_size, rest, dtype=np.float64)
    arr[0] = top
    return arr


def _two_token_log_probs():
    # Two generated tokens, vocab size 4; peaked distributions.
    a = np.log(np.array([0.7, 0.2, 0.07, 0.03]))
    b = np.log(np.array([0.6, 0.3, 0.07, 0.03]))
    return [a, b]


# --- probe disabled / uncalibrated path -------------------------------------


def test_returns_uncalibrated_safety_when_probe_not_loaded() -> None:
    _, safety, calib, _ = extract_metrics(
        token_log_probs=[_peaked_log_probs()],
        projections=[],
        probe_loaded=False,
        refusal_layer=14,
        calibration=Calibration.uncalibrated(),
    )
    assert safety.calibrated_pressure == "uncalibrated"
    assert safety.probe_version == "not_loaded"
    assert safety.refusal_projection_max == 0.0
    assert safety.refusal_projection_mean == 0.0
    assert safety.layer == 14


def test_probe_loaded_but_no_projections_falls_back_to_uncalibrated() -> None:
    # E.g., zero generated tokens: hook captured nothing.
    _, safety, calib, _ = extract_metrics(
        token_log_probs=[],
        projections=[],
        probe_loaded=True,
        refusal_layer=14,
        calibration=Calibration.uncalibrated(),
    )
    assert safety.calibrated_pressure == "uncalibrated"


# --- probe loaded paths ------------------------------------------------------


def test_low_projection_yields_low_pressure_label() -> None:
    cal = Calibration(
        spike=5.0, low_margin=0.3, pressure_low=0.5, pressure_moderate=1.5,
        calibrated=True, calibration_id="abc123", calibrated_at="2026-06-22T00:00:00Z",
        model="Qwen/Qwen2.5-7B-Instruct", source="calibration.json",
    )
    _, safety, calib, _ = extract_metrics(
        token_log_probs=[_peaked_log_probs()],
        projections=[0.1, 0.2, 0.3],   # all well below default low cutoff (0.5)
        probe_loaded=True,
        refusal_layer=14,
        calibration=cal,
    )
    assert safety.calibrated_pressure == "low"
    assert math.isclose(safety.refusal_projection_max, 0.3)
    assert math.isclose(safety.refusal_projection_mean, 0.2)


def test_moderate_projection_yields_moderate_pressure_label() -> None:
    cal = Calibration(
        spike=5.0, low_margin=0.3, pressure_low=0.5, pressure_moderate=1.5,
        calibrated=True, calibration_id="abc123", calibrated_at="2026-06-22T00:00:00Z",
        model="Qwen/Qwen2.5-7B-Instruct", source="calibration.json",
    )
    _, safety, calib, _ = extract_metrics(
        token_log_probs=[_peaked_log_probs()],
        projections=[0.8, 1.0, 1.2],   # max in [low=0.5, moderate=1.5)
        probe_loaded=True,
        refusal_layer=14,
        calibration=cal,
    )
    assert safety.calibrated_pressure == "moderate"


def test_high_projection_yields_high_pressure_label() -> None:
    cal = Calibration(
        spike=5.0, low_margin=0.3, pressure_low=0.5, pressure_moderate=1.5,
        calibrated=True, calibration_id="abc123", calibrated_at="2026-06-22T00:00:00Z",
        model="Qwen/Qwen2.5-7B-Instruct", source="calibration.json",
    )
    _, safety, calib, _ = extract_metrics(
        token_log_probs=[_peaked_log_probs()],
        projections=[0.4, 2.5, 0.6],   # max above moderate=1.5
        probe_loaded=True,
        refusal_layer=14,
        calibration=cal,
    )
    assert safety.calibrated_pressure == "high"
    assert math.isclose(safety.refusal_projection_max, 2.5)


def test_label_uses_max_projection_not_mean() -> None:
    # Mean is low (0.5), but one spike pushes max into "high".
    cal = Calibration(
        spike=5.0, low_margin=0.3, pressure_low=0.5, pressure_moderate=1.5,
        calibrated=True, calibration_id="abc123", calibrated_at="2026-06-22T00:00:00Z",
        model="Qwen/Qwen2.5-7B-Instruct", source="calibration.json",
    )
    _, safety, calib, _ = extract_metrics(
        token_log_probs=[_peaked_log_probs()],
        projections=[0.1, 0.1, 0.1, 0.1, 1.9],
        probe_loaded=True,
        refusal_layer=14,
        calibration=cal,
    )
    assert safety.calibrated_pressure == "high"


# --- threshold overrides -----------------------------------------------------


def test_custom_pressure_thresholds_override_defaults() -> None:
    # custom thresholds: low=2.0, moderate=5.0
    # projection=1.5 -> "low" under custom (below 2.0), would be "high" under defaults (>=1.5)
    cal = Calibration(
        spike=5.0, low_margin=0.3, pressure_low=2.0, pressure_moderate=5.0,
        calibrated=True, calibration_id="abc123", calibrated_at="2026-06-22T00:00:00Z",
        model="Qwen/Qwen2.5-7B-Instruct", source="calibration.json",
    )
    _, safety, calib, _ = extract_metrics(
        token_log_probs=[_peaked_log_probs()],
        projections=[1.5],
        probe_loaded=True,
        refusal_layer=14,
        calibration=cal,
    )
    # Under the default thresholds 1.5 would be "high"; with custom it's "low".
    assert safety.calibrated_pressure == "low"


def test_custom_spike_threshold_passed_through_to_confidence() -> None:
    # Construct two "tokens" with known entropies via uniform distributions
    # of different vocab sizes.
    high_entropy = np.full(1000, -math.log(1000), dtype=np.float64)  # H = ln(1000) ~ 6.9
    low_entropy = _peaked_log_probs(vocab_size=100)
    cal = Calibration(
        spike=2.0, low_margin=0.3, pressure_low=0.5, pressure_moderate=1.5,
        calibrated=False,
    )
    confidence, _, calib, _ = extract_metrics(
        token_log_probs=[high_entropy, low_entropy],
        projections=[],
        probe_loaded=False,
        refusal_layer=14,
        calibration=cal,
    )
    # Only high_entropy exceeds 2.0
    assert confidence.entropy_spike_count == 1


# --- probe version + layer wired through ------------------------------------


def test_probe_version_surfaced_on_loaded_probe() -> None:
    cal = Calibration(
        spike=5.0, low_margin=0.3, pressure_low=0.5, pressure_moderate=1.5,
        calibrated=True, calibration_id="abc123", calibrated_at="2026-06-22T00:00:00Z",
        model="Qwen/Qwen2.5-7B-Instruct", source="calibration.json",
    )
    _, safety, calib, _ = extract_metrics(
        token_log_probs=[_peaked_log_probs()],
        projections=[0.1],
        probe_loaded=True,
        refusal_layer=14,
        calibration=cal,
        probe_version="arditi_v2_custom",
    )
    assert safety.probe_version == "arditi_v2_custom"


def test_refusal_layer_surfaced_on_both_paths() -> None:
    cal = Calibration(
        spike=5.0, low_margin=0.3, pressure_low=0.5, pressure_moderate=1.5,
        calibrated=True, calibration_id="abc123", calibrated_at="2026-06-22T00:00:00Z",
        model="Qwen/Qwen2.5-7B-Instruct", source="calibration.json",
    )
    _, loaded, calib1, _ = extract_metrics(
        token_log_probs=[_peaked_log_probs()],
        projections=[0.1],
        probe_loaded=True,
        refusal_layer=22,
        calibration=cal,
    )
    _, unloaded, calib2, _ = extract_metrics(
        token_log_probs=[_peaked_log_probs()],
        projections=[],
        probe_loaded=False,
        refusal_layer=22,
        calibration=Calibration.uncalibrated(),
    )
    assert loaded.layer == 22
    assert unloaded.layer == 22


# --- debug_info shape --------------------------------------------------------


def test_debug_info_contains_raw_series() -> None:
    cal = Calibration(
        spike=5.0, low_margin=0.3, pressure_low=0.5, pressure_moderate=1.5,
        calibrated=True, calibration_id="abc123", calibrated_at="2026-06-22T00:00:00Z",
        model="Qwen/Qwen2.5-7B-Instruct", source="calibration.json",
    )
    confidence, _, calib, debug = extract_metrics(
        token_log_probs=[_peaked_log_probs(), _peaked_log_probs(200)],
        projections=[0.5, 1.0],
        probe_loaded=True,
        refusal_layer=14,
        calibration=cal,
    )
    assert set(debug.keys()) == {"raw_entropies", "raw_margins", "raw_projections"}
    assert len(debug["raw_entropies"]) == 2
    assert len(debug["raw_margins"]) == 2
    assert debug["raw_projections"] == [0.5, 1.0]
    # raw entropies / margins should align with the aggregate confidence values.
    assert math.isclose(confidence.mean_entropy, sum(debug["raw_entropies"]) / 2)


def test_empty_inputs_produce_zero_metrics() -> None:
    conf, safety, calib, debug = extract_metrics(
        token_log_probs=[],
        projections=[],
        probe_loaded=False,
        refusal_layer=14,
        calibration=Calibration.uncalibrated(),
    )
    assert conf.mean_entropy == 0.0
    assert conf.entropy_spike_count == 0
    assert safety.calibrated_pressure == "uncalibrated"
    assert debug["raw_entropies"] == []
    assert debug["raw_projections"] == []


# --- three-state honesty tests (Task 3) -------------------------------------


def test_probe_absent_is_uncalibrated_with_zero_projection() -> None:
    conf, safety, calib, _ = extract_metrics(
        token_log_probs=_two_token_log_probs(),
        projections=[],
        probe_loaded=False,
        refusal_layer=14,
        calibration=Calibration.uncalibrated(),
    )
    assert isinstance(calib, CalibrationInfo)
    assert calib.calibrated is False
    assert safety.calibrated_pressure == "uncalibrated"
    assert safety.probe_version == "not_loaded"
    assert safety.refusal_projection_max == 0.0


def test_probe_loaded_but_uncalibrated_shows_raw_projection() -> None:
    conf, safety, calib, _ = extract_metrics(
        token_log_probs=_two_token_log_probs(),
        projections=[0.8, 1.9, 0.5],
        probe_loaded=True,
        refusal_layer=14,
        calibration=Calibration.uncalibrated(),
    )
    assert calib.calibrated is False
    assert safety.calibrated_pressure == "uncalibrated"   # gated: no real label without calibration
    assert safety.probe_version != "not_loaded"           # probe IS loaded
    assert safety.refusal_projection_max == pytest.approx(1.9)   # real measurement surfaced


def test_probe_loaded_and_calibrated_emits_real_label() -> None:
    cal = Calibration(
        spike=5.0, low_margin=0.3, pressure_low=0.5, pressure_moderate=1.5,
        calibrated=True, calibration_id="abc123def456",
        calibrated_at="2026-06-22T00:00:00Z", model="Qwen/Qwen2.5-7B-Instruct",
        source="calibration.json",
    )
    conf, safety, calib, _ = extract_metrics(
        token_log_probs=_two_token_log_probs(),
        projections=[0.8, 1.9, 0.5],
        probe_loaded=True,
        refusal_layer=14,
        calibration=cal,
    )
    assert calib.calibrated is True
    assert calib.calibration_id == "abc123def456"
    assert safety.calibrated_pressure == "high"           # 1.9 >= pressure_moderate 1.5
    assert safety.refusal_projection_max == pytest.approx(1.9)
