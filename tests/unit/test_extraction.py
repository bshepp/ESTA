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

from esta.extraction import extract_metrics
from esta.probes.thresholds import PressureThresholds


def _peaked_log_probs(vocab_size: int = 100) -> np.ndarray:
    """Single near-deterministic next-token distribution."""
    top = -1e-6
    rest = math.log((1.0 - math.exp(top)) / (vocab_size - 1))
    arr = np.full(vocab_size, rest, dtype=np.float64)
    arr[0] = top
    return arr


# --- probe disabled / uncalibrated path -------------------------------------


def test_returns_uncalibrated_safety_when_probe_not_loaded() -> None:
    _, safety, _ = extract_metrics(
        token_log_probs=[_peaked_log_probs()],
        projections=[],
        probe_loaded=False,
        refusal_layer=14,
    )
    assert safety.calibrated_pressure == "uncalibrated"
    assert safety.probe_version == "not_loaded"
    assert safety.refusal_projection_max == 0.0
    assert safety.refusal_projection_mean == 0.0
    assert safety.layer == 14


def test_probe_loaded_but_no_projections_falls_back_to_uncalibrated() -> None:
    # E.g., zero generated tokens: hook captured nothing.
    _, safety, _ = extract_metrics(
        token_log_probs=[],
        projections=[],
        probe_loaded=True,
        refusal_layer=14,
    )
    assert safety.calibrated_pressure == "uncalibrated"


# --- probe loaded paths ------------------------------------------------------


def test_low_projection_yields_low_pressure_label() -> None:
    _, safety, _ = extract_metrics(
        token_log_probs=[_peaked_log_probs()],
        projections=[0.1, 0.2, 0.3],   # all well below default low cutoff (0.5)
        probe_loaded=True,
        refusal_layer=14,
    )
    assert safety.calibrated_pressure == "low"
    assert math.isclose(safety.refusal_projection_max, 0.3)
    assert math.isclose(safety.refusal_projection_mean, 0.2)


def test_moderate_projection_yields_moderate_pressure_label() -> None:
    _, safety, _ = extract_metrics(
        token_log_probs=[_peaked_log_probs()],
        projections=[0.8, 1.0, 1.2],   # max in [low=0.5, moderate=1.5)
        probe_loaded=True,
        refusal_layer=14,
    )
    assert safety.calibrated_pressure == "moderate"


def test_high_projection_yields_high_pressure_label() -> None:
    _, safety, _ = extract_metrics(
        token_log_probs=[_peaked_log_probs()],
        projections=[0.4, 2.5, 0.6],   # max above moderate=1.5
        probe_loaded=True,
        refusal_layer=14,
    )
    assert safety.calibrated_pressure == "high"
    assert math.isclose(safety.refusal_projection_max, 2.5)


def test_label_uses_max_projection_not_mean() -> None:
    # Mean is low (0.5), but one spike pushes max into "high".
    _, safety, _ = extract_metrics(
        token_log_probs=[_peaked_log_probs()],
        projections=[0.1, 0.1, 0.1, 0.1, 1.9],
        probe_loaded=True,
        refusal_layer=14,
    )
    assert safety.calibrated_pressure == "high"


# --- threshold overrides -----------------------------------------------------


def test_custom_pressure_thresholds_override_defaults() -> None:
    custom = PressureThresholds(low=2.0, moderate=5.0)
    _, safety, _ = extract_metrics(
        token_log_probs=[_peaked_log_probs()],
        projections=[1.5],
        probe_loaded=True,
        refusal_layer=14,
        pressure_thresholds=custom,
    )
    # Under the default thresholds 1.5 would be "high"; with custom it's "low".
    assert safety.calibrated_pressure == "low"


def test_custom_spike_threshold_passed_through_to_confidence() -> None:
    # Construct two "tokens" with known entropies via uniform distributions
    # of different vocab sizes.
    high_entropy = np.full(1000, -math.log(1000), dtype=np.float64)  # H = ln(1000) ~ 6.9
    low_entropy = _peaked_log_probs(vocab_size=100)
    confidence, _, _ = extract_metrics(
        token_log_probs=[high_entropy, low_entropy],
        projections=[],
        probe_loaded=False,
        refusal_layer=14,
        spike_threshold=2.0,   # both tokens? high_entropy is way above; low_entropy is ~0
    )
    # Only high_entropy exceeds 2.0
    assert confidence.entropy_spike_count == 1


# --- probe version + layer wired through ------------------------------------


def test_probe_version_surfaced_on_loaded_probe() -> None:
    _, safety, _ = extract_metrics(
        token_log_probs=[_peaked_log_probs()],
        projections=[0.1],
        probe_loaded=True,
        refusal_layer=14,
        probe_version="arditi_v2_custom",
    )
    assert safety.probe_version == "arditi_v2_custom"


def test_refusal_layer_surfaced_on_both_paths() -> None:
    _, loaded, _ = extract_metrics(
        token_log_probs=[_peaked_log_probs()],
        projections=[0.1],
        probe_loaded=True,
        refusal_layer=22,
    )
    _, unloaded, _ = extract_metrics(
        token_log_probs=[_peaked_log_probs()],
        projections=[],
        probe_loaded=False,
        refusal_layer=22,
    )
    assert loaded.layer == 22
    assert unloaded.layer == 22


# --- debug_info shape --------------------------------------------------------


def test_debug_info_contains_raw_series() -> None:
    confidence, _, debug = extract_metrics(
        token_log_probs=[_peaked_log_probs(), _peaked_log_probs(200)],
        projections=[0.5, 1.0],
        probe_loaded=True,
        refusal_layer=14,
    )
    assert set(debug.keys()) == {"raw_entropies", "raw_margins", "raw_projections"}
    assert len(debug["raw_entropies"]) == 2
    assert len(debug["raw_margins"]) == 2
    assert debug["raw_projections"] == [0.5, 1.0]
    # raw entropies / margins should align with the aggregate confidence values.
    assert math.isclose(confidence.mean_entropy, sum(debug["raw_entropies"]) / 2)


def test_empty_inputs_produce_zero_metrics() -> None:
    confidence, safety, debug = extract_metrics(
        token_log_probs=[],
        projections=[],
        probe_loaded=False,
        refusal_layer=14,
    )
    assert confidence.mean_entropy == 0.0
    assert confidence.entropy_spike_count == 0
    assert safety.calibrated_pressure == "uncalibrated"
    assert debug["raw_entropies"] == []
    assert debug["raw_projections"] == []
