"""Assemble the structured epistemic_state metrics from already-extracted arrays.

This is the pure-Python boundary between the torch-dependent generation path
(in `esta.inference.generation`) and the schema layer. By taking numpy arrays
and Python floats as input (not torch tensors), this function can be unit-tested
without torch installed — which is what CI uses.

The torch wrapper is responsible for:
  - log-softmaxing per-step scores from `outputs.scores` and converting to numpy
  - projecting captured activations onto the refusal direction (project_activations)
  - calling this function with the resulting numbers
"""

from __future__ import annotations

from typing import Any

import numpy as np

from esta.confidence.metrics import (
    DEFAULT_LOW_MARGIN_THRESHOLD,
    DEFAULT_SPIKE_THRESHOLD,
    aggregate_confidence,
    token_entropy_and_margin,
)
from esta.probes.thresholds import (
    DEFAULT_PRESSURE_THRESHOLDS,
    DEFAULT_PROBE_VERSION,
    PressureThresholds,
    label_pressure,
)
from esta.schema import ConfidenceMetrics, SafetyPressure


def extract_metrics(
    *,
    token_log_probs: list[np.ndarray],
    projections: list[float],
    probe_loaded: bool,
    refusal_layer: int,
    probe_version: str = DEFAULT_PROBE_VERSION,
    pressure_thresholds: PressureThresholds = DEFAULT_PRESSURE_THRESHOLDS,
    spike_threshold: float = DEFAULT_SPIKE_THRESHOLD,
    low_margin_threshold: float = DEFAULT_LOW_MARGIN_THRESHOLD,
) -> tuple[ConfidenceMetrics, SafetyPressure, dict[str, Any]]:
    """Build ConfidenceMetrics + SafetyPressure + a debug dict.

    Parameters
    ----------
    token_log_probs : list of (vocab,) numpy arrays, one per generated token.
        Already log-softmaxed (i.e., exp() sums to 1).
    projections : list of refusal-direction projection magnitudes, one per
        captured activation. Empty when the probe is not loaded.
    probe_loaded : whether the refusal direction tensor was loaded at startup.
        Determines whether to return calibrated SafetyPressure or the
        uncalibrated stub.
    refusal_layer : layer index used during refusal-direction extraction.
        Surfaced in SafetyPressure so consumers can audit the probe config.

    Returns
    -------
    (confidence, safety_pressure, debug_info)
        debug_info contains the raw per-token series for downstream review
        when the request opts in via `return_activations=true`.
    """
    entropies: list[float] = []
    margins: list[float] = []
    for lp in token_log_probs:
        e, m = token_entropy_and_margin(lp)
        entropies.append(e)
        margins.append(m)

    confidence = aggregate_confidence(
        entropies,
        margins,
        spike_threshold=spike_threshold,
        low_margin_threshold=low_margin_threshold,
    )

    if probe_loaded and projections:
        proj_arr = np.asarray(projections, dtype=np.float64)
        proj_max = float(np.max(proj_arr))
        safety = SafetyPressure(
            refusal_projection_max=proj_max,
            refusal_projection_mean=float(np.mean(proj_arr)),
            calibrated_pressure=label_pressure(proj_max, pressure_thresholds),
            probe_version=probe_version,
            layer=refusal_layer,
        )
    else:
        safety = SafetyPressure(
            refusal_projection_max=0.0,
            refusal_projection_mean=0.0,
            calibrated_pressure="uncalibrated",
            probe_version="not_loaded",
            layer=refusal_layer,
        )

    debug_info: dict[str, Any] = {
        "raw_entropies": entropies,
        "raw_margins": margins,
        "raw_projections": list(projections),
    }
    return confidence, safety, debug_info
