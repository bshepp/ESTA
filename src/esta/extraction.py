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

from esta.calibration import Calibration
from esta.confidence.metrics import aggregate_confidence, token_entropy_and_margin
from esta.probes.thresholds import DEFAULT_PROBE_VERSION, label_pressure
from esta.schema import CalibrationInfo, ConfidenceMetrics, SafetyPressure


def extract_metrics(
    *,
    token_log_probs: list[np.ndarray],
    projections: list[float],
    probe_loaded: bool,
    refusal_layer: int,
    calibration: Calibration,
    probe_version: str = DEFAULT_PROBE_VERSION,
) -> tuple[ConfidenceMetrics, SafetyPressure, CalibrationInfo, dict[str, Any]]:
    """Build ConfidenceMetrics + SafetyPressure + CalibrationInfo + a debug dict.

    The honesty rule: a real pressure label (low/moderate/high) is emitted only
    when the probe is loaded AND the calibration is calibrated. Otherwise the
    label is 'uncalibrated', even though the raw projection magnitude is still
    surfaced when the probe is loaded.

    Parameters
    ----------
    token_log_probs : list of (vocab,) numpy arrays, one per generated token.
        Already log-softmaxed (i.e., exp() sums to 1).
    projections : list of refusal-direction projection magnitudes, one per
        captured activation. Empty when the probe is not loaded.
    probe_loaded : whether the refusal direction tensor was loaded at startup.
        Determines whether raw projections are available.
    refusal_layer : layer index used during refusal-direction extraction.
        Surfaced in SafetyPressure so consumers can audit the probe config.
    calibration : Calibration value object governing thresholds + provenance.
        Use Calibration.uncalibrated() when no calibration file is configured.
    probe_version : optional version string for the refusal probe.
        Resolves to ``"not_loaded"`` when the probe is absent; otherwise the
        passed value is forwarded as-is into `SafetyPressure.probe_version`.

    Returns
    -------
    (confidence, safety_pressure, calibration_info, debug_info)
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
        spike_threshold=calibration.spike,
        low_margin_threshold=calibration.low_margin,
    )

    have_projection = probe_loaded and bool(projections)
    if have_projection:
        proj_arr = np.asarray(projections, dtype=np.float64)
        proj_max = float(np.max(proj_arr))
        proj_mean = float(np.mean(proj_arr))
    else:
        proj_max = 0.0
        proj_mean = 0.0

    if have_projection and calibration.calibrated:
        pressure_label = label_pressure(proj_max, calibration.pressure_thresholds)
        resolved_probe_version = probe_version
    else:
        pressure_label = "uncalibrated"
        resolved_probe_version = probe_version if probe_loaded else "not_loaded"

    safety = SafetyPressure(
        refusal_projection_max=proj_max,
        refusal_projection_mean=proj_mean,
        calibrated_pressure=pressure_label,
        probe_version=resolved_probe_version,
        layer=refusal_layer,
    )

    calibration_info = CalibrationInfo(
        calibrated=calibration.calibrated,
        calibration_id=calibration.calibration_id,
        calibrated_at=calibration.calibrated_at,
        model=calibration.model,
        source=calibration.source,
    )

    debug_info: dict[str, Any] = {
        "raw_entropies": entropies,
        "raw_margins": margins,
        "raw_projections": list(projections),
    }
    return confidence, safety, calibration_info, debug_info
