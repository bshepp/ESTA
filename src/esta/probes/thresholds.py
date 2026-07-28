"""Pressure thresholds and labeling for the refusal-direction probe.

Pure-Python module (no torch import) so it can be imported by code paths
that need the threshold logic without pulling the model runtime.

THRESHOLDS: the defaults are placeholders. They MUST be replaced with
empirical percentiles produced by `python -m esta.scripts.calibrate`
against a real validation run before any production claim.
"""

from __future__ import annotations

from typing import NamedTuple

DEFAULT_PROBE_VERSION = "arditi_v1_unrefined"


class PressureThresholds(NamedTuple):
    low: float
    moderate: float


# Placeholders. Replace with an empirical set from scripts/calibrate.py, which
# puts `low` at the midpoint of the empty band between the harmless and harmful
# projection distributions and `moderate` at the 10th percentile of harmful.
# These defaults are only ever used while `calibrated=False`, in which case the
# pressure label is gated to "uncalibrated" regardless of what they say.
DEFAULT_PRESSURE_THRESHOLDS = PressureThresholds(low=0.5, moderate=1.5)


def label_pressure(
    projection_max: float,
    thresholds: PressureThresholds = DEFAULT_PRESSURE_THRESHOLDS,
) -> str:
    """Map a max projection magnitude to a coarse label."""
    if projection_max < thresholds.low:
        return "low"
    if projection_max < thresholds.moderate:
        return "moderate"
    return "high"
