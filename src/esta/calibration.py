"""Load and validate a calibration set produced by `esta.scripts.calibrate`.

Torch-free: imported by the server at startup and by `extract_metrics`, neither
of which should pull the model runtime just to read a small JSON file. The
`Calibration` value object is injected explicitly (no globals).

A calibration is VALID only if its pressure thresholds are separable
(pressure_low < pressure_moderate) and it was computed against the model being
served. A configured-but-invalid calibration is a hard error (fail loud) rather
than a silent fallback: serving uncalibrated while the operator believes
calibration is active is the exact false-assurance failure ESTA exists to avoid.
An ABSENT calibration is a legitimate, honestly-labeled uncalibrated state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from esta.confidence.metrics import DEFAULT_LOW_MARGIN_THRESHOLD, DEFAULT_SPIKE_THRESHOLD
from esta.probes.thresholds import DEFAULT_PRESSURE_THRESHOLDS, PressureThresholds

_REQUIRED_KEYS = ("spike_threshold", "low_margin_threshold", "pressure_low", "pressure_moderate")


class CalibrationError(Exception):
    """A configured calibration file is malformed, inverted, or model-mismatched."""


@dataclass(frozen=True)
class Calibration:
    """Threshold set governing confidence + pressure metrics, plus provenance."""

    spike: float
    low_margin: float
    pressure_low: float
    pressure_moderate: float
    calibrated: bool
    calibration_id: str | None = None
    calibrated_at: str | None = None
    model: str | None = None
    source: str | None = None

    @property
    def pressure_thresholds(self) -> PressureThresholds:
        return PressureThresholds(low=self.pressure_low, moderate=self.pressure_moderate)

    @classmethod
    def uncalibrated(cls) -> Calibration:
        """Placeholder-backed: confidence counts still compute against documented
        default thresholds, but calibrated=False gates the pressure label to
        'uncalibrated' downstream."""
        return cls(
            spike=DEFAULT_SPIKE_THRESHOLD,
            low_margin=DEFAULT_LOW_MARGIN_THRESHOLD,
            pressure_low=DEFAULT_PRESSURE_THRESHOLDS.low,
            pressure_moderate=DEFAULT_PRESSURE_THRESHOLDS.moderate,
            calibrated=False,
        )


def load_calibration(path: Path | None, serving_model: str) -> Calibration:
    """Load + validate a calibration JSON. Returns uncalibrated() if path is None.

    Raises CalibrationError on a configured-but-invalid calibration.
    """
    if path is None:
        return Calibration.uncalibrated()
    if not path.exists():
        raise CalibrationError(f"calibration path configured but not found: {path}")

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"could not read calibration {path}: {exc}") from exc

    missing = [k for k in _REQUIRED_KEYS if k not in data]
    if missing:
        raise CalibrationError(f"calibration {path} missing keys: {missing}")

    pressure_low = float(data["pressure_low"])
    pressure_moderate = float(data["pressure_moderate"])
    if pressure_low >= pressure_moderate:
        raise CalibrationError(
            f"calibration {path} has inverted pressure thresholds "
            f"(pressure_low={pressure_low} >= pressure_moderate={pressure_moderate}); "
            "harmful/harmless projection distributions overlap — recalibrate."
        )

    provenance = data.get("provenance", {})
    calibrated_model = provenance.get("model")
    if calibrated_model is not None and calibrated_model != serving_model:
        raise CalibrationError(
            f"calibration {path} was computed against model {calibrated_model!r} "
            f"but the server is serving {serving_model!r}; recalibrate for this model."
        )

    return Calibration(
        spike=float(data["spike_threshold"]),
        low_margin=float(data["low_margin_threshold"]),
        pressure_low=pressure_low,
        pressure_moderate=pressure_moderate,
        calibrated=True,
        calibration_id=hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12],
        calibrated_at=provenance.get("timestamp"),
        model=calibrated_model,
        source=path.name,
    )
