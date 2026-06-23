"""Tests for the torch-free calibration loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from esta.calibration import Calibration, CalibrationError, load_calibration

SERVING_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _write(tmp_path: Path, payload: dict, name: str = "calibration.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _valid_payload(model: str = SERVING_MODEL) -> dict:
    return {
        "spike_threshold": 5.0,
        "low_margin_threshold": 0.3,
        "pressure_low": 0.4,
        "pressure_moderate": 1.2,
        "provenance": {"model": model, "timestamp": "2026-06-22T00:00:00Z"},
    }


def test_none_path_returns_uncalibrated() -> None:
    cal = load_calibration(None, SERVING_MODEL)
    assert cal.calibrated is False
    assert cal.calibration_id is None


def test_uncalibrated_uses_default_thresholds() -> None:
    from esta.confidence.metrics import DEFAULT_LOW_MARGIN_THRESHOLD, DEFAULT_SPIKE_THRESHOLD
    from esta.probes.thresholds import DEFAULT_PRESSURE_THRESHOLDS

    cal = Calibration.uncalibrated()
    assert cal.spike == DEFAULT_SPIKE_THRESHOLD
    assert cal.low_margin == DEFAULT_LOW_MARGIN_THRESHOLD
    assert cal.pressure_thresholds == DEFAULT_PRESSURE_THRESHOLDS


def test_valid_calibration_loads(tmp_path: Path) -> None:
    cal = load_calibration(_write(tmp_path, _valid_payload()), SERVING_MODEL)
    assert cal.calibrated is True
    assert cal.spike == 5.0
    assert cal.low_margin == 0.3
    assert cal.pressure_thresholds.low == 0.4
    assert cal.pressure_thresholds.moderate == 1.2
    assert cal.model == SERVING_MODEL
    assert cal.calibrated_at == "2026-06-22T00:00:00Z"
    assert cal.source == "calibration.json"
    assert cal.calibration_id and len(cal.calibration_id) == 12


def test_calibration_id_stable_for_identical_content(tmp_path: Path) -> None:
    a = load_calibration(_write(tmp_path, _valid_payload(), "a.json"), SERVING_MODEL)
    b = load_calibration(_write(tmp_path, _valid_payload(), "b.json"), SERVING_MODEL)
    assert a.calibration_id == b.calibration_id


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(CalibrationError, match="not found"):
        load_calibration(tmp_path / "nope.json", SERVING_MODEL)


def test_malformed_json_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(CalibrationError):
        load_calibration(p, SERVING_MODEL)


def test_missing_keys_raises(tmp_path: Path) -> None:
    payload = _valid_payload()
    del payload["pressure_moderate"]
    with pytest.raises(CalibrationError, match="missing keys"):
        load_calibration(_write(tmp_path, payload), SERVING_MODEL)


def test_inverted_pressure_thresholds_raise(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["pressure_low"] = 1.5
    payload["pressure_moderate"] = 1.0
    with pytest.raises(CalibrationError, match="inverted"):
        load_calibration(_write(tmp_path, payload), SERVING_MODEL)


def test_model_mismatch_raises(tmp_path: Path) -> None:
    payload = _valid_payload(model="some/other-model")
    with pytest.raises(CalibrationError, match="model"):
        load_calibration(_write(tmp_path, payload), SERVING_MODEL)


def test_missing_provenance_model_is_allowed(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["provenance"] = {"timestamp": "2026-06-22T00:00:00Z"}
    cal = load_calibration(_write(tmp_path, payload), SERVING_MODEL)
    assert cal.calibrated is True
    assert cal.model is None
