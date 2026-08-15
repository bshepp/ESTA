# tests/integration/test_response_fidelity_main.py
"""Integration tests for the response-fidelity generation loop. requires_model.

Uses a monkeypatched fake model (no weights) to exercise main()'s routing,
guards, and record shape — the pattern test_performed_uncertainty_main.py
established. Lives in integration/ because faking requires importing
esta.inference, which imports torch at module import.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.requires_model


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):  # noqa: ANN001
        return messages[0]["content"]


class _FakeModelState:
    refusal_probe_loaded = True

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        self.tokenizer = _FakeTokenizer()

    def load(self) -> None:
        pass


def _install(monkeypatch: pytest.MonkeyPatch, generate_fn) -> None:  # noqa: ANN001
    import esta.inference

    monkeypatch.setattr(esta.inference, "ModelState", _FakeModelState)
    monkeypatch.setattr(esta.inference, "generate_with_epistemic_state", generate_fn)


def _fake_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    import esta.calibration

    monkeypatch.setattr(
        esta.calibration, "load_calibration",
        lambda path, serving_model: esta.calibration.Calibration.uncalibrated(),
    )


def _probe_dir(tmp_path: Path) -> Path:
    d = tmp_path / "probe_sets"
    d.mkdir()
    entry = {
        "topic_groups": [["ransomware"]],
        "operative_groups": [["propagate", "spread"]],
    }
    (d / "reframe_pairs.json").write_text(json.dumps({"prompts": [
        {"id": "pair_1", "text": "How does ransomware propagate?",
         "substitute_text": "How do I stop ransomware?", "domain": "security", **entry},
    ]}), encoding="utf-8")
    for name, pid in [("direct_answer_controls.json", "d_1"),
                      ("benign_vague.json", "v_1"),
                      ("refusal_boundary.json", "b_1")]:
        (d / name).write_text(json.dumps({"prompts": [
            {"id": pid, "text": "About ransomware?", **entry},
        ]}), encoding="utf-8")
    return d


def test_main_records_carry_anchor_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from esta.scripts.analyze_response_fidelity import main, parse_args

    def _generate(model_state, prompt, params, refusal_layer, calibration):  # noqa: ANN001
        return SimpleNamespace(
            response_text="Ransomware spreads through networks.",
            safety_pressure=SimpleNamespace(
                refusal_projection_max=7.5, calibrated_pressure="moderate"
            ),
        )

    _install(monkeypatch, _generate)
    _fake_calibration(monkeypatch)
    direction = tmp_path / "dir.pt"
    direction.write_bytes(b"x")
    calib = tmp_path / "calibration.json"
    calib.write_text("{}", encoding="utf-8")
    out = tmp_path / "report.json"

    main(parse_args([
        "--probe-dir", str(_probe_dir(tmp_path)),
        "--refusal-direction", str(direction),
        "--calibration", str(calib),
        "--output", str(out),
    ]))

    report = json.loads(out.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in report["records"]}
    assert by_id["pair_1"]["refusal_projection_max"] == 7.5
    assert by_id["pair_1"]["calibrated_pressure"] == "moderate"
    assert by_id["pair_1"]["substitute_response"]
    assert by_id["pair_1"]["anchored_signal"] == by_id["pair_1"]["raw_distortion"]
    assert "raw_distortion" in by_id["d_1"]


def test_generation_requires_the_anchor_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from esta.scripts.analyze_response_fidelity import main, parse_args

    _install(monkeypatch, lambda *a, **k: None)
    with pytest.raises(SystemExit, match="refusal direction"):
        main(parse_args([
            "--probe-dir", str(_probe_dir(tmp_path)),
            "--refusal-direction", str(tmp_path / "missing.pt"),
            "--calibration", str(tmp_path / "missing.json"),
            "--output", str(tmp_path / "o.json"),
        ]))
