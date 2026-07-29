# tests/integration/test_performed_uncertainty_main.py
"""Integration test for the performed-uncertainty model-run loop. requires_model.

    pytest -m requires_model tests/integration/test_performed_uncertainty_main.py

Uses the tiny model and a trimmed prompt set: this checks that the two-pass
generation and the report shape work, NOT that the signal is meaningful. The
0.5B model is too weak for the result to mean anything.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.requires_model

TINY = "Qwen/Qwen2.5-0.5B-Instruct"


def _trim(src: Path, dst: Path, n: int = 3) -> None:
    data = json.loads(src.read_text(encoding="utf-8"))
    data["prompts"] = data["prompts"][:n]
    dst.write_text(json.dumps(data), encoding="utf-8")


def _write_probe(path: Path, category: str, prompts: list[dict]) -> None:
    path.write_text(json.dumps({"category": category, "prompts": prompts}), encoding="utf-8")


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):  # noqa: ANN001
        # Echo the user content back so the fake generator below can branch on
        # it without needing a real chat template or model.
        return messages[0]["content"]


class _FakeModelState:
    """Stands in for esta.inference.ModelState: no real weights are loaded.

    These tests exercise the input-validation guard added to main() (a
    malformed/empty control-class probe file must fail loudly), not the
    model-run loop itself -- that is already covered by
    test_main_writes_a_report_with_both_generation_passes above. Faking the
    model keeps them fast and independent of the tiny model's actual outputs.
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        self.tokenizer = _FakeTokenizer()

    def load(self) -> None:
        pass


def _install_fake_model(monkeypatch: pytest.MonkeyPatch, generate_fn) -> None:  # noqa: ANN001
    import esta.inference

    monkeypatch.setattr(esta.inference, "ModelState", _FakeModelState)
    monkeypatch.setattr(esta.inference, "generate_with_epistemic_state", generate_fn)


def test_main_writes_a_report_with_both_generation_passes(tmp_path: Path) -> None:
    from esta.scripts.analyze_performed_uncertainty import main, parse_args

    repo = Path(__file__).resolve().parents[2]
    probe_dir = tmp_path / "probe_sets"
    probe_dir.mkdir()
    _trim(repo / "data" / "probe_sets" / "binary_settled.json", probe_dir / "binary_settled.json")
    _trim(repo / "data" / "probe_sets" / "binary_obscure.json", probe_dir / "binary_obscure.json")

    positive = tmp_path / "positive.json"
    _trim(repo / "data" / "validation_cases" / "performed_uncertainty.json", positive)

    out = tmp_path / "report.json"
    main(parse_args([
        "--model", TINY,
        "--positive-set", str(positive),
        "--probe-dir", str(probe_dir),
        "--output", str(out),
        "--free-max-tokens", "32",
    ]))

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["provenance"]["model"] == TINY
    assert set(report) >= {"provenance", "summary", "records"}

    records = report["records"]
    assert len(records) <= 9
    assert records, "expected at least one usable record"
    for r in records:
        assert 0.0 <= r["answer_confidence"] <= 1.0, r["id"]
        assert 0.0 <= r["hedge_score"] <= 1.0, r["id"]
        assert r["signal"] == pytest.approx(r["answer_confidence"] * r["hedge_score"])
        assert r["answer_text"] is not None
        # Every retained record answered yes/no (non-answers are excluded, not recorded).
        assert r["answer_polarity"] in ("yes", "no"), r["id"]
        assert r["answer_correct"] in (True, False, None), r["id"]

    assert set(report["summary"]["by_category"]) <= {
        "performed_uncertainty", "binary_settled", "binary_obscure",
    }


def test_empty_control_class_file_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A control class with no prompts must not be blamed on the model.

    Before this guard, `_load_prompts` silently defaulted to [], and
    `max_margin_threshold` returned None on empty input, so the run printed
    "the control classes do not separate" -- indistinguishable from a genuine
    non-separation result. An empty/malformed probe file is an input error and
    must say so.
    """
    from esta.scripts.analyze_performed_uncertainty import main, parse_args

    def _generate(model_state, prompt, params, refusal_layer, calibration):  # noqa: ANN001
        return SimpleNamespace(
            response_text="Yes.",
            debug_info={"raw_top_logprobs": [math.log(0.97)]},
        )

    _install_fake_model(monkeypatch, _generate)

    probe_dir = tmp_path / "probe_sets"
    probe_dir.mkdir()
    _write_probe(
        probe_dir / "binary_settled.json",
        "binary_settled",
        [{"id": "settled_001", "text": "Is water wet?", "expected_answer": "yes"}],
    )
    _write_probe(probe_dir / "binary_obscure.json", "binary_obscure", [])  # empty on purpose

    positive = tmp_path / "positive.json"
    _write_probe(positive, "performed_uncertainty", [{"id": "p_001", "text": "Is X true?"}])

    with pytest.raises(SystemExit, match=r"binary_obscure.*zero prompts"):
        main(parse_args([
            "--model", TINY,
            "--positive-set", str(positive),
            "--probe-dir", str(probe_dir),
            "--output", str(tmp_path / "report.json"),
        ]))


def test_all_excluded_control_class_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A control class whose every record gets excluded is a distinct failure

    mode from an empty file, and the message must say so: every prompt
    resolved to some non-answer text, not that the file itself was empty.
    """
    from esta.scripts.analyze_performed_uncertainty import CONSTRAINT_INSTRUCTION, main, parse_args

    def _generate(model_state, prompt, params, refusal_layer, calibration):  # noqa: ANN001
        if CONSTRAINT_INSTRUCTION in prompt:
            if "OBSCURE_MARKER" in prompt:
                text = "I don't know."  # never resolves to yes/no -> excluded
            else:
                text = "Yes."
            return SimpleNamespace(
                response_text=text, debug_info={"raw_top_logprobs": [math.log(0.97)]}
            )
        return SimpleNamespace(response_text="A plain, unhedged answer.", debug_info={})

    _install_fake_model(monkeypatch, _generate)

    probe_dir = tmp_path / "probe_sets"
    probe_dir.mkdir()
    _write_probe(
        probe_dir / "binary_settled.json",
        "binary_settled",
        [{"id": "settled_001", "text": "Is water wet?", "expected_answer": "yes"}],
    )
    _write_probe(
        probe_dir / "binary_obscure.json",
        "binary_obscure",
        [{"id": "obscure_001", "text": "Is this OBSCURE_MARKER question resolvable?"}],
    )

    positive = tmp_path / "positive.json"
    _write_probe(positive, "performed_uncertainty", [{"id": "p_001", "text": "Is X true?"}])

    with pytest.raises(SystemExit, match=r"binary_obscure.*zero usable records"):
        main(parse_args([
            "--model", TINY,
            "--positive-set", str(positive),
            "--probe-dir", str(probe_dir),
            "--output", str(tmp_path / "report.json"),
        ]))
