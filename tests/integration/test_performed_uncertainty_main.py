# tests/integration/test_performed_uncertainty_main.py
"""Integration test for the performed-uncertainty model-run loop. requires_model.

    pytest -m requires_model tests/integration/test_performed_uncertainty_main.py

Uses the tiny model and a trimmed prompt set: this checks that the two-pass
generation and the report shape work, NOT that the signal is meaningful. The
0.5B model is too weak for the result to mean anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_model

TINY = "Qwen/Qwen2.5-0.5B-Instruct"


def _trim(src: Path, dst: Path, n: int = 3) -> None:
    data = json.loads(src.read_text(encoding="utf-8"))
    data["prompts"] = data["prompts"][:n]
    dst.write_text(json.dumps(data), encoding="utf-8")


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

    assert set(report["summary"]["by_category"]) <= {
        "performed_uncertainty", "binary_settled", "binary_obscure",
    }
