"""Integration test for conflict-state main() wiring. requires_model."""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.requires_model


def _series_record(rid, category, p_ref, p_eng, topic="neutral"):
    return {"id": rid, "category": category, "text": "q", "response": "r",
            "topic": topic, "p_ref_series": p_ref, "p_eng_series": p_eng}


def test_main_scores_and_writes_report(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    import esta.scripts.analyze_conflict_state as mod
    from esta.scripts.analyze_conflict_state import (
        CLASS_ANALYTICAL,
        CLASS_CONSTRAINT,
        CLASS_RECALL,
        main,
        parse_args,
    )

    records = (
        [_series_record(f"c{i}", CLASS_CONSTRAINT, [2.0, 2.0], [2.0, 2.0], "israel-palestine") for i in range(3)]
        + [_series_record(f"a{i}", CLASS_ANALYTICAL, [0.1], [2.0]) for i in range(20)]
        + [_series_record(f"r{i}", CLASS_RECALL, [0.1], [0.1]) for i in range(20)]
    )
    monkeypatch.setattr(mod, "_generate_records",
                        lambda args: (records, [], {"model": "fake", "theta_ref": 1.0}, 1.0))

    out = tmp_path / "report.json"
    main(parse_args(["--output", str(out)]))
    report = json.loads(out.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in report["records"]}
    # constraint fires (both axes lit on every token), recall does not
    assert by_id["c0"]["conflict_events"] == 2
    assert by_id["r0"]["conflict_events"] == 0
    assert report["summary"]["israel_palestine"]           # broken out by name
    assert report["summary"]["theta_eng"] is not None       # controls separated -> cutoff placed
