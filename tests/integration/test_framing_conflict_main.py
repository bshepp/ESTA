"""Integration test for framing-conflict main() wiring. requires_model (stubbed)."""
from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.requires_model


def _rec(rid, category, p_a, p_b, perts, topic="israel-palestine"):
    return {"id": rid, "category": category, "topic": topic, "text": "q", "response": perts[0],
            "p_a_series": p_a, "p_b_series": p_b, "perturbation_responses": perts}


def test_main_scores_and_writes_report(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    import esta.scripts.analyze_framing_conflict as mod
    from esta.scripts.analyze_framing_conflict import (
        CLASS_NEUTRAL,
        CLASS_ONE_A,
        CLASS_ONE_B,
        CLASS_TWO_SIDED,
        main,
        parse_args,
    )

    records = (
        [_rec(f"ts{i}", CLASS_TWO_SIDED, [2.0, 2.0], [2.0, 2.0], ["alpha beta", "gamma delta"])
         for i in range(4)]
        + [_rec(f"oa{i}", CLASS_ONE_A, [2.0, 2.0], [0.1, 0.1], ["alpha beta", "alpha beta"])
           for i in range(6)]
        + [_rec(f"ob{i}", CLASS_ONE_B, [0.1, 0.1], [2.0, 2.0], ["alpha beta", "alpha beta"])
           for i in range(6)]
        + [_rec(f"nu{i}", CLASS_NEUTRAL, [0.1, 0.1], [0.1, 0.1], ["alpha beta", "alpha beta"],
                topic="energy") for i in range(6)]
    )
    monkeypatch.setattr(mod, "_generate_records",
                        lambda args: (records, [], {"model": "fake", "topic": "israel-palestine",
                                                    "window": 2}))

    out = tmp_path / "report.json"
    main(parse_args(["--output", str(out)]))
    report = json.loads(out.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in report["records"]}
    # two-sided co-activates (both axes lit), one-sided does not
    assert by_id["ts0"]["coactivation_any"] is True
    assert by_id["oa0"]["coactivation_any"] is False
    # two-sided perturbations diverge (disjoint content), one-sided stable
    assert by_id["ts0"]["instability"] == pytest.approx(1.0)
    assert by_id["oa0"]["instability"] == pytest.approx(0.0)
    assert report["summary"]["israel_palestine"]           # broken out by name
    assert report["summary"]["theta_a"] is not None        # controls separated -> cutoff placed
