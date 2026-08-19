"""Tests for the torch-free layer of the conflict-state analysis."""

from __future__ import annotations

import pytest

from esta.scripts.analyze_conflict_state import (
    CLASS_ANALYTICAL,
    CLASS_CONSTRAINT,
    CLASS_RECALL,
    build_report,
    derive_theta_eng,
    score_records,
)


def test_theta_eng_separates_recall_from_analytical() -> None:
    recall = [0.1, 0.2, 0.15] * 8
    analytical = [0.8, 0.9, 0.85] * 8
    cut = derive_theta_eng(recall, analytical)
    assert cut is not None
    assert 0.2 < cut.cutoff < 0.8


def test_theta_eng_none_when_classes_overlap() -> None:
    same = [0.1, 0.5, 0.9] * 8
    assert derive_theta_eng(same, list(same)) is None


def _rec(rid, category, p_ref, p_eng):
    return {"id": rid, "category": category, "text": "q", "response": "r",
            "p_ref_series": p_ref, "p_eng_series": p_eng}


def test_score_records_adds_aggregates_using_both_thresholds() -> None:
    # one token both-lit (ref 2.0, eng 2.0), one token eng-cold
    records = [_rec("c1", CLASS_CONSTRAINT, [2.0, 2.0], [2.0, 0.1])]
    score_records(records, theta_ref=1.0, theta_eng=1.0)
    assert records[0]["conflict_events"] == 1
    assert records[0]["max_conflict_score"] == pytest.approx(2.0)


def test_build_report_summarizes_by_category_and_flags_israel_palestine() -> None:
    records = (
        [_rec(f"c{i}", CLASS_CONSTRAINT, [2.0], [2.0]) for i in range(3)]
        + [_rec(f"a{i}", CLASS_ANALYTICAL, [0.1], [2.0]) for i in range(3)]
        + [_rec(f"r{i}", CLASS_RECALL, [0.1], [0.1]) for i in range(3)]
    )
    records[0]["topic"] = "israel-palestine"
    score_records(records, theta_ref=1.0, theta_eng=1.0)
    report = build_report(records, excluded=[], provenance={"model": "m"},
                          theta_ref=1.0, theta_eng_cut=None)
    by_cat = report["summary"]["by_category"]
    assert by_cat[CLASS_CONSTRAINT]["mean_conflict_events"] == pytest.approx(1.0)
    assert by_cat[CLASS_RECALL]["mean_conflict_events"] == pytest.approx(0.0)
    # constraint fires, recall does not -> the intended contrast
    assert by_cat[CLASS_CONSTRAINT]["mean_max_score"] > by_cat[CLASS_RECALL]["mean_max_score"]


def _prior(rid, category, p_ref, p_eng, topic="neutral"):
    return {"id": rid, "category": category, "text": "q", "response": "r",
            "topic": topic, "p_ref_series": p_ref, "p_eng_series": p_eng,
            # stale fields rescore must overwrite:
            "conflict_events": 999, "max_conflict_score": 999.0}


def _write_prior(path, records, theta_ref=1.0):
    import json as _json
    path.write_text(_json.dumps({
        "provenance": {"model": "test-model", "theta_ref": theta_ref},
        "summary": {"excluded": [], "theta_ref": theta_ref}, "records": records,
    }), encoding="utf-8")


def test_rescore_recomputes_without_torch(tmp_path) -> None:  # noqa: ANN001
    import json as _json
    import sys

    from esta.scripts.analyze_conflict_state import main, parse_args

    records = (
        [_prior(f"c{i}", CLASS_CONSTRAINT, [2.0, 2.0], [2.0, 2.0], "israel-palestine") for i in range(3)]
        + [_prior(f"an{i}", CLASS_ANALYTICAL, [0.1], [2.0]) for i in range(20)]
        + [_prior(f"re{i}", CLASS_RECALL, [0.1], [0.1]) for i in range(20)]
    )
    prior = tmp_path / "prior.json"
    _write_prior(prior, records)
    out = tmp_path / "out.json"
    main(parse_args(["--rescore", str(prior), "--output", str(out)]))
    assert "torch" not in sys.modules
    report = _json.loads(out.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in report["records"]}
    assert by_id["c0"]["conflict_events"] == 2      # recomputed, not 999
    assert report["summary"]["israel_palestine"]    # broken out


def test_rescore_refuses_series_missing(tmp_path) -> None:  # noqa: ANN001
    from esta.scripts.analyze_conflict_state import main, parse_args
    rec = _prior("c0", CLASS_CONSTRAINT, [1.0], [1.0])
    del rec["p_eng_series"]
    prior = tmp_path / "p.json"
    _write_prior(prior, [rec])
    with pytest.raises(SystemExit, match="p_eng_series"):
        main(parse_args(["--rescore", str(prior), "--output", str(tmp_path / "o.json")]))


def test_print_report_ascii_safe_with_theta_eng_none(capsys) -> None:  # noqa: ANN001
    """Regression test: print_report must not crash with UnicodeEncodeError on Windows console (cp1252)."""
    from pathlib import Path

    from esta.scripts.analyze_conflict_state import print_report

    # Build a minimal report with theta_eng=None (triggers the NOTE line with θ_eng)
    records = [
        _prior(f"r{i}", CLASS_RECALL, [0.1], [0.1]) for i in range(3)
    ] + [
        _prior(f"a{i}", CLASS_ANALYTICAL, [0.1], [2.0]) for i in range(3)
    ]
    report = build_report(records, excluded=[], provenance={"model": "test"},
                          theta_ref=1.0, theta_eng_cut=None)

    # Call print_report and verify output is ASCII-encodable (Windows console constraint)
    output_path = Path("dummy.json")
    print_report(report, output_path)

    out = capsys.readouterr().out
    # This is the Windows cp1252 constraint: if the string contains θ, it will raise UnicodeEncodeError
    out.encode("cp1252")  # Must not raise


def test_print_report_ascii_safe_with_theta_eng_placed(capsys) -> None:  # noqa: ANN001
    """Regression test: print_report must handle placed theta_eng with ASCII-safe output."""
    from pathlib import Path

    from esta.scripts.analyze_conflict_state import print_report

    # Build a minimal report with theta_eng placed (triggers the θ_ref/θ_eng line)
    records = [
        _prior(f"r{i}", CLASS_RECALL, [0.1], [0.1]) for i in range(10)
    ] + [
        _prior(f"a{i}", CLASS_ANALYTICAL, [0.1], [2.0]) for i in range(10)
    ]
    score_records(records, theta_ref=1.0, theta_eng=1.5)
    theta_eng_cut = derive_theta_eng(
        [max(r["p_eng_series"]) for r in records if r["category"] == CLASS_RECALL],
        [max(r["p_eng_series"]) for r in records if r["category"] == CLASS_ANALYTICAL]
    )
    report = build_report(records, excluded=[], provenance={"model": "test"},
                          theta_ref=1.0, theta_eng_cut=theta_eng_cut)

    # Call print_report and verify output is ASCII-encodable
    output_path = Path("dummy.json")
    print_report(report, output_path)

    out = capsys.readouterr().out
    out.encode("cp1252")  # Must not raise
