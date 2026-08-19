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
