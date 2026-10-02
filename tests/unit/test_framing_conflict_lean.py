"""Tests for the torch-free LEAN-geometry layer of the framing-conflict analysis (v1b.1)."""
from __future__ import annotations

import json
import sys

import pytest

from esta.scripts.analyze_framing_conflict import (
    CLASS_NEUTRAL,
    CLASS_ONE_A,
    CLASS_ONE_B,
    CLASS_TWO_SIDED,
    GEOMETRY_LEAN,
    build_report_lean,
    main,
    parse_args,
    score_records_lean,
)


def _pert(kind, p_topic, p_lean):
    return {"kind": kind, "text": "q'", "response": "r'", "p_topic_series": p_topic, "p_lean_series": p_lean}


def _rec(rid, category, p_topic, p_lean, perts=(), topic="israel-palestine"):
    return {"id": rid, "category": category, "topic": topic, "text": "q", "response": "r",
            "p_topic_series": p_topic, "p_lean_series": p_lean, "perturbations": list(perts)}


def test_parse_args_defaults_to_lean_geometry() -> None:
    assert parse_args([]).geometry == GEOMETRY_LEAN
    assert parse_args(["--geometry", "narratives"]).geometry == "narratives"


def test_score_records_lean_sets_internal_signals_and_ground_truth() -> None:
    # Base: engaged, balanced (|lean| < theta) -> balance 1.0, no committed flips.
    # Swap pair flips sign (+ vs -) -> swap_flip True -> torn.
    rec = _rec("ts0", CLASS_TWO_SIDED, [2.0, 2.0], [0.5, -0.5], perts=[
        _pert("swap_a_first", [2.0, 2.0], [3.0, 3.0]),
        _pert("swap_b_first", [2.0, 2.0], [-3.0, -3.0]),
        _pert("paraphrase", [2.0, 2.0], [0.4, 0.4]),
    ])
    score_records_lean([rec], theta_topic=1.0, theta_lean=2.0)
    assert rec["n_engaged"] == 2
    assert rec["balance"] == pytest.approx(1.0)
    assert rec["mean_lean"] == pytest.approx(0.0)
    assert rec["swap_flip"] is True
    assert rec["lean_shift"] == pytest.approx(1.5)      # |(+1.5) - 0| is the largest move
    assert rec["torn"] is True


def test_score_records_lean_committed_stable_record_is_not_torn() -> None:
    rec = _rec("oa0", CLASS_ONE_A, [2.0, 2.0], [4.0, 4.0], perts=[
        _pert("paraphrase", [2.0, 2.0], [3.6, 4.4]),
    ])
    score_records_lean([rec], theta_topic=1.0, theta_lean=2.0)
    assert rec["balance"] == pytest.approx(0.0)
    assert rec["swap_flip"] is None                      # no swap pair on one_sided
    assert rec["lean_shift"] == pytest.approx(0.0)       # paraphrase mean lean 2.0 == base 2.0
    assert rec["torn"] is False


def test_score_records_lean_with_missing_theta_leaves_signals_unscored() -> None:
    rec = _rec("ts0", CLASS_TWO_SIDED, [2.0], [1.0], perts=[_pert("paraphrase", [2.0], [1.0])])
    score_records_lean([rec], theta_topic=None, theta_lean=2.0)
    assert rec["balance"] is None and rec["torn"] is None and rec["lean_shift"] is None


def _four_class_set():
    ts = [_rec(f"ts{i}", CLASS_TWO_SIDED, [2.0, 2.0], [0.5, -0.5], perts=[
        _pert("swap_a_first", [2.0, 2.0], [3.0, 3.0]), _pert("swap_b_first", [2.0, 2.0], [-3.0, -3.0])])
          for i in range(4)]
    oa = [_rec(f"oa{i}", CLASS_ONE_A, [2.0, 2.0], [4.0, 4.0], perts=[_pert("paraphrase", [2.0, 2.0], [4.0, 4.0])])
          for i in range(8)]
    ob = [_rec(f"ob{i}", CLASS_ONE_B, [2.0, 2.0], [-4.0, -4.0], perts=[_pert("paraphrase", [2.0, 2.0], [-4.0, -4.0])])
          for i in range(8)]
    nu = [_rec(f"nu{i}", CLASS_NEUTRAL, [0.1, 0.1], [0.1, 0.1], topic="energy") for i in range(8)]
    return ts + oa + ob + nu


def test_build_report_lean_summarizes_torn_rate_and_association() -> None:
    records = _four_class_set()
    score_records_lean(records, theta_topic=1.0, theta_lean=2.0)
    report = build_report_lean(records, excluded=[], provenance={"model": "m"},
                               theta_topic_cut=None, theta_lean_cut=None)
    by_cat = report["summary"]["by_category"]
    assert by_cat[CLASS_TWO_SIDED]["torn_rate"] == pytest.approx(1.0)
    assert by_cat[CLASS_ONE_A]["torn_rate"] == pytest.approx(0.0)
    assert by_cat[CLASS_TWO_SIDED]["mean_balance"] > by_cat[CLASS_ONE_A]["mean_balance"]
    assert by_cat[CLASS_NEUTRAL]["mean_n_engaged"] == pytest.approx(0.0)
    ivt = report["summary"]["internal_vs_torn"]["balance"]
    assert ivt["torn_mean"] > ivt["stable_mean"]
    assert report["summary"]["israel_palestine"]


def _write_prior(path, records, geometry=GEOMETRY_LEAN):
    path.write_text(json.dumps({"provenance": {"model": "test-model", "geometry": geometry},
                                "summary": {"excluded": []}, "records": records}), encoding="utf-8")


def test_rescore_lean_runs_end_to_end_without_torch(tmp_path) -> None:  # noqa: ANN001
    prior = tmp_path / "prior.json"
    _write_prior(prior, _four_class_set())
    out = tmp_path / "out.json"
    main(parse_args(["--rescore", str(prior), "--output", str(out)]))
    assert "torch" not in sys.modules
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["provenance"]["geometry"] == GEOMETRY_LEAN
    s = report["summary"]
    assert s["theta_topic"] is not None and s["theta_lean"] is not None   # controls separate
    by_id = {r["id"]: r for r in report["records"]}
    assert by_id["ts0"]["torn"] is True
    assert by_id["oa0"]["torn"] is False


def test_rescore_lean_refuses_a_corpus_missing_series(tmp_path) -> None:  # noqa: ANN001
    records = _four_class_set()
    del records[0]["p_lean_series"]
    prior = tmp_path / "prior.json"
    _write_prior(prior, records)
    with pytest.raises(SystemExit, match="p_lean_series"):
        main(parse_args(["--rescore", str(prior), "--output", str(tmp_path / "o.json")]))


def test_theta_lean_is_derived_from_engaged_tokens_only() -> None:
    """The lean is only defined while the model is ON the topic. Off-topic neutral prompts
    project arbitrary noise onto the lean axis, so a theta_lean built from |lean| over ALL
    tokens cannot separate one_sided from neutral (the 7B IP lean check: theta_lean None).
    Restrict the |lean| peak to engaged tokens (s_topic >= theta_topic): neutral then has no
    engaged tokens (peak 0) and one_sided separates cleanly."""
    from esta.scripts.analyze_framing_conflict import _thetas_lean_from_controls

    # one_sided: engaged (topic 2.0) with committed lean 4.0
    # neutral: NOT engaged (topic 0.1) but with LARGE spurious |lean| 9.0 over all tokens
    recs = (
        [_rec(f"oa{i}", CLASS_ONE_A, [2.0, 2.0], [4.0, 4.0]) for i in range(8)]
        + [_rec(f"ob{i}", CLASS_ONE_B, [2.0, 2.0], [-4.0, -4.0]) for i in range(8)]
        + [_rec(f"nu{i}", CLASS_NEUTRAL, [0.1, 0.1], [9.0, -9.0], topic="energy") for i in range(8)]
    )
    t_cut, l_cut, offset = _thetas_lean_from_controls(recs)
    assert t_cut is not None and t_cut.cutoff > 0
    assert l_cut is not None, "the one_sided classes separate on the engaged lean axis"
    assert offset == pytest.approx(0.0)                 # +4 and -4 are symmetric about zero
    assert l_cut.cutoff == pytest.approx(4.0)           # half the A-B gap: one_sided sits at +-1
    assert l_cut.p_value < 0.05 and l_cut.auc == pytest.approx(1.0)


def test_lean_midpoint_is_calibrated_from_the_one_sided_controls() -> None:
    """On 7B the raw lean axis separated the sides by MAGNITUDE, not sign (engaged mean-lean
    +4.77 for one_sided_a but still +0.63 for one_sided_b): the axis's zero, set by prompt
    activations, is not 'between the sides' at generation time. Calibrate the midpoint from the
    controls -- offset = mean of the two one-sided classes' engaged lean, two_sided never used --
    so that A is positive and B negative by construction, and 'balanced' means near that midpoint."""
    from esta.scripts.analyze_framing_conflict import _thetas_lean_from_controls

    recs = (
        [_rec(f"oa{i}", CLASS_ONE_A, [2.0, 2.0], [4.0, 4.0]) for i in range(8)]     # raw lean +4
        + [_rec(f"ob{i}", CLASS_ONE_B, [2.0, 2.0], [1.0, 1.0]) for i in range(8)]   # raw lean +1 (!)
        + [_rec(f"nu{i}", CLASS_NEUTRAL, [0.1, 0.1], [9.0, -9.0], topic="energy") for i in range(8)]
    )
    t_cut, l_cut, offset = _thetas_lean_from_controls(recs)
    assert offset == pytest.approx(2.5)                       # midpoint of +4 and +1
    assert l_cut is not None and l_cut.cutoff > 0
    score_records_lean(recs, theta_topic=t_cut.cutoff, theta_lean=l_cut.cutoff, lean_offset=offset)
    by_id = {r["id"]: r for r in recs}
    assert by_id["oa0"]["mean_lean"] > 0                        # A-leaning after centering
    assert by_id["ob0"]["mean_lean"] < 0                        # B-leaning after centering
    assert by_id["oa0"]["mean_lean"] == pytest.approx(-by_id["ob0"]["mean_lean"])  # symmetric
