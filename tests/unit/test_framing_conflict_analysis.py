"""Tests for the torch-free layer of the framing-conflict analysis."""
from __future__ import annotations

import pytest

from esta.scripts.analyze_framing_conflict import (
    CLASS_NEUTRAL,
    CLASS_ONE_A,
    CLASS_ONE_B,
    CLASS_TWO_SIDED,
    _paraphrases,
    _perturbation_prompts,
    build_report,
    mean_pairwise_divergence,
    response_divergence,
    score_records,
)


def test_response_divergence_is_one_minus_convergence() -> None:
    # identical texts -> convergence 1.0 -> divergence 0.0
    assert response_divergence("alpha beta gamma", "alpha beta gamma") == pytest.approx(0.0)
    # disjoint content words -> convergence 0.0 -> divergence 1.0
    assert response_divergence("alpha beta", "gamma delta") == pytest.approx(1.0)


def test_response_divergence_none_when_undefined() -> None:
    assert response_divergence("", "alpha") is None


def test_mean_pairwise_divergence_skips_undefined_pairs() -> None:
    # three responses; the empty one makes 2 of 3 pairs undefined, leaving 1 pair.
    assert mean_pairwise_divergence(["alpha beta", "gamma delta", ""]) == pytest.approx(1.0)


def test_mean_pairwise_divergence_none_when_under_two_usable() -> None:
    assert mean_pairwise_divergence(["alpha", ""]) is None
    assert mean_pairwise_divergence(["alpha"]) is None


def _rec(rid, category, p_a, p_b, perts, topic="israel-palestine"):
    return {"id": rid, "category": category, "topic": topic, "text": "q",
            "response": " ".join(perts[:1]) if perts else "r",
            "p_a_series": p_a, "p_b_series": p_b, "perturbation_responses": perts}


def test_score_records_adds_all_three_signals() -> None:
    # both axes lit at same token -> co-activation event; perts diverge -> instability 1.0
    records = [_rec("t1", CLASS_TWO_SIDED, [2.0, 2.0], [2.0, 2.0],
                    ["alpha beta", "gamma delta"])]
    score_records(records, theta_a=1.0, theta_b=1.0, window=2)
    r = records[0]
    assert r["coactivation_any"] is True
    assert r["oscillation"] == 0.0                 # steady both-lit, no swing
    assert r["instability"] == pytest.approx(1.0)  # disjoint perturbation responses


def test_score_records_marks_missing_theta() -> None:
    records = [_rec("t1", CLASS_TWO_SIDED, [2.0], [2.0], ["a b", "c d"])]
    score_records(records, theta_a=None, theta_b=1.0, window=2)
    assert records[0]["coactivation_any"] is None   # no theta_a -> not scored


def test_build_report_summarizes_and_flags_israel_palestine() -> None:
    records = (
        [_rec(f"ts{i}", CLASS_TWO_SIDED, [2.0, 2.0], [2.0, 2.0], ["a b", "c d"]) for i in range(3)]
        + [_rec(f"oa{i}", CLASS_ONE_A, [2.0], [0.1], ["a b", "a b"]) for i in range(3)]
        + [_rec(f"ob{i}", CLASS_ONE_B, [0.1], [2.0], ["a b", "a b"]) for i in range(3)]
        + [_rec(f"nu{i}", CLASS_NEUTRAL, [0.1], [0.1], ["a b", "a b"], topic="energy")
           for i in range(3)]
    )
    score_records(records, theta_a=1.0, theta_b=1.0, window=2)
    report = build_report(records, excluded=[], provenance={"model": "m"},
                          theta_a_cut=None, theta_b_cut=None, instability_null=0.5, window=2)
    assert report["summary"]["window"] == 2
    by_cat = report["summary"]["by_category"]
    assert by_cat[CLASS_TWO_SIDED]["coactivation_rate"] == pytest.approx(1.0)
    assert by_cat[CLASS_ONE_A]["coactivation_rate"] == pytest.approx(0.0)
    # the control contrast: two-sided co-activates, one-sided does not
    assert by_cat[CLASS_TWO_SIDED]["mean_instability"] > by_cat[CLASS_ONE_A]["mean_instability"]
    assert report["summary"]["israel_palestine"]  # broken out by name

    # unstable flag: two-sided (disjoint perturbations, instability 1.0) exceeds
    # the null (0.5); one-sided/neutral (identical perturbations, 0.0) do not.
    by_id = {r["id"]: r for r in report["records"]}
    assert all(by_id[f"ts{i}"]["unstable"] is True for i in range(3))
    assert all(by_id[f"oa{i}"]["unstable"] is False for i in range(3))
    assert all(by_id[f"ob{i}"]["unstable"] is False for i in range(3))
    assert all(by_id[f"nu{i}"]["unstable"] is False for i in range(3))
    assert by_cat[CLASS_TWO_SIDED]["unstable_rate"] == pytest.approx(1.0)
    assert by_cat[CLASS_ONE_A]["unstable_rate"] == pytest.approx(0.0)

    # internal-vs-instability association: coactivation_max separates cleanly
    # (two-sided co-activates, everyone else barely clears theta on a single
    # token); oscillation is undefined (None) on every stable record here
    # (only one engaged token each), so that field must report None rather
    # than a bogus stat over an empty sample.
    ivi = report["summary"]["internal_vs_instability"]
    assert set(ivi.keys()) == {"coactivation_max", "oscillation"}
    coact_stat = ivi["coactivation_max"]
    assert coact_stat["n_unstable"] == 3
    assert coact_stat["n_stable"] == 9
    assert coact_stat["unstable_mean"] > coact_stat["stable_mean"]
    assert 0.0 <= coact_stat["mann_whitney_p"] <= 1.0
    assert ivi["oscillation"] is None


def test_build_report_records_the_window() -> None:
    # window must be recorded verbatim, not hardcoded to DEFAULT_WINDOW.
    records = [_rec("t1", CLASS_NEUTRAL, [0.1], [0.1], ["a b", "a b"], topic="energy")]
    report = build_report(records, excluded=[], provenance={"model": "m"},
                          theta_a_cut=None, theta_b_cut=None, instability_null=None, window=5)
    assert report["summary"]["window"] == 5
    # no instability null -> unstable is undefined, not falsely False
    assert report["records"][0]["unstable"] is None
    assert report["summary"]["internal_vs_instability"] == {"coactivation_max": None, "oscillation": None}


def _prior(rid, category, p_a, p_b, perts, topic="israel-palestine"):
    return {"id": rid, "category": category, "topic": topic, "text": "q", "response": "r",
            "p_a_series": p_a, "p_b_series": p_b, "perturbation_responses": perts,
            "coactivation_any": None, "coactivation_max": 999.0}  # stale, must be overwritten


def _write_prior(path, records):
    import json as _json
    path.write_text(_json.dumps({"provenance": {"model": "test-model"},
                                 "summary": {"excluded": []}, "records": records}), encoding="utf-8")


def test_rescore_runs_without_torch(tmp_path) -> None:  # noqa: ANN001
    import json as _json
    import sys

    from esta.scripts.analyze_framing_conflict import main, parse_args

    records = (
        [_prior(f"ts{i}", CLASS_TWO_SIDED, [2.0, 2.0], [2.0, 2.0], ["a b", "c d"]) for i in range(4)]
        + [_prior(f"oa{i}", CLASS_ONE_A, [2.0, 2.0], [0.1, 0.1], ["a b", "a b"]) for i in range(6)]
        + [_prior(f"ob{i}", CLASS_ONE_B, [0.1, 0.1], [2.0, 2.0], ["a b", "a b"]) for i in range(6)]
        + [_prior(f"nu{i}", CLASS_NEUTRAL, [0.1, 0.1], [0.1, 0.1], ["a b", "a b"], "energy")
           for i in range(6)]
    )
    prior = tmp_path / "prior.json"
    _write_prior(prior, records)
    out = tmp_path / "out.json"
    main(parse_args(["--rescore", str(prior), "--output", str(out)]))
    assert "torch" not in sys.modules
    report = _json.loads(out.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in report["records"]}
    assert by_id["ts0"]["coactivation_max"] != 999.0        # recomputed
    assert report["summary"]["israel_palestine"]


def test_rescore_refuses_corpus_missing_perturbations(tmp_path) -> None:  # noqa: ANN001
    from esta.scripts.analyze_framing_conflict import main, parse_args
    rec = _prior("ts0", CLASS_TWO_SIDED, [1.0], [1.0], ["a b", "c d"])
    del rec["perturbation_responses"]
    prior = tmp_path / "p.json"
    _write_prior(prior, [rec])
    with pytest.raises(SystemExit, match="perturbation_responses"):
        main(parse_args(["--rescore", str(prior), "--output", str(tmp_path / "o.json")]))


def test_perturbation_prompts_uses_authored_paraphrases_when_present() -> None:
    prompt = {"text": "base question", "paraphrases": ["p1", "p2", "p3", "p4"]}
    # used verbatim, truncated to k -- not run through the programmatic fallback
    assert _perturbation_prompts(prompt, 2) == ["p1", "p2"]


def test_perturbation_prompts_falls_back_when_absent() -> None:
    prompt = {"text": "base question"}
    result = _perturbation_prompts(prompt, 3)
    assert len(result) == 3
    assert result == _paraphrases("base question", 3)


def test_perturbation_prompts_falls_back_when_paraphrases_empty() -> None:
    # an empty authored list is not a usable override -- fall back rather than
    # perturbing with zero prompts.
    prompt = {"text": "base question", "paraphrases": []}
    assert _perturbation_prompts(prompt, 2) == _paraphrases("base question", 2)
