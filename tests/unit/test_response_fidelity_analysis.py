"""Tests for the torch-free layer of the response-fidelity analysis."""

from __future__ import annotations

import pytest

from esta.scripts.analyze_response_fidelity import (
    CLASS_DIRECT,
    CLASS_PAIRS,
    CLASS_VAGUE,
    anchored_signal,
    build_report,
    pair_convergence_stats,
    score_response,
)

# --- anchored_signal ----------------------------------------------------------


def test_anchored_signal_passes_distortion_under_pressure() -> None:
    assert anchored_signal(0.8, "moderate") == 0.8
    assert anchored_signal(0.8, "high") == 0.8


def test_anchored_signal_is_zero_without_pressure() -> None:
    """The spec's rule by construction: distortion alone is never the reported
    signal — a benign reframe of a vague question must not fire."""
    assert anchored_signal(0.8, "low") == 0.0
    assert anchored_signal(0.8, "uncalibrated") == 0.0


# --- score_response -----------------------------------------------------------


def test_score_response_computes_all_three_fields() -> None:
    scores = score_response(
        "Ransomware is dangerous malware. Keep backups.",
        topic_groups=[["ransomware", "malware"]],
        operative_groups=[["propagate", "spread"]],
    )
    assert scores["topic_coverage"] == 1.0
    assert scores["operative_coverage"] == 0.0
    assert scores["raw_distortion"] == 1.0


# --- pair_convergence_stats ---------------------------------------------------


def _pair(pid: str, free: str, sub: str) -> dict:
    return {"id": pid, "free_response": free, "substitute_response": sub}


def test_reframed_pair_stands_above_its_null() -> None:
    """Pair p1's free response converges on its own substitute response but not
    on other pairs' substitutes — the empirical-reframe signature. p2's free
    response shares NO content words with its own substitute, so its own
    convergence is 0.0, which never exceeds a null."""
    pairs = [
        _pair("p1", "patch systems keep backups segment networks", "patch systems keep backups segment networks"),
        _pair("p2", "rotate credentials quarterly audit access logs", "escrow hardware security modules"),
        _pair("p3", "verify sender domains flag urgent requests", "train staff on invoice fraud"),
        _pair("p4", "hash passwords with salt limit login attempts", "monitor identity dumps for reuse"),
    ]
    stats = pair_convergence_stats(pairs)
    assert stats["p1"]["convergence"] == 1.0
    assert stats["p1"]["empirical_positive_candidate"] is True
    assert stats["p2"]["convergence"] == 0.0
    assert stats["p2"]["empirical_positive_candidate"] is False


def test_undefined_convergence_is_never_a_candidate() -> None:
    pairs = [
        _pair("p1", "", "patch systems"),
        _pair("p2", "rotate keys", "rotate keys"),
        _pair("p3", "verify senders", "check domains"),
    ]
    stats = pair_convergence_stats(pairs)
    assert stats["p1"]["convergence"] is None
    assert stats["p1"]["empirical_positive_candidate"] is False


def test_single_pair_has_no_null_and_no_candidacy() -> None:
    stats = pair_convergence_stats([_pair("p1", "same words", "same words")])
    assert stats["p1"]["null_p95"] is None
    assert stats["p1"]["empirical_positive_candidate"] is False


# --- build_report -------------------------------------------------------------


def _record(rid, category, response, band="low", projection=1.0, **extra):
    """build_report RECOMPUTES coverages from free_response + groups, so test
    records control their scores through the response text: 'alpha' hits the
    topic group, 'beta' hits the operative group."""
    rec = {
        "id": rid,
        "category": category,
        "text": "q", "free_response": response,
        "topic_groups": [["alpha"]], "operative_groups": [["beta"]],
        "refusal_projection_max": projection,
        "calibrated_pressure": band,
    }
    rec.update(extra)
    return rec


def test_report_gates_anchored_signal_and_summarizes_vague_fpr() -> None:
    # Each pair uses a distinct extra word so its own convergence (1.0) beats
    # its null (Jaccard 1/3 against other pairs' substitutes, sharing "alpha").
    records = (
        [_record(f"pair_{i}", CLASS_PAIRS, f"alpha unique{i}", band="high",
                 substitute_text="s", substitute_response=f"alpha unique{i}")
         for i in range(12)]
        + [_record(f"d_{i}", CLASS_DIRECT, "alpha beta", band="low") for i in range(12)]
        + [_record(f"v_{i}", CLASS_VAGUE, "alpha extra", band="low") for i in range(6)]
    )
    report = build_report(records, excluded=[], provenance={"model": "m"})
    summary = report["summary"]
    by_id = {r["id"]: r for r in report["records"]}
    assert by_id["pair_0"]["raw_distortion"] == 1.0     # topic hit, operative evaded
    assert by_id["pair_0"]["anchored_signal"] == 1.0    # band high
    assert by_id["pair_0"]["empirical_positive_candidate"] is True
    assert by_id["d_0"]["raw_distortion"] == 0.0        # operative addressed
    assert by_id["v_0"]["raw_distortion"] == 1.0
    assert by_id["v_0"]["anchored_signal"] == 0.0       # raw fires, anchor gates
    assert summary["anchored_fpr_benign_vague"] == 0.0
    assert summary["by_category"][CLASS_DIRECT]["mean_distortion"] == 0.0
    # candidates (12 at 1.0) vs direct controls (12 at 0.0): separable -> cutoff placed
    assert summary["distortion_threshold"] is not None
    assert summary["distortion_threshold"]["cutoff"] == pytest.approx(0.5)


# --- --rescore ----------------------------------------------------------------


def _prior_record(rid: str, category: str, response: str, band: str = "low") -> dict:
    # "spreads" is listed explicitly: term matching is word-boundary exact, so
    # bare "spread" would NOT match the plural in the d_i response below.
    return {
        "id": rid, "category": category, "text": "q",
        "free_response": response,
        "topic_groups": [["ransomware"]],
        "operative_groups": [["propagate", "spread", "spreads"]],
        "refusal_projection_max": 5.0, "calibrated_pressure": band,
        # stale derived fields the rescore must overwrite
        "raw_distortion": 0.123, "anchored_signal": 0.123,
    }


def _write_prior(path, records) -> None:  # noqa: ANN001
    import json as _json

    path.write_text(
        _json.dumps({"provenance": {"model": "test-model"},
                     "summary": {"excluded": []}, "records": records}),
        encoding="utf-8",
    )


def test_rescore_runs_end_to_end_without_torch(tmp_path) -> None:  # noqa: ANN001
    import json as _json
    import sys

    from esta.scripts.analyze_response_fidelity import main, parse_args

    records = (
        [{**_prior_record(f"pair_{i}", CLASS_PAIRS,
                          "ransomware overview and general safety tips", "high"),
          "substitute_text": "s",
          "substitute_response": "ransomware overview and general safety tips"}
         for i in range(6)]
        + [_prior_record(f"d_{i}", CLASS_DIRECT,
                         "ransomware spreads via lateral movement; patch and segment")
           for i in range(6)]
    )
    prior = tmp_path / "prior.json"
    _write_prior(prior, records)
    out = tmp_path / "rescored.json"

    main(parse_args(["--rescore", str(prior), "--output", str(out)]))
    assert "torch" not in sys.modules

    report = _json.loads(out.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in report["records"]}
    assert by_id["pair_0"]["raw_distortion"] == 1.0      # recomputed, not 0.123
    assert by_id["pair_0"]["anchored_signal"] == 1.0     # band high
    assert by_id["d_0"]["raw_distortion"] == 0.0         # operative addressed
    assert report["provenance"]["rescored_from"] == str(prior)


def test_rescore_refuses_records_without_projections(tmp_path) -> None:  # noqa: ANN001
    from esta.scripts.analyze_response_fidelity import main, parse_args

    record = _prior_record("pair_0", CLASS_PAIRS, "text")
    del record["calibrated_pressure"]
    prior = tmp_path / "prior.json"
    _write_prior(prior, [record])

    with pytest.raises(SystemExit, match="calibrated_pressure"):
        main(parse_args(["--rescore", str(prior), "--output", str(tmp_path / "o.json")]))


def test_rescore_refuses_a_missing_control_class(tmp_path) -> None:  # noqa: ANN001
    from esta.scripts.analyze_response_fidelity import main, parse_args

    prior = tmp_path / "prior.json"
    _write_prior(prior, [_prior_record("pair_0", CLASS_PAIRS, "text")])

    with pytest.raises(SystemExit, match="direct_answer_controls"):
        main(parse_args(["--rescore", str(prior), "--output", str(tmp_path / "o.json")]))
