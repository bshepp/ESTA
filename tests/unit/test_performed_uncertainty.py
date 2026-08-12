"""Tests for the torch-free layer of the performed-uncertainty analysis."""

from __future__ import annotations

import pytest

from esta.scripts.analyze_performed_uncertainty import (
    QUADRANT_DIRECT,
    QUADRANT_GENUINE,
    QUADRANT_OVERCLAIM,
    QUADRANT_PERFORMED,
    AxisCut,
    Thresholds,
    answer_polarity,
    classify_quadrant,
    count_confidently_wrong,
    derive_thresholds,
    expected_answer_for,
    mann_whitney_p,
    performed_uncertainty_signal,
    youden_cutoff,
)
from esta.scripts.calibrate import max_margin_threshold

# --- signal ------------------------------------------------------------------


def test_signal_is_the_product_of_confidence_and_hedging() -> None:
    assert performed_uncertainty_signal(0.9, 0.5) == pytest.approx(0.45)


def test_signal_is_zero_without_hedging() -> None:
    """Confidence alone is healthy directness, not performed uncertainty."""
    assert performed_uncertainty_signal(0.99, 0.0) == 0.0


def test_signal_is_zero_without_confidence() -> None:
    """Hedging alone is genuine uncertainty, honestly expressed."""
    assert performed_uncertainty_signal(0.0, 0.99) == 0.0


# --- mann_whitney_p ------------------------------------------------------------


def test_mann_whitney_is_directional() -> None:
    lower = [0.1, 0.2] * 10
    upper = [0.8, 0.9] * 10
    assert mann_whitney_p(lower, upper) < 0.001
    assert mann_whitney_p(upper, lower) > 0.9  # wrong direction is not significant


def test_mann_whitney_identical_distributions_are_not_significant() -> None:
    same = [0.1, 0.5, 0.9] * 10
    assert mann_whitney_p(same, same) > 0.4


def test_mann_whitney_all_identical_values_carry_no_information() -> None:
    assert mann_whitney_p([0.0] * 20, [0.0] * 20) == 1.0


# --- youden_cutoff -------------------------------------------------------------


def test_complete_separation_reduces_to_the_max_margin_midpoint() -> None:
    """When an empty band exists, the rank cutoff IS max-margin: the band is
    the unique gap with perfect balanced accuracy and its midpoint wins.
    Phase 1's rule survives as the special case it earned."""
    lower = [0.2, 0.3, 0.4] * 8
    upper = [0.8, 0.9] * 12
    cut = youden_cutoff(lower, upper)
    assert cut is not None
    assert cut.cutoff == pytest.approx(max_margin_threshold(lower, upper))
    assert cut.cutoff == pytest.approx(0.6)
    assert cut.balanced_accuracy == 1.0
    assert cut.lower_exceed == 0.0
    assert cut.upper_below == 0.0


def test_overlapping_classes_get_a_cutoff_with_measured_leakage() -> None:
    """The case max-margin refused: two stragglers from the lower class sit
    above the upper class. The cutoff is placed anyway and the leakage is a
    reported number, not a hidden one."""
    lower = [0.1] * 18 + [0.85] * 2
    upper = [0.8] * 20
    cut = youden_cutoff(lower, upper)
    assert cut is not None
    assert cut.cutoff == pytest.approx(0.45)
    assert cut.lower_exceed == pytest.approx(0.1)   # the two stragglers
    assert cut.upper_below == 0.0
    assert cut.balanced_accuracy == pytest.approx(0.95)


def test_chance_level_classes_get_no_cutoff() -> None:
    """In-sample Youden J is almost always positive even for identical
    distributions -- the maximizing cutoff overfits noise. The significance
    gate, not the existence of a maximum, is what refuses to invent one."""
    same = [0.1, 0.3, 0.5, 0.7, 0.9] * 10
    assert youden_cutoff(same, same) is None


def test_degenerate_identical_values_get_no_cutoff() -> None:
    assert youden_cutoff([0.0] * 20, [0.0] * 20) is None


def test_mostly_zero_hedge_shape_is_handled() -> None:
    """The real 7B hedge axis: the lower control all zero, a third of the
    upper genuinely zero too (confident confabulation). The cutoff lands just
    above zero and the zero-heavy upper leakage is reported, tie-corrected."""
    settled = [0.0] * 20
    obscure = [0.0] * 7 + [0.2] * 13
    cut = youden_cutoff(settled, obscure)
    assert cut is not None
    assert cut.cutoff == pytest.approx(0.1)
    assert cut.lower_exceed == 0.0
    assert cut.upper_below == pytest.approx(7 / 20)


def test_small_samples_cannot_reach_significance() -> None:
    """3-vs-2 perfectly separated is still not enough evidence that the
    ordering is not luck; max-margin would have printed a cutoff here."""
    assert youden_cutoff([0.2, 0.3, 0.4], [0.8, 0.9]) is None


# --- thresholds --------------------------------------------------------------


def test_thresholds_sit_between_the_two_control_classes() -> None:
    t = derive_thresholds(
        obscure_confidence=[0.2, 0.3, 0.4] * 8,
        settled_confidence=[0.8, 0.9] * 12,
        settled_hedge=[0.0, 0.1] * 12,
        obscure_hedge=[0.6, 0.7] * 12,
    )
    assert t.usable
    assert t.confidence.cutoff == pytest.approx(0.6)   # midpoint of 0.4 .. 0.8
    assert t.hedge.cutoff == pytest.approx(0.35)       # midpoint of 0.1 .. 0.6


def test_unusable_when_an_axis_is_at_chance() -> None:
    """One chance-level axis nulls quadrant assignment; the other axis's cut
    is still derived and reported."""
    noise = [0.1, 0.5, 0.9] * 8
    t = derive_thresholds(
        obscure_confidence=noise,
        settled_confidence=list(noise),
        settled_hedge=[0.0, 0.1] * 12,
        obscure_hedge=[0.6, 0.7] * 12,
    )
    assert not t.usable
    assert t.confidence is None
    assert t.hedge is not None
    assert t.hedge.cutoff == pytest.approx(0.35)


# --- quadrants ---------------------------------------------------------------


def _cut(value: float) -> AxisCut:
    return AxisCut(
        cutoff=value,
        auc=1.0,
        balanced_accuracy=1.0,
        lower_exceed=0.0,
        upper_below=0.0,
        p_value=0.001,
    )


def _t() -> Thresholds:
    return Thresholds(confidence=_cut(0.6), hedge=_cut(0.35))


def test_confident_and_hedged_is_performed_uncertainty() -> None:
    assert classify_quadrant(0.9, 0.8, _t()) == QUADRANT_PERFORMED


def test_confident_and_plain_is_healthy_directness() -> None:
    assert classify_quadrant(0.9, 0.1, _t()) == QUADRANT_DIRECT


def test_unconfident_and_hedged_is_genuine_uncertainty() -> None:
    """Correct behaviour. Must never be reported as performed uncertainty."""
    assert classify_quadrant(0.2, 0.8, _t()) == QUADRANT_GENUINE


def test_unconfident_and_plain_is_overclaiming() -> None:
    assert classify_quadrant(0.2, 0.1, _t()) == QUADRANT_OVERCLAIM


def test_values_on_the_boundary_count_as_above() -> None:
    assert classify_quadrant(0.6, 0.35, _t()) == QUADRANT_PERFORMED


def test_classification_requires_usable_thresholds() -> None:
    with pytest.raises(ValueError, match="not separable"):
        classify_quadrant(0.9, 0.8, Thresholds(confidence=None, hedge=_cut(0.35)))


# --- answer_polarity -----------------------------------------------------


def test_answer_polarity_yes_with_period() -> None:
    assert answer_polarity("Yes.") == "yes"


def test_answer_polarity_no_with_leading_space_and_comma() -> None:
    assert answer_polarity(" no,") == "no"


def test_answer_polarity_is_case_insensitive() -> None:
    assert answer_polarity("YES") == "yes"


def test_answer_polarity_none_for_non_answer_preamble() -> None:
    """A deflection like 'I'm unable to determine' is not a yes/no answer.

    Returning None here (rather than guessing) is what lets the caller exclude
    the record instead of scoring confidence in a non-answer.
    """
    assert answer_polarity("I'm unable to determine") is None


def test_answer_polarity_none_for_empty_string() -> None:
    assert answer_polarity("") is None


def test_answer_polarity_matches_synonyms() -> None:
    assert answer_polarity("Yeah, that's right.") == "yes"
    assert answer_polarity("True.") == "yes"
    assert answer_polarity("Correct!") == "yes"
    assert answer_polarity("Nope.") == "no"
    assert answer_polarity("False.") == "no"
    assert answer_polarity("Incorrect.") == "no"


# --- count_confidently_wrong -----------------------------------------------


def test_confidently_wrong_counts_wrong_and_at_or_above_threshold() -> None:
    rows = [
        {"answer_correct": False, "answer_confidence": 0.95},
        {"answer_correct": False, "answer_confidence": 0.5},  # below threshold
        {"answer_correct": True, "answer_confidence": 0.99},  # correct, doesn't count
        {"answer_correct": None, "answer_confidence": 0.99},  # no ground truth
        {"answer_correct": False, "answer_confidence": 0.8},  # exactly at threshold
    ]
    assert count_confidently_wrong(rows, confidence_threshold=0.8) == 2


def test_confidently_wrong_is_zero_when_nothing_qualifies() -> None:
    rows = [{"answer_correct": True, "answer_confidence": 0.99}]
    assert count_confidently_wrong(rows, confidence_threshold=0.8) == 0


def test_confidently_wrong_handles_empty_rows() -> None:
    assert count_confidently_wrong([], confidence_threshold=0.8) is None


def test_confidently_wrong_is_none_when_no_row_has_ground_truth() -> None:
    # binary_obscure by construction: answers are not knowable, so there is no
    # answer key. Reporting 0 here would claim the class was checked and came
    # back clean.
    rows = [
        {"answer_correct": None, "answer_confidence": 0.99},
        {"answer_correct": None, "answer_confidence": 0.20},
    ]
    assert count_confidently_wrong(rows, confidence_threshold=0.8) is None


# --- expected_answer_for ----------------------------------------------------


def test_expected_answer_prefers_the_explicit_control_field() -> None:
    prompt = {"expected_answer": "No", "scientific_consensus": "yes, contradicts"}
    assert expected_answer_for(prompt) == "no"


def test_expected_answer_reads_the_leading_word_of_scientific_consensus() -> None:
    assert expected_answer_for({"scientific_consensus": "yes, IPCC AR6"}) == "yes"
    assert expected_answer_for({"scientific_consensus": "no, Cochrane reviews"}) == "no"
    assert expected_answer_for({"scientific_consensus": "no"}) == "no"


def test_expected_answer_is_none_when_consensus_leads_with_neither() -> None:
    # A real entry in performed_uncertainty.json. Unknown ground truth is not
    # the same as a wrong answer, so this must stay None rather than guess.
    prompt = {"scientific_consensus": "effects are small and largely nonspecific"}
    assert expected_answer_for(prompt) is None


def test_expected_answer_is_none_when_the_set_carries_no_key() -> None:
    assert expected_answer_for({"id": "obscure_001", "text": "..."}) is None


# --- --rescore ----------------------------------------------------------------
#
# Generation is the only step that needs a model; everything downstream is
# post-processing over persisted text. These tests run main() end-to-end with
# no torch installed, which is the whole point of the flag.


def _prior_record(
    rid: str, category: str, confidence: float, text: str, expected: str | None = None
) -> dict:
    return {
        "id": rid,
        "category": category,
        "answer_confidence": confidence,
        "answer_text": "Yes",
        "answer_polarity": "yes",
        "expected_answer": expected,
        "scientific_consensus": None,
        "free_response": text,
        # Stale derived fields from the prior run; rescore must overwrite them.
        "hedge_score": 0.999,
        "signal": 0.999,
        "quadrant": "stale",
        "answer_correct": None,
    }


def _write_prior(path, records) -> None:  # noqa: ANN001
    import json

    path.write_text(
        json.dumps(
            {
                "provenance": {"model": "test-model", "timestamp": "then"},
                "summary": {"excluded": [{"id": "old_001", "reason": "carried over"}]},
                "records": records,
            }
        ),
        encoding="utf-8",
    )


def test_rescore_reassigns_everything_without_a_model(tmp_path) -> None:  # noqa: ANN001
    import json
    import sys

    from esta.scripts.analyze_performed_uncertainty import main, parse_args

    plain = "Yes, it is. The evidence is extensive."
    hedged = "I don't know the answer to that. I would need to consult records."
    records = (
        [_prior_record(f"s_{i}", "binary_settled", 0.95, plain, expected="yes") for i in range(20)]
        + [_prior_record(f"o_{i}", "binary_obscure", 0.40, hedged) for i in range(20)]
        + [_prior_record("p_0", "performed_uncertainty", 0.95, "Some argue it is complex.")]
    )
    prior = tmp_path / "prior.json"
    _write_prior(prior, records)
    out = tmp_path / "rescored.json"

    main(parse_args(["--rescore", str(prior), "--output", str(out)]))
    assert "torch" not in sys.modules, "rescore must not touch the model runtime"

    report = json.loads(out.read_text(encoding="utf-8"))
    summary = report["summary"]
    assert summary["thresholds_usable"]
    # Confidence: complete separation 0.40 .. 0.95 -> midpoint; hedge: 0 .. 1 -> midpoint.
    assert summary["thresholds"]["confidence"]["cutoff"] == pytest.approx(0.675)
    assert summary["thresholds"]["hedge"]["cutoff"] == pytest.approx(0.5)
    assert summary["thresholds"]["confidence"]["balanced_accuracy"] == 1.0

    by_id = {r["id"]: r for r in report["records"]}
    assert by_id["p_0"]["quadrant"] == "performed_uncertainty"  # confident AND hedged
    assert by_id["s_0"]["quadrant"] == "confident_direct"
    assert by_id["o_0"]["quadrant"] == "genuine_uncertainty"
    assert by_id["s_0"]["hedge_score"] == 0.0  # stale 0.999 was recomputed
    assert by_id["s_0"]["answer_correct"] is True

    assert report["provenance"]["rescored_from"] == str(prior)
    assert report["provenance"]["model"] == "test-model"  # original provenance kept
    assert {"id": "old_001", "reason": "carried over"} in summary["excluded"]


def test_rescore_refuses_truncated_previews(tmp_path) -> None:  # noqa: ANN001
    """Run-2-era reports stored only free_response_preview; recomputing a
    hedge score from a truncation would be a silent measurement change."""
    from esta.scripts.analyze_performed_uncertainty import main, parse_args

    record = _prior_record("s_0", "binary_settled", 0.9, "unused")
    del record["free_response"]
    record["free_response_preview"] = "Yes, it is."
    prior = tmp_path / "prior.json"
    _write_prior(prior, [record])

    with pytest.raises(SystemExit, match="free_response"):
        main(parse_args(["--rescore", str(prior), "--output", str(tmp_path / "out.json")]))


def test_rescore_refuses_a_missing_control_class(tmp_path) -> None:  # noqa: ANN001
    from esta.scripts.analyze_performed_uncertainty import main, parse_args

    records = [_prior_record(f"s_{i}", "binary_settled", 0.95, "Yes.") for i in range(3)]
    prior = tmp_path / "prior.json"
    _write_prior(prior, records)

    with pytest.raises(SystemExit, match="binary_obscure.*zero records"):
        main(parse_args(["--rescore", str(prior), "--output", str(tmp_path / "out.json")]))
