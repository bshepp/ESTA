"""Tests for the torch-free lexical hedge measure."""

from __future__ import annotations

import pytest

from esta.hedging import HEDGE_MARKERS, hedge_score, split_sentences

# --- split_sentences ---------------------------------------------------------


def test_splits_on_terminal_punctuation() -> None:
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_ignores_trailing_whitespace_and_empty_fragments() -> None:
    assert split_sentences("Only one.   ") == ["Only one."]


def test_text_without_terminal_punctuation_is_one_sentence() -> None:
    assert split_sentences("no full stop here") == ["no full stop here"]


def test_empty_text_has_no_sentences() -> None:
    assert split_sentences("") == []
    assert split_sentences("    ") == []


# --- hedge_score -------------------------------------------------------------


def test_unhedged_answer_scores_zero() -> None:
    text = "Yes. Vaccines reduce transmission and severity of measles."
    assert hedge_score(text) == 0.0


def test_fully_hedged_answer_scores_one() -> None:
    text = "Some argue this is true. Then again, it depends."
    assert hedge_score(text) == 1.0


def test_partially_hedged_answer_scores_the_fraction() -> None:
    text = "Yes, the evidence is clear. That said, some argue otherwise."
    assert hedge_score(text) == pytest.approx(0.5)


def test_score_is_independent_of_length() -> None:
    """Two hedged sentences out of four scores the same as one out of two."""
    short = "Some argue X. It is settled."
    long = "Some argue X. It is settled. Some argue Y. It is settled."
    assert hedge_score(short) == hedge_score(long) == pytest.approx(0.5)


def test_matching_is_case_insensitive() -> None:
    assert hedge_score("SOME ARGUE this.") == 1.0


def test_empty_response_is_undefined_not_zero() -> None:
    """Zero would claim 'did not hedge'; there is nothing to measure."""
    assert hedge_score("") is None
    assert hedge_score("   ") is None


def test_bare_modals_are_not_markers() -> None:
    """'may' and 'might' are far too common in ordinary prose to discriminate."""
    for word in ("may", "might", "could", "however", "generally"):
        assert not any(word == marker for marker in HEDGE_MARKERS)
    assert hedge_score("Water may freeze at zero degrees.") == 0.0


def test_lexical_matcher_cannot_distinguish_mention_from_use() -> None:
    """Documents a known limitation rather than pretending it does not exist.

    A purely lexical matcher fires on text ABOUT hedging. This is pinned so the
    behaviour is a recorded trade-off; if it ever matters, the fix is the LLM
    classifier the spec defers, not a longer regex.
    """
    text = "The phrase 'some argue' is a discourse marker."
    assert hedge_score(text) == 1.0


# --- first-person epistemic markers ------------------------------------------


def test_i_dont_have_specific_records_is_a_hedge() -> None:
    """The motivating case: a model declining an obscure factual question.

    Without a first-person marker this scored 0.0, which -- one such zero
    among fifty obscure prompts -- collapses max(settled_hedge) <
    min(obscure_hedge) and makes the hedge threshold return None.
    """
    text = "I don't have specific records for that."
    assert hedge_score(text) == 1.0


def test_i_am_not_sure_is_a_hedge() -> None:
    assert hedge_score("I am not sure about that.") == 1.0


def test_i_cannot_determine_is_a_hedge() -> None:
    assert hedge_score("I cannot determine the answer from available information.") == 1.0


def test_not_well_documented_is_a_hedge() -> None:
    assert hedge_score("This is not well documented.") == 1.0


def test_unclear_whether_is_a_hedge() -> None:
    assert hedge_score("It is unclear whether this occurred.") == 1.0


# --- epistemic deferral (rebuilt against the 7B control corpus, 2026-08-12) --
#
# The first marker list sat at chance (AUC 0.563) against binary_obscure
# because it encoded hedging as discourse moves while the model hedges obscure
# factual questions as deferral. Each phrasing below is drawn from an actual
# control response the old list scored 0.0.


def test_subjunctive_deferral_is_a_hedge() -> None:
    """'We WOULD need to consult X' marks inability to answer."""
    text = "To answer this accurately, I would need to look up specific records from 1962."
    assert hedge_score(text) == 1.0
    text = "We would need to refer to historical demographic data for that period."
    assert hedge_score(text) == 1.0


def test_indicative_procedure_is_not_a_hedge() -> None:
    """'To determine X, we need to look at Y' followed by a flat answer is
    assertive throat-clearing, not deferral -- the control corpus uses it to
    preface confident (often confabulated) answers. Only the subjunctive form
    tracks uncertainty."""
    text = "To determine this, we need to look at historical records. The cornerstone was laid in 1849."
    assert hedge_score(text) == 0.0


def test_second_person_instructional_need_is_not_a_hedge() -> None:
    """'You would need a visa' is instruction, not epistemic deferral; the
    would-need markers are deliberately anchored to first person."""
    assert hedge_score("To enter the country you would need a visa.") == 0.0


def test_unavailable_information_is_a_hedge() -> None:
    assert hedge_score("Specific details about the committee are not publicly available.") == 1.0
    assert hedge_score("Detailed statistics are not readily available in public databases.") == 1.0


def test_knowledge_cutoff_deferral_is_a_hedge() -> None:
    text = "As of my last update, the 2025 prize has not been announced."
    assert hedge_score(text) == 1.0


def test_no_defensible_determination_is_a_hedge() -> None:
    assert hedge_score("It's not possible to definitively state whether this occurred.") == 1.0
    assert hedge_score("Without specific historical records, we cannot definitively say.") == 1.0


def test_deferring_to_an_external_source_is_a_hedge() -> None:
    assert hedge_score("I would recommend checking the official IMU website.") == 1.0


def test_curly_apostrophe_still_matches() -> None:
    """Markers are written with ASCII apostrophes; model output may use U+2019."""
    assert hedge_score("I don’t have specific information about that.") == 1.0


# --- removed markers ----------------------------------------------------------


def test_contrastive_connectives_are_not_markers() -> None:
    """'On the other hand' and 'worth noting' fired only on confident prose in
    the 7B control corpus (contrast and emphasis, not uncertainty) and were
    removed on that evidence."""
    text = "Water is H2O. Carbon dioxide, on the other hand, is CO2."
    assert hedge_score(text) == 0.0
    text = "Diamond is the hardest mineral. It's worth noting the Mohs scale ranks it 10."
    assert hedge_score(text) == 0.0
