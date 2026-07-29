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
    text = "Some argue this is true. On the other hand, it depends."
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
    text = "The phrase 'on the other hand' is a discourse marker."
    assert hedge_score(text) == 1.0
