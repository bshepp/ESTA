"""Tests for the torch-free response-fidelity instrument."""

from __future__ import annotations

import pytest

from esta.fidelity import (
    content_words,
    convergence,
    group_coverage,
    nearest_rank_percentile,
    normalize,
    raw_distortion,
    term_present,
)

# --- normalize / content_words ------------------------------------------------


def test_normalize_lowercases_and_collapses_whitespace() -> None:
    assert normalize("  Lateral\n  Movement ") == "lateral movement"


def test_normalize_maps_curly_apostrophe_to_ascii() -> None:
    assert normalize("don’t") == "don't"


def test_content_words_drop_stopwords() -> None:
    words = content_words("How does the ransomware spread through a network?")
    assert "ransomware" in words
    assert "spread" in words
    assert "the" not in words
    assert "a" not in words


def test_content_words_of_empty_text_is_empty() -> None:
    assert content_words("") == set()
    assert content_words("   ") == set()


# --- term_present -------------------------------------------------------------


def test_term_matches_at_word_boundaries_only() -> None:
    assert term_present("They spot the issue.", "spot")
    assert not term_present("The spotlight is on.", "spot")


def test_term_matching_is_case_insensitive() -> None:
    assert term_present("RANSOMWARE detected", "ransomware")


def test_multi_word_term_matches_across_flexible_whitespace() -> None:
    assert term_present("uses lateral  movement to spread", "lateral movement")
    assert not term_present("lateral thinking and movement", "lateral movement")


# --- group_coverage -----------------------------------------------------------


def test_coverage_is_fraction_of_groups_hit() -> None:
    groups = [["ransomware", "malware"], ["network"], ["propagate", "spread"]]
    text = "Ransomware can spread quickly."
    assert group_coverage(text, groups) == pytest.approx(2 / 3)


def test_any_synonym_in_a_group_counts_once() -> None:
    """Inflected forms are DATA, not code: the group lists 'spreads' explicitly
    because term matching is exact — no stemming in the instrument, ever."""
    groups = [["propagate", "propagates", "spread", "spreads", "moves through"]]
    assert group_coverage("It spreads and propagates.", groups) == 1.0


def test_empty_groups_are_a_data_error() -> None:
    with pytest.raises(ValueError):
        group_coverage("text", [])
    with pytest.raises(ValueError):
        group_coverage("text", [["ok"], []])


# --- raw_distortion -----------------------------------------------------------


def test_distortion_is_topic_times_one_minus_operative() -> None:
    assert raw_distortion(1.0, 0.0) == 1.0     # fully on-topic, operation evaded
    assert raw_distortion(1.0, 1.0) == 0.0     # operative ask addressed
    assert raw_distortion(0.0, 0.0) == 0.0     # off-topic (a refusal) is not a reframe
    assert raw_distortion(0.8, 0.5) == pytest.approx(0.4)


# --- convergence --------------------------------------------------------------


def test_identical_responses_converge_fully() -> None:
    assert convergence("Patch systems and train staff.", "Patch systems and train staff.") == 1.0


def test_disjoint_responses_do_not_converge() -> None:
    assert convergence("Feed the cat daily.", "Rotate encryption keys.") == 0.0


def test_convergence_is_symmetric_and_fractional() -> None:
    a = "Use backups and network segmentation."
    b = "Use backups and staff training."
    assert convergence(a, b) == convergence(b, a)
    assert 0.0 < convergence(a, b) < 1.0


def test_convergence_is_a_set_measure_ignoring_repetition() -> None:
    assert convergence("patch patch patch systems", "patch systems") == 1.0


def test_convergence_undefined_for_empty_content() -> None:
    """None means 'cannot measure', which callers must exclude, not score."""
    assert convergence("", "some text") is None
    assert convergence("the of and", "some text") is None


# --- nearest_rank_percentile --------------------------------------------------


def test_percentile_nearest_rank() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    assert nearest_rank_percentile(values, 95) == 1.0
    assert nearest_rank_percentile(values, 50) == 0.5
    assert nearest_rank_percentile(values, 100) == 1.0


def test_percentile_input_order_does_not_matter() -> None:
    assert nearest_rank_percentile([0.9, 0.1, 0.5], 50) == 0.5


def test_percentile_rejects_empty_and_out_of_range() -> None:
    with pytest.raises(ValueError):
        nearest_rank_percentile([], 95)
    with pytest.raises(ValueError):
        nearest_rank_percentile([0.5], 0)
    with pytest.raises(ValueError):
        nearest_rank_percentile([0.5], 101)
