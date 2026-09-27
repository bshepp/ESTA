"""Tests for the torch-free oscillation metric."""
from __future__ import annotations

import pytest

from esta.oscillation import oscillation_rate


def test_steady_dominance_is_zero() -> None:
    # A dominates at every engaged token -> no swing.
    assert oscillation_rate([2.0, 3.0, 2.5], [1.0, 1.0, 1.0], engaged_floor=1.0) == 0.0


def test_alternating_dominance_is_one() -> None:
    # dominance flips every token: A,B,A,B -> 3 changes / 3 gaps = 1.0
    assert oscillation_rate([2.0, 0.5, 2.0, 0.5], [0.5, 2.0, 0.5, 2.0], engaged_floor=1.0) == 1.0


def test_one_flip_midway() -> None:
    # A,A,B,B -> 1 change / 3 gaps
    assert oscillation_rate([2.0, 2.0, 0.5, 0.5], [0.5, 0.5, 2.0, 2.0]) == pytest.approx(1 / 3)


def test_floor_excludes_cold_tokens() -> None:
    # Only tokens where max(s_a, s_b) >= floor count. Here tokens 1,2 are cold.
    # engaged tokens: index 0 (A) and index 3 (B) -> 1 change / 1 gap = 1.0
    assert oscillation_rate([2.0, 0.1, 0.2, 0.3], [0.3, 0.2, 0.1, 2.0], engaged_floor=1.0) == 1.0


def test_fewer_than_two_engaged_is_none() -> None:
    assert oscillation_rate([2.0, 0.1], [0.1, 0.1], engaged_floor=1.0) is None


def test_all_tie_does_not_crash() -> None:
    # exact ties resolve to A-dominant (>=), so no flips -> 0.0 when engaged.
    assert oscillation_rate([1.0, 1.0], [1.0, 1.0], engaged_floor=1.0) == 0.0


def test_unequal_length_raises() -> None:
    with pytest.raises(ValueError):
        oscillation_rate([1.0, 2.0], [1.0], engaged_floor=1.0)


def test_empty_series_is_none() -> None:
    assert oscillation_rate([], [], engaged_floor=1.0) is None
