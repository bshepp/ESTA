"""Tests for the torch-free lean-geometry signals (v1b.1)."""
from __future__ import annotations

import pytest

from esta.lean import lean_shift, lean_signals, swap_flip

# s_topic = p_topic / theta_topic ; engaged when >= 1. lean in theta_lean units, +A / -B.


def test_only_engaged_tokens_count() -> None:
    sig = lean_signals(p_topic=[0.1, 2.0, 2.0], p_lean=[9.0, 1.0, -1.0], theta_topic=1.0, theta_lean=2.0)
    assert sig["n_engaged"] == 2                      # token 0 is not engaged (s_topic 0.1)
    assert sig["mean_lean"] == pytest.approx(0.0)     # (+0.5 + -0.5) / 2 in theta units


def test_mean_lean_is_signed_and_theta_scaled() -> None:
    sig = lean_signals([2.0, 2.0], [-4.0, -4.0], theta_topic=1.0, theta_lean=2.0)
    assert sig["mean_lean"] == pytest.approx(-2.0)    # B-leaning, 2x the lean threshold


def test_balance_is_fraction_of_engaged_tokens_under_the_lean_threshold() -> None:
    # |lean| in theta units: 0.25, 0.5, 1.5, 3.0 -> two of four are "balanced" (< 1)
    sig = lean_signals([2.0] * 4, [0.5, -1.0, 3.0, -6.0], theta_topic=1.0, theta_lean=2.0)
    assert sig["balance"] == pytest.approx(0.5)


def test_oscillation_counts_committed_sign_flips_only() -> None:
    # committed (|lean| >= theta) and alternating sign -> maximal vacillation
    sig = lean_signals([2.0] * 4, [3.0, -3.0, 3.0, -3.0], theta_topic=1.0, theta_lean=2.0)
    assert sig["oscillation"] == pytest.approx(1.0)
    # steadily A-leaning -> no vacillation
    steady = lean_signals([2.0] * 4, [3.0, 4.0, 3.0, 5.0], theta_topic=1.0, theta_lean=2.0)
    assert steady["oscillation"] == pytest.approx(0.0)


def test_no_engaged_tokens_yields_no_measurement() -> None:
    sig = lean_signals([0.1, 0.2], [5.0, 5.0], theta_topic=1.0, theta_lean=1.0)
    assert sig["n_engaged"] == 0
    assert sig["mean_lean"] is None and sig["balance"] is None and sig["oscillation"] is None


def test_lean_signals_validate_inputs() -> None:
    with pytest.raises(ValueError):
        lean_signals([1.0, 1.0], [1.0], theta_topic=1.0, theta_lean=1.0)
    with pytest.raises(ValueError):
        lean_signals([1.0], [1.0], theta_topic=0.0, theta_lean=1.0)
    with pytest.raises(ValueError):
        lean_signals([1.0], [1.0], theta_topic=1.0, theta_lean=-1.0)


# --- lean_shift: how far the lean moves under perturbation (theta units) --------


def test_lean_shift_is_max_abs_delta_skipping_undefined() -> None:
    assert lean_shift(0.2, [1.5, None, -0.3]) == pytest.approx(1.3)


def test_lean_shift_none_without_usable_perturbations() -> None:
    assert lean_shift(0.2, [None, None]) is None
    assert lean_shift(None, [1.0]) is None


# --- swap_flip: does presentation order flip the lean? ------------------------


def test_swap_flip_true_when_sign_differs() -> None:
    assert swap_flip(0.8, -0.5) is True
    assert swap_flip(-0.2, 0.9) is True


def test_swap_flip_false_when_same_sign() -> None:
    assert swap_flip(0.8, 0.3) is False
    assert swap_flip(-0.8, -0.1) is False


def test_swap_flip_undefined_when_either_side_missing() -> None:
    assert swap_flip(None, 0.5) is None
    assert swap_flip(0.5, None) is None
