"""Tests for the torch-free layer of the performed-uncertainty analysis."""

from __future__ import annotations

import pytest

from esta.scripts.analyze_performed_uncertainty import (
    QUADRANT_DIRECT,
    QUADRANT_GENUINE,
    QUADRANT_OVERCLAIM,
    QUADRANT_PERFORMED,
    Thresholds,
    classify_quadrant,
    derive_thresholds,
    performed_uncertainty_signal,
)

# --- signal ------------------------------------------------------------------


def test_signal_is_the_product_of_confidence_and_hedging() -> None:
    assert performed_uncertainty_signal(0.9, 0.5) == pytest.approx(0.45)


def test_signal_is_zero_without_hedging() -> None:
    """Confidence alone is healthy directness, not performed uncertainty."""
    assert performed_uncertainty_signal(0.99, 0.0) == 0.0


def test_signal_is_zero_without_confidence() -> None:
    """Hedging alone is genuine uncertainty, honestly expressed."""
    assert performed_uncertainty_signal(0.0, 0.99) == 0.0


# --- thresholds --------------------------------------------------------------


def test_thresholds_sit_between_the_two_control_classes() -> None:
    t = derive_thresholds(
        obscure_confidence=[0.2, 0.3, 0.4],
        settled_confidence=[0.8, 0.9],
        settled_hedge=[0.0, 0.1],
        obscure_hedge=[0.6, 0.7],
    )
    assert t.usable
    assert t.confidence == pytest.approx(0.6)   # midpoint of 0.4 .. 0.8
    assert t.hedge == pytest.approx(0.35)       # midpoint of 0.1 .. 0.6


def test_unusable_when_confidence_classes_do_not_separate() -> None:
    """No empty band means no defensible cutoff; report rather than invent one."""
    t = derive_thresholds(
        obscure_confidence=[0.2, 0.95],
        settled_confidence=[0.8, 0.9],
        settled_hedge=[0.0, 0.1],
        obscure_hedge=[0.6, 0.7],
    )
    assert not t.usable
    assert t.confidence is None
    assert t.hedge == pytest.approx(0.35)


def test_unusable_when_hedge_classes_do_not_separate() -> None:
    t = derive_thresholds(
        obscure_confidence=[0.2, 0.3],
        settled_confidence=[0.8, 0.9],
        settled_hedge=[0.0, 0.9],
        obscure_hedge=[0.6, 0.7],
    )
    assert not t.usable
    assert t.hedge is None


# --- quadrants ---------------------------------------------------------------


def _t() -> Thresholds:
    return Thresholds(confidence=0.6, hedge=0.35)


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
        classify_quadrant(0.9, 0.8, Thresholds(confidence=None, hedge=0.35))
