"""Tests for the torch-free conflict-state metric layer."""

from __future__ import annotations

import math

import pytest

from esta.conflict import (
    conflict_aggregates,
    cosine_similarity,
    orthogonalize,
    token_conflict,
)

# --- cosine_similarity --------------------------------------------------------


def test_cosine_of_identical_vectors_is_one() -> None:
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_of_opposite_vectors_is_negative_one() -> None:
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


# --- orthogonalize ------------------------------------------------------------


def test_orthogonalize_removes_the_shared_component() -> None:
    # v has a component along x and along y; project out x -> only y remains.
    residual = orthogonalize([3.0, 4.0], [1.0, 0.0])
    assert residual == pytest.approx([0.0, 4.0])


def test_orthogonalized_residual_is_perpendicular_to_onto() -> None:
    onto = [2.0, 1.0]
    residual = orthogonalize([5.0, -1.0], onto)
    assert cosine_similarity(residual, onto) == pytest.approx(0.0, abs=1e-9)


def test_orthogonalize_against_collinear_leaves_near_zero() -> None:
    residual = orthogonalize([2.0, 4.0], [1.0, 2.0])  # v is exactly along onto
    assert math.sqrt(sum(x * x for x in residual)) == pytest.approx(0.0, abs=1e-9)


def test_orthogonalize_rejects_zero_onto() -> None:
    with pytest.raises(ValueError):
        orthogonalize([1.0, 2.0], [0.0, 0.0])


# --- token_conflict -----------------------------------------------------------


def test_token_conflict_is_min_of_threshold_ratios() -> None:
    # p_ref/theta_ref = 2.0 ; p_eng/theta_eng = 1.5 ; min = 1.5
    assert token_conflict(4.0, 3.0, theta_ref=2.0, theta_eng=2.0) == pytest.approx(1.5)


def test_token_conflict_below_one_when_either_axis_is_cold() -> None:
    # refusal lit (ratio 2.0) but reasoning cold (ratio 0.25) -> not a conflict
    assert token_conflict(4.0, 0.5, theta_ref=2.0, theta_eng=2.0) == pytest.approx(0.25)


def test_token_conflict_rejects_nonpositive_thresholds() -> None:
    with pytest.raises(ValueError):
        token_conflict(1.0, 1.0, theta_ref=0.0, theta_eng=1.0)


# --- conflict_aggregates ------------------------------------------------------


def test_aggregates_count_events_and_take_the_peak() -> None:
    # tokens:      (ref, eng) ratios vs theta=1.0 each
    p_ref = [2.0, 0.5, 3.0]   # ratios 2.0, 0.5, 3.0
    p_eng = [2.0, 2.0, 0.4]   # ratios 2.0, 2.0, 0.4
    # c(t) = min: 2.0, 0.5, 0.4  -> one event (c>=1), max 2.0, mean 0.9667
    agg = conflict_aggregates(p_ref, p_eng, theta_ref=1.0, theta_eng=1.0)
    assert agg["conflict_events"] == 1
    assert agg["max_conflict_score"] == pytest.approx(2.0)
    assert agg["mean_conflict_score"] == pytest.approx((2.0 + 0.5 + 0.4) / 3)
    assert agg["n_tokens"] == 3


def test_empty_series_yields_no_conflict_measurement() -> None:
    agg = conflict_aggregates([], [], theta_ref=1.0, theta_eng=1.0)
    assert agg["max_conflict_score"] is None
    assert agg["mean_conflict_score"] is None
    assert agg["conflict_events"] == 0
    assert agg["n_tokens"] == 0


def test_aggregates_require_equal_length_series() -> None:
    with pytest.raises(ValueError):
        conflict_aggregates([1.0, 2.0], [1.0], theta_ref=1.0, theta_eng=1.0)
