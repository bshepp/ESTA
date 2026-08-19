"""Torch-free tests for the reasoning-direction builder."""

from __future__ import annotations

import math

import pytest

from esta.conflict import cosine_similarity
from esta.scripts.extract_reasoning_direction import build_reasoning_direction


def test_reasoning_direction_is_unit_and_orthogonal_to_refusal() -> None:
    # high-reasoning mean minus low-reasoning mean has a component along refusal;
    # the builder must return the orthogonal, unit-norm residual.
    refusal = [1.0, 0.0, 0.0]
    high = [[2.0, 3.0, 0.0], [2.0, 3.0, 0.0]]   # mean (2,3,0)
    low = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]    # mean (1,0,0); diff (1,3,0)
    direction, cos_before = build_reasoning_direction(high, low, refusal)
    assert math.isclose(sum(x * x for x in direction), 1.0, abs_tol=1e-9)  # unit
    assert cosine_similarity(direction, refusal) == pytest.approx(0.0, abs=1e-9)
    # diff (1,3,0) had a positive refusal component, so cos_before > 0
    assert cos_before > 0.0


def test_reasoning_collinear_with_refusal_reports_high_cosine() -> None:
    refusal = [1.0, 0.0]
    high = [[3.0, 0.0]]
    low = [[1.0, 0.0]]     # diff (2,0) is exactly along refusal
    with pytest.raises(ValueError):
        # orthogonal residual is ~zero -> cannot normalize -> loud failure
        build_reasoning_direction(high, low, refusal)
