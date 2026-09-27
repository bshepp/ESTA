"""Tests for the torch-free layer of narrative-direction extraction."""
from __future__ import annotations

import math

import pytest

from esta.conflict import cosine_similarity
from esta.scripts.extract_narrative_directions import build_narrative_directions


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def test_directions_are_unit_and_b_is_orthogonal_to_a() -> None:
    # A-mean points +x from neutral, B-mean points +y from neutral.
    a_acts = [[2.0, 0.0], [2.0, 0.0]]
    b_acts = [[0.0, 2.0], [0.0, 2.0]]
    neutral = [[0.0, 0.0], [0.0, 0.0]]
    r_a, r_b_perp, cos_before = build_narrative_directions(a_acts, b_acts, neutral)
    assert math.isclose(math.sqrt(sum(x * x for x in r_a)), 1.0, abs_tol=1e-9)
    assert math.isclose(math.sqrt(sum(x * x for x in r_b_perp)), 1.0, abs_tol=1e-9)
    assert cosine_similarity(r_a, r_b_perp) == pytest.approx(0.0, abs=1e-9)
    assert cos_before == pytest.approx(0.0, abs=1e-9)  # x vs y before orthogonalization


def test_reports_cosine_before_orthogonalization() -> None:
    # A and B nearly aligned -> cos_before near +1.
    a_acts = [[2.0, 0.0]]
    b_acts = [[2.0, 0.1]]
    neutral = [[0.0, 0.0]]
    _, _, cos_before = build_narrative_directions(a_acts, b_acts, neutral)
    assert cos_before > 0.99


def test_collinear_opposite_narratives_raise() -> None:
    # B is the exact negative of A (the "engagement = -refusal" failure): the
    # residual after orthogonalization is ~zero -> not separable, report it.
    a_acts = [[3.0, 0.0]]
    b_acts = [[-3.0, 0.0]]
    neutral = [[0.0, 0.0]]
    with pytest.raises(ValueError, match="collinear|separable"):
        build_narrative_directions(a_acts, b_acts, neutral)
