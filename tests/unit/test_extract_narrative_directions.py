"""Tests for the torch-free layer of narrative-direction extraction."""
from __future__ import annotations

import math

import pytest

from esta.conflict import cosine_similarity
from esta.scripts.extract_narrative_directions import (
    build_narrative_directions,
    build_topic_and_lean,
    separability_warning,
)

# --- separability_warning ------------------------------------------------------
# The go/no-go diagnostic must flag BOTH collinear failure modes. The 7B
# Israel-Palestine check (2026-09-30) returned cos(A, B) = +0.987 -- the two
# narratives were nearly the SAME direction (the shared topic component
# dominated) -- and the original check, which only looked for the anticipated
# collinear-OPPOSITE case (cos near -1), stayed silent.


def test_positive_collinear_narratives_are_flagged() -> None:
    msg = separability_warning(0.987)
    assert msg is not None
    assert "same direction" in msg or "shared" in msg


def test_negative_collinear_narratives_are_flagged() -> None:
    msg = separability_warning(-0.95)
    assert msg is not None
    assert "opposed" in msg or "opposite" in msg


def test_separable_narratives_produce_no_warning() -> None:
    assert separability_warning(0.34) is None   # the v1a reasoning/refusal cosine
    assert separability_warning(-0.5) is None
    assert separability_warning(0.0) is None


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


# --- build_topic_and_lean (v1b.1) --------------------------------------------
# With two classes against one baseline there are only two directions: a SHARED
# one (the topic) and a CONTRAST one (the lean). v1b's cos(A, B) = +0.987 showed
# the shared direction dominating; the lean model represents that geometry
# honestly instead of pretending there are two independent narrative axes.
def test_topic_is_shared_mean_and_lean_is_orthogonal_contrast() -> None:
    # A and B share a big common component (+x) and differ along y.
    a_acts = [[4.0, 1.0], [4.0, 1.0]]
    b_acts = [[4.0, -1.0], [4.0, -1.0]]
    neutral = [[0.0, 0.0]]
    topic, lean, cos_before = build_topic_and_lean(a_acts, b_acts, neutral)
    assert topic == pytest.approx([1.0, 0.0])          # mean(A u B) - neutral = (4, 0), normalized
    assert lean == pytest.approx([0.0, 1.0])           # (A - B) = (0, 2) -> +y is A-leaning
    assert cosine_similarity(topic, lean) == pytest.approx(0.0, abs=1e-9)
    assert math.isclose(math.sqrt(sum(x * x for x in lean)), 1.0, abs_tol=1e-9)
    assert cos_before == pytest.approx(15 / 17)        # raw collinearity of r_A=(4,1), r_B=(4,-1)


def test_lean_sign_convention_is_positive_toward_a() -> None:
    a_acts = [[1.0, 3.0]]
    b_acts = [[1.0, 1.0]]
    neutral = [[0.0, 0.0]]
    _, lean, _ = build_topic_and_lean(a_acts, b_acts, neutral)
    # projecting a pure A activation onto lean must be positive
    assert sum(x * y for x, y in zip(a_acts[0], lean, strict=True)) > 0


def test_identical_narratives_have_no_lean() -> None:
    a_acts = [[2.0, 2.0]]
    b_acts = [[2.0, 2.0]]
    neutral = [[0.0, 0.0]]
    with pytest.raises(ValueError, match="no contrast|lean"):
        build_topic_and_lean(a_acts, b_acts, neutral)
