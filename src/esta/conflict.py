"""Torch-free conflict-state metrics.

Conflict-state is simultaneous high projection on two competing axes — the
refusal axis and a reasoning axis orthogonalized against it (see
docs/superpowers/specs/2026-08-18-conflict-state-probe-design.md). This module
is pure numpy, unit-tested without [model], like esta.fidelity and esta.hedging.

The vector math (cosine, orthogonalize) lives here so the torch extraction
script can build the reasoning direction by converting tensors to numpy at the
boundary and calling these, keeping the numeric logic testable.

Grounding: ESTA-original construct; method [arditi-2024], feature-competition
intuition [templeton-2024] — see docs/REFERENCES.md
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors, in [-1, 1]."""
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return float(va @ vb / denom)


def orthogonalize(vector: Sequence[float], onto: Sequence[float]) -> list[float]:
    """Component of `vector` orthogonal to `onto` (Gram-Schmidt), un-normalized.

    This is what makes the reasoning axis separable from refusal: the reasoning
    direction's projection along refusal is removed, leaving only the part that
    can fire independently. The caller normalizes the residual.
    """
    v, u = np.asarray(vector, dtype=np.float64), np.asarray(onto, dtype=np.float64)
    u_norm_sq = float(u @ u)
    if u_norm_sq == 0:
        raise ValueError("cannot orthogonalize against the zero vector")
    residual = v - (float(v @ u) / u_norm_sq) * u
    return [float(x) for x in residual]


def token_conflict(p_ref: float, p_eng: float, theta_ref: float, theta_eng: float) -> float:
    """Threshold-relative conflict at one token: min(p_ref/theta_ref, p_eng/theta_eng).

    Dominated by whichever axis is closer to not-firing — the conservative
    "both must clear the bar" reading. A value >= 1 means both axes are lit,
    which is a conflict event.
    """
    if theta_ref <= 0 or theta_eng <= 0:
        raise ValueError("thresholds must be positive")
    return min(p_ref / theta_ref, p_eng / theta_eng)


def conflict_aggregates(
    p_ref_series: Sequence[float],
    p_eng_series: Sequence[float],
    theta_ref: float,
    theta_eng: float,
) -> dict:
    """Per-response aggregates over the two per-token projection series.

    max/mean of the graded per-token score, and the count of tokens where both
    axes are lit (score >= 1). Empty series -> no measurement (None/0), so the
    caller excludes the record rather than scoring an absence.
    """
    if len(p_ref_series) != len(p_eng_series):
        raise ValueError("projection series must have equal length")
    scores = [
        token_conflict(r, e, theta_ref, theta_eng)
        for r, e in zip(p_ref_series, p_eng_series, strict=True)
    ]
    if not scores:
        return {
            "max_conflict_score": None,
            "mean_conflict_score": None,
            "conflict_events": 0,
            "n_tokens": 0,
        }
    return {
        "max_conflict_score": max(scores),
        "mean_conflict_score": sum(scores) / len(scores),
        "conflict_events": sum(1 for s in scores if s >= 1.0),
        "n_tokens": len(scores),
    }
