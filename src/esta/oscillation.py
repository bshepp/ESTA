"""Torch-free oscillation metric: sequential vacillation between two axes.

Where co-activation (esta.conflict) asks whether two competing directions are
lit AT ONCE, oscillation asks whether the model SWINGS between them across the
response -- the other way a generation can be "torn". Among engaged tokens (at
least one axis lit), it counts how often the dominant axis flips.

Pure Python, unit-tested without [model], like esta.hedging / esta.fidelity /
esta.conflict. See docs/superpowers/specs/2026-09-27-framing-conflict-probe-design.md.

Grounding: ESTA-original construct; competing-feature intuition [templeton-2024]
-- see docs/REFERENCES.md.
"""

from __future__ import annotations

from collections.abc import Sequence


def oscillation_rate(
    s_a: Sequence[float],
    s_b: Sequence[float],
    engaged_floor: float = 1.0,
) -> float | None:
    """Normalized dominance-flip rate between two threshold-ratio series, in [0, 1].

    An "engaged" token is one where at least one axis is lit: max(s_a, s_b) >=
    engaged_floor. Among engaged tokens, the dominant axis is A when s_a >= s_b.
    The rate is (dominance changes) / (engaged tokens - 1): 0.0 when one axis
    dominates throughout (steady, possibly co-active), 1.0 when dominance flips
    at every engaged step (maximal vacillation). None when fewer than two tokens
    are engaged (no swing is definable).
    """
    if len(s_a) != len(s_b):
        raise ValueError("series must have equal length")
    dominant_a = [a >= b for a, b in zip(s_a, s_b, strict=True) if max(a, b) >= engaged_floor]
    if len(dominant_a) < 2:
        return None
    changes = sum(1 for x, y in zip(dominant_a, dominant_a[1:], strict=False) if x != y)
    return changes / (len(dominant_a) - 1)
