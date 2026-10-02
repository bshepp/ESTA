"""Torch-free lean-geometry signals for the framing-conflict probe (v1b.1).

Two framings of one topic span only a SHARED direction (the topic) and a
CONTRAST direction (the lean), so the per-token state is modelled as:

    s_topic(t) = p_topic(t) / theta_topic   -- engaged when >= 1
    lean(t)    = p_lean(t)  / theta_lean    -- signed: +A-leaning, -B-leaning

"Torn" has three signatures over ENGAGED tokens: balance (lean near zero while
engaged), oscillation (the lean's sign flipping across tokens, reusing
esta.oscillation), and -- the behavioural ground truth -- the lean moving or
flipping under logically-equivalent reframing (lean_shift, swap_flip).

Pure Python, unit-tested without [model], like esta.conflict / esta.oscillation.
See the v1b.1 addendum in docs/superpowers/specs/2026-09-27-framing-conflict-probe-design.md.

Grounding: ESTA-original construct; competing-feature intuition [templeton-2024],
contrastive-direction method [arditi-2024] -- see docs/REFERENCES.md.
"""

from __future__ import annotations

from collections.abc import Sequence

from esta.oscillation import oscillation_rate


def lean_signals(
    p_topic: Sequence[float],
    p_lean: Sequence[float],
    theta_topic: float,
    theta_lean: float,
) -> dict:
    """Per-response lean signals over engaged tokens.

    Returns n_engaged, and (None when nothing is engaged) mean_lean in
    theta_lean units (signed), balance = fraction of engaged tokens with
    |lean| < 1 (near the midpoint while on-topic), and oscillation = the
    committed-token sign-flip rate (esta.oscillation over (lean, -lean) with
    the lean threshold as the floor, so only |lean| >= 1 tokens count as a
    dominance and balanced tokens never register as flips).
    """
    if len(p_topic) != len(p_lean):
        raise ValueError("projection series must have equal length")
    if theta_topic <= 0 or theta_lean <= 0:
        raise ValueError("thresholds must be positive")
    leans = [pl / theta_lean for pt, pl in zip(p_topic, p_lean, strict=True)
             if pt / theta_topic >= 1.0]
    if not leans:
        return {"n_engaged": 0, "mean_lean": None, "balance": None, "oscillation": None}
    return {
        "n_engaged": len(leans),
        "mean_lean": sum(leans) / len(leans),
        "balance": sum(1 for x in leans if abs(x) < 1.0) / len(leans),
        "oscillation": oscillation_rate(leans, [-x for x in leans], engaged_floor=1.0),
    }


def lean_shift(base_mean_lean: float | None, perturbed_mean_leans: Sequence[float | None]) -> float | None:
    """Largest |delta| between the base lean and any perturbed lean, in theta units.

    Undefined (None) perturbations are skipped, not scored as zero; None when
    there is no base lean or no usable perturbation.
    """
    if base_mean_lean is None:
        return None
    deltas = [abs(x - base_mean_lean) for x in perturbed_mean_leans if x is not None]
    return max(deltas) if deltas else None


def swap_flip(first: float | None, second: float | None) -> bool | None:
    """Did presenting the sides in the opposite order flip the lean's sign?

    None when either lean is undefined (no engaged tokens on that generation).
    """
    if first is None or second is None:
        return None
    return (first > 0) != (second > 0)
