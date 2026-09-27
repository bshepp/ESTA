"""Extract two competing NARRATIVE directions per contested topic.

Procedure (mirrors extract_reasoning_direction.py):
    1. Run argue-A, argue-B, and register-matched NEUTRAL prompts.
    2. Capture the residual stream at the target layer (same as refusal).
    3. r_A = mean(A) - mean(neutral) ; r_B = mean(B) - mean(neutral).
    4. Gram-Schmidt r_B against r_A, then normalize both. r_B_perp is the part
       of narrative B independent of narrative A.
    5. Report cos(r_A, r_B) BEFORE orthogonalization -- the go/no-go diagnostic.
       Contested narratives are expected to be opposed, so a strongly NEGATIVE
       cosine (near -1) means they are collinear-opposite and not separable.

The torch-free math is build_narrative_directions (unit-tested without [model]);
only model loading and activation capture need torch.

Grounding: [arditi-2024] (contrastive-direction method) -- see docs/REFERENCES.md
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from esta.conflict import cosine_similarity, orthogonalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("esta.extract_narrative")


def build_narrative_directions(
    a_acts: Sequence[Sequence[float]],
    b_acts: Sequence[Sequence[float]],
    neutral_acts: Sequence[Sequence[float]],
) -> tuple[list[float], list[float], float]:
    """Mean-difference each narrative against neutral, orthogonalize B against A.

    Returns (unit_r_a, unit_r_b_perp, cosine_A_B_before_orthogonalization).
    Raises ValueError if the orthogonal residual is ~zero (the two narratives
    are collinear on this model -- report this rather than proceeding).
    """
    import numpy as np

    neutral_mean = np.asarray(neutral_acts, dtype=np.float64).mean(axis=0)
    r_a = np.asarray(a_acts, dtype=np.float64).mean(axis=0) - neutral_mean
    r_b = np.asarray(b_acts, dtype=np.float64).mean(axis=0) - neutral_mean
    cos_before = cosine_similarity(r_a.tolist(), r_b.tolist())
    a_norm = float(np.linalg.norm(r_a))
    if a_norm < 1e-8:
        raise ValueError("narrative A direction is ~zero; check the contrast prompts")
    unit_a = (r_a / a_norm).tolist()
    residual = np.asarray(orthogonalize(r_b.tolist(), r_a.tolist()), dtype=np.float64)
    b_norm = float(np.linalg.norm(residual))
    if b_norm < 1e-8:
        raise ValueError(
            f"narrative B is collinear with A (cos={cos_before:.3f}); the orthogonal "
            "residual is ~zero. The two narratives are not separable on this model -- "
            "report this rather than proceeding."
        )
    return [float(x) for x in unit_a], [float(x) for x in residual / b_norm], cos_before
