"""Per-token confidence metrics aggregated across a generated sequence.

These functions operate on numpy arrays so they can be unit-tested without
torch installed. The server converts torch logits to numpy at the boundary.

THRESHOLDS: the default spike and low-margin thresholds below are educated
guesses. They MUST be replaced with empirical percentiles from a calibration
set before any production claim — run `python -m esta.scripts.calibrate` and
serve the result via ESTA_CALIBRATION. While no calibration is loaded, the
pressure label is gated to "uncalibrated" so these placeholders can never be
mistaken for measured thresholds.
"""

from __future__ import annotations

import numpy as np

from esta.schema import ConfidenceMetrics

# ln(vocab_size) for Llama-3 and Qwen-2.5 is ~11.7 (vocab ~128k for Llama, ~152k
# for Qwen). A spike threshold of 4.0 nats marks tokens where the effective
# number of choices was exp(4) ~= 55 — moderate uncertainty.
# TODO: replace with empirical p95 of per-token entropy from a calibration set.
DEFAULT_SPIKE_THRESHOLD: float = 4.0

# A 0.5 nat gap between top1 and top2 logprobs marks the model as roughly
# split between two near-equivalent next tokens (top1 ~1.65x more likely).
# TODO: replace with empirical p10 of per-token top1-top2 gap.
DEFAULT_LOW_MARGIN_THRESHOLD: float = 0.5


def token_entropy_and_margin(log_probs: np.ndarray) -> tuple[float, float]:
    """Compute entropy (nats) and top1-top2 margin for a single token's distribution.

    Parameters
    ----------
    log_probs : np.ndarray
        1-D array of log-probabilities over the vocabulary. Must already be
        log-softmaxed (i.e., exp(log_probs) sums to 1).

    Returns
    -------
    (entropy_nats, margin_logprob)
        entropy_nats : -sum(p * log p), in nats.
        margin_logprob : log_probs.max() - log_probs.second-max(), in nats.
    """
    if log_probs.ndim != 1:
        raise ValueError(f"log_probs must be 1-D; got shape {log_probs.shape}")
    if log_probs.size < 2:
        raise ValueError("log_probs must have at least 2 entries for margin")

    probs = np.exp(log_probs)
    # p * log p is 0 at p=0; numpy handles this if we mask, but exp(very_negative)*log_p
    # is finite and well-behaved here.
    entropy = float(-np.sum(probs * log_probs))

    # Top-2 logprobs via partition (O(V), no full sort needed).
    top2 = np.partition(log_probs, -2)[-2:]
    margin = float(np.max(top2) - np.min(top2))
    return entropy, margin


def aggregate_confidence(
    entropies: list[float],
    margins: list[float],
    *,
    spike_threshold: float = DEFAULT_SPIKE_THRESHOLD,
    low_margin_threshold: float = DEFAULT_LOW_MARGIN_THRESHOLD,
) -> ConfidenceMetrics:
    """Aggregate per-token (entropy, margin) lists into a ConfidenceMetrics record.

    Handles the empty-sequence case (no tokens generated) by returning zeros.
    """
    if len(entropies) != len(margins):
        raise ValueError(
            f"entropies and margins must have equal length; got {len(entropies)} and {len(margins)}"
        )

    if not entropies:
        return ConfidenceMetrics(
            mean_entropy=0.0,
            median_entropy=0.0,
            p90_entropy=0.0,
            max_entropy=0.0,
            mean_margin=0.0,
            low_margin_fraction=0.0,
            entropy_spike_count=0,
            calibrated_confidence=None,
        )

    ent = np.asarray(entropies, dtype=np.float64)
    mar = np.asarray(margins, dtype=np.float64)

    return ConfidenceMetrics(
        mean_entropy=float(np.mean(ent)),
        median_entropy=float(np.median(ent)),
        p90_entropy=float(np.percentile(ent, 90)),
        max_entropy=float(np.max(ent)),
        mean_margin=float(np.mean(mar)),
        low_margin_fraction=float(np.mean(mar < low_margin_threshold)),
        entropy_spike_count=int(np.sum(ent > spike_threshold)),
        calibrated_confidence=None,
    )
