"""Tests for confidence metric primitives.

All tests construct synthetic log-probability arrays directly with numpy, so
they run without torch installed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from esta.confidence import (
    DEFAULT_LOW_MARGIN_THRESHOLD,
    DEFAULT_SPIKE_THRESHOLD,
    aggregate_confidence,
    token_entropy_and_margin,
)


def _uniform_log_probs(vocab_size: int) -> np.ndarray:
    return np.full(vocab_size, -math.log(vocab_size), dtype=np.float64)


def _peaked_log_probs(vocab_size: int, top: float = -1e-6) -> np.ndarray:
    # Sharply peaked: one token near probability 1, the rest split tiny mass.
    rest = math.log((1.0 - math.exp(top)) / (vocab_size - 1))
    arr = np.full(vocab_size, rest, dtype=np.float64)
    arr[0] = top
    return arr


# --- token_entropy_and_margin ------------------------------------------------


def test_uniform_distribution_has_max_entropy_zero_margin() -> None:
    vocab_size = 100
    lp = _uniform_log_probs(vocab_size)
    entropy, margin = token_entropy_and_margin(lp)
    assert math.isclose(entropy, math.log(vocab_size), abs_tol=1e-9)
    assert math.isclose(margin, 0.0, abs_tol=1e-12)


def test_peaked_distribution_has_low_entropy_large_margin() -> None:
    lp = _peaked_log_probs(vocab_size=1000)
    entropy, margin = token_entropy_and_margin(lp)
    assert entropy < 0.01  # essentially deterministic
    assert margin > 5.0    # top1 vastly outweighs top2


def test_two_token_split_margin_matches_logprob_gap() -> None:
    # 60% / 40% split between two tokens, others negligible.
    p = np.array([0.6, 0.4, 1e-12, 1e-12, 1e-12], dtype=np.float64)
    p = p / p.sum()
    lp = np.log(p)
    entropy, margin = token_entropy_and_margin(lp)
    expected_margin = math.log(0.6) - math.log(0.4)
    assert math.isclose(margin, expected_margin, abs_tol=1e-9)
    # Entropy of [0.6, 0.4] is ~0.673 nats; full dist is barely larger.
    assert 0.65 < entropy < 0.70


def test_rejects_non_1d_input() -> None:
    with pytest.raises(ValueError, match="1-D"):
        token_entropy_and_margin(np.zeros((2, 3)))


def test_rejects_single_entry_input() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        token_entropy_and_margin(np.array([0.0]))


# --- aggregate_confidence ----------------------------------------------------


def test_aggregate_empty_sequence_returns_zeros() -> None:
    metrics = aggregate_confidence([], [])
    assert metrics.mean_entropy == 0.0
    assert metrics.entropy_spike_count == 0
    assert metrics.low_margin_fraction == 0.0
    assert metrics.calibrated_confidence is None


def test_aggregate_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="equal length"):
        aggregate_confidence([1.0, 2.0], [1.0])


def test_aggregate_basic_statistics() -> None:
    entropies = [0.1, 0.2, 0.3, 0.4, 5.0]
    margins = [2.0, 1.5, 1.0, 0.6, 0.1]
    m = aggregate_confidence(entropies, margins)
    assert math.isclose(m.mean_entropy, sum(entropies) / len(entropies))
    assert math.isclose(m.median_entropy, 0.3)
    assert m.max_entropy == 5.0
    assert math.isclose(m.mean_margin, sum(margins) / len(margins))


def test_aggregate_spike_count_uses_default_threshold() -> None:
    # Default spike threshold is 4.0 nats.
    entropies = [0.5, 1.0, 3.5, 4.1, 6.0]
    margins = [1.0] * 5
    m = aggregate_confidence(entropies, margins)
    assert m.entropy_spike_count == 2  # 4.1 and 6.0
    assert DEFAULT_SPIKE_THRESHOLD == 4.0  # guard against drift


def test_aggregate_low_margin_fraction_uses_default_threshold() -> None:
    # Default low-margin threshold is 0.5.
    entropies = [1.0] * 4
    margins = [0.1, 0.4, 0.5, 1.0]
    m = aggregate_confidence(entropies, margins)
    # Strict < threshold: 0.1, 0.4 qualify; 0.5 does not.
    assert math.isclose(m.low_margin_fraction, 2 / 4)
    assert DEFAULT_LOW_MARGIN_THRESHOLD == 0.5


def test_aggregate_custom_thresholds_override_defaults() -> None:
    entropies = [1.0, 2.0, 3.0]
    margins = [0.1, 0.2, 0.3]
    m = aggregate_confidence(
        entropies,
        margins,
        spike_threshold=1.5,
        low_margin_threshold=0.25,
    )
    assert m.entropy_spike_count == 2          # 2.0 and 3.0 > 1.5
    assert math.isclose(m.low_margin_fraction, 2 / 3)  # 0.1 and 0.2 < 0.25
