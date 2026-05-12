"""Refusal-direction probe per Arditi et al. (2024).

The refusal direction is extracted offline by scripts/extract_refusal_direction.py.
At inference time, residual-stream activations are projected onto it; the scalar
magnitude indicates how strongly the response is being shaped by safety training.

THRESHOLDS: the default pressure cutoffs (low<0.5, moderate<1.5, high>=1.5)
are placeholders. They MUST be replaced with empirical percentiles from a
validation set before any production claim.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import torch

DEFAULT_PROBE_VERSION = "arditi_v1_unrefined"


class PressureThresholds(NamedTuple):
    low: float
    moderate: float


# TODO: replace with empirical p50 / p90 of projection magnitudes on a
# validation set of harmful and harmless prompts.
DEFAULT_PRESSURE_THRESHOLDS = PressureThresholds(low=0.5, moderate=1.5)


def load_refusal_direction(path: Path | str, device: str | torch.device = "cpu") -> torch.Tensor:
    """Load a refusal direction tensor from disk and L2-normalize it."""
    direction = torch.load(Path(path), map_location=device)
    if not isinstance(direction, torch.Tensor):
        raise TypeError(f"refusal direction must be a torch.Tensor; got {type(direction).__name__}")
    norm = direction.norm()
    if norm.item() == 0.0:
        raise ValueError("refusal direction has zero norm")
    return direction / norm


def project_activations(
    activations: list[torch.Tensor],
    direction: torch.Tensor,
) -> list[float]:
    """Project each (batch=1, hidden) activation onto the (hidden,) direction.

    Returns a Python list of float projection magnitudes, one per activation.
    """
    if not activations:
        return []
    direction_cpu = direction.detach().to("cpu").float()
    out = []
    for act in activations:
        # act is (batch, hidden); take batch 0.
        proj = (act[0].detach().to("cpu").float() @ direction_cpu).item()
        out.append(proj)
    return out


def label_pressure(
    projection_max: float,
    thresholds: PressureThresholds = DEFAULT_PRESSURE_THRESHOLDS,
) -> str:
    """Map a max projection magnitude to a coarse label."""
    if projection_max < thresholds.low:
        return "low"
    if projection_max < thresholds.moderate:
        return "moderate"
    return "high"
