"""Calibrate ESTA's entropy / margin / pressure thresholds against a validation set.

The script runs the validation prompts through the configured model, captures
per-token (entropy, margin) and per-prompt refusal-direction projections, then
computes empirical percentiles to replace the placeholder thresholds in
`esta.confidence.metrics` and `esta.probes.refusal`.

Usage:
    python -m esta.scripts.calibrate \\
        --validation-dir data/validation_cases \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --refusal-direction data/refusal_direction.pt \\
        --output data/calibration.json

The model-running path is not yet implemented in this session — calling
`main()` raises `NotImplementedError` with instructions. The pure-function
`compute_calibration()` is implemented and unit-tested; it is what `main()`
will call once the model-run path lands.

Phase: needs model — the generation loop is added in the post-test session.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_SPIKE_PERCENTILE = 95.0
DEFAULT_LOW_MARGIN_PERCENTILE = 10.0
DEFAULT_PRESSURE_LOW_PERCENTILE = 95.0       # of HARMLESS projections
DEFAULT_PRESSURE_MODERATE_PERCENTILE = 10.0  # of HARMFUL projections


@dataclass
class CalibrationOutput:
    """A calibrated threshold set plus provenance for reproducibility."""

    spike_threshold: float
    low_margin_threshold: float
    pressure_low: float
    pressure_moderate: float
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def compute_percentile(values: list[float], percentile: float) -> float:
    """Wrap numpy.percentile with empty-input and bounds checks."""
    if not values:
        raise ValueError("cannot compute percentile of empty sequence")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100]; got {percentile}")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, percentile))


def compute_calibration(
    entropies: list[float],
    margins: list[float],
    harmless_projections: list[float],
    harmful_projections: list[float],
    *,
    spike_percentile: float = DEFAULT_SPIKE_PERCENTILE,
    low_margin_percentile: float = DEFAULT_LOW_MARGIN_PERCENTILE,
    pressure_low_percentile: float = DEFAULT_PRESSURE_LOW_PERCENTILE,
    pressure_moderate_percentile: float = DEFAULT_PRESSURE_MODERATE_PERCENTILE,
    provenance: dict[str, Any] | None = None,
) -> CalibrationOutput:
    """Derive empirical threshold values from observed distributions.

    Parameters
    ----------
    entropies : list of per-token entropies across the validation set
    margins   : list of per-token top1-top2 margins across the validation set
    harmless_projections : refusal-direction projections on prompts that are
        EXPECTED to be answered (the negative class for the refusal probe)
    harmful_projections : refusal-direction projections on prompts that are
        EXPECTED to be refused (the positive class)

    The pressure thresholds use the percentile of the *opposite* distribution
    so that the moderate-pressure cutoff sits at the low end of the harmful
    distribution and the low-pressure cutoff sits at the high end of the
    harmless distribution. If `pressure_low >= pressure_moderate`, the two
    distributions overlap and the probe is not well-calibrated; the caller
    should investigate.
    """
    return CalibrationOutput(
        spike_threshold=compute_percentile(entropies, spike_percentile),
        low_margin_threshold=compute_percentile(margins, low_margin_percentile),
        pressure_low=compute_percentile(harmless_projections, pressure_low_percentile),
        pressure_moderate=compute_percentile(harmful_projections, pressure_moderate_percentile),
        provenance=provenance or {},
    )


def load_validation_set(validation_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Read every *.json file under validation_dir; return {category: prompts}."""
    out: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(validation_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        category = data.get("category", path.stem)
        out[category] = data.get("prompts", [])
    return out


def _build_provenance(args: argparse.Namespace, prompts: dict[str, list[Any]]) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": args.model,
        "validation_dir": str(args.validation_dir.resolve()),
        "refusal_direction_path": str(args.refusal_direction)
            if args.refusal_direction else None,
        "categories": {cat: len(p) for cat, p in prompts.items()},
        "percentiles": {
            "spike": args.percentile_spike,
            "low_margin": args.percentile_low_margin,
            "pressure_low": args.percentile_pressure_low,
            "pressure_moderate": args.percentile_pressure_moderate,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate ESTA thresholds against a validation set."
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path("data/validation_cases"),
        help="Directory containing the validation_cases/*.json prompt files.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model id to calibrate against.",
    )
    parser.add_argument(
        "--refusal-direction",
        type=Path,
        default=Path("data/refusal_direction.pt"),
        help="Path to the refusal direction tensor produced by extract_refusal_direction.",
    )
    parser.add_argument(
        "--refusal-layer",
        type=int,
        default=14,
        help="Layer index used during refusal-direction extraction.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/calibration.json"),
        help="Where to write the calibrated thresholds + provenance.",
    )
    parser.add_argument(
        "--percentile-spike", type=float, default=DEFAULT_SPIKE_PERCENTILE,
        help="Per-token entropy percentile that defines a 'spike'.",
    )
    parser.add_argument(
        "--percentile-low-margin", type=float, default=DEFAULT_LOW_MARGIN_PERCENTILE,
        help="Per-token margin percentile that defines 'low margin'.",
    )
    parser.add_argument(
        "--percentile-pressure-low", type=float, default=DEFAULT_PRESSURE_LOW_PERCENTILE,
        help="Upper-tail percentile of HARMLESS projections used as the 'low' cutoff.",
    )
    parser.add_argument(
        "--percentile-pressure-moderate", type=float, default=DEFAULT_PRESSURE_MODERATE_PERCENTILE,
        help="Lower-tail percentile of HARMFUL projections used as the 'moderate' cutoff.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    prompts = load_validation_set(args.validation_dir)

    # Provenance is built early so the failure message can include it.
    _ = _build_provenance(args, prompts)

    raise NotImplementedError(
        "The model-run path of calibrate.py is implemented in the post-test "
        "session. The pure function `compute_calibration` is available and "
        "tested in tests/unit/test_calibration.py; once the model loop is "
        "wired in, main() will call it with the captured distributions. "
        "Use `python -m esta.scripts.calibrate --help` to inspect the CLI."
    )


if __name__ == "__main__":
    main()
