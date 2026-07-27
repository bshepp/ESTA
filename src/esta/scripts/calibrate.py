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

`main()` loads the model + tokenizer + refusal direction via `ModelState`,
runs each validation prompt through `generate_with_epistemic_state`, pools
per-token entropies and margins across ALL responses, and pools per-prompt
MAX refusal-direction projections into the harmful / harmless classes chosen
by `resolve_probe_class()`.  Categories with an intermediate expected
pressure (or an explicit `probe_class: "excluded"`) feed neither projection
pool — see that function for why — and the held-out categories are printed
and recorded in the output provenance.  `main()` then calls the pure
`compute_calibration()` and writes the result to `--output`.  The pure
functions here are separately unit-tested and importable without torch.
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


def _build_provenance(
    args: argparse.Namespace,
    prompts: dict[str, list[Any]],
    probe_classes: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": args.model,
        "validation_dir": str(args.validation_dir.resolve()),
        "refusal_direction_path": str(args.refusal_direction)
            if args.refusal_direction else None,
        "categories": {cat: len(p) for cat, p in prompts.items()},
        # Which pool each category fed, so a calibration can be audited without
        # re-reading the corpus it came from.
        "probe_classes": dict(sorted((probe_classes or {}).items())),
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


CALIBRATION_MAX_TOKENS = 64

PROBE_CLASS_HARMFUL = "harmful"
PROBE_CLASS_HARMLESS = "harmless"
PROBE_CLASS_EXCLUDED = "excluded"
VALID_PROBE_CLASSES = frozenset({PROBE_CLASS_HARMFUL, PROBE_CLASS_HARMLESS, PROBE_CLASS_EXCLUDED})


class ProbeClassError(ValueError):
    """A validation-case file declares a probe_class that is not recognized."""


def resolve_probe_class(data: dict[str, Any]) -> str:
    """Decide which refusal-probe pool a validation category's projections feed.

    The pressure thresholds are tail statistics of two opposing distributions:
    `pressure_low` is the upper tail of the HARMLESS projections and
    `pressure_moderate` is the lower tail of the HARMFUL ones. Only categories
    that are unambiguously one or the other may contribute. Anything in
    between — `low_to_moderate`, `moderate_to_high`, an unlabeled file — is
    EXCLUDED, because pooling an elevated-pressure category with the negatives
    inflates exactly the tail that sets `pressure_low`, which widens the "low"
    band and makes the probe under-report pressure.

    A file may state `probe_class` explicitly, which wins over inference; that
    is how a category can be topically harmful-adjacent yet deliberately held
    out of both pools. An unrecognized explicit value is a hard error rather
    than a silent fallback.
    """
    explicit = data.get("probe_class")
    if explicit is not None:
        value = str(explicit).strip().lower()
        if value not in VALID_PROBE_CLASSES:
            raise ProbeClassError(
                f"category {data.get('category', '<unnamed>')!r} declares "
                f"probe_class={explicit!r}; expected one of {sorted(VALID_PROBE_CLASSES)}"
            )
        return value

    pressure = str(data.get("expected_state", {}).get("safety_pressure", "")).strip().lower()
    if pressure == "high":
        return PROBE_CLASS_HARMFUL
    if pressure == "low":
        return PROBE_CLASS_HARMLESS
    return PROBE_CLASS_EXCLUDED


def main(args: argparse.Namespace | None = None) -> None:
    # Imported here so importing this module stays torch-free (CI imports the
    # pure functions above without [model] installed).
    import torch

    from esta.calibration import Calibration
    from esta.inference import GenerationParams, ModelState, generate_with_epistemic_state

    if args is None:
        args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    state = ModelState(
        model_name=args.model,
        device=device,
        dtype=dtype,
        refusal_direction_path=args.refusal_direction,
    )
    state.load()
    if not state.refusal_probe_loaded:
        raise SystemExit(
            f"refusal direction not found at {args.refusal_direction}; "
            "pressure calibration requires it. Run extract_refusal_direction first."
        )

    entropies: list[float] = []
    margins: list[float] = []
    harmless_projections: list[float] = []
    harmful_projections: list[float] = []
    counts: dict[str, int] = {}
    probe_classes: dict[str, str] = {}

    uncalibrated = Calibration.uncalibrated()
    gen_params = GenerationParams(max_tokens=CALIBRATION_MAX_TOKENS, temperature=0.0)

    for path in sorted(args.validation_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        category = data.get("category", path.stem)
        prompts = data.get("prompts", [])
        counts[category] = len(prompts)
        probe_class = resolve_probe_class(data)
        probe_classes[category] = probe_class

        for prompt in prompts:
            text = prompt["text"]
            chat_prompt = state.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            result = generate_with_epistemic_state(
                model_state=state,
                prompt=chat_prompt,
                params=gen_params,
                refusal_layer=args.refusal_layer,
                calibration=uncalibrated,
            )
            entropies.extend(result.debug_info["raw_entropies"])
            margins.extend(result.debug_info["raw_margins"])
            projs = result.debug_info["raw_projections"]
            # Entropies/margins above pool across every category; only the
            # projections are class-gated, and an excluded category feeds
            # neither pool.
            if projs and probe_class != PROBE_CLASS_EXCLUDED:
                pool = (
                    harmful_projections
                    if probe_class == PROBE_CLASS_HARMFUL
                    else harmless_projections
                )
                pool.append(max(projs))

    for cls in (PROBE_CLASS_HARMFUL, PROBE_CLASS_HARMLESS, PROBE_CLASS_EXCLUDED):
        members = {c: counts[c] for c, k in probe_classes.items() if k == cls}
        total = sum(members.values())
        print(f"probe class {cls:9} {total:4} prompts across {members or '{}'}")
    if excluded := [c for c, k in probe_classes.items() if k == PROBE_CLASS_EXCLUDED]:
        print(
            f"NOTE: {sorted(excluded)} contributed entropy/margin samples but were held out "
            "of both projection pools (ambiguous or explicitly excluded probe class)."
        )

    if not harmful_projections:
        raise SystemExit("no harmful-class prompts found (expected_state.safety_pressure='high').")
    if not harmless_projections:
        raise SystemExit(
            "no harmless-class prompts found (expected_state.safety_pressure='low'). "
            "Categories with an intermediate pressure label are excluded by design."
        )
    if not entropies:
        raise SystemExit("no tokens generated; cannot calibrate entropy/margin thresholds.")

    output = compute_calibration(
        entropies=entropies,
        margins=margins,
        harmless_projections=harmless_projections,
        harmful_projections=harmful_projections,
        spike_percentile=args.percentile_spike,
        low_margin_percentile=args.percentile_low_margin,
        pressure_low_percentile=args.percentile_pressure_low,
        pressure_moderate_percentile=args.percentile_pressure_moderate,
        provenance=_build_provenance(
            args, {c: [None] * n for c, n in counts.items()}, probe_classes
        ),
    )

    if output.pressure_low >= output.pressure_moderate:
        print(
            "WARNING: pressure_low >= pressure_moderate — harmful/harmless projection "
            "distributions overlap; this calibration will be REJECTED at server load. "
            "Add more/clearer refusal_expected prompts and recalibrate."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output.to_json(), encoding="utf-8")
    print(f"Wrote calibration to {args.output}")


if __name__ == "__main__":
    main()
