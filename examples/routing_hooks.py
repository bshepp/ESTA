#!/usr/bin/env python3
"""Consume an ESTA `epistemic_state` block to make a routing decision.

This is the downstream side of ESTA: the server exposes state, and a caller
decides what to *do* with a response based on that state. ESTA never blocks or
filters — routing policy lives here, in the consumer.

`route()` is a pure function over the Phase 1 epistemic_state dict, so it is
trivially testable and has no dependencies. Run this file directly to see it
applied to two illustrative sample states, or pass --live "<prompt>" to route a
real response from a running server.

IMPORTANT: the thresholds below are ILLUSTRATIVE, not calibrated. Phase 1 ships
placeholder thresholds (see esta.probes.thresholds); real cutoffs come from
`python -m esta.scripts.calibrate`. Treat this as a template for your own policy,
not as production-ready numbers.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

# --- Illustrative policy thresholds (NOT calibrated) -------------------------
ENTROPY_SPIKE_LIMIT = 3        # tokens above the spike threshold tolerated
LOW_MARGIN_FRACTION_LIMIT = 0.25  # fraction of near-tie tokens tolerated
ELEVATED_PRESSURE_LABELS = {"moderate", "high"}

# Routing actions, most-to-least severe.
PROCEED = "proceed"
REVIEW = "human_review"
GATHER_SOURCES = "require_additional_sources"
WARN_UNCALIBRATED = "proceed_with_uncalibrated_warning"


@dataclass
class RoutingDecision:
    action: str
    reasons: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        head = f"action: {self.action}"
        if not self.reasons:
            return head
        return head + "".join(f"\n  - {r}" for r in self.reasons)


def route(epistemic_state: dict) -> RoutingDecision:
    """Map an epistemic_state dict to a routing decision.

    Policy (illustrative):
      - elevated safety pressure  -> human review
      - low token-level confidence -> require additional sources before acting
      - uncalibrated pressure probe -> proceed, but flag that the signal is a stub
      - otherwise                  -> proceed
    The most severe applicable action wins; all triggered reasons are reported.
    """
    conf = epistemic_state["confidence"]
    pressure = epistemic_state["safety_pressure"]
    reasons: list[str] = []

    pressure_label = pressure["calibrated_pressure"]
    elevated_pressure = pressure_label in ELEVATED_PRESSURE_LABELS
    if elevated_pressure:
        reasons.append(
            f"safety pressure '{pressure_label}' "
            f"(refusal_projection_max={pressure['refusal_projection_max']:.3f})"
        )

    low_confidence = (
        conf["entropy_spike_count"] > ENTROPY_SPIKE_LIMIT
        or conf["low_margin_fraction"] > LOW_MARGIN_FRACTION_LIMIT
    )
    if low_confidence:
        reasons.append(
            f"low confidence (entropy_spike_count={conf['entropy_spike_count']}, "
            f"low_margin_fraction={conf['low_margin_fraction']:.3f})"
        )

    uncalibrated = pressure_label == "uncalibrated"
    if uncalibrated:
        reasons.append("safety-pressure probe not loaded/calibrated; pressure signal is a stub")

    # Severity ordering: review > gather-sources > uncalibrated-warning > proceed.
    if elevated_pressure:
        return RoutingDecision(REVIEW, reasons)
    if low_confidence:
        return RoutingDecision(GATHER_SOURCES, reasons)
    if uncalibrated:
        return RoutingDecision(WARN_UNCALIBRATED, reasons)
    return RoutingDecision(PROCEED, reasons)


# --- Sample states for the offline demo --------------------------------------
SAMPLE_CLEAN = {
    "confidence": {"entropy_spike_count": 0, "low_margin_fraction": 0.02},
    "safety_pressure": {"calibrated_pressure": "low", "refusal_projection_max": 0.12},
}
SAMPLE_PRESSURED = {
    "confidence": {"entropy_spike_count": 5, "low_margin_fraction": 0.31},
    "safety_pressure": {"calibrated_pressure": "high", "refusal_projection_max": 1.84},
}


def _demo() -> int:
    for name, state in (("clean factual answer", SAMPLE_CLEAN),
                        ("pressured / uncertain answer", SAMPLE_PRESSURED)):
        print(f"\n# {name}")
        print(route(state))
    return 0


def _live(prompt: str, url: str) -> int:
    try:
        from basic_request import chat  # sibling example; run from examples/ or repo root
    except ImportError:
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
        from basic_request import chat
    body = chat(url, prompt, max_tokens=256, return_activations=False)
    print(f"\n# live response from {url}")
    print(body["choices"][0]["message"]["content"].strip())
    print()
    print(route(body["epistemic_state"]))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Route on an ESTA epistemic_state.")
    parser.add_argument("--live", metavar="PROMPT", help="Route a real response for this prompt.")
    parser.add_argument("--url", default="http://localhost:8000", help="Server base URL.")
    args = parser.parse_args(argv)

    if args.live:
        return _live(args.live, args.url)
    return _demo()


if __name__ == "__main__":
    raise SystemExit(main())
