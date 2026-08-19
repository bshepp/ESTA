"""Detect conflict-state: refusal and reasoning axes both lit during generation.

Component 1, v1a. Each validation prompt is generated once with the residual
stream hooked; every generated token is projected onto the refusal direction
and the orthogonalized reasoning direction. Conflict is the conjunction (both
above their calibrated thresholds). See
docs/superpowers/specs/2026-08-18-conflict-state-probe-design.md.

Everything except _generate_records() is torch-free; --rescore runs with no
model, no GPU, no torch.

Grounding: ESTA-original construct; method [arditi-2024], feature-competition intuition [templeton-2024] — see docs/REFERENCES.md.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from esta.conflict import conflict_aggregates
from esta.scripts.analyze_performed_uncertainty import youden_cutoff

CLASS_CONSTRAINT = "constraint_region"
CLASS_ANALYTICAL = "uncontested_analytical"
CLASS_RECALL = "direct_recall"
CLASS_REFUSAL = "refusal_boundary"
ALL_CLASSES = (CLASS_CONSTRAINT, CLASS_ANALYTICAL, CLASS_RECALL, CLASS_REFUSAL)

RESPONSE_MAX_TOKENS = 256


def derive_theta_eng(recall_peaks: Sequence[float], analytical_peaks: Sequence[float]):
    """θ_eng between the low- and high-reasoning control classes (peak p_eng per
    response), via the shared Youden machinery. None if they do not separate."""
    return youden_cutoff(recall_peaks, analytical_peaks)


def _peak_eng(record: dict[str, Any]) -> float:
    series = record["p_eng_series"]
    return max(series) if series else 0.0


def score_records(records: list[dict[str, Any]], theta_ref: float, theta_eng: float) -> None:
    """Add conflict aggregates to each record in place."""
    for r in records:
        r.update(
            conflict_aggregates(r["p_ref_series"], r["p_eng_series"], theta_ref, theta_eng)
        )


def build_report(
    records: list[dict[str, Any]],
    excluded: list[dict[str, str]],
    provenance: dict[str, Any],
    theta_ref: float,
    theta_eng_cut,
) -> dict[str, Any]:
    """Summarize by category. Torch-free: pure post-processing over persisted series."""
    summary: dict[str, Any] = {
        "theta_ref": theta_ref,
        "theta_eng": asdict(theta_eng_cut) if theta_eng_cut else None,
        "excluded": excluded,
        "by_category": {},
    }
    for category in dict.fromkeys(r["category"] for r in records):
        rows = [r for r in records if r["category"] == category]
        scored = [r for r in rows if r["max_conflict_score"] is not None]
        n = len(scored) or 1
        summary["by_category"][category] = {
            "n": len(rows),
            "n_scored": len(scored),
            "mean_conflict_events": sum(r["conflict_events"] for r in scored) / n,
            "mean_max_score": sum(r["max_conflict_score"] for r in scored) / n,
            "any_conflict_rate": sum(1 for r in scored if r["conflict_events"] > 0) / n,
        }
    # Israel-Palestine broken out by name (the originator is curious).
    ip = [r for r in records if r.get("topic", "").lower().startswith("israel")]
    summary["israel_palestine"] = [
        {"id": r["id"], "conflict_events": r["conflict_events"],
         "max_conflict_score": r["max_conflict_score"]} for r in ip
    ]
    return {"provenance": provenance, "summary": summary, "records": records}


def print_report(report: dict[str, Any], output: Path) -> None:
    s = report["summary"]
    print(f"\nwrote {output}  ({len(report['records'])} records, {len(s['excluded'])} excluded)")
    if s["theta_eng"] is None:
        print("\nNOTE: reasoning controls did not separate; θ_eng not placed, no conflict scored.")
    else:
        c = s["theta_eng"]
        print(f"\nθ_ref={s['theta_ref']:.3f}  θ_eng={c['cutoff']:.3f} "
              f"(AUC {c['auc']:.2f}, p={c['p_value']:.1e})")
    print("\nby category:")
    for cat, st in s["by_category"].items():
        print(f"  {cat:24} n={st['n']:3}  events/resp={st['mean_conflict_events']:.2f}  "
              f"max={st['mean_max_score']:.2f}  fired={st['any_conflict_rate']:.0%}")
    if s["israel_palestine"]:
        print(f"\nIsrael-Palestine: {s['israel_palestine']}")
    if s["excluded"]:
        print(f"\nexcluded {len(s['excluded'])}: {s['excluded']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure conflict-state: refusal and reasoning both lit.")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--probe-dir", type=Path, default=Path("data/probe_sets"))
    p.add_argument("--refusal-direction", type=Path, default=Path("data/refusal_direction.pt"))
    p.add_argument("--reasoning-direction", type=Path, default=Path("data/reasoning_direction.pt"))
    p.add_argument("--refusal-layer", type=int, default=14)
    p.add_argument("--calibration", type=Path, default=Path("data/calibration.json"),
                   help="Provides θ_ref = pressure_moderate.")
    p.add_argument("--output", type=Path, default=Path("data/conflict_state_analysis.json"))
    p.add_argument("--max-tokens", type=int, default=RESPONSE_MAX_TOKENS)
    p.add_argument("--rescore", type=Path, default=None, metavar="PRIOR_REPORT",
                   help="Recompute thresholds and conflict from a prior report's persisted "
                        "per-token series. No model, no GPU, no torch.")
    return p.parse_args(argv)
