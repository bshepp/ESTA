"""Detect framing-conflict (v1b): two competing narrative axes both engaged.

Component 1, v1b. Reuses the v2 windowed co-activation machinery
(esta.conflict.windowed_conflict_aggregates), adds oscillation
(esta.oscillation) and an INDEPENDENT perturbation-instability signal
(esta.fidelity content-divergence over re-framed re-generations). Everything
except _generate_records() is torch-free; --rescore runs with no model.

See docs/superpowers/specs/2026-09-27-framing-conflict-probe-design.md.

Grounding: ESTA-original construct; method [arditi-2024], feature-competition
intuition [templeton-2024] -- see docs/REFERENCES.md.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

from esta.conflict import windowed_conflict_aggregates
from esta.fidelity import convergence, nearest_rank_percentile  # noqa: F401 (Task 5)
from esta.oscillation import oscillation_rate
from esta.scripts.analyze_performed_uncertainty import youden_cutoff

CLASS_TWO_SIDED = "two_sided"
CLASS_ONE_A = "one_sided_a"
CLASS_ONE_B = "one_sided_b"
CLASS_NEUTRAL = "neutral"
ALL_CLASSES = (CLASS_TWO_SIDED, CLASS_ONE_A, CLASS_ONE_B, CLASS_NEUTRAL)

DEFAULT_WINDOW = 2          # co-activation token gap (v2 windowed conjunction)
RESPONSE_MAX_TOKENS = 256


def response_divergence(a: str, b: str) -> float | None:
    """1 - Jaccard content overlap; None when either side has no content words."""
    conv = convergence(a, b)
    return None if conv is None else 1.0 - conv


def mean_pairwise_divergence(responses: Sequence[str]) -> float | None:
    """Mean divergence over all response pairs; None when there are no usable pairs.

    Undefined pairs (empty response) are skipped, not scored as zero -- an
    absence of content is not agreement.
    """
    divs = [d for a, b in combinations(responses, 2)
            if (d := response_divergence(a, b)) is not None]
    return sum(divs) / len(divs) if divs else None


def _peak(series: Sequence[float]) -> float:
    return max(series) if series else 0.0


def derive_theta(high_peaks: Sequence[float], low_peaks: Sequence[float]):
    """Youden cutoff between a narrative's own class (high) and the rest (low)."""
    return youden_cutoff(low_peaks, high_peaks)


def score_records(records: list[dict[str, Any]], theta_a, theta_b, window: int) -> None:
    """Add co-activation, oscillation, and instability to each record in place.

    When either theta is None the two axes have no calibrated threshold, so
    co-activation and oscillation are left None (not scored); instability is
    independent of theta and is always computed.
    """
    for r in records:
        r["instability"] = mean_pairwise_divergence(r.get("perturbation_responses", []))
        if theta_a is None or theta_b is None:
            r["coactivation_any"] = None
            r["coactivation_max"] = None
            r["oscillation"] = None
            continue
        agg = windowed_conflict_aggregates(
            r["p_a_series"], r["p_b_series"], theta_a, theta_b, window)
        r["coactivation_any"] = agg["any_conflict"] if agg["windowed_max_conflict_score"] is not None else None
        r["coactivation_max"] = agg["windowed_max_conflict_score"]
        s_a = [p / theta_a for p in r["p_a_series"]]
        s_b = [p / theta_b for p in r["p_b_series"]]
        r["oscillation"] = oscillation_rate(s_a, s_b)


def _mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _rate(flags):
    vals = [bool(v) for v in flags if v is not None]
    return sum(vals) / len(vals) if vals else None


def build_report(records, excluded, provenance, theta_a_cut, theta_b_cut, instability_null, window):  # noqa: ANN001
    """Summarize by category and report the internal-vs-instability contrast. Torch-free."""
    summary: dict[str, Any] = {
        "theta_a": asdict(theta_a_cut) if theta_a_cut else None,
        "theta_b": asdict(theta_b_cut) if theta_b_cut else None,
        "instability_null": instability_null,
        "window": window,
        "excluded": excluded,
        "by_category": {},
    }
    for category in dict.fromkeys(r["category"] for r in records):
        rows = [r for r in records if r["category"] == category]
        summary["by_category"][category] = {
            "n": len(rows),
            "coactivation_rate": _rate([r.get("coactivation_any") for r in rows]),
            "mean_coactivation_max": _mean([r.get("coactivation_max") for r in rows]),
            "mean_oscillation": _mean([r.get("oscillation") for r in rows]),
            "mean_instability": _mean([r.get("instability") for r in rows]),
        }
    ip = [r for r in records if r.get("topic", "").lower().startswith("israel")]
    summary["israel_palestine"] = [
        {"id": r["id"], "category": r["category"],
         "coactivation_max": r.get("coactivation_max"),
         "oscillation": r.get("oscillation"), "instability": r.get("instability")} for r in ip
    ]
    return {"provenance": provenance, "summary": summary, "records": records}


def print_report(report: dict[str, Any], output: Path) -> None:
    """ASCII-safe console summary (Windows cp1252)."""
    s = report["summary"]
    print(f"\nwrote {output}  ({len(report['records'])} records, {len(s['excluded'])} excluded)")
    ta = s["theta_a"]["cutoff"] if s["theta_a"] else None
    tb = s["theta_b"]["cutoff"] if s["theta_b"] else None
    print(f"theta_a={ta}  theta_b={tb}  window={s['window']}  instability_null={s['instability_null']}")
    print("\nby category (coact rate / mean coact max / mean oscillation / mean instability):")
    for cat, st in s["by_category"].items():
        def _f(x):
            return "n/a" if x is None else f"{x:.2f}"
        print(f"  {cat:14} n={st['n']:3}  coact={_f(st['coactivation_rate'])}  "
              f"max={_f(st['mean_coactivation_max'])}  osc={_f(st['mean_oscillation'])}  "
              f"instab={_f(st['mean_instability'])}")
    if s["israel_palestine"]:
        print(f"\nIsrael-Palestine: {s['israel_palestine']}")
    if s["excluded"]:
        print(f"\nexcluded {len(s['excluded'])}: {s['excluded']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure framing-conflict: two narrative axes engaged.")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--probe-dir", type=Path, default=Path("data/probe_sets"))
    p.add_argument("--topic", default="israel-palestine")
    p.add_argument("--direction-prefix", type=Path, default=Path("data/narrative_direction"))
    p.add_argument("--refusal-layer", type=int, default=14)
    p.add_argument("--paraphrases", type=int, default=3, help="Neutral rewordings per prompt.")
    p.add_argument("--output", type=Path, default=Path("data/framing_conflict_analysis.json"))
    p.add_argument("--max-tokens", type=int, default=RESPONSE_MAX_TOKENS)
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--rescore", type=Path, default=None, metavar="PRIOR_REPORT",
                   help="Recompute all scores from a prior report's persisted series and "
                        "perturbation responses. No model, no GPU, no torch.")
    return p.parse_args(argv)


def _load_rescore(path: Path):
    prior = json.loads(path.read_text(encoding="utf-8"))
    records = prior.get("records", [])
    if not records:
        raise SystemExit(f"{path} has no records to rescore.")
    for field in ("p_a_series", "p_b_series", "perturbation_responses"):
        missing = [r.get("id", "?") for r in records if field not in r]
        if missing:
            raise SystemExit(f"{len(missing)} record(s) in {path} lack {field!r}; re-run the model pass.")
    provenance = dict(prior.get("provenance", {}))
    provenance["rescored_from"] = str(path)
    provenance["rescored_at"] = datetime.now(UTC).isoformat()
    return records, list(prior.get("summary", {}).get("excluded", [])), provenance
