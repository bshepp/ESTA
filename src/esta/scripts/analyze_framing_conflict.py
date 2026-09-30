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
from esta.fidelity import convergence, nearest_rank_percentile
from esta.oscillation import oscillation_rate
from esta.scripts.analyze_performed_uncertainty import mann_whitney_p, youden_cutoff

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
    # instability_null is None means no calibrated threshold exists yet -- leave
    # every record's flag undefined (None) rather than defaulting it to "stable".
    for r in records:
        inst = r.get("instability")
        r["unstable"] = (
            None if instability_null is None
            else bool(inst is not None and inst > instability_null)
        )

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
            "unstable_rate": _rate([r.get("unstable") for r in rows]),
        }

    # Does the internal (co-activation/oscillation) signal track the independent
    # perturbation-instability signal? Compare records flagged unstable against
    # records flagged stable; None when either side lacks a usable sample.
    internal_vs_instability: dict[str, Any] = {}
    for field in ("coactivation_max", "oscillation"):
        unstable_vals = [r[field] for r in records
                         if r.get("unstable") is True and r.get(field) is not None]
        stable_vals = [r[field] for r in records
                       if r.get("unstable") is False and r.get(field) is not None]
        if not unstable_vals or not stable_vals:
            internal_vs_instability[field] = None
        else:
            internal_vs_instability[field] = {
                "unstable_mean": _mean(unstable_vals),
                "stable_mean": _mean(stable_vals),
                "n_unstable": len(unstable_vals),
                "n_stable": len(stable_vals),
                "mann_whitney_p": mann_whitney_p(stable_vals, unstable_vals),
            }
    summary["internal_vs_instability"] = internal_vs_instability

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

    def _f(x):
        return "n/a" if x is None else f"{x:.2f}"

    print("\nby category (coact rate / mean coact max / mean oscillation / mean instability / "
          "unstable rate):")
    for cat, st in s["by_category"].items():
        print(f"  {cat:14} n={st['n']:3}  coact={_f(st['coactivation_rate'])}  "
              f"max={_f(st['mean_coactivation_max'])}  osc={_f(st['mean_oscillation'])}  "
              f"instab={_f(st['mean_instability'])}  unstable={_f(st['unstable_rate'])}")
    print("\ninternal signal vs instability (unstable mean / stable mean / mann-whitney p):")
    for field, stat in s.get("internal_vs_instability", {}).items():
        if stat is None:
            print(f"  {field:18} n/a")
        else:
            print(f"  {field:18} unstable={_f(stat['unstable_mean'])}  "
                  f"stable={_f(stat['stable_mean'])}  p={_f(stat['mann_whitney_p'])}")
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


def _theta_from_controls(records):  # noqa: ANN001
    """theta_a from one_sided_a (high on A) vs the rest (low); theta_b symmetric."""
    a_high = [_peak(r["p_a_series"]) for r in records if r["category"] == CLASS_ONE_A]
    a_low = [_peak(r["p_a_series"]) for r in records
             if r["category"] in (CLASS_ONE_B, CLASS_NEUTRAL)]
    b_high = [_peak(r["p_b_series"]) for r in records if r["category"] == CLASS_ONE_B]
    b_low = [_peak(r["p_b_series"]) for r in records
             if r["category"] in (CLASS_ONE_A, CLASS_NEUTRAL)]
    if not (a_high and a_low and b_high and b_low):
        raise SystemExit("theta needs one_sided_a, one_sided_b, and neutral control records.")
    return derive_theta(a_high, a_low), derive_theta(b_high, b_low)


def _instability_null(records):  # noqa: ANN001
    """p95 of one-sided within-topic instability -- 'unstable' means beyond this."""
    one_sided = [r["instability"] for r in records
                 if r["category"] in (CLASS_ONE_A, CLASS_ONE_B) and r.get("instability") is not None]
    return nearest_rank_percentile(one_sided, 95) if one_sided else None


def _scorable_theta(cut):  # noqa: ANN001
    """The cutoff to score with, or None when the axis cannot be scored.

    Projections can be negative, so Youden can place a cutoff <= 0 (the
    narrative's own class peaks at or below zero, merely above the rest). A
    threshold RATIO p/theta is meaningless there and windowed_conflict_aggregates
    rejects it -- so treat it as unscorable, a reported condition (the report
    still shows the placed cutoff and its sign), rather than crash the run.
    """
    return cut.cutoff if cut is not None and cut.cutoff > 0 else None


def _finish(records, excluded, provenance, window):  # noqa: ANN001
    theta_a_cut, theta_b_cut = _theta_from_controls(records)
    # score_records computes instability for every record (it is theta-independent)
    # before _instability_null reads it below, so no separate pre-pass is needed.
    score_records(records, _scorable_theta(theta_a_cut), _scorable_theta(theta_b_cut), window)
    return build_report(records, excluded, provenance, theta_a_cut, theta_b_cut,
                        _instability_null(records), window)


# Neutral paraphrase templates, shared perturbation family across all classes.
def _paraphrases(text: str, k: int) -> list[str]:
    """WEAK deterministic perturbation fallback: reword the instruction framing only.

    These stems vary the wrapper around the question, not the core ask itself,
    so they are a weak proxy for a real re-framing perturbation -- a model that
    is insensitive to instruction wording will look "stable" here even if it
    would diverge under an actual re-framing. Authored per-prompt
    `prompt["paraphrases"]` (see _perturbation_prompts) are preferred whenever
    available, since they give a much stronger perturbation signal; this
    fallback exists only for probe items that don't have any authored yet.
    Deterministic and always returns exactly k items.
    """
    stems = [
        "Consider the following question. {t}",
        "Please address this: {t}",
        "Here is a question to work through. {t}",
        "Respond to the following prompt. {t}",
        "Take a moment to think through this. {t}",
        "Here's something worth considering. {t}",
    ]
    return [stems[i % len(stems)].format(t=text) for i in range(k)]


def _perturbation_prompts(prompt: dict, k: int) -> list[str]:
    """Perturbation prompts for one probe item: authored paraphrases if present, else the fallback.

    Authored `prompt["paraphrases"]` are curated re-framings that give a strong,
    intentional perturbation signal; when absent (or empty), fall back to the
    weak deterministic `_paraphrases`.
    """
    authored = prompt.get("paraphrases")
    if authored:
        return list(authored)[:k]
    return _paraphrases(prompt["text"], k)


def _generate_records(args):  # noqa: ANN001
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from esta.inference.hooks import HookCapture
    from esta.probes.refusal import load_refusal_direction, project_activations

    prefix = args.direction_prefix
    dir_a = prefix.with_name(f"{prefix.name}_{args.topic}_a.pt")
    dir_b = prefix.with_name(f"{prefix.name}_{args.topic}_b.pt")
    for path in (dir_a, dir_b):
        if not path.exists():
            raise SystemExit(f"narrative direction not found at {path}; extract it first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device)
    model.train(False)
    r_a = load_refusal_direction(dir_a, device="cpu")   # same loader: a (hidden,) tensor
    r_b = load_refusal_direction(dir_b, device="cpu")

    def _generate(text: str, hook_it: bool):
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(templated, return_tensors="pt").to(device)
        if hook_it:
            with HookCapture() as hook:
                hook.attach(model, args.refusal_layer)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=args.max_tokens,
                                         do_sample=False, pad_token_id=tokenizer.pad_token_id)
            text_out = tokenizer.decode(out[0, inputs.input_ids.shape[1]:],
                                        skip_special_tokens=True).strip()
            return text_out, hook.activations
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip(), None

    records, excluded = [], []
    for cls in ALL_CLASSES:
        path = args.probe_dir / f"framing_{args.topic}.json" if cls != CLASS_NEUTRAL \
            else args.probe_dir / "uncontested_analytical.json"
        prompts = [p for p in json.loads(path.read_text(encoding="utf-8")).get("prompts", [])
                   if cls == CLASS_NEUTRAL or p.get("class") == cls]
        print(f"running {cls} ({len(prompts)} prompts) ...")
        for prompt in prompts:
            base, acts = _generate(prompt["text"], hook_it=True)
            p_a = project_activations(acts, r_a)
            p_b = project_activations(acts, r_b)
            if not p_a:
                excluded.append({"id": prompt["id"], "reason": "no tokens generated"})
                continue
            perts = [base] + [_generate(t, hook_it=False)[0]
                              for t in _perturbation_prompts(prompt, args.paraphrases)]
            records.append({
                "id": prompt["id"], "category": cls, "topic": prompt.get("topic", args.topic),
                "text": prompt["text"], "response": base,
                "p_a_series": p_a, "p_b_series": p_b, "perturbation_responses": perts,
            })
    provenance = {
        "timestamp": datetime.now(UTC).isoformat(), "model": args.model, "topic": args.topic,
        "max_tokens": args.max_tokens, "refusal_layer": args.refusal_layer,
        "paraphrases": args.paraphrases, "window": args.window,
        "direction_a": str(dir_a), "direction_b": str(dir_b),
    }
    return records, excluded, provenance


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    if args.rescore is not None:
        records, excluded, provenance = _load_rescore(args.rescore)
        window = int(provenance.get("window", DEFAULT_WINDOW))
    else:
        records, excluded, provenance = _generate_records(args)
        window = args.window
    report = _finish(records, excluded, provenance, window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_report(report, args.output)


if __name__ == "__main__":
    main()
