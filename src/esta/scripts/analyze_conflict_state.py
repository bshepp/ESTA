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
import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
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
        print("\nNOTE: reasoning controls did not separate; theta_eng not placed, no conflict scored.")
    else:
        c = s["theta_eng"]
        print(f"\ntheta_ref={s['theta_ref']:.3f}  theta_eng={c['cutoff']:.3f} "
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
                   help="Recompute θ_eng and conflict from a prior report's persisted series; "
                        "inherit θ_ref from that report. No model, no GPU, no torch.")
    return p.parse_args(argv)


def _load_prompts(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")).get("prompts", [])


def _theta_eng_from_controls(records: list[dict[str, Any]]):
    recall = [_peak_eng(r) for r in records if r["category"] == CLASS_RECALL]
    analytical = [_peak_eng(r) for r in records if r["category"] == CLASS_ANALYTICAL]
    if not recall or not analytical:
        raise SystemExit(
            "θ_eng needs both direct_recall and uncontested_analytical records; one is empty."
        )
    return derive_theta_eng(recall, analytical)


def _finish(records, excluded, provenance, theta_ref):  # noqa: ANN001
    theta_eng_cut = _theta_eng_from_controls(records)
    theta_eng = theta_eng_cut.cutoff if theta_eng_cut is not None else None
    if theta_eng is not None:
        score_records(records, theta_ref, theta_eng)
    else:
        for r in records:  # no cutoff -> no conflict measurement, but keep records
            r.update({"max_conflict_score": None, "mean_conflict_score": None,
                      "conflict_events": 0, "n_tokens": len(r["p_ref_series"])})
    return build_report(records, excluded, provenance, theta_ref, theta_eng_cut)


def _load_rescore(path: Path):
    prior = json.loads(path.read_text(encoding="utf-8"))
    records = prior.get("records", [])
    if not records:
        raise SystemExit(f"{path} has no records to rescore.")
    for field in ("p_ref_series", "p_eng_series"):
        missing = [r.get("id", "?") for r in records if field not in r]
        if missing:
            raise SystemExit(f"{len(missing)} record(s) in {path} lack {field!r}; re-run the model pass.")
    theta_ref = float(prior.get("summary", {}).get("theta_ref", prior.get("provenance", {}).get("theta_ref")))
    provenance = dict(prior.get("provenance", {}))
    provenance["rescored_from"] = str(path)
    provenance["rescored_at"] = datetime.now(UTC).isoformat()
    return records, list(prior.get("summary", {}).get("excluded", [])), provenance, theta_ref


def _generate_records(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from esta.calibration import load_calibration
    from esta.inference.hooks import HookCapture
    from esta.probes.refusal import load_refusal_direction, project_activations

    for path, what in ((args.refusal_direction, "refusal"), (args.reasoning_direction, "reasoning")):
        if not path.exists():
            raise SystemExit(f"{what} direction not found at {path}; extract it first.")
    if not args.calibration or not args.calibration.exists():
        raise SystemExit(
            f"calibration not found at {args.calibration}; θ_ref gates the refusal axis, "
            "so without it every conflict score would rest on a placeholder threshold. "
            "Run esta.scripts.calibrate first."
        )
    calibration = load_calibration(args.calibration, args.model)
    theta_ref = float(calibration.pressure_moderate)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32, device_map=device)
    model.train(False)
    r_ref = load_refusal_direction(args.refusal_direction, device="cpu")
    r_eng = load_refusal_direction(args.reasoning_direction, device="cpu")  # same loader: a (hidden,) tensor

    records: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for cls in ALL_CLASSES:
        prompts = _load_prompts(args.probe_dir / f"{cls}.json")
        print(f"running {cls} ({len(prompts)} prompts) ...")
        for prompt in prompts:
            with HookCapture() as hook:
                hook.attach(model, args.refusal_layer)
                templated = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt["text"]}], tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(templated, return_tensors="pt").to(device)
                with torch.no_grad():
                    # Greedy (deterministic). Omit temperature/top_p entirely — passing them with
                    # do_sample=False triggers "generation flags not valid" warnings on some
                    # transformers versions.
                    out = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False,
                                         pad_token_id=tokenizer.pad_token_id)
            response = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            p_ref = project_activations(hook.activations, r_ref)
            p_eng = project_activations(hook.activations, r_eng)
            # Defensive: HookCapture yields the prefill residual, so p_ref is rarely empty in practice.
            if not p_ref:
                excluded.append({"id": prompt["id"], "reason": "no tokens generated"})
                continue
            records.append({
                "id": prompt["id"], "category": cls, "text": prompt["text"],
                "topic": prompt.get("topic", ""), "response": response,
                "p_ref_series": p_ref, "p_eng_series": p_eng,
            })
    provenance = {
        "timestamp": datetime.now(UTC).isoformat(), "model": args.model,
        "max_tokens": args.max_tokens, "refusal_layer": args.refusal_layer,
        "refusal_direction": str(args.refusal_direction),
        "reasoning_direction": str(args.reasoning_direction),
        "calibration": str(args.calibration), "theta_ref": theta_ref,
        "calibrated": calibration.calibrated, "calibration_id": calibration.calibration_id,
    }
    return records, excluded, provenance, theta_ref


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    if args.rescore is not None:
        records, excluded, provenance, theta_ref = _load_rescore(args.rescore)
    else:
        records, excluded, provenance, theta_ref = _generate_records(args)
    report = _finish(records, excluded, provenance, theta_ref)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_report(report, args.output)


if __name__ == "__main__":
    main()
