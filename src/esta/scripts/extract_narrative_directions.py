"""Extract two competing NARRATIVE directions per contested topic.

Procedure (mirrors extract_reasoning_direction.py):
    1. Run argue-A, argue-B, and register-matched NEUTRAL prompts.
    2. Capture the residual stream at the target layer (same as refusal).
    3. r_A = mean(A) - mean(neutral) ; r_B = mean(B) - mean(neutral).
    4. Gram-Schmidt r_B against r_A, then normalize both. r_B_perp is the part
       of narrative B independent of narrative A.
    5. Report cos(r_A, r_B) BEFORE orthogonalization -- the go/no-go diagnostic.
       Contested narratives are expected to be opposed, so a strongly NEGATIVE
       cosine (near -1) means they are collinear-opposite and not separable.

The torch-free math is build_narrative_directions (unit-tested without [model]);
only model loading and activation capture need torch.

Grounding: [arditi-2024] (contrastive-direction method) -- see docs/REFERENCES.md
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from esta.conflict import cosine_similarity, orthogonalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("esta.extract_narrative")


def build_narrative_directions(
    a_acts: Sequence[Sequence[float]],
    b_acts: Sequence[Sequence[float]],
    neutral_acts: Sequence[Sequence[float]],
) -> tuple[list[float], list[float], float]:
    """Mean-difference each narrative against neutral, orthogonalize B against A.

    Returns (unit_r_a, unit_r_b_perp, cosine_A_B_before_orthogonalization).
    Raises ValueError if the orthogonal residual is ~zero (the two narratives
    are collinear on this model -- report this rather than proceeding).
    """
    import numpy as np

    neutral_mean = np.asarray(neutral_acts, dtype=np.float64).mean(axis=0)
    r_a = np.asarray(a_acts, dtype=np.float64).mean(axis=0) - neutral_mean
    r_b = np.asarray(b_acts, dtype=np.float64).mean(axis=0) - neutral_mean
    cos_before = cosine_similarity(r_a.tolist(), r_b.tolist())
    a_norm = float(np.linalg.norm(r_a))
    if a_norm < 1e-8:
        raise ValueError("narrative A direction is ~zero; check the contrast prompts")
    unit_a = (r_a / a_norm).tolist()
    residual = np.asarray(orthogonalize(r_b.tolist(), r_a.tolist()), dtype=np.float64)
    b_norm = float(np.linalg.norm(residual))
    if b_norm < 1e-8:
        raise ValueError(
            f"narrative B is collinear with A (cos={cos_before:.3f}); the orthogonal "
            "residual is ~zero. The two narratives are not separable on this model -- "
            "report this rather than proceeding."
        )
    return [float(x) for x in unit_a], [float(x) for x in residual / b_norm], cos_before


# Smoke defaults for ONE topic (israel-palestine). Production supplies
# --a-file/--b-file/--neutral-file per topic, HELD OUT from the validation and
# calibration classes in data/probe_sets/. Register-matched: each is an
# analytical ask; only the narrative content differs.
DEFAULT_A = [
    "Make the strongest case for the Israeli government's security rationale in the conflict.",
    "Explain the most compelling version of Israel's argument for its territorial claims.",
]
DEFAULT_B = [
    "Make the strongest case for Palestinian claims to statehood and rights in the conflict.",
    "Explain the most compelling version of the Palestinian argument regarding displacement.",
]
DEFAULT_NEUTRAL = [
    "Analyze the trade-offs between renewable and nuclear energy for grid stability.",
    "Compare the mechanisms by which inflation and unemployment interact in an economy.",
]


def _capture(model, tokenizer, prompts, layer_idx, device):  # noqa: ANN001
    import torch

    from esta.inference.hooks import HookCapture

    acts: list[list[float]] = []
    for prompt in prompts:
        with HookCapture() as hook:
            hook.attach(model, layer_idx)
            templated = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(templated, return_tensors="pt").to(device)
            with torch.no_grad():
                model(**inputs)
        if not hook.activations:
            raise RuntimeError("no activations captured; check layer/architecture")
        acts.append([float(x) for x in hook.activations[-1][0].detach().cpu().float()])
    return acts


def _read_lines(path):  # noqa: ANN001
    return [ln for ln in path.read_text(encoding="utf-8").strip().splitlines() if ln.strip()]


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from esta.probes.refusal import load_refusal_direction

    parser = argparse.ArgumentParser(description="Extract two narrative directions for a topic.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--topic", default="israel-palestine",
                        help="Topic slug; names the output tensors.")
    parser.add_argument("--refusal-direction", type=Path, default=Path("data/refusal_direction.pt"),
                        help="Only used to REPORT cos(narrative, refusal) as a sanity check.")
    parser.add_argument("--output-prefix", type=Path, default=Path("data/narrative_direction"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--a-file", type=Path, default=None, help="One argue-A prompt per line.")
    parser.add_argument("--b-file", type=Path, default=None)
    parser.add_argument("--neutral-file", type=Path, default=None)
    args = parser.parse_args()

    a = _read_lines(args.a_file) if args.a_file else DEFAULT_A
    b = _read_lines(args.b_file) if args.b_file else DEFAULT_B
    neutral = _read_lines(args.neutral_file) if args.neutral_file else DEFAULT_NEUTRAL
    if not (args.a_file and args.b_file and args.neutral_file):
        log.warning("Using built-in smoke prompts; supply --a-file/--b-file/--neutral-file "
                    "(held out) for production.")

    log.info("Loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        device_map=args.device,
    )
    model.train(False)

    a_acts = _capture(model, tokenizer, a, args.layer, args.device)
    b_acts = _capture(model, tokenizer, b, args.layer, args.device)
    neutral_acts = _capture(model, tokenizer, neutral, args.layer, args.device)

    r_a, r_b_perp, cos_before = build_narrative_directions(a_acts, b_acts, neutral_acts)
    log.info("topic=%s  cos(A, B) before orthogonalization: %.4f", args.topic, cos_before)
    if cos_before < -0.9:
        log.warning("Narratives are strongly opposed (cos=%.3f): near collinear-opposite; "
                    "co-activation may be undetectable on this model.", cos_before)

    if args.refusal_direction.exists():
        refusal = [float(x) for x in load_refusal_direction(args.refusal_direction, device="cpu")]
        log.info("cos(A, refusal)=%.3f  cos(B_perp, refusal)=%.3f",
                 cosine_similarity(r_a, refusal), cosine_similarity(r_b_perp, refusal))

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_a = args.output_prefix.with_name(f"{args.output_prefix.name}_{args.topic}_a.pt")
    out_b = args.output_prefix.with_name(f"{args.output_prefix.name}_{args.topic}_b.pt")
    torch.save(torch.tensor(r_a, dtype=torch.float32), out_a)
    torch.save(torch.tensor(r_b_perp, dtype=torch.float32), out_b)
    log.info("Saved narrative directions to %s and %s", out_a, out_b)


if __name__ == "__main__":
    main()
