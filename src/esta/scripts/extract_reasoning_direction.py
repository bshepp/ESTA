"""Extract the reasoning direction and orthogonalize it against refusal.

Procedure (mirrors extract_refusal_direction.py):
    1. Run HIGH-reasoning prompts (multi-step analysis) and LOW-reasoning
       prompts (direct recall), both benign so refusal is held constant.
    2. Capture the residual stream at the target layer (same layer as refusal).
    3. reasoning = mean(high) - mean(low).
    4. Gram-Schmidt against the refusal direction, then normalize. The result
       is the reasoning component that can fire INDEPENDENTLY of refusal.
    5. Report cosine(reasoning, refusal) BEFORE orthogonalization — the go/no-go
       diagnostic. Near +/-1 means the axes are inseparable on this model.

Usage:
    python -m esta.scripts.extract_reasoning_direction \\
        --model Qwen/Qwen2.5-7B-Instruct --layer 14 \\
        --refusal-direction data/refusal_direction.pt \\
        --output data/reasoning_direction.pt

The torch-free math is esta.conflict.orthogonalize / cosine_similarity; only
model loading and activation capture need torch, so build_reasoning_direction
below is unit-tested without [model].

Grounding: [arditi-2024] (contrastive-direction method) — see docs/REFERENCES.md
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from esta.conflict import cosine_similarity, orthogonalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("esta.extract_reasoning")

# Smoke defaults. Production supplies larger --high-file / --low-file sets that
# are HELD OUT from the validation classes (data/probe_sets/*), the same
# discipline extract_refusal_direction uses with AdvBench/Alpaca.
DEFAULT_HIGH_REASONING = [
    "Analyze the trade-offs between renewable and nuclear energy for grid stability.",
    "Walk through why a bridge might resonate and collapse under rhythmic loading.",
    "Compare the mechanisms by which inflation and unemployment interact in an economy.",
    "Explain step by step how a compiler turns source code into machine instructions.",
    "Reason through why antibiotic resistance spreads faster in some populations than others.",
    "Work out the trade-offs a city faces between density and transit cost.",
    "Explain the causal chain from atmospheric CO2 to ocean acidification.",
    "Analyze why two sorting algorithms differ in worst-case versus average performance.",
    "Reason through how compound feedback loops amplify a small interest-rate change.",
    "Compare the structural reasons some materials are ductile and others brittle.",
]
DEFAULT_LOW_REASONING = [
    "What is the capital of Australia?",
    "When did the first human land on the moon?",
    "Define the word 'photosynthesis'.",
    "What is the chemical symbol for gold?",
    "How many continents are there?",
    "What year did World War II end?",
    "What is the boiling point of water at sea level in Celsius?",
    "Name the largest planet in the solar system.",
    "What is the square root of 144?",
    "Who wrote the play 'Romeo and Juliet'?",
]


def build_reasoning_direction(
    high_acts: Sequence[Sequence[float]],
    low_acts: Sequence[Sequence[float]],
    refusal_direction: Sequence[float],
) -> tuple[list[float], float]:
    """Mean-difference, orthogonalize against refusal, normalize.

    Returns (unit_reasoning_direction, cosine_to_refusal_before_orthogonalization).
    Raises ValueError if the orthogonal residual is ~zero (reasoning is
    collinear with refusal on this model — the mechanistic approach fails, and
    that is the reported result).
    """
    import numpy as np

    high_mean = np.asarray(high_acts, dtype=np.float64).mean(axis=0)
    low_mean = np.asarray(low_acts, dtype=np.float64).mean(axis=0)
    diff = high_mean - low_mean
    cos_before = cosine_similarity(diff.tolist(), list(refusal_direction))
    residual = np.asarray(orthogonalize(diff.tolist(), list(refusal_direction)), dtype=np.float64)
    norm = float(np.linalg.norm(residual))
    if norm < 1e-8:
        raise ValueError(
            f"reasoning direction is collinear with refusal (cos={cos_before:.3f}); "
            "the orthogonal residual is ~zero. The two axes are not separable on "
            "this model — report this rather than proceeding."
        )
    return [float(x) for x in residual / norm], cos_before


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


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from esta.probes.refusal import load_refusal_direction

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--refusal-direction", type=Path, default=Path("data/refusal_direction.pt"))
    parser.add_argument("--output", type=Path, default=Path("data/reasoning_direction.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--high-file", type=Path, default=None,
                        help="One high-reasoning prompt per line. Built-in smoke set if omitted.")
    parser.add_argument("--low-file", type=Path, default=None)
    args = parser.parse_args()

    if not args.refusal_direction.exists():
        raise SystemExit(
            f"refusal direction not found at {args.refusal_direction}; extract it first "
            "(the reasoning axis is orthogonalized against it)."
        )
    high = args.high_file.read_text(encoding="utf-8").strip().splitlines() if args.high_file else DEFAULT_HIGH_REASONING
    low = args.low_file.read_text(encoding="utf-8").strip().splitlines() if args.low_file else DEFAULT_LOW_REASONING
    if not args.high_file or not args.low_file:
        log.warning("Using built-in smoke prompts; supply --high-file/--low-file (held out) for production.")

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

    refusal = load_refusal_direction(args.refusal_direction, device="cpu")
    high_acts = _capture(model, tokenizer, high, args.layer, args.device)
    low_acts = _capture(model, tokenizer, low, args.layer, args.device)

    direction, cos_before = build_reasoning_direction(high_acts, low_acts, [float(x) for x in refusal])
    log.info("cosine(reasoning, refusal) before orthogonalization: %.4f", cos_before)
    if abs(cos_before) > 0.9:
        log.warning("HIGH cosine to refusal (%.3f): axes are nearly inseparable; "
                    "interpret conflict results with caution.", cos_before)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(direction, dtype=torch.float32), args.output)
    log.info("Saved reasoning direction (%d-dim, unit-norm, orthogonal to refusal) to %s",
             len(direction), args.output)


if __name__ == "__main__":
    main()
