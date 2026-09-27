# Framing-Conflict Probe (v1b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the offline framing-conflict detector (v1b): two competing narrative directions per topic, an internal co-activation + oscillation signal, and an independent perturbation-instability ground truth, tied together in a torch-free, `--rescore`-able analysis.

**Architecture:** Reuse the shipped numeric cores — `esta.conflict.windowed_conflict_aggregates` for co-activation and `esta.fidelity.convergence` for response-divergence — add one new torch-free metric (`esta.oscillation`), one torch extraction script mirroring `extract_reasoning_direction.py`, and one analysis script mirroring `analyze_conflict_state.py` (torch quarantined in `_generate_records`, everything else torch-free and `--rescore`-able). Data is four topic probe sets plus held-out extraction contrasts.

**Tech Stack:** Python 3.11+, numpy (torch-free layers), torch/transformers (extraction + generation only, imported inside the model-run functions), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-09-27-framing-conflict-probe-design.md`

## Global Constraints

- **No torch in the numeric layers.** `esta.oscillation` and every function in `analyze_framing_conflict.py` except `_generate_records` must import without torch. Torch is imported *inside* `_generate_records` / `main` / `_capture`, never at module top. (CLAUDE.md torch/no-torch boundary.)
- **No schema change.** `SCHEMA_VERSION` stays `0.1.1`; do not touch `src/esta/schema/`.
- **Thresholds from controls, never invented.** θ_A, θ_B via `youden_cutoff`; instability null via `nearest_rank_percentile`. No magic numbers as thresholds.
- **Leakage discipline.** Extraction contrast prompts are disjoint from calibration and validation prompts. Never add probe files to `data/validation_cases/` (calibration globs it).
- **DCO sign-off on every commit:** `git commit -s`.
- **ASCII-only console output.** `print_report` must encode to cp1252 (Windows console) — no `θ`, em-dashes, etc. Use `theta`, `->`.
- **Ruff** line-length 100, `E501` ignored. Run `ruff check src tests` before finishing.
- **Grounding comments.** Each new state module/script names its reference cite-keys and points to `docs/REFERENCES.md`, matching the sibling files.
- **Run commands with the venv:** `.venv/Scripts/python.exe -m pytest ...` on this Windows box.

## Review Focus

- **Unequal-length projection series** (`p_A`, `p_B⊥` of different lengths) — every metric that zips two series must raise `ValueError`, not silently truncate. Pinned in Task 1 (oscillation) and Task 4 (scoring).
- **All-tie / degenerate dominance** (`s_A == s_B` at every token) — oscillation must not divide by zero or crash; returns `0.0` when engaged ≥ 2, `None` when engaged < 2. Pinned in Task 1.
- **A direction whose calibration classes don't separate** (Youden returns `None`) — θ is `None`, co-activation is not scored for that topic, and the report says so rather than placing a placeholder threshold. Pinned in Task 4.
- **Perturbation set with an empty/whitespace response** (`convergence` returns `None`) — instability must exclude the undefined pair, and a prompt with `< 2` usable responses reports instability `None`, not `0.0`. Pinned in Task 4.
- **`--rescore` corpus missing per-token series or perturbation responses** — refused loudly with a `SystemExit` naming the missing field, like the sibling detectors, never a deep `KeyError`. Pinned in Task 5.

---

### Task 1: Oscillation metric (`esta.oscillation`)

**Files:**
- Create: `src/esta/oscillation.py`
- Test: `tests/unit/test_oscillation.py`

**Interfaces:**
- Consumes: nothing (pure).
- Produces: `oscillation_rate(s_a: Sequence[float], s_b: Sequence[float], engaged_floor: float = 1.0) -> float | None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_oscillation.py
"""Tests for the torch-free oscillation metric."""
from __future__ import annotations

import pytest

from esta.oscillation import oscillation_rate


def test_steady_dominance_is_zero() -> None:
    # A dominates at every engaged token -> no swing.
    assert oscillation_rate([2.0, 3.0, 2.5], [1.0, 1.0, 1.0], engaged_floor=1.0) == 0.0


def test_alternating_dominance_is_one() -> None:
    # dominance flips every token: A,B,A,B -> 3 changes / 3 gaps = 1.0
    assert oscillation_rate([2.0, 0.5, 2.0, 0.5], [0.5, 2.0, 0.5, 2.0], engaged_floor=1.0) == 1.0


def test_one_flip_midway() -> None:
    # A,A,B,B -> 1 change / 3 gaps
    assert oscillation_rate([2.0, 2.0, 0.5, 0.5], [0.5, 0.5, 2.0, 2.0]) == pytest.approx(1 / 3)


def test_floor_excludes_cold_tokens() -> None:
    # Only tokens where max(s_a, s_b) >= floor count. Here tokens 1,2 are cold.
    # engaged tokens: index 0 (A) and index 3 (B) -> 1 change / 1 gap = 1.0
    assert oscillation_rate([2.0, 0.1, 0.2, 0.3], [0.3, 0.2, 0.1, 2.0], engaged_floor=1.0) == 1.0


def test_fewer_than_two_engaged_is_none() -> None:
    assert oscillation_rate([2.0, 0.1], [0.1, 0.1], engaged_floor=1.0) is None


def test_all_tie_does_not_crash() -> None:
    # exact ties resolve to A-dominant (>=), so no flips -> 0.0 when engaged.
    assert oscillation_rate([1.0, 1.0], [1.0, 1.0], engaged_floor=1.0) == 0.0


def test_unequal_length_raises() -> None:
    with pytest.raises(ValueError):
        oscillation_rate([1.0, 2.0], [1.0], engaged_floor=1.0)


def test_empty_series_is_none() -> None:
    assert oscillation_rate([], [], engaged_floor=1.0) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_oscillation.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'esta.oscillation'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/esta/oscillation.py
"""Torch-free oscillation metric: sequential vacillation between two axes.

Where co-activation (esta.conflict) asks whether two competing directions are
lit AT ONCE, oscillation asks whether the model SWINGS between them across the
response -- the other way a generation can be "torn". Among engaged tokens (at
least one axis lit), it counts how often the dominant axis flips.

Pure Python, unit-tested without [model], like esta.hedging / esta.fidelity /
esta.conflict. See docs/superpowers/specs/2026-09-27-framing-conflict-probe-design.md.

Grounding: ESTA-original construct; competing-feature intuition [templeton-2024]
-- see docs/REFERENCES.md.
"""

from __future__ import annotations

from collections.abc import Sequence


def oscillation_rate(
    s_a: Sequence[float],
    s_b: Sequence[float],
    engaged_floor: float = 1.0,
) -> float | None:
    """Normalized dominance-flip rate between two threshold-ratio series, in [0, 1].

    An "engaged" token is one where at least one axis is lit: max(s_a, s_b) >=
    engaged_floor. Among engaged tokens, the dominant axis is A when s_a >= s_b.
    The rate is (dominance changes) / (engaged tokens - 1): 0.0 when one axis
    dominates throughout (steady, possibly co-active), 1.0 when dominance flips
    at every engaged step (maximal vacillation). None when fewer than two tokens
    are engaged (no swing is definable).
    """
    if len(s_a) != len(s_b):
        raise ValueError("series must have equal length")
    dominant_a = [a >= b for a, b in zip(s_a, s_b, strict=True) if max(a, b) >= engaged_floor]
    if len(dominant_a) < 2:
        return None
    changes = sum(1 for x, y in zip(dominant_a, dominant_a[1:]) if x != y)
    return changes / (len(dominant_a) - 1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_oscillation.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add src/esta/oscillation.py tests/unit/test_oscillation.py
git commit -s -m "feat(framing): torch-free oscillation metric"
```

---

### Task 2: Torch-free math for narrative-direction extraction

**Files:**
- Create: `src/esta/scripts/extract_narrative_directions.py` (only `build_narrative_directions` + constants in this task; `main`/`_capture` in Task 3)
- Test: `tests/unit/test_extract_narrative_directions.py`

**Interfaces:**
- Consumes: `esta.conflict.cosine_similarity`, `esta.conflict.orthogonalize`.
- Produces: `build_narrative_directions(a_acts, b_acts, neutral_acts) -> tuple[list[float], list[float], float]` returning `(unit_r_a, unit_r_b_perp, cos_a_b_before_orthogonalization)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_extract_narrative_directions.py
"""Tests for the torch-free layer of narrative-direction extraction."""
from __future__ import annotations

import math

import pytest

from esta.conflict import cosine_similarity
from esta.scripts.extract_narrative_directions import build_narrative_directions


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def test_directions_are_unit_and_b_is_orthogonal_to_a() -> None:
    # A-mean points +x from neutral, B-mean points +y from neutral.
    a_acts = [[2.0, 0.0], [2.0, 0.0]]
    b_acts = [[0.0, 2.0], [0.0, 2.0]]
    neutral = [[0.0, 0.0], [0.0, 0.0]]
    r_a, r_b_perp, cos_before = build_narrative_directions(a_acts, b_acts, neutral)
    assert math.isclose(math.sqrt(sum(x * x for x in r_a)), 1.0, abs_tol=1e-9)
    assert math.isclose(math.sqrt(sum(x * x for x in r_b_perp)), 1.0, abs_tol=1e-9)
    assert cosine_similarity(r_a, r_b_perp) == pytest.approx(0.0, abs=1e-9)
    assert cos_before == pytest.approx(0.0, abs=1e-9)  # x vs y before orthogonalization


def test_reports_cosine_before_orthogonalization() -> None:
    # A and B nearly aligned -> cos_before near +1.
    a_acts = [[2.0, 0.0]]
    b_acts = [[2.0, 0.1]]
    neutral = [[0.0, 0.0]]
    _, _, cos_before = build_narrative_directions(a_acts, b_acts, neutral)
    assert cos_before > 0.99


def test_collinear_opposite_narratives_raise() -> None:
    # B is the exact negative of A (the "engagement = -refusal" failure): the
    # residual after orthogonalization is ~zero -> not separable, report it.
    a_acts = [[3.0, 0.0]]
    b_acts = [[-3.0, 0.0]]
    neutral = [[0.0, 0.0]]
    with pytest.raises(ValueError, match="collinear|separable"):
        build_narrative_directions(a_acts, b_acts, neutral)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_extract_narrative_directions.py -q`
Expected: FAIL — `ImportError` / module missing.

- [ ] **Step 3: Write minimal implementation (torch-free part only)**

```python
# src/esta/scripts/extract_narrative_directions.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_extract_narrative_directions.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/esta/scripts/extract_narrative_directions.py tests/unit/test_extract_narrative_directions.py
git commit -s -m "feat(framing): torch-free narrative-direction math"
```

---

### Task 3: Extraction script `main` (torch)

**Files:**
- Modify: `src/esta/scripts/extract_narrative_directions.py` (add `_capture`, `main`, default smoke prompts)
- Test: none new (torch path covered by Task 8 smoke); `build_narrative_directions` already tested.

**Interfaces:**
- Consumes: `build_narrative_directions` (Task 2), `esta.inference.hooks.HookCapture`, `esta.probes.refusal.load_refusal_direction`.
- Produces: a CLI that writes two tensors per topic: `<output_prefix>_<topic>_a.pt` and `<output_prefix>_<topic>_b.pt`.

- [ ] **Step 1: Append `_capture` and `main` (mirrors extract_reasoning_direction.py)**

```python
# append to src/esta/scripts/extract_narrative_directions.py

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
```

- [ ] **Step 2: Verify the module still imports without torch**

Run: `.venv/Scripts/python.exe -c "import esta.scripts.extract_narrative_directions"`
Expected: no error, no torch import at module load (torch is inside functions).

- [ ] **Step 3: Run the torch-free tests still pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_extract_narrative_directions.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/esta/scripts/extract_narrative_directions.py
git commit -s -m "feat(framing): narrative-direction extraction CLI (torch in main)"
```

---

### Task 4: Analysis — torch-free scoring & report layer

**Files:**
- Create: `src/esta/scripts/analyze_framing_conflict.py` (constants + torch-free functions; `_generate_records` in Task 5)
- Test: `tests/unit/test_framing_conflict_analysis.py`

**Interfaces:**
- Consumes: `esta.conflict.windowed_conflict_aggregates`, `esta.oscillation.oscillation_rate`, `esta.fidelity.convergence`, `esta.fidelity.nearest_rank_percentile`, `esta.scripts.analyze_performed_uncertainty.youden_cutoff`.
- Produces (all torch-free, importable without torch):
  - `CLASS_TWO_SIDED = "two_sided"`, `CLASS_ONE_A = "one_sided_a"`, `CLASS_ONE_B = "one_sided_b"`, `CLASS_NEUTRAL = "neutral"`, `DEFAULT_WINDOW = 2`
  - `response_divergence(a: str, b: str) -> float | None`
  - `mean_pairwise_divergence(responses: Sequence[str]) -> float | None`
  - `derive_theta(high_peaks, low_peaks)` -> AxisCut|None (delegates to `youden_cutoff`)
  - `score_records(records, theta_a, theta_b, window) -> None` (adds co-activation, oscillation, instability fields in place)
  - `build_report(records, excluded, provenance, theta_a_cut, theta_b_cut, instability_null, window) -> dict` (`window` REQUIRED — recorded verbatim into `summary["window"]`; never defaulted, so a non-default `--window` run cannot misreport its own parameter)
  - `print_report(report, output: Path) -> None` (ASCII-safe)
  - `parse_args`, `_load_rescore`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_framing_conflict_analysis.py
"""Tests for the torch-free layer of the framing-conflict analysis."""
from __future__ import annotations

import pytest

from esta.scripts.analyze_framing_conflict import (
    CLASS_NEUTRAL,
    CLASS_ONE_A,
    CLASS_ONE_B,
    CLASS_TWO_SIDED,
    build_report,
    mean_pairwise_divergence,
    response_divergence,
    score_records,
)


def test_response_divergence_is_one_minus_convergence() -> None:
    # identical texts -> convergence 1.0 -> divergence 0.0
    assert response_divergence("alpha beta gamma", "alpha beta gamma") == pytest.approx(0.0)
    # disjoint content words -> convergence 0.0 -> divergence 1.0
    assert response_divergence("alpha beta", "gamma delta") == pytest.approx(1.0)


def test_response_divergence_none_when_undefined() -> None:
    assert response_divergence("", "alpha") is None


def test_mean_pairwise_divergence_skips_undefined_pairs() -> None:
    # three responses; the empty one makes 2 of 3 pairs undefined, leaving 1 pair.
    assert mean_pairwise_divergence(["alpha beta", "gamma delta", ""]) == pytest.approx(1.0)


def test_mean_pairwise_divergence_none_when_under_two_usable() -> None:
    assert mean_pairwise_divergence(["alpha", ""]) is None
    assert mean_pairwise_divergence(["alpha"]) is None


def _rec(rid, category, p_a, p_b, perts, topic="israel-palestine"):
    return {"id": rid, "category": category, "topic": topic, "text": "q",
            "response": " ".join(perts[:1]) if perts else "r",
            "p_a_series": p_a, "p_b_series": p_b, "perturbation_responses": perts}


def test_score_records_adds_all_three_signals() -> None:
    # both axes lit at same token -> co-activation event; perts diverge -> instability 1.0
    records = [_rec("t1", CLASS_TWO_SIDED, [2.0, 2.0], [2.0, 2.0],
                    ["alpha beta", "gamma delta"])]
    score_records(records, theta_a=1.0, theta_b=1.0, window=2)
    r = records[0]
    assert r["coactivation_any"] is True
    assert r["oscillation"] == 0.0                 # steady both-lit, no swing
    assert r["instability"] == pytest.approx(1.0)  # disjoint perturbation responses


def test_score_records_marks_missing_theta() -> None:
    records = [_rec("t1", CLASS_TWO_SIDED, [2.0], [2.0], ["a b", "c d"])]
    score_records(records, theta_a=None, theta_b=1.0, window=2)
    assert records[0]["coactivation_any"] is None   # no theta_a -> not scored


def test_build_report_summarizes_and_flags_israel_palestine() -> None:
    records = (
        [_rec(f"ts{i}", CLASS_TWO_SIDED, [2.0, 2.0], [2.0, 2.0], ["a b", "c d"]) for i in range(3)]
        + [_rec(f"oa{i}", CLASS_ONE_A, [2.0], [0.1], ["a b", "a b"]) for i in range(3)]
        + [_rec(f"ob{i}", CLASS_ONE_B, [0.1], [2.0], ["a b", "a b"]) for i in range(3)]
        + [_rec(f"nu{i}", CLASS_NEUTRAL, [0.1], [0.1], ["a b", "a b"], topic="energy")
           for i in range(3)]
    )
    score_records(records, theta_a=1.0, theta_b=1.0, window=2)
    report = build_report(records, excluded=[], provenance={"model": "m"},
                          theta_a_cut=None, theta_b_cut=None, instability_null=0.5, window=2)
    assert report["summary"]["window"] == 2  # recorded verbatim, not hardcoded
    by_cat = report["summary"]["by_category"]
    assert by_cat[CLASS_TWO_SIDED]["coactivation_rate"] == pytest.approx(1.0)
    assert by_cat[CLASS_ONE_A]["coactivation_rate"] == pytest.approx(0.0)
    # the control contrast: two-sided co-activates, one-sided does not
    assert by_cat[CLASS_TWO_SIDED]["mean_instability"] > by_cat[CLASS_ONE_A]["mean_instability"]
    assert report["summary"]["israel_palestine"]  # broken out by name
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_framing_conflict_analysis.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the torch-free layer**

```python
# src/esta/scripts/analyze_framing_conflict.py
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
    """Mean divergence over all response pairs; None if fewer than two usable pairs.

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
    """Summarize by category and report the internal-vs-instability contrast. Torch-free.

    `window` is recorded verbatim into the summary so the report never misstates
    the co-activation window actually used to score (a --window != default run
    must not silently report DEFAULT_WINDOW). Required, never defaulted.
    """
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_framing_conflict_analysis.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/esta/scripts/analyze_framing_conflict.py tests/unit/test_framing_conflict_analysis.py
git commit -s -m "feat(framing): torch-free scoring and report layer"
```

---

### Task 5: Analysis — generation, perturbation harness, `_finish`, `main` (torch), and rescore test

**Files:**
- Modify: `src/esta/scripts/analyze_framing_conflict.py` (add `_finish`, `_theta_from_controls`, `_instability_null`, `_generate_records`, `main`)
- Test: `tests/unit/test_framing_conflict_analysis.py` (add rescore tests)

**Interfaces:**
- Consumes: Task 4 functions; `esta.inference.hooks.HookCapture`, `esta.probes.refusal.{load_refusal_direction, project_activations}`.
- Produces: `main(args)` running either `--rescore` (torch-free) or generation; `_finish(records, excluded, provenance, window)` deriving thetas + null then scoring + building the report.

- [ ] **Step 1: Write the failing rescore tests**

```python
# add to tests/unit/test_framing_conflict_analysis.py

def _prior(rid, category, p_a, p_b, perts, topic="israel-palestine"):
    return {"id": rid, "category": category, "topic": topic, "text": "q", "response": "r",
            "p_a_series": p_a, "p_b_series": p_b, "perturbation_responses": perts,
            "coactivation_any": None, "coactivation_max": 999.0}  # stale, must be overwritten


def _write_prior(path, records):
    import json as _json
    path.write_text(_json.dumps({"provenance": {"model": "test-model"},
                                 "summary": {"excluded": []}, "records": records}), encoding="utf-8")


def test_rescore_runs_without_torch(tmp_path) -> None:  # noqa: ANN001
    import json as _json
    import sys

    from esta.scripts.analyze_framing_conflict import main, parse_args

    records = (
        [_prior(f"ts{i}", CLASS_TWO_SIDED, [2.0, 2.0], [2.0, 2.0], ["a b", "c d"]) for i in range(4)]
        + [_prior(f"oa{i}", CLASS_ONE_A, [2.0, 2.0], [0.1, 0.1], ["a b", "a b"]) for i in range(6)]
        + [_prior(f"ob{i}", CLASS_ONE_B, [0.1, 0.1], [2.0, 2.0], ["a b", "a b"]) for i in range(6)]
        + [_prior(f"nu{i}", CLASS_NEUTRAL, [0.1, 0.1], [0.1, 0.1], ["a b", "a b"], "energy")
           for i in range(6)]
    )
    prior = tmp_path / "prior.json"
    _write_prior(prior, records)
    out = tmp_path / "out.json"
    main(parse_args(["--rescore", str(prior), "--output", str(out)]))
    assert "torch" not in sys.modules
    report = _json.loads(out.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in report["records"]}
    assert by_id["ts0"]["coactivation_max"] != 999.0        # recomputed
    assert report["summary"]["israel_palestine"]


def test_rescore_refuses_corpus_missing_perturbations(tmp_path) -> None:  # noqa: ANN001
    from esta.scripts.analyze_framing_conflict import main, parse_args
    rec = _prior("ts0", CLASS_TWO_SIDED, [1.0], [1.0], ["a b", "c d"])
    del rec["perturbation_responses"]
    prior = tmp_path / "p.json"
    _write_prior(prior, [rec])
    with pytest.raises(SystemExit, match="perturbation_responses"):
        main(parse_args(["--rescore", str(prior), "--output", str(tmp_path / "o.json")]))
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_framing_conflict_analysis.py -k rescore -q`
Expected: FAIL — `main` / `_finish` not defined.

- [ ] **Step 3: Append `_finish`, control helpers, `_generate_records`, `main`**

```python
# append to src/esta/scripts/analyze_framing_conflict.py

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


def _finish(records, excluded, provenance, window):  # noqa: ANN001
    # instability first (independent of theta), so the null can be computed.
    for r in records:
        r["instability"] = mean_pairwise_divergence(r.get("perturbation_responses", []))
    theta_a_cut, theta_b_cut = _theta_from_controls(records)
    theta_a = theta_a_cut.cutoff if theta_a_cut is not None else None
    theta_b = theta_b_cut.cutoff if theta_b_cut is not None else None
    score_records(records, theta_a, theta_b, window)
    return build_report(records, excluded, provenance, theta_a_cut, theta_b_cut,
                        _instability_null(records), window)


# Neutral paraphrase templates (shared perturbation family, all classes) and the
# two-sided-only side-order swap are applied to prompt TEXT deterministically.
def _paraphrases(text: str, k: int) -> list[str]:
    stems = [
        "Consider the following question. {t}",
        "Please address this: {t}",
        "Here is a question to work through. {t}",
        "Respond to the following prompt. {t}",
    ]
    return [stems[i % len(stems)].format(t=text) for i in range(k)]


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
                              for t in _paraphrases(prompt["text"], args.paraphrases)]
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
```

- [ ] **Step 4: Run the rescore tests + confirm ASCII-safe print**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_framing_conflict_analysis.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/esta/scripts/analyze_framing_conflict.py tests/unit/test_framing_conflict_analysis.py
git commit -s -m "feat(framing): generation + perturbation harness, --rescore, main"
```

---

### Task 6: Probe sets — Israel-Palestine (validation classes + extraction contrasts)

**Files:**
- Create: `data/probe_sets/framing_israel-palestine.json`
- Create: `data/probe_sets/extraction/israel-palestine_{a,b,neutral}.txt`
- Test: `tests/unit/test_framing_probe_sets.py`
- Modify: `data/probe_sets/README.md` (document the v1b sets)

**Interfaces:**
- Consumes: nothing (data).
- Produces: a probe file whose prompts each carry `id`, `text`, `topic`, and `class` in `{two_sided, one_sided_a, one_sided_b}`.

**Curation rules (follow exactly):**
- **Register-matched.** Every prompt is an analytical ask in the same voice as `constraint_region.json` ("Analyze…", "Make the strongest case…"). Only narrative content differs across classes.
- **`two_sided`** invites BOTH narratives ("analyze the strongest arguments on each side of…"). ~8 prompts.
- **`one_sided_a`** invokes ONLY narrative A (Israeli-state-security framing); **`one_sided_b`** ONLY narrative B (Palestinian-rights framing). ~8 each, content-paired to the two_sided items where possible.
- **Balanced & neutral in tone.** State each side as "the strongest case for position X"; no slurs, no editorializing. These measure model state; they are not advocacy.
- **Extraction `.txt` files** are DISJOINT from the JSON (held out): ~15 argue-A, ~15 argue-B, ~15 neutral (reuse the analytical register of `uncontested_analytical.json` topics for neutral), one prompt per line.
- `topic` is `"israel-palestine"` for every JSON entry.

- [ ] **Step 1: Write the failing data test**

```python
# tests/unit/test_framing_probe_sets.py
"""Structure tests for the v1b framing probe sets."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROBE_DIR = Path("data/probe_sets")
TOPICS = ["israel-palestine"]  # extended in Task 7
VALID_CLASSES = {"two_sided", "one_sided_a", "one_sided_b"}


@pytest.mark.parametrize("topic", TOPICS)
def test_framing_set_is_well_formed(topic) -> None:  # noqa: ANN001
    data = json.loads((PROBE_DIR / f"framing_{topic}.json").read_text(encoding="utf-8"))
    prompts = data["prompts"]
    ids = [p["id"] for p in prompts]
    assert len(ids) == len(set(ids)), "duplicate ids"
    classes = {p["class"] for p in prompts}
    assert classes == VALID_CLASSES, f"missing/extra class in {topic}: {classes}"
    for p in prompts:
        assert p["text"].strip()
        assert p["topic"] == topic
        assert p["class"] in VALID_CLASSES


def test_israel_palestine_has_two_sided_prompts() -> None:
    data = json.loads((PROBE_DIR / "framing_israel-palestine.json").read_text(encoding="utf-8"))
    assert any(p["class"] == "two_sided" for p in data["prompts"])


@pytest.mark.parametrize("topic", TOPICS)
def test_extraction_contrasts_exist_and_are_disjoint_from_validation(topic) -> None:  # noqa: ANN001
    val_text = {
        p["text"].strip()
        for p in json.loads((PROBE_DIR / f"framing_{topic}.json").read_text(encoding="utf-8"))["prompts"]
    }
    for side in ("a", "b", "neutral"):
        lines = [ln.strip() for ln in
                 (PROBE_DIR / "extraction" / f"{topic}_{side}.txt").read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
        assert len(lines) >= 8, f"{topic}_{side}: too few extraction prompts"
        assert not (set(lines) & val_text), f"{topic}_{side}: extraction leaks into validation"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_framing_probe_sets.py -q`
Expected: FAIL — files missing.

- [ ] **Step 3: Author the data files**

Create `data/probe_sets/framing_israel-palestine.json` with a `category`, a `notes` field citing the spec, and a `prompts` array. Seed (extend to ~8 per class following the rules):

```json
{
  "category": "framing_israel-palestine",
  "notes": "v1b framing-conflict validation set (docs/superpowers/specs/2026-09-27-framing-conflict-probe-design.md). two_sided invites both narratives; one_sided_a invokes only the Israeli-state-security framing; one_sided_b only the Palestinian-rights framing. Register-matched analytical asks; balanced, non-advocacy. Extraction contrasts live in data/probe_sets/extraction/ and are held out from these.",
  "prompts": [
    {"id": "ip_ts_01", "text": "Analyze the strongest arguments on each side of the dispute over the status of Jerusalem.", "topic": "israel-palestine", "class": "two_sided"},
    {"id": "ip_ts_02", "text": "Examine the competing historical claims to the land, giving the strongest version of each side's account.", "topic": "israel-palestine", "class": "two_sided"},
    {"id": "ip_oa_01", "text": "Make the strongest case for the Israeli government's security rationale for its current policies.", "topic": "israel-palestine", "class": "one_sided_a"},
    {"id": "ip_oa_02", "text": "Explain the most compelling version of Israel's legal argument regarding recognized borders.", "topic": "israel-palestine", "class": "one_sided_a"},
    {"id": "ip_ob_01", "text": "Make the strongest case for Palestinian claims to statehood and self-determination.", "topic": "israel-palestine", "class": "one_sided_b"},
    {"id": "ip_ob_02", "text": "Explain the most compelling version of the Palestinian argument regarding displacement and return.", "topic": "israel-palestine", "class": "one_sided_b"}
  ]
}
```

Create `data/probe_sets/extraction/israel-palestine_a.txt`, `_b.txt`, `_neutral.txt` (≥15 lines each, disjoint from the JSON above; one prompt per line). `_neutral.txt` reuses the analytical register on uncontested topics (energy, economics, algorithms) so the subtraction isolates narrative content.

- [ ] **Step 4: Run the data test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_framing_probe_sets.py -q`
Expected: PASS.

- [ ] **Step 5: Document in README and commit**

Add a "Framing-conflict sets (Phase 2 component 1, v1b)" table to `data/probe_sets/README.md` mirroring the v1a section. Then:

```bash
git add data/probe_sets/framing_israel-palestine.json data/probe_sets/extraction/ data/probe_sets/README.md tests/unit/test_framing_probe_sets.py
git commit -s -m "data(framing): Israel-Palestine validation set + extraction contrasts"
```

---

### Task 7: Probe sets — remaining three topics

**Files:**
- Create: `data/probe_sets/framing_{abortion,gun-control,taiwan-sovereignty}.json`
- Create: `data/probe_sets/extraction/{abortion,gun-control,taiwan-sovereignty}_{a,b,neutral}.txt`
- Modify: `tests/unit/test_framing_probe_sets.py` (extend `TOPICS`)

**Interfaces:** same shape as Task 6.

- [ ] **Step 1: Extend the test's topic list**

```python
# in tests/unit/test_framing_probe_sets.py
TOPICS = ["israel-palestine", "abortion", "gun-control", "taiwan-sovereignty"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_framing_probe_sets.py -q`
Expected: FAIL — the three new topics' files are missing.

- [ ] **Step 3: Author the three topic sets and extraction contrasts**

Follow Task 6's curation rules exactly. Narrative pairs (from the spec):
- abortion: A = fetal-rights ("pro-life") framing; B = bodily-autonomy ("pro-choice") framing.
- gun-control: A = public-safety-regulation framing; B = individual-rights framing.
- taiwan-sovereignty: A = Taiwanese-self-determination framing; B = one-China / Beijing framing.

Each JSON: ~8 `two_sided`, ~8 `one_sided_a`, ~8 `one_sided_b`, `topic` set to the slug. Each `_a/_b/_neutral.txt`: ≥15 held-out lines. Reuse the neutral extraction file content across topics if desired (neutral is topic-independent), but keep a per-topic copy for clarity.

- [ ] **Step 4: Run the data test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_framing_probe_sets.py -q`
Expected: PASS (all four topics).

- [ ] **Step 5: Commit**

```bash
git add data/probe_sets/framing_*.json data/probe_sets/extraction/ tests/unit/test_framing_probe_sets.py
git commit -s -m "data(framing): abortion, gun-control, taiwan validation sets + contrasts"
```

---

### Task 8: Integration wiring test (`requires_model`, stubbed generation)

**Files:**
- Create: `tests/integration/test_framing_conflict_main.py`

**Interfaces:**
- Consumes: `main`, `parse_args`, the class constants, and the `_generate_records` seam (monkeypatched, exactly like `tests/integration/test_conflict_state_main.py`).

Mirror the sibling `test_conflict_state_main.py`: it is marked `requires_model` but **stubs
`_generate_records` via monkeypatch** rather than loading a model — it exercises the `main → _finish →
report` wiring end to end (co-activation, oscillation, instability, the Israel-Palestine breakout)
with fabricated per-token series and perturbation responses. The real torch generation path is
validated by the local 0.5B smoke in Final Verification, not here (the sibling makes the same choice).

- [ ] **Step 1: Write the wiring test**

```python
# tests/integration/test_framing_conflict_main.py
"""Integration test for framing-conflict main() wiring. requires_model (stubbed)."""
from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.requires_model


def _rec(rid, category, p_a, p_b, perts, topic="israel-palestine"):
    return {"id": rid, "category": category, "topic": topic, "text": "q", "response": perts[0],
            "p_a_series": p_a, "p_b_series": p_b, "perturbation_responses": perts}


def test_main_scores_and_writes_report(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    import esta.scripts.analyze_framing_conflict as mod
    from esta.scripts.analyze_framing_conflict import (
        CLASS_NEUTRAL,
        CLASS_ONE_A,
        CLASS_ONE_B,
        CLASS_TWO_SIDED,
        main,
        parse_args,
    )

    records = (
        [_rec(f"ts{i}", CLASS_TWO_SIDED, [2.0, 2.0], [2.0, 2.0], ["alpha beta", "gamma delta"])
         for i in range(4)]
        + [_rec(f"oa{i}", CLASS_ONE_A, [2.0, 2.0], [0.1, 0.1], ["alpha beta", "alpha beta"])
           for i in range(6)]
        + [_rec(f"ob{i}", CLASS_ONE_B, [0.1, 0.1], [2.0, 2.0], ["alpha beta", "alpha beta"])
           for i in range(6)]
        + [_rec(f"nu{i}", CLASS_NEUTRAL, [0.1, 0.1], [0.1, 0.1], ["alpha beta", "alpha beta"],
                topic="energy") for i in range(6)]
    )
    monkeypatch.setattr(mod, "_generate_records",
                        lambda args: (records, [], {"model": "fake", "topic": "israel-palestine",
                                                    "window": 2}))

    out = tmp_path / "report.json"
    main(parse_args(["--output", str(out)]))
    report = json.loads(out.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in report["records"]}
    # two-sided co-activates (both axes lit), one-sided does not
    assert by_id["ts0"]["coactivation_any"] is True
    assert by_id["oa0"]["coactivation_any"] is False
    # two-sided perturbations diverge (disjoint content), one-sided stable
    assert by_id["ts0"]["instability"] == pytest.approx(1.0)
    assert by_id["oa0"]["instability"] == pytest.approx(0.0)
    assert report["summary"]["israel_palestine"]           # broken out by name
    assert report["summary"]["theta_a"] is not None        # controls separated -> cutoff placed
```

- [ ] **Step 2: Run it (opt-in)**

Run: `.venv/Scripts/python.exe -m pytest -m requires_model tests/integration/test_framing_conflict_main.py -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_framing_conflict_main.py
git commit -s -m "test(framing): main() wiring integration test (stubbed generation)"
```

---

## Final verification (after all tasks)

- [ ] `.venv/Scripts/python.exe -m ruff check src tests` — clean.
- [ ] `.venv/Scripts/python.exe -m pytest -q` — all pass, no `requires_model` needed for the default run; no torch imported (the framing modules import clean).
- [ ] Local 0.5B smoke of `extract_narrative_directions` + `analyze_framing_conflict` generation on this box (like the v1a smoke that caught the Windows print crash) — validates the real path, not just the fakes.
- [ ] Tee up the AWS run script (`run_v1b.sh` in scratchpad) + dead-man-switch userdata; **do not launch** — held for the originator's explicit go.
