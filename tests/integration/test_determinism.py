"""Identical input must reproduce identical epistemic state. requires_model.

    pytest -m requires_model tests/integration/test_determinism.py

This covers the MVP deliverable "deterministic reproduction of state metadata
given the same input" (docs/epistemic-transparency-agent (1).md).

Why it matters more here than in a normal API: every response is written to a
hash-chained audit log and is meant to be reviewable after the fact. If the
same prompt produced drifting metrics, an auditor could not distinguish "the
model behaved differently" from "the measurement is noisy", and the audit trail
would not support the claims made for it.

Determinism is asserted only for greedy decoding (temperature=0). Sampling is
deliberately not covered — it is nondeterministic by construction, and pinning
it would test the seed rather than the system.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.requires_model

TINY = "Qwen/Qwen2.5-0.5B-Instruct"
LAYER = 6
PROMPT = "What is the boiling point of water at standard atmospheric pressure?"

# Floating-point accumulation can reorder between runs on some backends, so
# exact bitwise equality is too strong a contract to promise. Anything looser
# than this, though, would let a real regression through.
TOL = 1e-9


@pytest.fixture(scope="module")
def state():
    import torch

    from esta.inference.model_state import ModelState

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ms = ModelState(
        model_name=TINY,
        device=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        refusal_direction_path=None,
    )
    ms.load()
    return ms


def _generate(state, prompt: str):
    from esta.calibration import Calibration
    from esta.inference import GenerationParams, generate_with_epistemic_state

    chat = state.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    return generate_with_epistemic_state(
        model_state=state,
        prompt=chat,
        params=GenerationParams(max_tokens=32, temperature=0.0),
        refusal_layer=LAYER,
        calibration=Calibration.uncalibrated(),
    )


def test_same_prompt_reproduces_same_text_and_metrics(state) -> None:
    first = _generate(state, PROMPT)
    second = _generate(state, PROMPT)

    assert first.response_text == second.response_text

    a, b = first.confidence, second.confidence
    for field in (
        "mean_entropy",
        "median_entropy",
        "p90_entropy",
        "max_entropy",
        "mean_margin",
        "low_margin_fraction",
    ):
        assert getattr(a, field) == pytest.approx(getattr(b, field), rel=TOL, abs=TOL), field
    assert a.entropy_spike_count == b.entropy_spike_count


def test_per_token_series_are_reproduced_not_just_their_aggregates(state) -> None:
    """Aggregates can coincide while the underlying series differs."""
    first = _generate(state, PROMPT)
    second = _generate(state, PROMPT)

    for key in ("raw_entropies", "raw_margins"):
        x, y = first.debug_info[key], second.debug_info[key]
        assert len(x) == len(y), key
        for i, (u, v) in enumerate(zip(x, y, strict=True)):
            assert u == pytest.approx(v, rel=TOL, abs=TOL), f"{key}[{i}]"


def test_different_prompts_do_not_collapse_to_the_same_state(state) -> None:
    """Guards the above: a stubbed pipeline would pass determinism trivially."""
    factual = _generate(state, PROMPT)
    obscure = _generate(
        state, "What was the exact attendance at the 1923 county fair in Ackley, Iowa?"
    )

    assert factual.response_text != obscure.response_text
    assert factual.confidence.mean_entropy != pytest.approx(
        obscure.confidence.mean_entropy, rel=TOL, abs=TOL
    )
