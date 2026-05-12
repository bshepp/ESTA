"""Generate a response while capturing the epistemic-state metrics.

Wraps the torch / HuggingFace generation path. The metric-assembly logic
proper lives in `esta.extraction.extract_metrics`, which is pure numpy and
tested without torch installed. This module is responsible for:

  1. Tokenizing the prompt.
  2. Attaching the residual-stream hook at the refusal layer.
  3. Running model.generate(...) with output_scores=True.
  4. Converting torch scores to numpy log-probabilities.
  5. Projecting captured activations onto the refusal direction.
  6. Calling extract_metrics() with the numerical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812 — universal PyTorch convention

from esta.extraction import extract_metrics
from esta.inference.hooks import HookCapture
from esta.inference.model_state import ModelState
from esta.probes.refusal import project_activations
from esta.schema import ConfidenceMetrics, SafetyPressure


@dataclass(frozen=True)
class GenerationParams:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95


@dataclass
class GenerationResult:
    response_text: str
    confidence: ConfidenceMetrics
    safety_pressure: SafetyPressure
    debug_info: dict[str, Any]


def generate_with_epistemic_state(
    model_state: ModelState,
    prompt: str,
    params: GenerationParams,
    refusal_layer: int,
) -> GenerationResult:
    """Run generation and return the response plus structured state metrics."""
    if model_state.model is None or model_state.tokenizer is None:
        raise RuntimeError("model_state.load() must be called before generation")

    tokenizer = model_state.tokenizer
    model = model_state.model
    device = model_state.device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    with HookCapture() as hook:
        if model_state.refusal_probe_loaded:
            hook.attach(model, refusal_layer)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=params.max_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                do_sample=params.temperature > 0,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
            )

    generated_ids = outputs.sequences[0, input_len:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Convert torch scores -> numpy log-probabilities at the boundary.
    token_log_probs = [
        F.log_softmax(step_scores[0], dim=-1).float().cpu().numpy()
        for step_scores in outputs.scores
    ]

    projections: list[float] = []
    if model_state.refusal_probe_loaded and hook.activations:
        projections = project_activations(hook.activations, model_state.refusal_direction)

    confidence, safety, debug_info = extract_metrics(
        token_log_probs=token_log_probs,
        projections=projections,
        probe_loaded=model_state.refusal_probe_loaded,
        refusal_layer=refusal_layer,
    )

    # Augment debug_info with generation-shape info that only the torch side
    # has access to (token counts).
    debug_info["input_tokens"] = int(input_len)
    debug_info["generated_tokens"] = int(len(generated_ids))

    return GenerationResult(
        response_text=response_text,
        confidence=confidence,
        safety_pressure=safety,
        debug_info=debug_info,
    )
