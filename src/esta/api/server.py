"""ESTA FastAPI server with the OpenAI-compatible /v1/chat/completions endpoint.

Run:
    uvicorn esta.api.server:app --host 0.0.0.0 --port 8000

Test:
    curl -X POST http://localhost:8000/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model":"local","messages":[{"role":"user","content":"Hello"}]}'

Configuration (environment variables):
    ESTA_MODEL          HF model id (default: Qwen/Qwen2.5-7B-Instruct)
    ESTA_DEVICE         cuda or cpu (default: auto)
    ESTA_REFUSAL_DIR    path to refusal_direction.pt (default: ./data/refusal_direction.pt)
    ESTA_REFUSAL_LAYER  layer index for residual-stream extraction (default: 14)
    ESTA_AUDIT_DIR      audit log directory (default: ./audit_logs)
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 — universal PyTorch convention
from fastapi import FastAPI, HTTPException

from esta.audit import AuditLogger
from esta.confidence import aggregate_confidence, token_entropy_and_margin
from esta.inference import HookCapture, ModelState
from esta.probes import (
    DEFAULT_PROBE_VERSION,
    label_pressure,
    project_activations,
)
from esta.schema import (
    SCHEMA_VERSION,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ConfidenceMetrics,
    EpistemicState,
    ModelInfo,
    Provenance,
    SafetyPressure,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("ESTA_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DEVICE = os.environ.get("ESTA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
REFUSAL_DIRECTION_PATH = Path(os.environ.get("ESTA_REFUSAL_DIR", "data/refusal_direction.pt"))
REFUSAL_HOOK_LAYER = int(os.environ.get("ESTA_REFUSAL_LAYER", "14"))
AUDIT_LOG_DIR = Path(os.environ.get("ESTA_AUDIT_DIR", "audit_logs"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("esta.api")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

state = ModelState(
    model_name=MODEL_NAME,
    device=DEVICE,
    dtype=DTYPE,
    refusal_direction_path=REFUSAL_DIRECTION_PATH,
)
audit = AuditLogger(AUDIT_LOG_DIR)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate_with_state(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, ConfidenceMetrics, SafetyPressure, dict[str, Any]]:
    assert state.model is not None and state.tokenizer is not None

    inputs = state.tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs.input_ids.shape[1]

    with HookCapture() as hook:
        if state.refusal_probe_loaded:
            hook.attach(state.model, REFUSAL_HOOK_LAYER)

        with torch.no_grad():
            outputs = state.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=state.tokenizer.pad_token_id,
            )

    generated_ids = outputs.sequences[0, input_len:]
    response_text = state.tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Per-token confidence: convert torch logits -> numpy log_probs at the boundary.
    entropies: list[float] = []
    margins: list[float] = []
    for step_scores in outputs.scores:
        log_probs = F.log_softmax(step_scores[0], dim=-1).float().cpu().numpy()
        e, m = token_entropy_and_margin(log_probs)
        entropies.append(e)
        margins.append(m)

    confidence = aggregate_confidence(entropies, margins)

    # Safety pressure: refusal-direction projection over captured activations.
    projections: list[float] = []
    if state.refusal_probe_loaded and hook.activations:
        projections = project_activations(hook.activations, state.refusal_direction)
        proj_arr = np.asarray(projections)
        safety = SafetyPressure(
            refusal_projection_max=float(np.max(proj_arr)),
            refusal_projection_mean=float(np.mean(proj_arr)),
            calibrated_pressure=label_pressure(float(np.max(proj_arr))),
            probe_version=DEFAULT_PROBE_VERSION,
            layer=REFUSAL_HOOK_LAYER,
        )
    else:
        safety = SafetyPressure(
            refusal_projection_max=0.0,
            refusal_projection_mean=0.0,
            calibrated_pressure="uncalibrated",
            probe_version="not_loaded",
            layer=REFUSAL_HOOK_LAYER,
        )

    debug_info: dict[str, Any] = {
        "input_tokens": int(input_len),
        "generated_tokens": int(len(generated_ids)),
        "raw_entropies": entropies,
        "raw_margins": margins,
        "raw_projections": projections,
    }

    return response_text, confidence, safety, debug_info


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    state.load()
    yield


app = FastAPI(
    title="ESTA - Epistemic State Transparency Agent",
    description="Local LLM with internal-state metadata for high-assurance use cases.",
    version=SCHEMA_VERSION,
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_loaded": state.model is not None,
        "refusal_probe_loaded": state.refusal_probe_loaded,
        "device": DEVICE,
        "schema_version": SCHEMA_VERSION,
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    if state.model is None or state.tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_id = f"esta-{uuid.uuid4().hex}"
    created = int(time.time())

    messages_dict = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt = state.tokenizer.apply_chat_template(
        messages_dict, tokenize=False, add_generation_prompt=True
    )

    response_text, confidence, safety, debug_info = _generate_with_state(
        prompt=prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )

    audit_record = {
        "request_id": request_id,
        "model": MODEL_NAME,
        "prompt": prompt,
        "response": response_text,
        "confidence": confidence.model_dump(),
        "safety_pressure": safety.model_dump(),
        "debug": debug_info if req.return_activations else None,
    }
    audit_log_path = audit.write(audit_record)

    provenance = Provenance(
        timestamp=datetime.now(UTC).isoformat(),
        request_id=request_id,
        audit_log_path=audit_log_path,
    )

    epistemic_state = EpistemicState(
        model=ModelInfo(
            name=MODEL_NAME,
            quantization=str(DTYPE).replace("torch.", ""),
        ),
        confidence=confidence,
        safety_pressure=safety,
        provenance=provenance,
    )

    return ChatCompletionResponse(
        id=request_id,
        created=created,
        model=req.model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=response_text),
            )
        ],
        epistemic_state=epistemic_state,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
