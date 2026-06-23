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
    ESTA_CALIBRATION    path to calibration.json produced by esta.scripts.calibrate
                        (unset = serve uncalibrated; set = must be valid or startup fails)
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

import torch
from fastapi import FastAPI, HTTPException

from esta.audit import AuditLogger
from esta.calibration import Calibration, load_calibration
from esta.inference import GenerationParams, ModelState, generate_with_epistemic_state
from esta.schema import (
    SCHEMA_VERSION,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    EpistemicState,
    ModelInfo,
    Provenance,
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
_calibration_env = os.environ.get("ESTA_CALIBRATION")
CALIBRATION_PATH = Path(_calibration_env) if _calibration_env else None

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
calibration: Calibration = Calibration.uncalibrated()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global calibration
    state.load()
    calibration = load_calibration(CALIBRATION_PATH, MODEL_NAME)
    log.info(
        "Calibration: %s (id=%s)",
        "calibrated" if calibration.calibrated else "uncalibrated",
        calibration.calibration_id,
    )
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
        "calibrated": calibration.calibrated,
        "calibration_id": calibration.calibration_id,
    }


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """Minimal OpenAI-compatible model list so standard clients can discover the model."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "esta",
            }
        ],
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

    try:
        result = generate_with_epistemic_state(
            model_state=state,
            prompt=prompt,
            params=GenerationParams(
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            ),
            refusal_layer=REFUSAL_HOOK_LAYER,
            calibration=calibration,
        )
    except Exception as exc:  # noqa: BLE001 — full detail is logged; client gets a clean 500.
        log.exception("Generation failed for request %s", request_id)
        raise HTTPException(status_code=500, detail="Generation failed") from exc

    audit_record = {
        "request_id": request_id,
        "model": MODEL_NAME,
        "prompt": prompt,
        "response": result.response_text,
        "confidence": result.confidence.model_dump(),
        "safety_pressure": result.safety_pressure.model_dump(),
        "calibration": result.calibration.model_dump(),
        "calibration_path": str(CALIBRATION_PATH) if CALIBRATION_PATH else None,
        "debug": result.debug_info if req.return_activations else None,
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
        confidence=result.confidence,
        safety_pressure=result.safety_pressure,
        calibration=result.calibration,
        provenance=provenance,
    )

    return ChatCompletionResponse(
        id=request_id,
        created=created,
        model=req.model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=result.response_text),
            )
        ],
        epistemic_state=epistemic_state,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
