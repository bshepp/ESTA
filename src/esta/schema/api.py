"""OpenAI-compatible request/response DTOs, plus the ESTA `epistemic_state` extension.

Standard OpenAI clients ignore unknown fields, so the extra `epistemic_state`
field on the response is invisible to them. ESTA-aware clients can read it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from esta.schema.epistemic_state import EpistemicState


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "local"
    # At least one message is required; an empty list would otherwise produce a
    # degenerate chat-template prompt and fail deep in generation with a 500.
    messages: list[ChatMessage] = Field(..., min_length=1)
    # Bounds give callers a clean 422 instead of a torch error from a negative or
    # absurd value. The ceilings are generous, not model limits.
    max_tokens: int = Field(512, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    # ESTA extension: include the raw per-token entropies/margins/projections in the
    # response for research/debug. Kept as a request body field rather than a query
    # param so it travels with the rest of the configuration.
    return_activations: bool = False


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    epistemic_state: EpistemicState = Field(
        ..., description="ESTA extension; standard OpenAI clients ignore unknown fields."
    )
