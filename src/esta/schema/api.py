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
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
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
