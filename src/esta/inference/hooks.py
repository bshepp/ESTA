"""Activation hooks for capturing residual-stream state during generation.

Architecture-agnostic across the HF decoder families ESTA supports
(Llama, Qwen2, Mistral, Phi3 — all use `model.model.layers[i]`).
"""

from __future__ import annotations

import torch

# Common paths to the residual-stream block list on HF decoder models.
_RESIDUAL_PATHS = (
    "model.layers",        # Llama, Qwen2, Mistral, Phi3
    "transformer.h",       # GPT-2, GPT-Neo, Falcon
    "gpt_neox.layers",     # Pythia, GPT-NeoX
)


def _follow_path(obj: object, path: str) -> object | None:
    for attr in path.split("."):
        if not hasattr(obj, attr):
            return None
        obj = getattr(obj, attr)
    return obj


def resolve_residual_layer(model: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
    """Return the residual-stream block at `layer_idx` for an HF causal LM.

    Tries known paths in order. Raises ValueError if none match or the index
    is out of range.
    """
    for path in _RESIDUAL_PATHS:
        layers = _follow_path(model, path)
        if layers is None:
            continue
        try:
            return layers[layer_idx]  # type: ignore[index]
        except IndexError as e:
            raise ValueError(
                f"Layer {layer_idx} out of range on {type(model).__name__} at path {path!r}"
            ) from e
    raise ValueError(
        f"No known residual-stream path on {type(model).__name__}. "
        f"Tried {_RESIDUAL_PATHS}."
    )


class HookCapture:
    """Captures the last-token residual stream from a target layer on each forward pass.

    During autoregressive generation, the forward hook fires once per decoded
    token. We keep the last position only — that's the residual state used to
    decide the next token.
    """

    def __init__(self) -> None:
        self.activations: list[torch.Tensor] = []
        self._handle = None

    def _hook_fn(self, module, _input, output):  # noqa: ANN001
        hidden = output[0] if isinstance(output, tuple) else output
        # (batch, seq, hidden) -> (batch, hidden) at the last position.
        self.activations.append(hidden[:, -1, :].detach().cpu().float())

    def attach(self, model: torch.nn.Module, layer_idx: int) -> None:
        target_layer = resolve_residual_layer(model, layer_idx)
        self._handle = target_layer.register_forward_hook(self._hook_fn)

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> HookCapture:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.detach()
