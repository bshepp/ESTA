"""Inference subpackage.

All exports here are torch-dependent. Callers that need torch-free utilities
(e.g., the threshold logic in `esta.probes.thresholds` or metric assembly in
`esta.extraction`) should import those directly.
"""

from esta.inference.generation import (
    GenerationParams,
    GenerationResult,
    generate_with_epistemic_state,
)
from esta.inference.hooks import HookCapture, resolve_residual_layer
from esta.inference.model_state import ModelState

__all__ = [
    "GenerationParams",
    "GenerationResult",
    "HookCapture",
    "ModelState",
    "generate_with_epistemic_state",
    "resolve_residual_layer",
]
