"""ModelState: holds the loaded LM, tokenizer, and refusal probe.

Single-process; not threadsafe. The server's lifespan handler instantiates one
ModelState at startup and reuses it across requests.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from esta.probes import load_refusal_direction

log = logging.getLogger(__name__)


class ModelState:
    def __init__(
        self,
        model_name: str,
        device: str,
        dtype: torch.dtype,
        refusal_direction_path: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.refusal_direction_path = refusal_direction_path

        self.tokenizer: AutoTokenizer | None = None
        self.model: AutoModelForCausalLM | None = None
        self.refusal_direction: torch.Tensor | None = None

    @property
    def refusal_probe_loaded(self) -> bool:
        return self.refusal_direction is not None

    def load(self) -> None:
        log.info("Loading tokenizer: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        log.info("Loading model: %s on %s (dtype=%s)", self.model_name, self.device, self.dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
        )
        # Equivalent to .eval(); using .train(False) for clarity.
        self.model.train(False)

        if self.refusal_direction_path and self.refusal_direction_path.exists():
            log.info("Loading refusal direction from %s", self.refusal_direction_path)
            self.refusal_direction = load_refusal_direction(
                self.refusal_direction_path, device=self.device
            )
        else:
            log.warning(
                "Refusal direction not found at %s. "
                "Safety pressure metrics will be uncalibrated until "
                "scripts/extract_refusal_direction.py is run.",
                self.refusal_direction_path,
            )
