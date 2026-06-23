"""Integration test for the calibrate.py model-run loop. requires_model.

Run on a box with [model] + weights:
    pytest -m requires_model tests/integration/test_calibrate_main.py

Generates a tiny refusal direction on-the-fly by calling the module-level
`extract_refusal_direction` function directly (the script's `main()` takes no
argv, so it cannot be driven programmatically).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_model

TINY = "Qwen/Qwen2.5-0.5B-Instruct"
TINY_LAYER = 6


def test_calibrate_main_writes_valid_calibration(tmp_path: Path) -> None:
    import torch

    from esta.inference.model_state import ModelState
    from esta.scripts.calibrate import main, parse_args
    from esta.scripts.extract_refusal_direction import (
        DEFAULT_HARMFUL_PROMPTS,
        DEFAULT_HARMLESS_PROMPTS,
        extract_refusal_direction,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Load the tiny model once to extract a refusal direction.
    model_state = ModelState(
        model_name=TINY,
        device=device,
        dtype=dtype,
        refusal_direction_path=None,
    )
    model_state.load()

    refusal_path = tmp_path / "refusal.pt"
    direction = extract_refusal_direction(
        model_state.model,
        model_state.tokenizer,
        DEFAULT_HARMFUL_PROMPTS,
        DEFAULT_HARMLESS_PROMPTS,
        layer_idx=TINY_LAYER,
        device=device,
    )
    torch.save(direction, refusal_path)

    out = tmp_path / "calibration.json"
    main(parse_args([
        "--validation-dir", "data/validation_cases",
        "--model", TINY,
        "--refusal-direction", str(refusal_path),
        "--refusal-layer", str(TINY_LAYER),
        "--output", str(out),
    ]))

    data = json.loads(out.read_text(encoding="utf-8"))

    # Well-formedness checks — pressure separation is model-quality dependent
    # on a 0.5B model with few prompts; main() already warns on inversion.
    assert set(data) >= {
        "spike_threshold",
        "low_margin_threshold",
        "pressure_low",
        "pressure_moderate",
        "provenance",
    }, f"Missing keys in calibration output: {set(data)}"
    assert isinstance(data["spike_threshold"], float), "spike_threshold must be float"
    assert isinstance(data["low_margin_threshold"], float), "low_margin_threshold must be float"
    assert isinstance(data["pressure_low"], float), "pressure_low must be float"
    assert isinstance(data["pressure_moderate"], float), "pressure_moderate must be float"
    assert data["provenance"]["model"] == TINY, (
        f"provenance.model mismatch: {data['provenance'].get('model')!r} != {TINY!r}"
    )
