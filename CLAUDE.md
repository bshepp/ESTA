# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What ESTA is

A local, self-hosted wrapper around an open-weights LLM that returns the usual OpenAI-compatible
chat completion **plus** an `epistemic_state` block describing the internal state under which the
response was generated (token confidence, refusal-direction projection, provenance). It does not
block, filter, or modify responses — it only exposes state. The full spec is
`docs/epistemic-transparency-agent (1).md`; the README is the user-facing overview.

## Commands

```bash
pip install -e .[dev]          # core + test tooling, NO torch — this is what CI uses
pip install -e .[model]        # add torch/transformers/accelerate — needed only to run the server

ruff check src tests           # lint (run before every push; CI runs the same)
pytest -q                      # unit tests; requires_model/requires_gpu deselected by default
pytest tests/unit/test_audit_chain.py::test_name   # single test
pytest -m requires_model tests/integration/        # opt-in; needs [model] + weights, ~minutes on CPU/8GB GPU
```

Run the server (needs `[model]` + an extracted refusal direction):

```bash
python -m esta.scripts.extract_refusal_direction --model Qwen/Qwen2.5-7B-Instruct --layer 14 --output data/refusal_direction.pt
$env:ESTA_MODEL = "Qwen/Qwen2.5-7B-Instruct"   # PowerShell; bash uses `export`
uvicorn esta.api.server:app --port 8000
```

`ESTA_*` env vars (see `.env.example` / `src/esta/api/server.py`): `ESTA_MODEL`, `ESTA_DEVICE`,
`ESTA_REFUSAL_DIR`, `ESTA_REFUSAL_LAYER`, `ESTA_AUDIT_DIR`, `ESTA_CALIBRATION`.

Produce a calibration, then check what the refusal probe is actually responding to (both need
`[model]` + a refusal direction; outputs are gitignored):

```bash
python -m esta.scripts.calibrate --model Qwen/Qwen2.5-7B-Instruct --refusal-direction data/refusal_direction.pt --refusal-layer 14 --output data/calibration.json
python -m esta.scripts.analyze_dual_use --model Qwen/Qwen2.5-7B-Instruct --refusal-direction data/refusal_direction.pt --refusal-layer 14 --calibration data/calibration.json --output data/dual_use_analysis.json
python -m esta.scripts.analyze_performed_uncertainty --model Qwen/Qwen2.5-7B-Instruct --output data/performed_uncertainty_analysis.json
python -m esta.scripts.analyze_response_fidelity --model Qwen/Qwen2.5-7B-Instruct --refusal-direction data/refusal_direction.pt --calibration data/calibration.json --output data/response_fidelity_analysis.json
```

## Architecture: the torch / no-torch boundary

The single most important structural fact. CI and the default `pytest` run install **without**
`[model]`, so torch is absent. Code is split so the numeric logic is torch-free and unit-tested,
and torch is quarantined behind the inference layer:

- **Torch-free (must stay importable without torch):** `esta.extraction`, `esta.calibration`, `esta.confidence.metrics`,
  `esta.probes.thresholds`, `esta.schema.*`, `esta.audit.logger`, `esta.hedging`, `esta.fidelity`, and everything except
  the model-run function in `esta.scripts.calibrate`, `esta.scripts.analyze_dual_use`, `esta.scripts.analyze_performed_uncertainty`, and
  `esta.scripts.analyze_response_fidelity` (each imports torch *inside* the function that runs the
  model — `main()`, or `_generate_records()` in the analyze_performed_uncertainty and analyze_response_fidelity scripts —
  so the modules stay CI-importable; both --rescore paths run entirely torch-free).
- **Torch-dependent:** `esta.inference.*` (`generation`, `hooks`, `model_state`),
  `esta.probes.refusal`, `esta.api.server`, `esta.scripts.extract_refusal_direction`.

When adding code: keep numeric/metric logic in the torch-free modules and pass it numpy arrays /
Python floats. The torch side converts at the boundary (e.g. `generation.py` log-softmaxes
`outputs.scores` to numpy before calling `extract_metrics`). Tests that need torch must import it
lazily inside the test/fixture (see `tests/integration/test_smoke_tiny_model.py`) so they don't
drag torch into the default run, and must be marked `@pytest.mark.requires_model`.

Request flow: `api/server.py` → `inference.generation.generate_with_epistemic_state` (torch:
tokenize, attach residual hook, `model.generate`, project activations) → `extraction.extract_metrics`
(numpy: entropy/margin aggregation + pressure labeling) → pydantic `EpistemicState` → response,
with the record also written to the audit log.

## Invariants that have dedicated guards

- **Schema drift.** `src/esta/schema/epistemic_state.schema.json` is the canonical on-disk contract
  and must match the pydantic models. Any change to `schema/epistemic_state.py` requires
  `python -m esta.scripts.dump_schema` + committing the regenerated JSON, or
  `tests/unit/test_schema_drift.py` fails. Adding fields also requires bumping `SCHEMA_VERSION`
  (Phase 2 takes it to `0.2.0`) and a migration note. The prose contract is `docs/SCHEMA.md`,
  guarded separately by `tests/unit/test_schema_reference.py` — a new field must be documented
  there too, or that test fails.
- **Audit integrity.** `audit/logger.py` hash-chains JSONL records (SHA-256, daily rotation,
  `verify_chain()`). The chain is *locally verifiable only* — this is a documented limitation, not
  a bug to "fix" by adding local-only hardening; external anchoring is deferred to Phase 3.
- **No false precision.** Default thresholds in `confidence/metrics.py` and `probes/thresholds.py`
  are deliberate placeholders. When the refusal probe is absent, `safety_pressure` returns the
  explicit stub (`calibrated_pressure="uncalibrated"`, `probe_version="not_loaded"`); uncalibrated
  metrics must stay labeled as such rather than presented as calibrated. Real thresholds come from
  running `esta.scripts.calibrate` (fully implemented, needs `[model]` + a refusal direction)
  against `data/validation_cases/` and serving the resulting JSON via `ESTA_CALIBRATION`.

## Conventions

- Commits require DCO sign-off: `git commit -s` (CONTRIBUTING.md).
- Ruff config in `pyproject.toml` (line-length 100, `E501` ignored). Target Python 3.11+.
- Don't commit model weights, refusal-direction tensors, or calibration sets — they live in
  gitignored `data/`.
- Default model target is `Qwen/Qwen2.5-7B-Instruct`; integration smoke tests use the tiny
  `Qwen/Qwen2.5-0.5B-Instruct`. Dev box is Windows + 8GB GPU (smoke only); heavy runs go to AWS.
- Residual-layer resolution is architecture-agnostic via `inference/hooks.py:resolve_residual_layer`
  (known HF decoder paths); extend that function to support new model families.
