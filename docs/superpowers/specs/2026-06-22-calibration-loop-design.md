# Design: Close the calibration loop (Phase 1)

**Date:** 2026-06-22
**Status:** Approved (design); implementation pending
**Schema impact:** `SCHEMA_VERSION` 0.1.0 → 0.1.1 (additive: new top-level `calibration` block)

## Problem

Phase 1 emits `epistemic_state` using **placeholder** thresholds and never consumes
calibration output, producing a false-precision leak:

- `confidence/metrics.py` defines `DEFAULT_SPIKE_THRESHOLD = 4.0` and
  `DEFAULT_LOW_MARGIN_THRESHOLD = 0.5` (documented guesses).
- `probes/thresholds.py` defines `DEFAULT_PRESSURE_THRESHOLDS = (0.5, 1.5)` (guesses).
- `inference/generation.py` calls `extract_metrics()` **without** passing any of the
  three threshold params, so all three always use the placeholders.
- Nothing reads `data/calibration.json`. `scripts/calibrate.py` can *write* it
  (via the tested `compute_calibration`), but its `main()` model-run path is a
  `NotImplementedError` stub, and no runtime code *loads* it.
- Consequence: when a refusal direction is loaded but thresholds are still
  placeholders, the response reports `calibrated_pressure: "low"/"moderate"/"high"`
  **as if calibrated**. The honest `"uncalibrated"` value only appears when the
  probe is entirely absent.

This violates the project's core "no false precision" invariant (CLAUDE.md), which
is the whole reason ESTA exists. The spec's "Notes for the Implementing Agent"
sequence calibration as the Phase 1 closure before Phase 2.

## Goal

Close the loop end to end:

1. Implement the torch model-run loop in `calibrate.py:main()` so a real
   `calibration.json` can be produced (run on AWS; the dev box is smoke-only).
2. Load and validate that calibration at server startup.
3. Thread the calibrated thresholds through generation into `extract_metrics`.
4. Make every field honest about whether it is calibrated, and surface calibration
   provenance in the response and audit record.

Non-goals: Phase 2 work (conflict probe, SAE, performed-uncertainty,
response-fidelity); Phase 3 *domain* calibration (a different concept — input-topic
reliability, not threshold calibration). The `calibrated_confidence` 0–1 score stays
`null`; it is a separate future signal, not threshold calibration.

## Approach

Chosen: **load-once value object, explicit injection**, matching the codebase's
existing dependency-injection + torch-free-pure-function pattern. `extract_metrics`
already accepts `pressure_thresholds`, `spike_threshold`, and `low_margin_threshold`
— they are simply never passed today; this wires them.

Rejected alternatives: loading inside `extract_metrics` (pushes file IO into the
pure-numpy boundary); mutable module-level globals (hidden coupling, untestable).

## Components

### 1. New torch-free module `src/esta/calibration.py`

Must stay importable without torch (CI imports it). Adds it to the torch-free list
in CLAUDE.md.

- `CalibrationError(Exception)` — raised on any invalid configured calibration.
- `Calibration` (frozen dataclass):
  - thresholds: `spike`, `low_margin`, `pressure_low`, `pressure_moderate`
  - provenance: `calibrated: bool`, `calibration_id: str | None`,
    `calibrated_at: str | None`, `model: str | None`, `source: str | None`
  - `pressure_thresholds` property → `PressureThresholds(pressure_low, pressure_moderate)`
  - `uncalibrated()` classmethod → placeholder-backed instance with
    `calibrated=False` (thresholds = the documented defaults, so confidence counts
    still compute; pressure labeling is gated separately — see §4).
- `load_calibration(path: Path | None, serving_model: str) -> Calibration`:
  - `path` unset/`None` → `Calibration.uncalibrated()` (a legitimate state).
  - else read the JSON written by `CalibrationOutput.to_json()`
    (`{spike_threshold, low_margin_threshold, pressure_low, pressure_moderate, provenance}`).
  - `calibration_id = sha256(canonical_json)[:12]`.
  - **Validate, raising `CalibrationError` (fail loud) on:**
    - malformed JSON / missing keys
    - inversion: `pressure_low >= pressure_moderate` (overlapping harmful/harmless
      distributions ⇒ probe not separable ⇒ unusable)
    - model mismatch: `provenance.model != serving_model`
  - `source` = basename of the file (not full path, to avoid leaking server FS layout;
    consistent with not over-sharing infra detail).

### 2. Schema change — `src/esta/schema/epistemic_state.py`

- Bump `SCHEMA_VERSION` to `"0.1.1"`.
- New model:
  ```python
  class CalibrationInfo(BaseModel):
      calibrated: bool
      calibration_id: str | None = None
      calibrated_at: str | None = None
      model: str | None = None
      source: str | None = None
  ```
- Add `calibration: CalibrationInfo` to `EpistemicState` (top-level, because the
  calibration governs both confidence and pressure thresholds — not specific to
  `safety_pressure`).
- Regenerate `epistemic_state.schema.json` via `python -m esta.scripts.dump_schema`
  and commit it (satisfies `test_schema_drift`).
- Add a migration note (in this spec's Migration section and a one-line CHANGELOG-style
  note in the schema module docstring).

### 3. Threading — `extraction.py`, `generation.py`, `server.py`

- `extract_metrics(...)` gains a `calibration: Calibration` parameter. It uses
  `calibration.spike`, `calibration.low_margin`, and `calibration.pressure_thresholds`
  for the metric computations, builds the `CalibrationInfo` block from the
  `Calibration`, and applies the three-way honesty branch (§4). Returns
  `(ConfidenceMetrics, SafetyPressure, CalibrationInfo, debug_info)`.
- `generate_with_epistemic_state(...)` gains a `calibration: Calibration` parameter
  and forwards it; returns the `CalibrationInfo` in its result.
- `server.py`: load calibration once at startup (`lifespan`) from `ESTA_CALIBRATION`,
  store alongside `state`; pass to generation; place the returned `CalibrationInfo`
  into the `EpistemicState`; include calibration provenance (incl. full path +
  `calibration_id`) in the audit record (audit dict is free-form, not schema-guarded).

### 4. Honesty semantics

| state | `calibration.calibrated` | `calibrated_pressure` | `refusal_projection_max/mean` | `entropy_spike_count` / `low_margin_fraction` |
|---|---|---|---|---|
| probe absent | false | `"uncalibrated"` | 0.0 (no measurement) | placeholder thresholds |
| probe loaded, not calibrated | false | `"uncalibrated"` | **real measured values** | placeholder thresholds |
| probe loaded and calibrated | true | real label (low/moderate/high) | real values | calibrated thresholds |

- A real pressure *label* requires `probe_loaded AND calibration.calibrated`. Otherwise
  `"uncalibrated"`. This is the core honesty fix.
- (a) Approved: surface raw `refusal_projection_max/mean` even when uncalibrated
  (middle row) — it is a real measurement, explicitly marked uncalibrated by the label;
  hiding it loses signal. (Today the stub zeroes it.)
- (b) Approved: `entropy_spike_count` / `low_margin_fraction` keep computing against the
  documented placeholder thresholds when uncalibrated — they are descriptive counts whose
  names make no calibration claim; the `calibration` block carries the truth.
- `probe_version`: `"not_loaded"` when probe absent; the probe id (e.g.
  `"arditi_v1_unrefined"`) when loaded; unchanged by calibration (calibration identity
  is carried by `calibration.calibration_id`).
- `calibrated_confidence` stays `null` (separate Phase 2+ signal).

### 5. `calibrate.py:main()` model-run loop (torch, `requires_model`)

Replace the `NotImplementedError`. Reuse the existing generation path rather than
re-implementing forward/hook logic:

1. `load_validation_set(validation_dir)` (exists). The refusal direction is
   **required** for pressure calibration; `main()` errors if it is absent.
2. Load `ModelState` (model + tokenizer + refusal direction) and the refusal layer.
3. For each prompt, call `generate_with_epistemic_state` with greedy decoding
   (temperature 0) for reproducibility, and read its `GenerationResult.debug_info`
   (which always carries `raw_entropies`, `raw_margins`, `raw_projections` — the
   `return_activations` flag only gates the audit record, not computation).
4. Aggregate to match the statistics each threshold governs at runtime:
   - **entropies / margins** are pooled **per-token** across every prompt
     (`spike_threshold` = p95 of per-token entropy; `low_margin_threshold` = p10 of
     per-token margin — the same per-token series `aggregate_confidence` counts over).
   - **projections** reduce to **one value per prompt — that prompt's `max`
     projection** — because the runtime labels on `refusal_projection_max`. Each
     prompt's max projection joins the harmful or harmless pool, classified by the
     validation file's `expected_state.safety_pressure` (`"high"` → harmful;
     otherwise harmless), giving `pressure_low` = p95 of harmless maxes and
     `pressure_moderate` = p10 of harmful maxes.
5. Call the existing, tested `compute_calibration(...)`; attach provenance
   (`_build_provenance`, which already records model/dir/percentiles/timestamp);
   write with `CalibrationOutput.to_json()` to `--output`.

Only the loop is new torch code; the percentile math is already covered by
`tests/unit/test_calibration.py`.

### 6. Config & failure policy

- New env var **`ESTA_CALIBRATION`** — path to `calibration.json`. Unset → uncalibrated.
  Documented in `.env.example`, README ("Run the server"), and CLAUDE.md (env var list +
  torch-free module list).
- **Fail loud** at startup on malformed / inverted / model-mismatched calibration
  (`CalibrationError` propagates out of `lifespan`, aborting startup). Absent file is a
  legitimate, honestly-labeled uncalibrated state, not an error.

## Testing

**Torch-free unit tests (run in CI):**
- `load_calibration`: happy path returns populated `Calibration`; raises
  `CalibrationError` on inversion, model-mismatch, malformed JSON, missing keys;
  unset path → `uncalibrated()`; `calibration_id` is stable for identical content.
- `extract_metrics`: three-way honesty branch — asserts `calibrated_pressure`,
  `refusal_projection_*`, and `calibration.calibrated` for each of the three states;
  calibrated thresholds change `entropy_spike_count` / `low_margin_fraction` vs defaults.
- Schema: `CalibrationInfo` defaults + round-trip; `EpistemicState` carries the block;
  `test_schema_drift` passes against the regenerated JSON; `SCHEMA_VERSION == "0.1.1"`.

**`requires_model` (run on AWS, deselected by default):**
- `calibrate.py:main()` produces a well-formed `calibration.json` with no inversion on
  `data/validation_cases/`.
- Server loads that file and emits `calibration.calibrated == true` with a real
  pressure label.

## Migration (0.1.0 → 0.1.1)

Additive, backward-compatible: existing consumers ignore the new `calibration` block.
Producers must populate it. No field removed or renamed. Regenerated
`epistemic_state.schema.json` is the canonical artifact. Audit records written before
0.1.1 simply lack the block.

## File-by-file change summary

- **new** `src/esta/calibration.py` — `Calibration`, `load_calibration`, `CalibrationError`
- `src/esta/schema/epistemic_state.py` — `CalibrationInfo`, `EpistemicState.calibration`, version bump
- `src/esta/schema/epistemic_state.schema.json` — regenerated
- `src/esta/schema/__init__.py` — export `CalibrationInfo`
- `src/esta/extraction.py` — `calibration` param, `CalibrationInfo` build, honesty branch
- `src/esta/inference/generation.py` — `calibration` param, return `CalibrationInfo`
- `src/esta/api/server.py` — load calibration at startup, env var, inject, audit provenance
- `src/esta/scripts/calibrate.py` — implement `main()` model-run loop
- `.env.example`, `README.md`, `CLAUDE.md` — `ESTA_CALIBRATION`, torch-free module note
- **tests** — `tests/unit/test_calibration_loader.py` (new), extend `test_extraction.py`,
  `test_schema.py`/`test_schema_drift.py`; `requires_model` additions in integration
