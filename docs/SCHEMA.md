# `epistemic_state` schema reference

**Version 0.1.1.** The canonical machine-readable contract is
[`src/esta/schema/epistemic_state.schema.json`](../src/esta/schema/epistemic_state.schema.json),
generated from the pydantic models in
[`src/esta/schema/epistemic_state.py`](../src/esta/schema/epistemic_state.py). This page is the
prose reference for what the fields *mean* and how to read them.

Regenerate the JSON after any model change — do not hand-edit it:

```bash
python -m esta.scripts.dump_schema
```

`tests/unit/test_schema_drift.py` fails if the committed JSON and the pydantic models disagree,
and `tests/unit/test_schema_reference.py` fails if a field exists that this page does not
document.

## Versioning

`schema_version` is present on every response. Through Phase 1, additive changes bump the patch
digit; Phase 2 (conflict detection and feature attribution) takes the schema to `0.2.0`. Adding a
field requires bumping `SCHEMA_VERSION`, regenerating the JSON, and a migration note.

| Version | Change |
|---|---|
| `0.1.0` | Initial Phase 1 fields. |
| `0.1.1` | Adds the top-level `calibration` block. Additive — `0.1.0` consumers that ignore unknown keys keep working. |

**Consume defensively.** Treat unknown fields as ignorable and check `schema_version` before
relying on anything version-specific.

## Shape

`epistemic_state` (the `EpistemicState` model) is an extra top-level key on an otherwise standard
OpenAI chat-completion response. Standard clients ignore it.

```
epistemic_state   (EpistemicState)
├── schema_version   str
├── model            ModelInfo
├── confidence       ConfidenceMetrics
├── safety_pressure  SafetyPressure
├── calibration      CalibrationInfo
└── provenance       Provenance
```

---

## `model` — ModelInfo

Which model produced the response.

| Field | Type | Notes |
|---|---|---|
| `name` | `str` | HuggingFace model id, e.g. `Qwen/Qwen2.5-7B-Instruct`. |
| `revision` | `str \| null` | Model revision when pinned. Currently always `null`; reserved. |
| `quantization` | `str` | The torch dtype in use, e.g. `bfloat16` or `float32`. Named "quantization" for historical reasons; it reports dtype, not a quantization scheme. |

## `confidence` — ConfidenceMetrics

Aggregates over the **generated** tokens only; prompt tokens are not included. Entropy is in
**nats** (natural log), not bits. An empty generation yields all zeros.

| Field | Type | Notes |
|---|---|---|
| `mean_entropy` | `float` | Mean per-token entropy, `-Σ p log p`. Higher means the model was spreading probability across more alternatives. |
| `median_entropy` | `float` | Median per-token entropy. Less sensitive than the mean to a few uncertain tokens. |
| `p90_entropy` | `float` | 90th percentile per-token entropy. |
| `max_entropy` | `float` | Highest single-token entropy in the sequence. |
| `mean_margin` | `float` | Mean gap between the top-1 and top-2 log-probabilities, in nats. Small means the top two candidates were near-equivalent. |
| `low_margin_fraction` | `float` | Fraction of tokens whose top1−top2 gap fell below the low-margin threshold. Range 0–1. |
| `entropy_spike_count` | `int` | Number of tokens whose entropy exceeded the spike threshold. |
| `calibrated_confidence` | `float \| null` | **Always `null` in 0.1.1.** Reserved for a fitted 0–1 confidence score; nothing populates it yet. Do not treat `null` here as "low confidence" — it means "not implemented". |

The thresholds behind `low_margin_fraction` and `entropy_spike_count` come from the active
calibration, or from documented placeholders when none is loaded. Both are therefore only
comparable across responses sharing a `calibration.calibration_id`.

## `safety_pressure` — SafetyPressure

Derived from projecting the residual stream onto a refusal direction extracted per
[Arditi et al. (2024)](https://arxiv.org/abs/2406.11717).

| Field | Type | Notes |
|---|---|---|
| `refusal_projection_max` | `float` | Largest projection across generated tokens. This is the value the pressure label is computed from. `0.0` when no probe is loaded. |
| `refusal_projection_mean` | `float` | Mean projection across generated tokens. May be negative. `0.0` when no probe is loaded. |
| `calibrated_pressure` | `"low" \| "moderate" \| "high" \| "uncalibrated"` | Banded label. See the honesty contract below. |
| `probe_version` | `str` | Identifies the extraction procedure, e.g. `arditi_v1_unrefined`. The literal `not_loaded` means no probe was available. |
| `layer` | `int` | Residual-stream layer the projection was taken from. Projections from different layers are not comparable. |

**Projection units are arbitrary.** The refusal direction is unit-norm, but the projection scale
depends on the model, the layer, and the prompt set used during extraction. A value of 11 means
nothing on its own. Never compare raw projections across models, layers, or calibrations — compare
only within a single `calibration_id`, and prefer the banded label for cross-run reasoning.

## `calibration` — CalibrationInfo

Which threshold set governed this response. Present since `0.1.1`.

| Field | Type | Notes |
|---|---|---|
| `calibrated` | `bool` | Whether empirical thresholds were in use. When `false`, `calibrated_pressure` is `"uncalibrated"`. |
| `calibration_id` | `str \| null` | 12-character SHA-256 prefix of the calibration file. Stable for identical content, so it identifies the exact thresholds. `null` when uncalibrated. |
| `calibrated_at` | `str \| null` | When the calibration was computed. |
| `model` | `str \| null` | Model the calibration was computed against. The server refuses to start if this disagrees with the model being served. |
| `source` | `str \| null` | Calibration filename. |

## `provenance` — Provenance

| Field | Type | Notes |
|---|---|---|
| `timestamp` | `str` | ISO-8601 UTC generation time. |
| `request_id` | `str` | Correlates the response with its audit record. |
| `audit_log_path` | `str` | Path of the hash-chained JSONL log holding the full record. |

---

## The honesty contract

The system reports a real pressure label **only** when it can back one. Three states, and
consumers must handle all three:

| Probe loaded | Calibration loaded | `refusal_projection_*` | `calibrated_pressure` | `probe_version` |
|---|---|---|---|---|
| no | — | `0.0` | `uncalibrated` | `not_loaded` |
| yes | no | real measurement | `uncalibrated` | real version |
| yes | yes | real measurement | `low` / `moderate` / `high` | real version |

The middle row is deliberate: a projection without calibrated thresholds is a number with no
scale, so the raw measurement is surfaced while the *label* is withheld. **`uncalibrated` never
means "low".** Treating it as low is the false-assurance failure this project exists to prevent.

## Band boundaries

`calibrated_pressure` compares `refusal_projection_max` against two thresholds:

- **`pressure_low`** — below it, `low`. When the harmless and harmful calibration classes are
  separable, this sits at the midpoint of the empty band between them (`max-margin` policy). If
  they overlap, it falls back to the 95th percentile of the harmless class, and an inverted result
  is rejected at load rather than served.
- **`pressure_moderate`** — at or above it, `high`. The 10th percentile of the harmful class.

Between them is `moderate`: the genuinely ambiguous region.

### Measured reference points

Qwen 2.5 7B Instruct, layer 14, direction from 200 held-out AdvBench + 200 Alpaca prompts
(2026-07-28). Thresholds `pressure_low` 13.08, `pressure_moderate` 24.22.

| Prompt class | Mean `refusal_projection_max` | Typical label |
|---|---:|---|
| Unambiguous factual questions | 3.4 | `low` |
| Ordinary how-to / analytic requests | 8.7 | `low` |
| Defensive framings of harmful topics | 11.6 | `low` (14% read `moderate`) |
| Requests the model refuses | 27.6 | `high` |

These are orientation for your own calibration, **not** defaults to copy. They do not transfer to
another model, layer, or extraction set.

## Stability

For a fixed model and greedy decoding (`temperature=0`), identical input reproduces identical
metadata — the response text, the aggregates, and the underlying per-token series. Guarded by
`tests/integration/test_determinism.py`. Under sampling, metadata varies with the output, as it
should.

## What this does not tell you

- **Not a correctness signal.** Confident-looking metrics accompany hallucinations. Low entropy
  means the model was internally decided, not that it was right.
- **Not a refusal predictor.** `moderate` does not mean the model declined. In the 7B run above it
  answered every prompt in the defensive set while a portion still read `moderate`.
- **Not comparable across calibrations.** Two responses are only quantitatively comparable when
  their `calibration_id` matches.
- **Not a safety control.** ESTA never blocks, filters, or alters responses. Any gating is the
  downstream consumer's decision.
