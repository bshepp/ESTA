# Validation Cases

Known-state prompts used to characterize ESTA's epistemic_state metadata against expected behaviors. Per `docs/epistemic-transparency-agent (1).md` § Validation Methodology.

Each file is a JSON object with the shape:

```json
{
  "category": "<category_name>",
  "experimental": true,                  // optional; absent means stable
  "expected_state": { /* qualitative expectations per probe */ },
  "notes": "<curation notes, references, warnings>",
  "prompts": [
    {"id": "<unique_id>", "text": "<prompt>", "...": "..."}
  ]
}
```

| File | Category | Count | Expected confidence | Expected safety pressure | Expected conflict |
|------|----------|-------|---------------------|--------------------------|-------------------|
| `factual_unambiguous.json` | Single-answer facts | 50 | high | low | low |
| `refusal_expected.json`   | Trained refusals    | 50 | high | high | low |
| `low_confidence.json`     | Obscure / post-cutoff | 50 | low | low | low |
| `performed_uncertainty.json` | Empirically resolved but hedged in RLHF | 50 | high | low–moderate | moderate (Phase 2) |
| `conflict_region.json`    | EXPERIMENTAL — competing features | 6 | varies | moderate–high | high (Phase 2) |
| `dual_use_defensive.json` | EXPERIMENTAL — defensive framings of refusal-set topics | 50 | varies | *measurement target* | moderate |

## How these are consumed

- The calibration script `python -m esta.scripts.calibrate` loads these files, runs each prompt through the configured model, and computes empirical percentiles for the entropy / margin / refusal-projection thresholds.
- **Refusal-probe class assignment**: `calibrate.py:resolve_probe_class()` sorts each category into one of three pools. `expected_state.safety_pressure == "high"` → **harmful** (positive class, sets `pressure_moderate` from its lower tail); `== "low"` → **harmless** (negative class, sets `pressure_low` from its upper tail); anything else — an intermediate label, a missing label, or an explicit `"probe_class": "excluded"` — is **held out of both projection pools**. Excluded categories still contribute per-token entropies and margins to the confidence thresholds; only their projections are withheld. Held-out categories are printed by the script and recorded in the output under `provenance.probe_classes`.
- **Why exclusion matters**: both pressure thresholds are *tail* statistics, so a category placed in the wrong pool distorts them out of proportion to its size. Pooling an elevated-pressure category with the negatives inflates `pressure_low`, which widens the `"low"` band and makes the probe under-report pressure — the false-assurance direction. A file whose expected pressure is genuinely intermediate is neither a clean positive nor a clean negative and must be excluded rather than assigned.
- **Adding a category**: set `probe_class` explicitly unless the category is unambiguously all-benign or all-refused. An unrecognized value is a hard error, not a silent fallback.
- The validation harness (planned) runs each prompt and asserts that the produced `epistemic_state` is consistent with `expected_state`. Mismatches are reported as a calibration table, not as test failures, since these are qualitative expectations.

## Curation guidelines

- **One prompt per row**, with a stable `id` like `<category>_<NNN>`.
- **Don't paraphrase**: if a prompt's phrasing matters (e.g., for hedging detection), document why in `notes`.
- **Post-cutoff items in `low_confidence.json`**: verify the date is past the *chosen model's* training cutoff. Qwen 2.5 has data through approximately mid-2024.
- **`conflict_region.json` is experimental** and requires extra care; see the file's own `notes` field.
- **`dual_use_defensive.json` is paired**: each `defensive` prompt carries a `pair_id` naming its `refusal_expected` counterpart on the same topic, so the analysis is a per-pair projection *delta*, not a distribution comparison. Keep pairs one-to-one, and keep `probe_class` at `excluded` — these prompts are what the probe is being measured against, so they can define neither class.

## What these are NOT

- Not training data. ESTA does not fine-tune. These prompts are evaluation only.
- Not jailbreak attempts. The `refusal_expected` set is for calibrating the refusal-direction probe by measuring its response on prompts the model is **supposed to** refuse.
- Not an exhaustive corpus. The four non-experimental sets were expanded to 50 prompts each (2026-07-22) to meet the production-calibration floor; `conflict_region.json` stays at its starter size until a maintainer with the relevant domain knowledge curates it, per that file's own `notes`.
