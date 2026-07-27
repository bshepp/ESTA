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

## How these are consumed

- The calibration script `python -m esta.scripts.calibrate` loads these files, runs each prompt through the configured model, and computes empirical percentiles for the entropy / margin / refusal-projection thresholds.
- **Refusal-probe class assignment**: `calibrate.py` pools a category's per-prompt max projection into the *harmful* class when its `expected_state.safety_pressure` is exactly `"high"`, and into the *harmless* class otherwise. Categories with an ambiguous middle label (`low_to_moderate`, `moderate_to_high`) therefore land in the harmless class today, which inflates the harmless upper tail that `pressure_low` is drawn from. Keep this in mind before enlarging `performed_uncertainty.json` or `conflict_region.json`.
- The validation harness (planned) runs each prompt and asserts that the produced `epistemic_state` is consistent with `expected_state`. Mismatches are reported as a calibration table, not as test failures, since these are qualitative expectations.

## Curation guidelines

- **One prompt per row**, with a stable `id` like `<category>_<NNN>`.
- **Don't paraphrase**: if a prompt's phrasing matters (e.g., for hedging detection), document why in `notes`.
- **Post-cutoff items in `low_confidence.json`**: verify the date is past the *chosen model's* training cutoff. Qwen 2.5 has data through approximately mid-2024.
- **`conflict_region.json` is experimental** and requires extra care; see the file's own `notes` field.

## What these are NOT

- Not training data. ESTA does not fine-tune. These prompts are evaluation only.
- Not jailbreak attempts. The `refusal_expected` set is for calibrating the refusal-direction probe by measuring its response on prompts the model is **supposed to** refuse.
- Not an exhaustive corpus. The four non-experimental sets were expanded to 50 prompts each (2026-07-22) to meet the production-calibration floor; `conflict_region.json` stays at its starter size until a maintainer with the relevant domain knowledge curates it, per that file's own `notes`.
