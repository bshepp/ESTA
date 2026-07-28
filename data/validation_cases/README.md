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
| `benign_instructional.json` | Neutral how-to / analytic requests | 50 | high | low | low |
| `refusal_expected.json`   | Trained refusals    | 50 | high | high | low |
| `low_confidence.json`     | Obscure / post-cutoff | 50 | low | low | low |
| `performed_uncertainty.json` | Empirically resolved but hedged in RLHF | 50 | high | low–moderate | moderate (Phase 2) |
| `conflict_region.json`    | EXPERIMENTAL — competing features | 6 | varies | moderate–high | high (Phase 2) |
| `dual_use_defensive.json` | EXPERIMENTAL — defensive framings of refusal-set topics | 50 | varies | *measurement target* | moderate |

## How these are consumed

- The calibration script `python -m esta.scripts.calibrate` loads these files, runs each prompt through the configured model, and computes empirical percentiles for the entropy / margin / refusal-projection thresholds.
- **Refusal-probe class assignment**: `calibrate.py:resolve_probe_class()` sorts each category into one of three pools. `expected_state.safety_pressure == "high"` → **harmful** (positive class, sets `pressure_moderate` from its lower tail); `== "low"` → **harmless** (negative class, sets `pressure_low` from its upper tail); anything else — an intermediate label, a missing label, or an explicit `"probe_class": "excluded"` — is **held out of both projection pools**. Excluded categories still contribute per-token entropies and margins to the confidence thresholds; only their projections are withheld. Held-out categories are printed by the script and recorded in the output under `provenance.probe_classes`.
- **Why exclusion matters**: both pressure thresholds are set from the *facing edges* of the two class distributions, so a category placed in the wrong pool distorts them out of proportion to its size — a single elevated category in the harmless pool moves the edge, and with it the boundary. A file whose expected pressure is genuinely intermediate is neither a clean positive nor a clean negative and must be excluded rather than assigned. This also keeps the dual-use set out of the fit: its labels are a *measured outcome* of the calibration, and tuning the threshold to make them come out nicely would be fitting to the thing being measured.
- **Adding a category**: set `probe_class` explicitly unless the category is unambiguously all-benign or all-refused. An unrecognized value is a hard error, not a silent fallback.
- The validation harness (planned) runs each prompt and asserts that the produced `epistemic_state` is consistent with `expected_state`. Mismatches are reported as a calibration table, not as test failures, since these are qualitative expectations.

## Curation guidelines

- **One prompt per row**, with a stable `id` like `<category>_<NNN>`.
- **Don't paraphrase**: if a prompt's phrasing matters (e.g., for hedging detection), document why in `notes`.
- **Post-cutoff items in `low_confidence.json`**: verify the date is past the *chosen model's* training cutoff. Qwen 2.5 has data through approximately mid-2024.
- **`conflict_region.json` is experimental** and requires extra care; see the file's own `notes` field.
- **`dual_use_defensive.json` is paired**: each `defensive` prompt carries a `pair_id` naming its `refusal_expected` counterpart on the same topic, so the analysis is a per-pair projection *delta*, not a distribution comparison. Keep pairs one-to-one, and keep `probe_class` at `excluded` — these prompts are what the probe is being measured against, so they can define neither class.
- **`benign_instructional.json` must stay vocabulary-neutral**: it is the matched control that separates *alarming wording* from *how-to phrasing*, and one leaked word ("kill", "hack", "blow up") collapses that comparison. A unit test enforces the word list. Match `register` when comparing — `everyday` against the `benign_lexical` items, `professional` against the defensive framings.
- **Harmless-class composition changes the answer — and breadth alone is not a fix.** The harmless class defines where "benign" ends, so its composition moves `pressure_low` directly. (These runs predate the `max-margin` policy; `pressure_low` was then the 95th percentile of the harmless class.) Two runs on Qwen2.5-0.5B with a layer-6 direction:
  - *Narrow class* (factual recall only, 2026-07-27): `pressure_low` 1.140 against a factual maximum of 1.150 — the threshold sat at its own class ceiling, `pressure_moderate` was 1.694, and 100% of should-answer dual-use prompts were labeled moderate/high.
  - *Broadened class* (adding `benign_instructional`, 2026-07-28): `pressure_low` rose to **1.992, above `pressure_moderate` at 1.694** — an inversion, correctly rejected at load.

  - *Same broadened class at 7B* (Qwen2.5-7B-Instruct, layer 14, direction extracted from 200 AdvBench + 200 Alpaca prompts held out from this corpus, 2026-07-28): **valid, and separated by a wide margin** — `pressure_low` 9.76, `pressure_moderate` 24.22. Extraction separation was 22.89 (harmful mean 27.44 vs harmless 4.55) against 0.51 on the 0.5B.

  **The 0.5B inversion was an artifact of a weakly identified direction, not a property of the broadened class.** The lesson that survives is narrower but still holds: a comfortable-looking gap is evidence about the harmless class as much as about the probe, so re-check separation whenever that class changes. What the 7B run adds is that the *composition* effect is real and measurable — `factual_unambiguous` projects at mean 3.41 while `benign_instructional` projects at 8.69, so a harmless class of pure factual recall sets `pressure_low` far below where ordinary benign traffic actually sits.

## What these are NOT

- Not training data. ESTA does not fine-tune. These prompts are evaluation only.
- Not jailbreak attempts. The `refusal_expected` set is for calibrating the refusal-direction probe by measuring its response on prompts the model is **supposed to** refuse.
- Not an exhaustive corpus. The non-experimental sets hold 50 prompts each (expanded 2026-07-22; `benign_instructional.json` added 2026-07-28); `conflict_region.json` stays at its starter size until a maintainer with the relevant domain knowledge curates it, per that file's own `notes`.
