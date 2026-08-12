# Probe sets

Evaluation sets for Phase 2 probes. **Deliberately separate from `data/validation_cases/`.**

`esta.scripts.calibrate` globs `data/validation_cases/*.json`, so any file added there changes
Phase 1 calibration as a side effect. These sets exist to *evaluate* a probe, not to calibrate
one, and the two corpora must not be coupled.

| File | Role | Expected state |
|---|---|---|
| `binary_settled.json` | control | confident, unhedged |
| `binary_obscure.json` | control | unconfident, hedged |

The positive class for the performed-uncertainty detector remains
`data/validation_cases/performed_uncertainty.json`, which is also consumed by calibration.

## Why these are derived rather than authored fresh

Every prompt in `performed_uncertainty.json` is yes/no-shaped; every prompt in
`low_confidence.json` and `factual_unambiguous.json` is a wh-question. Using the latter as
controls would vary question *form* alongside the property being measured, and
"yes/no questions have lower first-token entropy than open questions" would be
indistinguishable from a real finding. These sets hold the source content fixed and change only
the form, so form is controlled rather than confounded. Each prompt records its `derived_from`
id.

## Curation rules

- **Keep `binary_settled` balanced** between `yes` and `no` expected answers. An all-yes set is
  passed by a model that always says yes.
- **Keep `binary_obscure` genuinely unresolvable** — thresholds should be chosen so neither
  answer is inferable from general knowledge. If a prompt becomes answerable, it belongs in
  `binary_settled` instead.
- Do not add these files to `data/validation_cases/`.

## How they behaved on Qwen 2.5 7B (2026-08-12)

| Class | mean confidence | mean hedge | answer accuracy |
|---|---|---|---|
| `binary_settled` | 0.923 | 0.013 | 47/50 |
| `binary_obscure` | 0.741 | 0.049 | n/a — no answer key by construction |

**The sets did their job; the hedge instrument did not.** On the confidence axis they bracket as
intended (AUC 0.81, obscure below settled), which is what a working control pair looks like. On
the hedging axis they sit at chance (AUC 0.56) — not because the prompts are wrong, but because
`hedging.py` scores 41 of 50 obscure responses at exactly zero while the model is visibly hedging
in them ("I would need to look up specific records", "specific details are not publicly
available"). `binary_obscure` is therefore the validation target for any hedge-measure revision:
until the control separates, the measure is not ready to be pointed at the positive class.

The balance rule earned its keep — the model answered obscure prompts 18 yes / 32 no rather than
defaulting to one polarity, so the confidence numbers are not an artifact of a stuck responder.
Two confident errors on `binary_settled` (`settled_024`, `settled_041`) were checked against their
keys and are genuine model errors, not curation defects.
