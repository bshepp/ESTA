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
