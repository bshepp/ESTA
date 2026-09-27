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

| Class | mean confidence | mean hedge (instrument v2) | answer accuracy |
|---|---|---|---|
| `binary_settled` | 0.923 | 0.000 (0/50 nonzero) | 47/50 |
| `binary_obscure` | 0.741 | 0.274 (33/50 nonzero) | n/a — no answer key by construction |

**The sets did their job.** On the confidence axis they bracket as intended (AUC 0.81, obscure
below settled). On the hedging axis the first marker list sat at chance (AUC 0.56), scoring 41 of
50 obscure responses at zero while the model visibly hedged in them ("I would need to look up
specific records", "specific details are not publicly available"); rebuilding the list against
these two classes — deferral phrasings drawn from `binary_obscure`, false positives checked
against `binary_settled`, the positive class never consulted — brought the pair to AUC 0.830
with the settled class perfectly clean.

The 17 obscure responses still scoring zero under v2 are genuine: the model **confidently
confabulates** rather than hedges on them. That is model behaviour the set correctly elicits, and
it means no hedge instrument can fully separate these classes on this model — a finding, not a
defect.

The balance rule earned its keep — the model answered obscure prompts 18 yes / 32 no rather than
defaulting to one polarity, so the confidence numbers are not an artifact of a stuck responder.
Two confident errors on `binary_settled` (`settled_024`, `settled_041`) were checked against their
keys and are genuine model errors, not curation defects.

## Response-fidelity sets (Phase 2 component 4)

| File | Role | Expected behavior |
|---|---|---|
| `reframe_pairs.json` | harness corpus | borderline ask + predicted substitute; convergence flags empirical reframes |
| `direct_answer_controls.json` | negative control | direct answers; instrument stays quiet |
| `benign_vague.json` | negative control (anchored) | raw distortion may fire; anchored signal must not |
| `refusal_boundary.json` | boundary map | refusals score distortion ≈ 0 via the topic gate |

Term groups (`topic_groups`, `operative_groups`) are curated data, reviewed like prompts:
synonym allowances live here, not in code. A prompt without non-empty groups cannot be scored
and fails the structure tests. Derived sets record `derived_from`; source files in
`data/validation_cases/` are never modified.

## Conflict-probe sets (Phase 2 component 1, v1a)

| File | Role | Expected axes |
|---|---|---|
| `constraint_region.json` | positive | high refusal AND high reasoning → conflict fires |
| `uncontested_analytical.json` | negative control + θ_eng upper class | high reasoning, low refusal |
| `direct_recall.json` | negative control + θ_eng lower class | neither axis |

The refusal-bait negative (`refusal_boundary.json`, high refusal / low reasoning) is reused. Only
`constraint_region` should fire conflict; if `refusal_boundary` fires, the probe is detecting
refusal, not conflict. The reasoning *direction* is extracted from held-out prompts built into
`extract_reasoning_direction.py`, kept distinct from these calibration/validation classes.

## Framing-conflict sets (Phase 2 component 1, v1b)

Israel-Palestine is the first of four topics
(`docs/superpowers/specs/2026-09-27-framing-conflict-probe-design.md`); Israel-Palestine is built
and validated first, then the pattern scales to abortion, gun-control, and taiwan-sovereignty.

| File | Role | Expected axes |
|---|---|---|
| `framing_israel-palestine.json` (`two_sided`) | positive | both narrative directions co-active; the model may be torn |
| `framing_israel-palestine.json` (`one_sided_a`) | control | only the Israeli-state-security direction; stable under perturbation |
| `framing_israel-palestine.json` (`one_sided_b`) | control | only the Palestinian-rights direction; stable under perturbation |
| `uncontested_analytical.json` | easy negative (reused) | neither narrative direction |

The one-sided classes are content-paired to the two_sided prompts where possible (same sub-topic —
Jerusalem, the barrier, settlements, the blockade, refugee return — argued from one side only), so
the two_sided-vs-one_sided contrast isolates narrative co-activation rather than topic choice. Both
one-sided classes are curated with equal care and equal strength of framing: neutral, analytical
register throughout ("Analyze the strongest arguments on each side of...", "Make the strongest case
for..."), no slurs, no editorializing, and no side favored in the `notes`. These prompts measure a
model's internal state on a contested topic; they are not advocacy.

`data/probe_sets/extraction/israel-palestine_{a,b,neutral}.txt` are the held-out contrast prompts
that `extract_narrative_directions.py` uses to build the two narrative directions
(`r_A` = Israeli-state-security, `r_B_perp` = Palestinian-rights orthogonalized against A). They are
disjoint from the validation prompts above (enforced by
`tests/unit/test_framing_probe_sets.py`) and register-matched: `_neutral.txt` reuses the same
analytical-ask register on uncontested topics (energy, finance, algorithms, ...) so the
mean-difference against neutral isolates narrative content rather than sentence form. Thresholds
(theta_A, theta_B) and the instability null are placed from the `one_sided` controls only; the
`two_sided` positives never touch calibration.
