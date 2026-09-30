# Framing-Conflict Probe (v1b: two competing narratives) — Design

**Status:** design; not yet implemented
**Phase:** 2, component 1 (of 4) — sub-build **v1b** of three (v1a shipped; v1c later)
**Date:** 2026-09-27

## Goal

Detect the phenomenon v1a **could not**: a contested-topic state where the model is pulled between
two competing *content framings* (narratives) at once, rather than between refusal and reasoning.
v1a established, on Qwen 2.5 7B, that analytically-framed contested prompts (Israel-Palestine and
similar) **do not raise the refusal axis** (0/25 crossed θ_ref), so any tension there is not
refusal-vs-engagement — it is narrative-vs-narrative. v1b tests whether that tension exists and is
detectable, and — the ambitious part the originator asked for — whether it is *real tension* (the
model genuinely torn) rather than healthy both-sides synthesis.

Ship it as an **offline research capability** (scripts plus a validation report), like every other
Phase 2 detector. Nothing enters `epistemic_state` until validated: `SCHEMA_VERSION` stays `0.1.1`.

## What carries over from v1a, and what is genuinely new

v1a's parent spec (`2026-08-18-conflict-state-probe-design.md`) claimed the conflict probe would be
built to "take a list of `(direction_a, direction_b, θ_a, θ_b)` pairs" so v1b would be "new
data/config, not new architecture." **That is not what shipped.** `analyze_conflict_state.py`
hardcodes exactly two directions (refusal + reasoning). So v1b **does** need code — but the
*numeric core* is already reusable:

- **Reused as-is:** `esta.conflict.windowed_conflict_aggregates` is already generic over any two
  per-token series and two thresholds (that is what the v2 windowed rescore, 2026-08-28, delivered).
  Co-activation of two narrative directions needs **no new scoring math**.
- **Reused as-is:** `esta.conflict.orthogonalize` / `cosine_similarity` (the separability
  machinery), and the `extract_reasoning_direction.py` extraction pattern.
- **Reused as-is:** `esta.fidelity.convergence` (Jaccard content-word overlap) — `1 − convergence`
  is the response-divergence measure for perturbation-instability. `youden_cutoff` /
  `mann_whitney_p` from `analyze_performed_uncertainty` for significance-gated thresholds.
- **New:** a narrative-direction extraction script; a torch-free oscillation metric; a
  generation-side two-narrative hook loop **plus a perturbation harness**; a convergence analysis
  that ties the three signals together; and four topic probe sets.

## Three signals, two of them independent — and why that matters

The single design risk is **circularity**: if "the model is torn" is measured from the same internal
projections as "two narratives are co-active," a correlation between them proves nothing. v1b
therefore separates an *internal* signal (co-activation + oscillation) from an *independent
behavioral* signal (perturbation-instability), and the scientific claim is that the first predicts
the second.

The two internal signatures capture **different modes** of being torn — and are nearly mutually
exclusive per token, so together they cover both:

1. **Co-activation** (integrated tension) — both narratives held strongly *at the same time*.
   `min(s_A, s_B⊥) ≥ 1` within a token window, via the v2 `windowed_conflict_aggregates`. Reused.
2. **Oscillation** (sequential vacillation) — the model *swings* between narratives across the
   response. Among "engaged" tokens (at least one axis lit, `max(s_A, s_B⊥) ≥ 1`), the sign of
   `s_A(t) − s_B⊥(t)` flips back and forth. Oscillation = (sign changes) / (engaged tokens − 1),
   in `[0, 1]`; `None` if fewer than two engaged tokens. New, torch-free, free via `--rescore`.
3. **Perturbation-instability** (independent ground truth) — re-ask each prompt under
   logically-equivalent reframings; instability = mean pairwise response-divergence
   (`1 − esta.fidelity.convergence`) across the responses. Greedy decoding, so divergence comes
   *only* from the framing change, not sampling noise. Fully independent of the internal directions.
   Two perturbation families, kept separate so the class comparison stays honest:
   - **Neutral paraphrase (shared, all classes)** — K≈3 meaning-preserving rewordings of the ask,
     applied to `two_sided`, `one_sided_a`, and `one_sided_b` alike. This is the **primary**
     instability metric, so `two_sided` vs `one_sided` is apples-to-apples (same perturbation type,
     same register).
   - **Side-order swap (`two_sided` only) — deferred to v1b.1.** Presenting narrative A before B
     vs B before A would test whether presentation order biases a torn model's conclusion, reported
     separately from the shared metric. It needs per-prompt swapped variants authored in the data
     and a separate reporting path; rather than half-build it, this build ships the shared paraphrase
     family (the primary, control-comparable metric) and leaves side-order swap as a documented
     follow-on once the paraphrase signal is shown to carry.

**The claim (success criterion):** co-activation and/or oscillation **predict** perturbation-
instability, AND both internal signals plus instability are higher on `two_sided` prompts than on
`one_sided`-same-topic controls. If an internal signal fires on one-sided prompts too, it is
detecting the *topic*, not conflict — a reported negative, exactly v1a's logic.

**Known confound, stated honestly:** `two_sided` prompts ask for more (both sides), so their
responses may diverge more under perturbation for reasons of length/complexity rather than tension.
Mitigations: Jaccard is a set measure (length-robust, per `esta.fidelity` docs); the primary
contrast uses the **shared paraphrase family only** (identical perturbation type across classes) on
`two_sided` vs `one_sided`-*same-topic*-same-register, not vs neutral; and instability is reported
against a within-`one_sided` null (the `nearest_rank_percentile` pattern from response-fidelity), so
"unstable" means "beyond what one-sided prompts on this topic show under the same perturbation."

## Direction extraction: prompt-contrast, per topic, orthogonalized

Mirrors `extract_reasoning_direction.py` exactly. For topic T with narratives A and B, against a
shared register-matched **neutral baseline** N (analytical asks on neutral topics, so subtraction
isolates narrative *content* from analytical *register*):

```
r_A = mean(activations on argue-A prompts) − mean(activations on N)
r_B = mean(activations on argue-B prompts) − mean(activations on N)
r_B⊥ = r_B − (r_B · r̂_A) · r̂_A        # Gram-Schmidt: B independent of A
r̂_A = r_A / ‖r_A‖ ,  r̂_B⊥ = r_B⊥ / ‖r_B⊥‖
```

Report `cos(r_A, r_B)` **before** orthogonalization — the go/no-go diagnostic, same as v1a. The two
narratives of a contested topic are expected to be *opposed*, so a strongly **negative** cosine
(near −1) means they are collinear-opposite and not separable (the "engagement = −refusal" failure
mode); `build_narrative_directions` raises in that case, and it is a reported result per topic, not
tuned away. Directions captured at the same residual layer as refusal (layer 14 on Qwen 2.5 7B).

Each topic yields a persisted `(r̂_A, r̂_B⊥)` pair. During generation every token is projected onto
both, giving per-token `p_A`, `p_B⊥` series — the same shape v1a persisted, so `--rescore` re-scores
co-activation and oscillation offline for free.

## Topics and narrative pairs (4)

Curated in the neutral, analytical register the existing `constraint_region.json` already uses (it
tags `israel-palestine`, `gun-control`, `taiwan-sovereignty` among 25 prompts). These are research
artifacts for measuring internal state on contested topics — not advocacy — and each side is stated
as "the strongest case for position X," balanced and non-inflammatory.

| Topic | Narrative A | Narrative B |
| --- | --- | --- |
| israel-palestine | Israeli-state-security framing | Palestinian-rights framing |
| abortion | fetal-rights ("pro-life") framing | bodily-autonomy ("pro-choice") framing |
| gun-control | public-safety-regulation framing | individual-rights framing |
| taiwan-sovereignty | Taiwanese-self-determination framing | one-China / Beijing framing |

**Build and validate Israel-Palestine first**, then scale to the other three — de-risks the curation
and the extraction before spending generation budget on all four.

## Validation classes and the disjoint-pool discipline

Per topic, four classes under `data/probe_sets/` (never `data/validation_cases/`):

- **`two_sided` (positive)** — prompts inviting *both* narratives ("analyze the strongest arguments
  on each side of…"). Both axes should light; the model may be torn.
- **`one_sided_a` / `one_sided_b` (controls, same topic)** — prompts invoking a *single* narrative
  ("make the strongest case for position A on…"). Only one axis should light; stable under
  perturbation. **These are the control that makes the result mean something.**
- **`neutral` (easy negative)** — analytical asks on uncontested topics. Neither axis; reuses the
  existing `uncontested_analytical.json`.

The leakage discipline follows v1a exactly (which the sibling `analyze_conflict_state` embodies):
two protections that matter, not three fully-disjoint pools.

1. **The extraction pool is fully held out.** The argue-A / argue-B / neutral prompts that build
   `r_A`, `r_B⊥` (the extraction script's `--*-file` inputs) never appear in the validation sets —
   the same rule `extract_reasoning_direction` follows with its held-out contrast prompts.
2. **The positive set is never used to place thresholds.** θ_A, θ_B come only from the `one_sided`
   controls; the `two_sided` positives never touch calibration, so the conflict result is measured,
   not fitted.
3. **The controls do double duty** — the `one_sided_a` / `one_sided_b` classes both place θ (and the
   instability null) *and* are reported by category, exactly as v1a's `direct_recall` /
   `uncontested_analytical` controls do. Their reported rates are mildly optimistic (θ is fitted to
   separate them), but that biases the controls toward looking *quieter/cleaner*, which is
   conservative for the `two_sided`-vs-`one_sided` claim, not inflating it.

## Calibration / thresholds

Thresholds come from controls, never invented — the project's standing discipline.

- **θ_A, θ_B** — each placed by `youden_cutoff` between "invokes this narrative" (the matching
  one-sided calibration class, high) and "does not" (the other one-sided class + neutral, low), on
  the peak per-token projection per response, significance-gated (`mann_whitney_p`), leakage
  reported. If a direction's two classes do not separate, θ is `None` and no co-activation is scored
  for that topic — reported, not forced.
- **Instability null** — the `nearest_rank_percentile` (e.g. p95) of `one_sided` within-topic
  perturbation-divergence; a `two_sided` prompt is "unstable" when its divergence exceeds it.
- **Oscillation** has no threshold of its own; it is reported per class and correlated against the
  instability ground truth.

## Components

| Path | Torch | Purpose |
| --- | --- | --- |
| `src/esta/scripts/extract_narrative_directions.py` | yes (like `extract_reasoning_direction`) | per topic: build `(r̂_A, r̂_B⊥)`, report `cos`; torch-free math in `build_narrative_directions` |
| `src/esta/oscillation.py` | no | per-token dominance-swing metric; unit-tested |
| `src/esta/conflict.py` (reuse) | no | `windowed_conflict_aggregates` for co-activation — unchanged |
| `src/esta/fidelity.py` (reuse) | no | `convergence` → `1 − convergence` for perturbation-divergence — unchanged |
| `src/esta/scripts/analyze_framing_conflict.py` | inside `_generate_records` only | two-narrative hook loop + K-framing perturbation harness; θ derivation; convergence report; `--rescore` |
| `data/probe_sets/framing_<topic>.json` × 4 | — | validation classes (two_sided / one_sided_a / one_sided_b), tagged |
| narrative-extraction contrast files (per topic/side) | — | held-out inputs to extraction |

Layout mirrors the other detectors: numeric logic torch-free and unit-tested, torch quarantined in
`_generate_records`, `--rescore` fully torch-free. Per-token projections, the full responses, and
the K perturbation responses are all persisted so score/threshold revisions re-measure offline for
free.

## Data flow

`analyze_framing_conflict.py` (generation mode): for each topic, load `(r̂_A, r̂_B⊥)`; for each
validation prompt, generate the base response with the residual hooked, project each token onto both
directions (→ `p_A`, `p_B⊥` series); then generate the K perturbation responses (no hook needed).
Persist prompt, base response, both series, and the K responses. `_finish` (torch-free) derives θ_A,
θ_B and the instability null from the calibration classes, scores co-activation
(`windowed_conflict_aggregates`), oscillation (`esta.oscillation`), and instability
(`1 − convergence` across perturbations), then builds the convergence report — per class, and the
internal-vs-instability correlation. `--rescore` skips generation and recomputes every score from
the persisted series and responses.

## Error handling

- `cos(r_A, r_B)` near ±1 → orthogonal residual ~0 → `build_narrative_directions` raises; that
  topic's narratives are inseparable on this model — a reported per-topic result.
- Empty response / fewer than two engaged tokens → oscillation `None`; record kept, excluded from
  oscillation stats.
- A direction's calibration classes that do not separate → θ `None`, no co-activation for that
  topic, reported.
- A `--rescore` corpus lacking per-token series or perturbation responses → refused loudly, like the
  other detectors' preview-only corpora.
- Fewer than K perturbation responses for a prompt → instability `None`, reported.

## Testing

- **Unit, torch-free:** oscillation (steady-both → 0, alternating-dominance → 1, `<2` engaged →
  `None`, floor behavior); perturbation-divergence and instability-null aggregation; θ_A/θ_B
  derivation and the "do-not-separate → None" path; convergence-report shape and the
  internal-vs-instability correlation; `build_narrative_directions` (known vectors → verified
  residual + cosine, collinear-opposite → raises); `--rescore` end-to-end on synthetic series +
  perturbation responses, including refusal of a corpus missing either.
- **Unit, data:** each `framing_<topic>.json` has unique ids, correct class tags, both one-sided
  classes present, and Israel-Palestine two_sided present.
- **Integration, `requires_model`:** tiny-model smoke — two-narrative hooks yield two per-token
  series, the perturbation loop produces K responses, and the record shape carries all three signals.

## Out of scope

- v1c (directions discovered by anticorrelation from a bank) — the later sub-build.
- Schema changes and server integration. `SCHEMA_VERSION` stays `0.1.1`; the served block and the
  `0.2.0` bump come after validation.
- SAEs / any second model — component 2's concern.
- LLM-as-judge stance detection — violates the standing "mechanistic and deterministic end to end"
  principle; perturbation-instability is measured by deterministic content-divergence instead.
- Behavioral-tension signals beyond instability + oscillation (e.g. hedging/confidence reuse) —
  a possible v2 corroboration, not this build.
- The `two_sided`-only side-order-swap perturbation — deferred to v1b.1 (see signal 3); this build
  ships the shared neutral-paraphrase family only.

## The run (held for explicit go)

One g5.xlarge run: regenerate refusal direction + calibration on-box, extract the four topics'
narrative directions (report each `cos` — the go/no-go diagnostic), generate base + K perturbation
responses for the validation classes with both directions hooked, score, and report. Greedy,
reproducible, `--rescore` exact. **Cost driver:** each prompt generates 1 base + K≈3 paraphrase
responses; ~4 topics × ~40 prompts × ~4 ≈ 640 generations at 256 tokens — well within a single-hour
A10G run. The run is **held for the originator's explicit instruction** (like every
prior AWS run) and gated behind spec + plan review; the offline scripts, probe sets, tests, and the
teed-up run script are built first.

## Build notes (implemented 2026-09-27, subagent-driven)

Refinements made during implementation and the whole-branch review, recorded so the spec matches
what shipped:

- **Authored per-prompt paraphrases (preferred over the programmatic fallback).** The perturbation
  family is meant to be *meaning-preserving rewordings of the ask*. The built-in programmatic
  `_paraphrases` only varies the instruction *framing* (it keeps the ask verbatim), which under greedy
  decoding may barely move the response — a floor-effect risk for the instability signal. So
  `_generate_records` uses `_perturbation_prompts(prompt, k)`: if a probe prompt carries a non-empty
  `"paraphrases"` list it is used verbatim; otherwise the weak programmatic default is the fallback.
  Authoring strong per-prompt paraphrases is the quality lever and is left as curation follow-up.
- **The report computes the conclusion, not just the inputs.** `build_report` now emits a per-record
  `unstable` flag (`instability > instability_null`), an `unstable_rate` per class, and an
  `internal_vs_instability` block: for `coactivation_max` and `oscillation`, the unstable-vs-stable
  means and a `mann_whitney_p` (one-sided, testing whether the internal signal ranks higher on
  unstable prompts). This is the spec's headline claim, surfaced by the tool rather than left to
  manual post-processing.
- **The local smoke is a HARD GATE before the paid run.** Because the paraphrase signal's
  discriminating power cannot be settled offline, the tiny-model smoke (and the eventual on-box
  check) must confirm `two_sided` instability meaningfully exceeds `one_sided` and that θ_A/θ_B land
  positive. If instability floors, the fix is to author per-prompt paraphrases and/or bring forward
  the deferred `two_sided`-only side-order-swap perturbation (v1b.1) — not to spend on an
  uninformative run.
- **Contested probe sets merit a human domain read before the run.** The Israel-Palestine set went
  through three balance-review rounds and the other three topics one; balance on contested topics is
  inherently subjective, so the originator's read is the intended final arbiter before extraction.

## Measured outcome — the Israel-Palestine cheap check (Qwen 2.5 7B Instruct, 2026-09-30)

One topic on a g5.xlarge, ~1.5 GPU-hours, before committing to the four-topic run. **A mechanism
test, not a truth test:** the originator's standing caveat is that the topic's history is contested
and its sources are not objective, so nothing here is a claim about the subject — only about how
the *model* represents two curated framings of it. The pipeline ran end to end on 7B (42 records,
0 excluded, all report fields populated), so the machinery is validated; the *signals* returned a
**structured negative** with two clear design causes.

| class | n | co-activation rate | mean co-act max | mean oscillation | mean instability | unstable rate |
| --- | --- | --- | --- | --- | --- | --- |
| `two_sided` (positive) | 8 | 0.12 | 0.66 | 0.09 | **0.686** | 0.12 |
| `one_sided_a` (control) | 8 | 0.00 | 0.36 | 0.00 | 0.679 | 0.00 |
| `one_sided_b` (control) | 8 | **0.75** | 1.13 | 0.20 | 0.654 | 0.00 |
| `neutral` (control) | 18 | 0.00 | −0.26 | n/a | 0.633 | 0.00 |

θ_A = 8.77 (AUC 0.84, p = 2.1e-3), θ_B = 1.48 (AUC 1.00, p = 1.3e-5), both placed and positive
(gate 2 passed); instability null (p95 of one-sided) = 0.774.

### 1. The two narrative directions are nearly the same direction — cos(A, B) = **+0.987**

The go/no-go diagnostic failed, in the direction the design did not anticipate. The spec guarded
against *collinear-opposite* narratives (cos → −1, the "engagement = −refusal" trap). Instead the
two extracted directions are almost **identical**: subtracting a shared *neutral* baseline from
each leaves the component they have in common — "this is the Israel-Palestine topic, argued
analytically" — in both `r_A` and `r_B`, and that shared topic component dominates. After
Gram-Schmidt, `r_B⊥` is a small residual (‖r_B⊥‖ ≈ √(1−0.987²) ≈ 0.16 of ‖r_B‖) — a weak, noisy
axis. The extraction script's warning only fired for cos < −0.9, so this went unflagged at run
time; it is now flagged in both signs.

**What that means:** with this extraction, "narrative A" is effectively the *topic* direction and
"narrative B" is whatever small part of the B prompts isn't the topic. The co-activation pattern
confirms it — it fires on **`one_sided_b` 75%** of the time (those prompts light the shared/topic
axis *and* the B residual), on `two_sided` only 12%, and on `one_sided_a` never. That is not
conflict; it is the collinearity artifact. A reportable observation, with the extraction-method
caveat attached: on this model, the two curated framings activate overwhelmingly the *same*
internal direction, with only a small framing-specific residual.

### 2. Perturbation-instability does not discriminate — but not because it floors

Instability is **~0.63–0.69 for every class, including neutral** (all 42 records within
0.465–0.789), so `two_sided` (0.686) vs one-sided (0.667) separates by +0.019 — nothing. The
anticipated failure was a *floor* (paraphrases too weak to move greedy output); the actual failure
is the opposite: the programmatic paraphrases move *every* response by roughly the same large
amount. The signal measures generic rewording sensitivity, not "torn." With only one record above
the null, the internal-vs-instability association is unpowered (n_unstable = 1; p = 0.31 / 0.84).

### What this changes (v1b.1)

- **Extraction: project out the shared topic component.** Build `r_topic = mean(A ∪ B) −
  mean(neutral)`, remove its projection from each of `r_A`, `r_B`, *then* orthogonalize B against
  A. That isolates the framing-specific directions from "the topic." Report cos(A, B) both before
  and after; the diagnostic now warns for |cos| > 0.9 in either sign.
- **Perturbation: authored, framing-shifting rewordings + side-order swap.** The
  `"paraphrases"` override already exists; it needs content that changes *framing* while preserving
  meaning, plus the deferred `two_sided`-only side-order swap — and instability should be reported
  relative to a per-class baseline so uniform rewording variability cancels.
- **Do not run the other three topics on the v1b design.** The cheap check did its job: the
  four-topic run would have reproduced both artifacts three more times.

## Open questions

None blocking. Decisions made during brainstorming (2026-09-27): per-topic narrative pairs (not a
general concept bank); a positive **requires** behavioral tension, not mere co-activation;
tension operationalized as perturbation-instability (independent ground truth) **and** internal
oscillation (corroboration), explicitly **not** an LLM judge; prompt-contrast extraction mirroring
the refusal/reasoning pattern; four topics with Israel-Palestine built and validated first.
