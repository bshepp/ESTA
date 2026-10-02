# Response fidelity

## Question

When a borderline ask is answered on-topic but the *operation* asked about is quietly replaced by a
safer adjacent one, can that reframing be measured deterministically — and can it be anchored to
the refusal-pressure measurement so that a benign reframe of a vague question never fires?

## Verdict

**The anchor discipline holds; the convergence harness over-flags** (Qwen 2.5 7B Instruct,
2026-08-19). The anchored signal fired **0/25** on benign-vague asks — the design's core safety
promise, on real data — and a handful of genuine candidate reframes surfaced under refusal pressure.
But the Jaccard convergence harness flagged 42 of 55 pairs, 35 of them faithful answers, so no
distortion threshold could be placed; the run reported that rather than inventing one.

## Measured

| class | n | mean raw distortion | anchored signal fires | mean refusal projection |
| --- | --- | --- | --- | --- |
| `direct_answer_controls` | 25 | 0.040 | 0/25 | 9.48 |
| `benign_vague` | 25 | **0.320** | **0/25** | 9.88 |
| `reframe_pairs` | 55 | 0.118 | 4/55 | 13.98 |
| `refusal_boundary` | 15 | 0.300 | 7/15 | 27.77 |

Convergence harness: **42/55** pairs flagged as empirical-positive candidates; **7 of those 42** carry
any distortion at all. Candidate distortion (mean 0.131) does not rank-separate from the direct
controls (0.040), so `youden_cutoff` returned `None`. Anchored hits with raw distortion 1.0:
`reframe_009` (privilege escalation), `reframe_015`, `reframe_024` (account takeover), each with the
refusal projection ~15, above the moderate band.

## Figures

![Response fidelity: raw distortion by class](figures/response_fidelity.png)

*Raw distortion (topic hit × operative evaded) per response, by class, from the persisted run. Raw
distortion is never the reported signal on its own — see Reasoning.*

## Reasoning

- `benign_vague` carries the *highest* raw distortion (0.320: vague asks legitimately get narrowed)
  yet an anchored rate of zero, because its refusal projection (9.88) sits below the moderate band.
  *(inference)* This is the whole argument for anchoring: distortion alone is a property of the
  question, not of the model's state; gated on refusal pressure it becomes a state signal.
- The convergence harness measures content-word Jaccard overlap between a pair's two responses, and
  topically-related pairs overlap heavily whether or not either reframes. *(inference)* The 42/55
  rate is the symptom of measuring the wrong thing; a v2 must discount shared *topic* vocabulary or
  require *operative*-vocabulary divergence.
- Two of the three anchored hits (`reframe_009`, `reframe_024`) are entries the probe-set audit had
  fixed by moving a shared topic noun out of `operative_groups`; without that fix they would have
  scored `operative_coverage = 1.0` and been missed. *(inference)* The audit earned its place on live
  data, and term-group curation is where this instrument's precision actually lives.

## What it changes

- Convergence v2 is queued and free: the persisted corpus re-scores offline.
- The ~4 anchored candidates are the read-worthy responses; the next step on them is a human read,
  not a threshold.
- Nothing is served; the anchored-FPR-0 result is the strongest evidence so far that the anchor
  design works as intended.

## Source

[2026-08-12 response-fidelity design](../superpowers/specs/2026-08-12-response-fidelity-design.md) and the
[2026-08-17 probe-set audit](../superpowers/specs/2026-08-17-response-fidelity-probe-audit.md). Persisted
report (local, gitignored): `data/response_fidelity_analysis.json`.
