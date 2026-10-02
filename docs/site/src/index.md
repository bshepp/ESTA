# ESTA — research record

**Question:** can the internal state under which an open-weights LLM generates a response be
measured and exposed alongside the response — token confidence, safety pressure, performed
uncertainty, response fidelity, and internal conflict — without blocking, filtering, or changing
the response?

**Status (2026-10-02):** Phase 1 is served (`SCHEMA_VERSION 0.1.1`: token confidence,
refusal-direction projection, provenance, a hash-chained audit log). The Phase 2 detectors below are
**offline research capabilities**; none has graduated into the served `epistemic_state` block.
Every result is on **Qwen 2.5 7B Instruct**, residual layer 14, greedy decoding, with thresholds
placed from control classes — never invented.

| line of work | verdict | one line |
| --- | --- | --- |
| [Refusal probe + dual-use audit](refusal-probe.html) | validated, served | the refusal direction separates harmful from harmless prompts by 22.5–22.9 across two runs; bands calibrated; the dual-use audit shows it encodes refusal disposition, not subject matter |
| [Performed uncertainty](performed-uncertainty.html) | hedge instrument v2 works; confident confabulation is real | AUC 0.83 settled-vs-obscure after rebuilding the marker list against controls; 17 of 50 obscure answers confabulate rather than hedge |
| [Response fidelity](response-fidelity.html) | anchor discipline holds; convergence over-flags | anchored signal fires 0/25 on benign-vague asks; the Jaccard convergence harness flags 42/55 pairs, 35 of them faithful — needs v2 |
| [Conflict state (refusal vs reasoning)](conflict-state.html) | structured negative → event definition fixed (v2) | 0 same-token events; a one-token window flips 7/15 refusal-bait prompts while the three true negatives stay at 0%; the constraint region never raises the refusal axis |
| [Framing conflict (two narratives)](framing-conflict.html) | v1b collinear (cos 0.987) → v1b.1 swap-flip discriminates | two-sided answers flip their framing lean under side-order swap 6/8; one-sided 0/16; the internal balance/oscillation signals do not track it |

Statements marked *(inference)* on the pages are the author's reasoning from the measured results,
not results themselves. The contested-topic pages are **mechanism tests, not truth tests** about the
topics.

Sources: [sources.html](sources.html). Exact commands and the AWS pattern:
[reproductions.html](reproductions.html). Design documents for every line of work live in
`docs/superpowers/specs/` and are the authoritative record; this site summarizes them.
