# Integrating Another Project into ESTA

This document is for an agent or maintainer of a separate project that is
considering folding that project into ESTA. Read it before proposing anything.
It tells you what ESTA is, the non-negotiable constraints any integration must
satisfy, and how to bring a proposal.

It is descriptive, not a how-to: ESTA does **not** yet have a formal plugin or
probe interface. Your job after reading this is to assess whether your project
fits these constraints and to propose how it would map onto ESTA — not to
implement against an interface that does not exist.

Before proposing, also read:

- [`../README.md`](../README.md) — user-facing overview
- [`../CLAUDE.md`](../CLAUDE.md) — the architectural ground truth (the
  torch/no-torch boundary, the guarded invariants)
- [`epistemic-transparency-agent (1).md`](<epistemic-transparency-agent (1).md>) — the full spec and 3-phase roadmap
- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — workflow and DCO requirement

## The two integration models

ESTA absorbs work in one of two shapes. Which one applies depends on the
maturity of the incoming project; decide this with the maintainer early.

- **Model A — absorbed as a subpackage.** Your code is merged into this
  repository as a module under `src/esta/`. It shares ESTA's schema, audit,
  server, and test infrastructure, and must conform to every constraint in
  this document.
- **Model B — ESTA consumes the capability.** ESTA pulls a capability from
  your project (e.g. a new detector or probe) through the generation pipeline.
  There is no formal interface for this yet. The integrating project proposes
  how its capability maps onto the pipeline; the interface is designed jointly
  and is out of scope for this document.

Either way, the constraints below apply to the code that ends up running inside
ESTA.

## What ESTA is — and explicitly is not

ESTA is a local, self-hosted wrapper around an open-weights LLM that returns
the normal OpenAI-compatible chat completion **plus** an `epistemic_state`
metadata block describing the internal state under which the response was
generated (token confidence, refusal-direction projection, provenance).

It is **not**:

- a safety system — it does not block, filter, or modify responses;
- a hallucination detector — it reports mechanistic signatures that correlate
  imperfectly with correctness;
- a replacement for expert review.

This scope is load-bearing. A capability whose job is to **gate, rewrite, or
suppress** model output does not belong in ESTA's core, no matter how useful —
ESTA exposes state so downstream systems can route; it does not intervene. If
your project's value is intervention, integration means contributing the
*observability* part, not the intervention part.

## Repository and licensing status

- **License: Apache-2.0.** Incoming code must be license-compatible. Code under
  a copyleft or otherwise incompatible license cannot be absorbed under Model A.
- **The repository is currently private and pre-public.** It will be made
  public later. Do not introduce anything that would have to be removed or
  rewritten before a public release (secrets, proprietary data, vendored code
  with unclear provenance, internal-only references). Treat every change as if
  it will be public.
- **DCO sign-off is mandatory.** Every commit needs a `Signed-off-by` trailer:
  `git commit -s`. See [`../CONTRIBUTING.md`](../CONTRIBUTING.md).
- **Do not commit model weights, refusal-direction tensors, or calibration
  sets.** Those live in the gitignored `data/`.

## Hard architectural constraints

Any integration must satisfy all of these. They are enforced by tests and CI;
a proposal that cannot meet them is not yet ready.

### 1. The torch / no-torch boundary

CI and the default `pytest` run install **without** the `[model]` extra, so
torch is absent. The codebase is deliberately split:

- **Torch-free** (must stay importable and unit-testable without torch):
  numeric and metric logic — e.g. `esta.extraction`,
  `esta.confidence.metrics`, `esta.probes.thresholds`, `esta.schema.*`,
  `esta.audit.logger`.
- **Torch-dependent** (quarantined behind the inference layer):
  `esta.inference.*`, `esta.probes.refusal`, `esta.api.server`.

Incoming numeric/scoring logic must be torch-free and operate on numpy
arrays / Python floats, with any torch conversion done at the inference
boundary. Heavy dependencies go behind an optional extra in `pyproject.toml`,
never in the base `dependencies`.

### 2. Schema discipline

`epistemic_state` is a versioned contract.

- Any new metadata field requires a `SCHEMA_VERSION` bump in
  `src/esta/schema/epistemic_state.py`, a regenerated
  `src/esta/schema/epistemic_state.schema.json`
  (`python -m esta.scripts.dump_schema`), and a migration note. The
  `test_schema_drift` test fails otherwise.
- **No false precision.** Uncalibrated or experimental output must be labeled
  as such in the schema (as the refusal probe does with
  `calibrated_pressure="uncalibrated"` / `probe_version="not_loaded"`). A
  metric that asserts confidence it has not earned will be rejected. Real
  thresholds come from a calibration run, not hand-picked constants presented
  as calibrated.

### 3. Audit integrity

Decision-relevant output flows through the SHA-256 hash-chained JSONL audit
log (`esta.audit.logger`). Do not add a path that produces routable metadata
while bypassing the audit log. The chain being only locally verifiable is a
known, documented limitation (external anchoring is deferred to Phase 3) — do
not "fix" it with local-only hardening that implies stronger guarantees than
exist.

### 4. CI discipline

- Unit tests must pass with only `pip install -e .[dev]` (no `[model]`).
- Tests that load model weights go in `tests/integration/` and are marked
  `@pytest.mark.requires_model` (deselected by default).
- `ruff check src tests` must pass (config in `pyproject.toml`).

## How to propose an integration

1. Confirm with the maintainer whether this is Model A or Model B.
2. Write a short fit assessment that answers, explicitly, each constraint in
   the section above: license, public-readiness, the torch-free surface of your
   capability, its schema impact (new fields? version bump?), its audit impact,
   and how it stays within ESTA's observe-don't-intervene scope.
3. Point your agent at this document, `CLAUDE.md`, and the spec.
4. Open the discussion **before** a large PR — substantive work is coordinated
   up front, per `CONTRIBUTING.md`.

A proposal that cannot honestly answer step 2 is feedback that the boundary
needs to move or that the capability belongs downstream of ESTA rather than
inside it. Either outcome is a valid result of this assessment.

## Pointers

| Topic | File |
|-------|------|
| User-facing overview | [`../README.md`](../README.md) |
| Architecture ground truth | [`../CLAUDE.md`](../CLAUDE.md) |
| Full spec / roadmap | [`epistemic-transparency-agent (1).md`](<epistemic-transparency-agent (1).md>) |
| Workflow & DCO | [`../CONTRIBUTING.md`](../CONTRIBUTING.md) |
| Vulnerability reporting | [`../SECURITY.md`](../SECURITY.md) |
