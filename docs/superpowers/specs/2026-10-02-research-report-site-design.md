# ESTA Research-Report Site — Design

**Status:** design; not yet implemented
**Date:** 2026-10-02

## Goal

A **local, static website** that presents ESTA's Phase 1–2 findings in the shape of a research
report — modelled on `F:\science-projects\baryogenesis\REPORT.md`: *question → verdict → measured
table → figures → reasoning (the author's own inferences tagged) → what it changes*, plus sources and
exact reproductions. It opens from a fresh clone via `file://docs/site/index.html` with no server, no
JavaScript, and no network.

Audience: the originator and research readers. It is a **record of what was measured**, on which
model, with which caveats. It never implies a served capability that does not exist
(`SCHEMA_VERSION` is `0.1.1`; nothing has graduated into `epistemic_state`).

## Why markdown sources, generated figures, thin HTML (approach C)

The baryogenesis "content" is a markdown report. Keeping the site's source in markdown keeps it
diffable, readable on GitHub, and in the same voice. Figures are the one thing that must come from
data, so they are regenerated from the persisted JSON reports by a script; everything else is a
thin rendering layer. Numbers in the prose are typed from the design docs — the authoritative record
— and every page links its design doc as the primary source. (Pulling every number from JSON at
build time was considered and rejected: it forces templated prose around data and fights the
written-report voice.)

## Layout

```
docs/site/
  src/                         markdown sources (committed)
    index.md                   the project question, status, verdict table across detectors
    refusal-probe.md           Phase 1 refusal direction, calibration, dual-use audit
    performed-uncertainty.md   hedge instrument v2, the confabulation finding
    response-fidelity.md       raw vs anchored distortion, convergence caveat
    conflict-state.md          v1a structured negative -> v2 windowed sweep
    framing-conflict.md        v1b collinearity -> v1b.1 lean geometry, swap-flip
    sources.md                 authored from docs/REFERENCES.md: each cite-key, its role, the ESTA-original constructs
    reproductions.md           exact commands per run; the AWS pattern; the DLAMI xet lesson
  style.css                    one shared stylesheet, no JS
  figures/*.png                generated from data/*.json (committed)
  *.html                       rendered output (committed)
src/esta/scripts/build_site.py torch-free builder
tests/unit/test_build_site.py
```

All of `docs/site/` is committed so the site is browsable from a fresh clone. `data/` stays
gitignored; the builder tolerates its absence.

## The builder: `python -m esta.scripts.build_site`

Torch-free, importable in CI without `[model]` or `[site]`.

- **`figures`** — for each known report, if `data/<report>.json` exists, regenerate its PNG(s) with
  matplotlib; if absent, print a one-line notice and **skip** (the committed PNG stands). Never an
  error. Requires the `[site]` extra; importing matplotlib happens inside this step only.
- **`render`** — markdown → HTML for every `src/*.md` with a shared header, nav, footer and
  `style.css`; relative links only. `(inference)` is rendered as a visible, styled marker so the
  reader can tell the author's reasoning from stated results. Requires `markdown` (in `[site]`).
- **`--check`** — asserts every `figures/*.png` referenced from the sources exists and every
  internal link resolves. Needs neither matplotlib nor markdown. This is what CI runs.
- Default (no flag) = `figures` then `render`.

Figure functions are **pure** (`(report_dict) -> matplotlib Figure`) and kept separate from file
I/O, so the data→figure math is unit-tested on tiny synthetic reports without matplotlib by testing
the array-shaping helpers, and the Figure-producing wrappers are smoke-tested only when matplotlib
is importable (`pytest.importorskip`).

## Page contract (every detector page, in this order)

1. **Question** — one sentence, what was being tested.
2. **Verdict** — one line, honest and categorical (e.g. *structured negative*, *validated
   machinery, no positive set*, *discriminates*), with the model and date.
3. **Measured table** — the table from the design doc, verbatim numbers.
4. **Figures** — generated; captioned with what the axes are and where the data came from.
5. **Reasoning** — the author's inferences tagged *(inference)*; **model-behaviour claims kept
   separate from claims about the subject matter** (the standing caveat for the contested-topic
   work: a mechanism test, not a truth test).
6. **What it changes** — the decisions the result forced.
7. **Source** — link to the design doc; the persisted report's filename (gitignored, local).

`index.md` carries the project question, current status, and a verdict table (one row per detector,
like baryogenesis's mechanism table). `sources.md` lists every cite-key in `docs/REFERENCES.md` with
its one-line role, plus the ESTA-original constructs. `reproductions.md` lists the exact commands
per run, the AWS run pattern (tagged key/SG, dead-man switch, teardown verification), and the DLAMI
xet pre-materialization lesson.

## Figures (one function each; data source in parentheses)

| figure | from | shows |
| --- | --- | --- |
| `refusal_calibration.png` | `calibration*.json` + `dual_use_analysis*.json` | pressure band boundaries; projection distributions by class |
| `performed_uncertainty.png` | `performed_uncertainty_analysis.json` | confidence and hedge (v2) distributions per class, AUCs |
| `response_fidelity.png` | `response_fidelity_analysis.json` | raw vs anchored distortion by class |
| `conflict_window_sweep.png` | `conflict_state_analysis_v2.json` | any-conflict rate vs gap per class (the v2 sweep) |
| `framing_v1b_classes.png` | `framing_conflict_israel-palestine.json` | v1b per-class co-activation / instability (the collinearity artefact) |
| `framing_swap_flip.png` | `framing_lean_israel-palestine_rescore2.json` | per two-sided prompt: lean under A-first vs B-first ordering, flips marked |

Each function reads only the report's `summary`/`records` and degrades to "n/a" annotations when a
field is `None` (the θ=None paths are real data).

## Dependencies, tests, CI

- New optional extra `site = ["matplotlib>=3.8", "markdown>=3.5"]`. Core, `[dev]`, and CI stay free
  of both.
- `tests/unit/test_build_site.py` (torch-free, no `[site]`): `--check` passes on the committed
  sources; the array-shaping helpers behave on synthetic summaries (including `None` fields); every
  source page follows the page contract (has the required headings in order); `(inference)` tags
  are rendered as the marker when `markdown` is importable (`importorskip`), else skipped.
- CI runs `build_site --check` via the existing test suite; it does not regenerate figures.
- Ruff as usual; ASCII-safe console output.

## Tone

Short, declarative, verdict-first. Tables over adjectives. Every negative result stated as such.
The contested-topic pages carry the mechanism-not-truth caveat at the top, not buried.

## Out of scope

- A hosted site, JavaScript, search, or theming beyond one stylesheet.
- Decision-maker ("layer 4") framing — a possible later page set with plainer language; this site
  is the research record.
- Pulling prose numbers from JSON at build time (see "Why" above).
- Any change to the served schema or the detectors themselves.

## Open questions

None blocking. Decisions made in brainstorming (2026-10-02): approach C; all six lines of work
included; committed build output; figures regenerated from data with graceful skip; markdown as the
source of truth for prose, design docs as the source of truth for numbers.
