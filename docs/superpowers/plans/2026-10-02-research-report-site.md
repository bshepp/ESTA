# ESTA Research-Report Site Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local, static, no-JS website under `docs/site/` that presents ESTA's Phase 1–2 findings as a research report (question → verdict → table → figures → tagged reasoning → what it changes), with figures regenerated from the persisted JSON reports by a torch-free builder.

**Architecture:** Markdown sources in `docs/site/src/` are the content; `src/esta/scripts/build_site.py` has three steps — `figures` (matplotlib, optional, skips missing reports), `render` (markdown → HTML with shared chrome), `--check` (link/figure/contract validation, no optional deps). Built HTML and PNGs are committed so the site opens via `file://` from a fresh clone. Figure data-shaping is pure and unit-tested; matplotlib wrappers are smoke-tested only when importable.

**Tech Stack:** Python 3.11+, `markdown>=3.5` and `matplotlib>=3.8` behind a new `[site]` extra (core/dev/CI stay free of both), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-02-research-report-site-design.md`

## Global Constraints

- **No torch, no matplotlib, no markdown at module import** of `esta.scripts.build_site`. Optional deps import inside the `figures` / `render` functions only. `--check` must run with none of them installed (that is what CI does).
- **Missing data is never an error.** `figures` prints `skip: data/<name>.json not found` and continues; the committed PNG stands.
- **Everything under `docs/site/` is committed** (sources, figures, HTML, CSS). `data/` stays gitignored and is never read by tests.
- **Page contract**, in this order, as `##` headings on every detector page: `Question`, `Verdict`, `Measured`, `Figures`, `Reasoning`, `What it changes`, `Source`. `index.md`, `sources.md`, `reproductions.md` are exempt.
- **`(inference)`** is the literal tag for the author's own reasoning; render wraps it in `<span class="inference">(inference)</span>`.
- **Contested-topic pages carry the mechanism-not-truth caveat at the top** (a blockquote under Question).
- **Numbers are typed from the design docs** (the authoritative record); each page's `Source` section links the design doc and names the gitignored report file.
- **Never implies a served capability**: `SCHEMA_VERSION` is `0.1.1`; say so on `index.md`.
- Ruff line-length 100, `E501` ignored; ASCII-safe console output; DCO `git commit -s`; venv python `.venv/Scripts/python.exe`.

## Review Focus

- A markdown page that references `figures/x.png` which does not exist → `--check` must fail naming the page and figure (Task 1 test).
- A page missing or misordering a contract heading → `--check` must fail naming the page and the first bad heading (Task 1 test).
- A report JSON whose summary field is `None` (θ=None paths are real data) → the figure helper returns `"n/a"`-annotated series, never raises (Task 2 test).
- Running the builder with no `data/` directory at all → `figures` reports N skips and exits 0; `render` still produces every HTML (Task 2 test).
- An internal link like `[x](conflict-state.html)` to a page that is not rendered → `--check` fails naming it (Task 1 test).

---

### Task 1: Builder skeleton — `render` and `--check`

**Files:**
- Create: `src/esta/scripts/build_site.py`
- Create: `docs/site/style.css`
- Create: `docs/site/src/index.md` (minimal, replaced in Task 3)
- Modify: `pyproject.toml` (add `site` extra)
- Test: `tests/unit/test_build_site.py`

**Interfaces:**
- Produces: `SITE = Path("docs/site")`, `SRC = SITE / "src"`, `FIGURES = SITE / "figures"`, `CONTRACT = ("Question", "Verdict", "Measured", "Figures", "Reasoning", "What it changes", "Source")`, `EXEMPT = {"index", "sources", "reproductions"}`;
  `page_names(src: Path) -> list[str]`; `headings(md_text: str) -> list[str]` (the `## ` texts in order); `check_contract(name: str, md_text: str) -> list[str]` (problems); `referenced_figures(md_text: str) -> set[str]`; `internal_links(md_text: str) -> set[str]`; `check(site: Path = SITE) -> list[str]` (all problems, empty = ok); `render(site: Path = SITE) -> list[Path]`; `main(argv) -> int`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_build_site.py
"""Tests for the torch-free research-report site builder (no [site] extra needed)."""
from __future__ import annotations

from pathlib import Path

import pytest

from esta.scripts.build_site import (
    CONTRACT,
    check,
    check_contract,
    headings,
    internal_links,
    referenced_figures,
)

GOOD = """# Conflict

## Question
q
## Verdict
v
## Measured
| a | b |
|---|---|
| 1 | 2 |
## Figures
![sweep](figures/conflict_window_sweep.png)
## Reasoning
x *(inference)* y
## What it changes
z
## Source
[design doc](../superpowers/specs/2026-08-18-conflict-state-probe-design.md)
"""


def test_headings_in_order() -> None:
    assert headings(GOOD) == list(CONTRACT)


def test_contract_passes_on_a_conforming_page() -> None:
    assert check_contract("conflict-state", GOOD) == []


def test_contract_names_the_first_bad_heading() -> None:
    bad = GOOD.replace("## Measured", "## Results")
    problems = check_contract("conflict-state", bad)
    assert problems and "conflict-state" in problems[0] and "Measured" in problems[0]


def test_exempt_pages_skip_the_contract() -> None:
    assert check_contract("index", "# ESTA\n\nhello\n") == []


def test_referenced_figures_and_links_are_extracted() -> None:
    md = "![a](figures/one.png) text [p](conflict-state.html) ![b](figures/two.png) [ext](https://x.y/)"
    assert referenced_figures(md) == {"one.png", "two.png"}
    assert internal_links(md) == {"conflict-state.html"}


def _site(tmp_path: Path, pages: dict[str, str], figures: list[str]) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "figures").mkdir()
    for name, text in pages.items():
        (tmp_path / "src" / f"{name}.md").write_text(text, encoding="utf-8")
    for f in figures:
        (tmp_path / "figures" / f).write_bytes(b"\x89PNG")
    return tmp_path


def test_check_passes_on_a_consistent_site(tmp_path: Path) -> None:
    site = _site(tmp_path, {"index": "# i\n[c](conflict-state.html)\n", "conflict-state": GOOD},
                 ["conflict_window_sweep.png"])
    assert check(site) == []


def test_check_fails_on_a_missing_figure(tmp_path: Path) -> None:
    site = _site(tmp_path, {"index": "# i\n", "conflict-state": GOOD}, [])
    problems = check(site)
    assert any("conflict-state" in p and "conflict_window_sweep.png" in p for p in problems)


def test_check_fails_on_a_dangling_internal_link(tmp_path: Path) -> None:
    site = _site(tmp_path, {"index": "# i\n[nope](ghost.html)\n"}, [])
    problems = check(site)
    assert any("index" in p and "ghost.html" in p for p in problems)


def test_render_wraps_inference_tag_and_writes_every_page(tmp_path: Path) -> None:
    pytest.importorskip("markdown")
    from esta.scripts.build_site import render

    site = _site(tmp_path, {"index": "# i\n", "conflict-state": GOOD}, ["conflict_window_sweep.png"])
    (site / "style.css").write_text("body{}", encoding="utf-8")
    out = render(site)
    assert {p.name for p in out} == {"index.html", "conflict-state.html"}
    html = (site / "conflict-state.html").read_text(encoding="utf-8")
    assert '<span class="inference">(inference)</span>' in html
    assert 'href="style.css"' in html and 'href="index.html"' in html   # shared chrome + nav
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_build_site.py -q`
Expected: FAIL — `ModuleNotFoundError: esta.scripts.build_site`.

- [ ] **Step 3: Write the builder (render + check), stylesheet, extra, placeholder index**

```python
# src/esta/scripts/build_site.py
"""Build the local research-report site under docs/site/.

Three steps, all torch-free; the optional deps are imported inside the step
that needs them so this module imports in CI without the [site] extra:

    figures  -- regenerate docs/site/figures/*.png from data/*.json (matplotlib);
                a missing report is SKIPPED with a notice, never an error.
    render   -- docs/site/src/*.md -> docs/site/*.html with shared chrome (markdown).
    --check  -- every referenced figure exists, every internal link resolves, and
                every detector page follows the page contract. No optional deps.

See docs/superpowers/specs/2026-10-02-research-report-site-design.md.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SITE = Path("docs/site")
SRC = SITE / "src"
FIGURES = SITE / "figures"
CONTRACT = ("Question", "Verdict", "Measured", "Figures", "Reasoning", "What it changes", "Source")
EXEMPT = {"index", "sources", "reproductions"}

NAV = (("index", "Overview"), ("refusal-probe", "Refusal probe"),
       ("performed-uncertainty", "Performed uncertainty"), ("response-fidelity", "Response fidelity"),
       ("conflict-state", "Conflict state"), ("framing-conflict", "Framing conflict"),
       ("sources", "Sources"), ("reproductions", "Reproductions"))

_H2 = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_FIG = re.compile(r"!\[[^\]]*\]\(figures/([^)\s]+)\)")
_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s#]+\.html)\)")
_INFERENCE = re.compile(r"\*?\(inference\)\*?")


def page_names(src: Path = SRC) -> list[str]:
    return sorted(p.stem for p in src.glob("*.md"))


def headings(md_text: str) -> list[str]:
    return _H2.findall(md_text)


def check_contract(name: str, md_text: str) -> list[str]:
    """Problems with the page contract: the ## headings must be exactly CONTRACT, in order."""
    if name in EXEMPT:
        return []
    found = headings(md_text)
    for i, want in enumerate(CONTRACT):
        got = found[i] if i < len(found) else "<missing>"
        if got != want:
            return [f"{name}.md: expected heading '## {want}' at position {i + 1}, found '## {got}'"]
    if len(found) > len(CONTRACT):
        return [f"{name}.md: unexpected extra heading '## {found[len(CONTRACT)]}'"]
    return []


def referenced_figures(md_text: str) -> set[str]:
    return set(_FIG.findall(md_text))


def internal_links(md_text: str) -> set[str]:
    return {link for link in _LINK.findall(md_text) if "://" not in link}


def check(site: Path = SITE) -> list[str]:
    src, figures = site / "src", site / "figures"
    names = page_names(src)
    problems: list[str] = []
    for name in names:
        text = (src / f"{name}.md").read_text(encoding="utf-8")
        problems += check_contract(name, text)
        for fig in sorted(referenced_figures(text)):
            if not (figures / fig).exists():
                problems.append(f"{name}.md: figure 'figures/{fig}' does not exist")
        for link in sorted(internal_links(text)):
            if Path(link).stem not in names:
                problems.append(f"{name}.md: link '{link}' has no source page")
    return problems


def _chrome(title: str, body_html: str, current: str) -> str:
    def _link(n: str, label: str) -> str:
        cls = ' class="current"' if n == current else ""   # no backslashes inside f-string expressions (py3.11)
        return f'<a href="{n}.html"{cls}>{label}</a>'

    nav = " | ".join(_link(n, label) for n, label in NAV)
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>{title}</title>"
            "<link rel=\"stylesheet\" href=\"style.css\"></head><body>\n"
            f"<nav>{nav}</nav>\n<main>\n{body_html}\n</main>\n"
            "<footer>ESTA research record. Mechanism tests on Qwen 2.5 7B Instruct; "
            "SCHEMA_VERSION 0.1.1 — nothing here is a served capability.</footer>\n</body></html>\n")


def render(site: Path = SITE) -> list[Path]:
    import markdown  # [site] extra; imported here so --check and CI need nothing

    src = site / "src"
    written: list[Path] = []
    for name in page_names(src):
        text = (src / f"{name}.md").read_text(encoding="utf-8")
        text = _INFERENCE.sub('<span class="inference">(inference)</span>', text)
        body = markdown.markdown(text, extensions=["tables", "fenced_code"])
        title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        title = title_match.group(1) if title_match else name
        out = site / f"{name}.html"
        out.write_text(_chrome(title, body, name), encoding="utf-8")
        written.append(out)
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the ESTA research-report site.")
    p.add_argument("--check", action="store_true", help="validate sources only (no optional deps)")
    p.add_argument("--no-figures", action="store_true", help="render only; skip figure regeneration")
    p.add_argument("--site", type=Path, default=SITE)
    args = p.parse_args(argv)
    if args.check:
        problems = check(args.site)
        for line in problems:
            print("CHECK:", line)
        print("check:", "ok" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    if not args.no_figures:
        from esta.scripts.site_figures import build_all_figures  # Task 2

        build_all_figures(args.site / "figures")
    for out in render(args.site):
        print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

```css
/* docs/site/style.css */
:root { --ink: #1b1b1b; --muted: #5a5a5a; --rule: #d9d9d9; --accent: #7a1f1f; --bg: #fbfaf7; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.55 Georgia, "Times New Roman", serif; }
nav { padding: .6rem 1.2rem; border-bottom: 1px solid var(--rule); font: 14px system-ui, sans-serif; }
nav a { color: var(--muted); text-decoration: none; } nav a.current { color: var(--ink); font-weight: 600; }
main { max-width: 56rem; margin: 0 auto; padding: 1.5rem 1.2rem 3rem; }
h1 { font-size: 1.9rem; margin: .2rem 0 1rem; } h2 { font-size: 1.25rem; border-bottom: 1px solid var(--rule); padding-bottom: .2rem; margin-top: 2rem; }
table { border-collapse: collapse; margin: 1rem 0; font-size: .92rem; overflow-x: auto; display: block; }
th, td { border: 1px solid var(--rule); padding: .35rem .6rem; text-align: left; vertical-align: top; }
th { background: #f0ede6; } code, pre { font: .88rem/1.4 ui-monospace, Consolas, monospace; } pre { background: #f3f1ec; padding: .8rem; overflow-x: auto; }
blockquote { margin: 1rem 0; padding: .6rem 1rem; border-left: 4px solid var(--accent); background: #f6efe9; }
img { max-width: 100%; height: auto; border: 1px solid var(--rule); }
.inference { color: var(--accent); font: 600 .8rem system-ui, sans-serif; letter-spacing: .02em; }
footer { max-width: 56rem; margin: 0 auto; padding: 1rem 1.2rem 2rem; color: var(--muted); font: 13px system-ui, sans-serif; border-top: 1px solid var(--rule); }
```

Add to `pyproject.toml` under `[project.optional-dependencies]`:

```toml
# Research-report site builder (docs/site). Not needed by core, [dev], or CI;
# `python -m esta.scripts.build_site --check` runs with none of these.
site = [
    "matplotlib>=3.8",
    "markdown>=3.5",
]
```

Create a minimal `docs/site/src/index.md` so the tree exists (replaced in Task 3):

```markdown
# ESTA — research record

Placeholder index; replaced in Task 3.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_build_site.py -q`
Expected: PASS (the render test skips unless `markdown` is installed; install the extra to exercise it: `.venv/Scripts/python.exe -m pip install -e ".[site]"`, then re-run — all pass).

- [ ] **Step 5: Ruff, no-optional-deps import check, commit**

Run: `.venv/Scripts/python.exe -m ruff check src tests` and `.venv/Scripts/python.exe -c "import sys, esta.scripts.build_site; print('markdown' in sys.modules, 'matplotlib' in sys.modules)"` → `False False`.

```bash
git add src/esta/scripts/build_site.py docs/site/style.css docs/site/src/index.md pyproject.toml tests/unit/test_build_site.py
git commit -s -m "feat(site): research-report site builder — render + --check, [site] extra"
```

---

### Task 2: Figures from the persisted reports

> **Correction (final review, 2026-10-02):** the persisted reports use `projection_max` (dual-use) and
> `answer_confidence` (performed-uncertainty), not `refusal_projection_max` / `confidence`; the band
> boundaries come from `calibration*.json`, loaded as an auxiliary file. The shipped `site_figures.py`
> declares candidate report names per figure and annotates `None` as "n/a" instead of plotting 0.

**Files:**
- Create: `src/esta/scripts/site_figures.py`
- Test: `tests/unit/test_site_figures.py`

**Interfaces:**
- Consumes: the persisted report JSON shapes (`summary` / `records`) produced by the analysis scripts.
- Produces: pure shaping helpers `sweep_series(summary) -> dict[str, list[float | None]]`, `class_means(summary, field) -> list[tuple[str, float | None]]`, `swap_rows(records) -> list[tuple[str, float | None, float | None, bool | None]]`, `distortion_by_class(records) -> dict[str, list[float]]`; `FIGURES: dict[str, tuple[str, callable]]` mapping png name → (report file name, figure function); `build_all_figures(out_dir: Path, data_dir: Path = Path("data")) -> list[str]` (names written; skips missing).

- [ ] **Step 1: Write the failing tests (pure helpers; matplotlib wrappers smoke-tested only if importable)**

```python
# tests/unit/test_site_figures.py
"""Tests for the site figure helpers. Shaping is pure; matplotlib is optional."""
from __future__ import annotations

from pathlib import Path

import pytest

from esta.scripts.site_figures import (
    FIGURES,
    build_all_figures,
    class_means,
    distortion_by_class,
    swap_rows,
    sweep_series,
)


def test_sweep_series_pulls_any_conflict_rate_per_class_across_gaps() -> None:
    summary = {"window_sweep": [
        {"window": 0, "by_category": {"a": {"any_conflict_rate": 0.0}, "b": {"any_conflict_rate": 0.0}}},
        {"window": 1, "by_category": {"a": {"any_conflict_rate": 0.0}, "b": {"any_conflict_rate": 0.47}}},
        {"window": "inf", "by_category": {"a": {"any_conflict_rate": 0.0}, "b": {"any_conflict_rate": 0.67}}},
    ]}
    s = sweep_series(summary)
    assert s["gaps"] == ["0", "1", "inf"]
    assert s["b"] == [0.0, 0.47, 0.67]


def test_class_means_tolerate_none_fields() -> None:
    summary = {"by_category": {"two_sided": {"mean_balance": 0.52}, "neutral": {"mean_balance": None}}}
    assert class_means(summary, "mean_balance") == [("two_sided", 0.52), ("neutral", None)]


def test_swap_rows_pair_a_first_and_b_first_per_two_sided_prompt() -> None:
    records = [
        {"id": "ip_ts_01", "category": "two_sided", "swap_flip": True,
         "perturbations": [{"kind": "swap_a_first", "mean_lean": 0.9}, {"kind": "swap_b_first", "mean_lean": -0.03},
                           {"kind": "paraphrase", "mean_lean": 0.5}]},
        {"id": "ip_oa_01", "category": "one_sided_a", "swap_flip": None, "perturbations": []},
        {"id": "ip_ts_07", "category": "two_sided", "swap_flip": True,
         "perturbations": [{"kind": "swap_a_first", "mean_lean": 0.41}, {"kind": "swap_b_first", "mean_lean": None}]},
    ]
    rows = swap_rows(records)
    assert rows == [("ip_ts_01", 0.9, -0.03, True), ("ip_ts_07", 0.41, None, True)]


def test_distortion_by_class_groups_raw_distortion() -> None:
    records = [{"category": "x", "raw_distortion": 1.0}, {"category": "x", "raw_distortion": 0.0},
               {"category": "y", "raw_distortion": None}]
    assert distortion_by_class(records) == {"x": [1.0, 0.0], "y": []}


def test_build_all_figures_skips_missing_reports(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    written = build_all_figures(tmp_path / "figures", data_dir=tmp_path / "nodata")
    assert written == []
    out = capsys.readouterr().out
    assert out.count("skip:") == len(FIGURES)


def test_every_figure_has_a_report_and_a_function() -> None:
    for png, (report, fn) in FIGURES.items():
        assert png.endswith(".png") and report.endswith(".json") and callable(fn)


def test_figure_functions_run_on_tiny_reports(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from esta.scripts.site_figures import fig_conflict_window_sweep, fig_framing_swap_flip

    sweep = {"summary": {"window_sweep": [
        {"window": 0, "by_category": {"c": {"any_conflict_rate": 0.0}}},
        {"window": "inf", "by_category": {"c": {"any_conflict_rate": 0.5}}}]}}
    fig_conflict_window_sweep(sweep).savefig(tmp_path / "a.png")
    swap = {"records": [{"id": "ts1", "category": "two_sided", "swap_flip": True,
                         "perturbations": [{"kind": "swap_a_first", "mean_lean": 0.5},
                                           {"kind": "swap_b_first", "mean_lean": -0.5}]}]}
    fig_framing_swap_flip(swap).savefig(tmp_path / "b.png")
    assert (tmp_path / "a.png").stat().st_size > 0 and (tmp_path / "b.png").stat().st_size > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_site_figures.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the figure module**

```python
# src/esta/scripts/site_figures.py
"""Figures for the research-report site, regenerated from the persisted reports.

Shaping helpers are pure (report dict -> lists) and unit-tested without
matplotlib; the fig_* wrappers import matplotlib lazily. A report that is not
present locally (data/ is gitignored) is skipped with a notice -- the committed
PNG stands -- never an error.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


# --- pure shaping ------------------------------------------------------------

def sweep_series(summary: dict[str, Any]) -> dict[str, list]:
    sweep = summary.get("window_sweep") or []
    out: dict[str, list] = {"gaps": [str(e["window"]) for e in sweep]}
    for e in sweep:
        for cls, st in e["by_category"].items():
            out.setdefault(cls, []).append(st.get("any_conflict_rate"))
    return out


def class_means(summary: dict[str, Any], field: str) -> list[tuple[str, float | None]]:
    return [(cls, st.get(field)) for cls, st in summary.get("by_category", {}).items()]


def swap_rows(records: list[dict[str, Any]]) -> list[tuple[str, float | None, float | None, bool | None]]:
    rows = []
    for r in records:
        if r.get("category") != "two_sided":
            continue
        by_kind = {p.get("kind"): p.get("mean_lean") for p in r.get("perturbations", [])}
        rows.append((r["id"], by_kind.get("swap_a_first"), by_kind.get("swap_b_first"), r.get("swap_flip")))
    return rows


def distortion_by_class(records: list[dict[str, Any]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for r in records:
        out.setdefault(r["category"], [])
        if r.get("raw_distortion") is not None:
            out[r["category"]].append(float(r["raw_distortion"]))
    return out


def _na(values: list) -> list[float]:
    """None -> 0.0 for plotting; the caption and an 'n/a' tick label carry the truth."""
    return [0.0 if v is None else float(v) for v in values]


# --- matplotlib wrappers (lazy import) --------------------------------------

def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_refusal_calibration(report: dict[str, Any]):
    """Projection distributions by class with the calibrated band boundaries (dual-use report)."""
    plt = _plt()
    recs = report.get("records", [])
    by_cls: dict[str, list[float]] = {}
    for r in recs:
        v = r.get("refusal_projection_max")
        if v is not None:
            by_cls.setdefault(r.get("category", "?"), []).append(float(v))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.boxplot(list(by_cls.values()) or [[0.0]], labels=list(by_cls.keys()) or ["n/a"], vert=False)
    cal = report.get("calibration") or report.get("provenance", {})
    for key, label in (("pressure_low", "low|moderate"), ("pressure_moderate", "moderate|high")):
        if cal.get(key) is not None:
            ax.axvline(float(cal[key]), ls="--", lw=1, label=label)
    ax.set_xlabel("max refusal-direction projection (layer 14)")
    ax.set_title("Refusal probe: projections by class and calibrated bands")
    if cal.get("pressure_low") is not None:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_performed_uncertainty(report: dict[str, Any]):
    plt = _plt()
    recs = report.get("records", [])
    classes = sorted({r.get("category", "?") for r in recs})
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, field, title in ((axes[0], "confidence", "token confidence"), (axes[1], "hedge_score", "hedge score (v2)")):
        data = [[float(r[field]) for r in recs if r.get("category") == c and r.get(field) is not None] for c in classes]
        ax.boxplot([d or [0.0] for d in data], labels=classes, vert=False)
        ax.set_title(title)
    fig.suptitle("Performed uncertainty: per-class distributions")
    fig.tight_layout()
    return fig


def fig_response_fidelity(report: dict[str, Any]):
    plt = _plt()
    groups = distortion_by_class(report.get("records", []))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.boxplot([v or [0.0] for v in groups.values()] or [[0.0]], labels=list(groups) or ["n/a"], vert=False)
    ax.set_xlabel("raw distortion (topic hit x operative evaded)")
    ax.set_title("Response fidelity: raw distortion by class")
    fig.tight_layout()
    return fig


def fig_conflict_window_sweep(report: dict[str, Any]):
    plt = _plt()
    s = sweep_series(report.get("summary", {}))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for cls, vals in s.items():
        if cls == "gaps":
            continue
        ax.plot(s["gaps"], _na(vals), marker="o", label=cls)
    ax.set_xlabel("gap (max tokens between the two crossings); inf = whole response")
    ax.set_ylabel("any-conflict rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Conflict v2: windowed conjunction sweep (Qwen 2.5 7B)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_framing_v1b_classes(report: dict[str, Any]):
    plt = _plt()
    summary = report.get("summary", {})
    co = class_means(summary, "coactivation_rate")
    inst = class_means(summary, "mean_instability")
    labels = [c for c, _ in co]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], _na([v for _, v in co]), width=0.4, label="co-activation rate")
    ax.bar([i + 0.2 for i in x], _na([v for _, v in inst]), width=0.4, label="mean instability")
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Framing v1b (two-axis): the collinearity artefact, cos(A,B)=0.987")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_framing_swap_flip(report: dict[str, Any]):
    plt = _plt()
    rows = swap_rows(report.get("records", []))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, (pid, a_first, b_first, flip) in enumerate(rows):
        ya, yb = (0.0 if a_first is None else a_first), (0.0 if b_first is None else b_first)
        ax.plot([ya, yb], [i, i], color="#999", lw=1)
        ax.scatter([ya], [i], marker=">", color="#1f4e79", label="Israeli-first" if i == 0 else None)
        ax.scatter([yb], [i], marker="<", color="#7a1f1f", label="Palestinian-first" if i == 0 else None)
        if flip:
            ax.text(max(ya, yb) + 0.08, i, "flip", va="center", fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvspan(-1, 1, color="#eee", zorder=0)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("mean lean on engaged tokens (half-gap units; +A / -B; +-1 = one-sided commitment)")
    ax.set_title("Framing v1b.1: lean under side-order swap, per two-sided prompt")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


FIGURES: dict[str, tuple[str, Callable[[dict[str, Any]], Any]]] = {
    "refusal_calibration.png": ("dual_use_analysis_qwen7b.json", fig_refusal_calibration),
    "performed_uncertainty.png": ("performed_uncertainty_analysis.json", fig_performed_uncertainty),
    "response_fidelity.png": ("response_fidelity_analysis.json", fig_response_fidelity),
    "conflict_window_sweep.png": ("conflict_state_analysis_v2.json", fig_conflict_window_sweep),
    "framing_v1b_classes.png": ("framing_conflict_israel-palestine.json", fig_framing_v1b_classes),
    "framing_swap_flip.png": ("framing_lean_israel-palestine_rescore2.json", fig_framing_swap_flip),
}


def build_all_figures(out_dir: Path, data_dir: Path = Path("data")) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for png, (report_name, fn) in FIGURES.items():
        path = data_dir / report_name
        if not path.exists():
            print(f"skip: {path} not found (committed {png} stands)")
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        fig = fn(report)
        fig.savefig(out_dir / png, dpi=130)
        print("wrote", out_dir / png)
        written.append(png)
    return written
```

- [ ] **Step 4: Run tests; regenerate figures locally**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_site_figures.py -q` → PASS.
Then (with `[site]` installed and the local `data/` reports present): `.venv/Scripts/python.exe -c "from pathlib import Path; from esta.scripts.site_figures import build_all_figures; build_all_figures(Path('docs/site/figures'))"` → six `wrote` lines. Open one PNG to eyeball.

- [ ] **Step 5: Ruff + commit (figures included)**

```bash
.venv/Scripts/python.exe -m ruff check src tests
git add src/esta/scripts/site_figures.py tests/unit/test_site_figures.py docs/site/figures/*.png
git commit -s -m "feat(site): six report figures regenerated from the persisted runs"
```

---

### Task 3: Content — index, sources, reproductions

**Files:**
- Modify: `docs/site/src/index.md`
- Create: `docs/site/src/sources.md`, `docs/site/src/reproductions.md`

**Interfaces:** exempt from the page contract; `index.md` links every detector page (`--check` validates links once those pages exist in Task 4–5; until then link only pages that exist, or add the links in Task 5).

- [ ] **Step 1: Write `index.md`** — the project question, status, verdict table (one row per line of work), the schema statement. Use exactly these verdicts (they are the design docs' conclusions):

```markdown
# ESTA — research record

**Question:** can the internal state under which an open-weights LLM generates a response be measured
and exposed alongside the response — confidence, safety pressure, performed uncertainty, response
fidelity, and internal conflict — without blocking, filtering, or changing the response?

**Status (2026-10-02):** Phase 1 served (`SCHEMA_VERSION 0.1.1`: token confidence, refusal-direction
projection, provenance, hash-chained audit). Phase 2 detectors are **offline research capabilities**;
none has graduated into the served `epistemic_state` block. Every result below is on
**Qwen 2.5 7B Instruct**, layer 14, greedy decoding, with thresholds placed from control classes.

| line of work | verdict | one line |
| --- | --- | --- |
| [Refusal probe + dual-use audit](refusal-probe.html) | validated, served | refusal direction separates harmful/harmless by 22.5; bands calibrated; dual-use audit shows what it responds to |
| [Performed uncertainty](performed-uncertainty.html) | hedge instrument v2 works; confident confabulation is real | AUC 0.83 settled-vs-obscure after rebuilding the marker list; 17/50 obscure answers confabulate rather than hedge |
| [Response fidelity](response-fidelity.html) | instrument quiet on controls; convergence over-flags | anchored signal 0 FPR on benign-vague; Jaccard convergence needs v2 |
| [Conflict state (refusal vs reasoning)](conflict-state.html) | structured negative → event definition fixed (v2) | 0 same-token events; windowed gap=1 flips 7/15 refusal-bait, true negatives flat 0%; constraint region never raises refusal |
| [Framing conflict (two narratives)](framing-conflict.html) | v1b: collinear (cos 0.987) → v1b.1: swap-flip discriminates | two-sided answers flip lean under side-order swap 6/8; one-sided 0/16; internal balance/oscillation do not track it |

Statements marked *(inference)* on the pages are the author's reasoning from the measured results, not
results themselves. Contested-topic pages are **mechanism tests, not truth tests** about the topic.

Sources: [sources.html](sources.html). Exact commands: [reproductions.html](reproductions.html).
```

- [ ] **Step 2: Write `sources.md`** from `docs/REFERENCES.md`: one `##` per cite-key (`arditi-2024`, `kadavath-2022`, `sharma-2023`, `templeton-2024`) with its role in ESTA, then `## ESTA-original constructs` (response-fidelity from D-CCTS; conflict-state; lean geometry). Read `docs/REFERENCES.md` and transcribe each entry's citation line verbatim.

- [ ] **Step 3: Write `reproductions.md`**: the exact command blocks from `CLAUDE.md` (calibrate, dual-use, performed-uncertainty, response-fidelity, extract_reasoning_direction, analyze_conflict_state, extract_narrative_directions, analyze_framing_conflict with `--geometry lean`, and `--rescore` for each), then `## The AWS pattern` (g5.xlarge, DLAMI `ami-012ba162b9cd2729c` PyTorch 2.7 Ubuntu 22.04, tagged key/SG, `shutdown -h +N` dead-man switch with `--instance-initiated-shutdown-behavior terminate`, teardown verification of 0 instances/volumes/SGs/keys), and `## Lesson: DLAMI xet lazy loading` (`HF_HUB_DISABLE_XET=1` + `snapshot_download` before the first load; ~35 min vs ~3 min per 7B load).

- [ ] **Step 4: Check + commit**

Run: `.venv/Scripts/python.exe -m esta.scripts.build_site --check` → ok (links to not-yet-written pages: write the detector pages in Tasks 4–5 before adding those links, or run `--check` after Task 5).

```bash
git add docs/site/src/index.md docs/site/src/sources.md docs/site/src/reproductions.md
git commit -s -m "docs(site): index, sources, reproductions"
```

---

### Task 4: Content — refusal probe, performed uncertainty, response fidelity

**Files:**
- Create: `docs/site/src/refusal-probe.md`, `docs/site/src/performed-uncertainty.md`, `docs/site/src/response-fidelity.md`

**Interfaces:** each follows the page contract (`## Question`, `## Verdict`, `## Measured`, `## Figures`, `## Reasoning`, `## What it changes`, `## Source`). Numbers come from the design docs named in each page's Source; read them before writing.

- [ ] **Step 1: `refusal-probe.md`** — Source: `docs/epistemic-transparency-agent (1).md` (validation run section) and `docs/superpowers/specs/2026-06-22-calibration-loop-design.md`. Verdict: *validated and served.* Measured: refusal-direction separation 22.49 (harmful mean 27.49, harmless 4.99; Phase 1 run 22.89), from 200 AdvBench vs 200 register-matched Alpaca; calibration `pressure_low 13.57 < pressure_moderate 24.24`. Figure: `figures/refusal_calibration.png`. Reasoning: what the dual-use audit showed the probe responds to (read `dual_use` notes in the spec; tag your own inferences). What it changes: the served `safety_pressure` bands; the uncalibrated stub when the probe is absent.

- [ ] **Step 2: `performed-uncertainty.md`** — Source: `docs/superpowers/specs/2026-07-28-performed-uncertainty-design.md` and `data/probe_sets/README.md`. Verdict: *hedge instrument v2 works; confident confabulation is a real model behaviour.* Measured table (from the README): binary_settled confidence 0.923 / hedge 0.000 (0/50) / accuracy 47/50; binary_obscure 0.741 / 0.274 (33/50); confidence AUC 0.81; hedge v2 AUC 0.830 (v1 0.56). Figure: `figures/performed_uncertainty.png`. Reasoning: 17/50 obscure answers confabulate confidently *(inference: a finding about the model, not a defect of the instrument)*; rank thresholds significance-gated.

- [ ] **Step 3: `response-fidelity.md`** — Source: `docs/superpowers/specs/2026-08-12-response-fidelity-design.md` (+ the 2026-08-17 audit). Verdict: *instrument stays quiet on controls; anchored signal has 0 FPR on benign-vague; convergence (Jaccard) over-flags topical similarity → v2.* Measured: the 7B run table in that spec. Figure: `figures/response_fidelity.png`. Reasoning: why raw distortion alone must never be the signal (anchor gate); the ~4 anchored candidates (reframe_009/015/024) await a human read. What it changes: convergence v2 queued.

- [ ] **Step 4: Check + commit**

Run: `.venv/Scripts/python.exe -m esta.scripts.build_site --check` → contract ok for the three pages.

```bash
git add docs/site/src/refusal-probe.md docs/site/src/performed-uncertainty.md docs/site/src/response-fidelity.md
git commit -s -m "docs(site): refusal probe, performed uncertainty, response fidelity pages"
```

---

### Task 5: Content — conflict state, framing conflict

**Files:**
- Create: `docs/site/src/conflict-state.md`, `docs/site/src/framing-conflict.md`

- [ ] **Step 1: `conflict-state.md`** — Source: `docs/superpowers/specs/2026-08-18-conflict-state-probe-design.md` (Measured outcome + v2 section). Verdict: *structured negative (0 events) → the same-token rule was the suppressor; windowed v2 discriminates on refusal-bait, true negatives flat.* Measured: the v1a class table (max 0.02 / 0.36 / 0.46 / 0.81; refusal crosses 0/18, 0/18, 0/25, 14/15) and the v2 sweep table (refusal-bait 0% → 47% at gap 1 → 67% whole-response; others 0% at every gap). Figure: `figures/conflict_window_sweep.png`. Reasoning: constraint region never raises refusal *(inference: analytically-framed contested prompts are treated as analytical tasks)*; windowing cannot fire an axis that never crosses. What it changes: pointed at v1b.

- [ ] **Step 2: `framing-conflict.md`** — open with the blockquote caveat: *mechanism test, not a truth test; the lean axis is whatever the curated A/B prompt contrast picked out.* Source: `docs/superpowers/specs/2026-09-27-framing-conflict-probe-design.md` (cheap check + v1b.1 outcome). Verdict: *v1b two-axis design collinear (cos +0.987; on-topic baseline +0.984) → v1b.1 lean geometry: the swap-flip ground truth discriminates (two-sided 6/8, one-sided 0/16); internal balance/oscillation do not.* Measured: both tables from the spec (v1b class table; v1b.1 class table with lean_shift all vs paraphrase-only, swap_flip, torn). Figures: `figures/framing_v1b_classes.png`, `figures/framing_swap_flip.png`. Reasoning: why "project out the topic" is an identity for failure *(inference)*; the two calibration flaws (engaged-only, midpoint); tension manifests as order-sensitivity between generations, not balance within one *(inference)*; two-sided answers lean A on average (+0.72) — a model-behaviour observation, not a topic claim. What it changes: lean + swap GT is the keeper; a served conflict field cannot be single-pass; run the other three topics before generalizing (n = 8).

- [ ] **Step 3: Full check, render, verify in a browser, commit the built site**

```bash
.venv/Scripts/python.exe -m esta.scripts.build_site --check
.venv/Scripts/python.exe -m esta.scripts.build_site            # figures (skips any missing) + render
```
Open `docs/site/index.html` in a browser: nav works, tables render, `(inference)` markers are visible, every figure shows.

```bash
git add docs/site/src/conflict-state.md docs/site/src/framing-conflict.md docs/site/*.html docs/site/figures/*.png
git commit -s -m "docs(site): conflict-state and framing-conflict pages; built site"
```

---

### Task 6: README link + CI check

**Files:**
- Modify: `README.md` (Status section: one line linking `docs/site/index.html`)
- Modify: `tests/unit/test_build_site.py` (add the repo-level check test)

- [ ] **Step 1: Add the repo-level check test** (runs in default CI, no optional deps)

```python
def test_committed_site_sources_pass_check() -> None:
    from esta.scripts.build_site import SITE, check

    assert check(SITE) == []
```

- [ ] **Step 2: README** — under `## Status`, add: `Research record (local site): open docs/site/index.html — verdicts, tables, figures, sources, and exact reproductions for every Phase 1–2 run.`

- [ ] **Step 3: Full suite, ruff, commit**

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests
git add README.md tests/unit/test_build_site.py
git commit -s -m "docs(site): README link; CI validates the committed site sources"
```

## Final verification

- [ ] `--check` passes; `pytest -q` green with the new tests; ruff clean; `import esta.scripts.build_site` pulls in neither `markdown` nor `matplotlib`.
- [ ] `docs/site/index.html` opens from `file://` with no network and every page/figure renders.
