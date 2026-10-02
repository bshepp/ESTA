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
