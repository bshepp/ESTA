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

# Anchored to the repo, not the cwd: --check from anywhere validates the committed site,
# and never passes vacuously on an empty glob.
SITE = Path(__file__).resolve().parents[3] / "docs" / "site"
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
# every non-image link target; the <...> form carries targets with spaces or parentheses
_LINK = re.compile(r"(?<!!)\[[^\]]*\]\((?:<([^>]+)>|([^)\s]+))\)")
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
    """Relative link targets, fragment stripped: pages (.html) and files such as ../specs/x.md."""
    out: set[str] = set()
    for angled, plain in _LINK.findall(md_text):
        link = angled or plain
        if "://" in link or link.startswith(("#", "mailto:")):
            continue
        target = link.split("#", 1)[0]
        if target:
            out.add(target)
    return out


def check(site: Path = SITE) -> list[str]:
    src, figures = site / "src", site / "figures"
    names = page_names(src)
    if not names:
        return [f"no source pages found under {src}"]
    problems: list[str] = []
    for name in names:
        text = (src / f"{name}.md").read_text(encoding="utf-8")
        problems += check_contract(name, text)
        for fig in sorted(referenced_figures(text)):
            if not (figures / fig).exists():
                problems.append(f"{name}.md: figure 'figures/{fig}' does not exist")
        for link in sorted(internal_links(text)):
            if link.endswith(".html"):
                if Path(link).stem not in names:
                    problems.append(f"{name}.md: link '{link}' has no source page")
            elif not (site / link).resolve().exists():
                problems.append(f"{name}.md: link '{link}' does not resolve from docs/site/")
    return problems


def _chrome(title: str, body_html: str, current: str) -> str:
    def _link(n: str, label: str) -> str:
        cls = ' class="current"' if n == current else ""   # no backslashes in f-string exprs (py3.11)
        return f'<a href="{n}.html"{cls}>{label}</a>'

    nav = " | ".join(_link(n, label) for n, label in NAV)
    return ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{title}</title>"
            '<link rel="stylesheet" href="style.css"></head><body>\n'
            f"<nav>{nav}</nav>\n<main>\n{body_html}\n</main>\n"
            "<footer>ESTA research record. Mechanism tests on Qwen 2.5 7B Instruct; SCHEMA_VERSION 0.1.1 -- "
            "the Phase 1 refusal probe is the only served signal; nothing else here is a served capability."
            "</footer>\n</body></html>\n")


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
        from esta.scripts.site_figures import build_all_figures

        build_all_figures(args.site / "figures")
    for out in render(args.site):
        print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
