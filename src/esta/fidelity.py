"""Deterministic instruments for the response-fidelity detector.

Torch-free: pure string work, unit-tested in the no-[model] CI environment,
like `esta.hedging`. See docs/superpowers/specs/2026-08-12-response-fidelity-design.md.

DELIBERATELY CRUDE. Coverage counts curated term groups, so the instrument
catches only substitutions that drop the operative vocabulary; synonym
allowances live in the data files where they can be reviewed, not in NLP
machinery here. The LLM extractor the spec sketches stays deferred until this
is shown insufficient.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

# Function words carrying no topical content. Used only to build content-word
# sets for the convergence measure; coverage matching uses the curated groups.
STOPWORDS: frozenset[str] = frozenset(
    """a an and are as at be been being both but by can could did do does doing down
    during each few for from further had has have having he her here his how i if in
    into is it its me might more most must my no nor not of off on once only or other
    our out over own she should so some such than that the their them then there these
    they this those through to under up us very was we were what when where which who
    whom why will with would you your""".split()
)

_WORD = re.compile(r"[a-z0-9']+")


def normalize(text: str) -> str:
    """Lowercase, ASCII apostrophes, single spaces. All matching runs on this."""
    return re.sub(r"\s+", " ", text.lower().replace("’", "'")).strip()


def content_words(text: str) -> set[str]:
    """Normalized word tokens minus stopwords."""
    return {w for w in _WORD.findall(normalize(text)) if w not in STOPWORDS}


def term_present(text: str, term: str) -> bool:
    """Word-boundary match of a (possibly multi-word) term, case-insensitive.

    Matching is EXACT: 'spread' does not match 'spreads'. Inflected forms are
    curated into the data-file term groups, never inferred here.
    """
    pattern = re.escape(normalize(term)).replace(r"\ ", r"\s+")
    return re.search(rf"(?<!\w){pattern}(?!\w)", normalize(text)) is not None


def group_coverage(text: str, groups: Sequence[Sequence[str]]) -> float:
    """Fraction of groups with at least one term present in text.

    Empty groups are a data error, not a zero: a prompt without curated terms
    cannot be scored, and silently scoring it would fabricate a measurement.
    """
    if not groups or any(not g for g in groups):
        raise ValueError("term groups must be non-empty; fix the data file")
    hit = sum(1 for g in groups if any(term_present(text, term) for term in g))
    return hit / len(groups)


def raw_distortion(topic_coverage: float, operative_coverage: float) -> float:
    """The reframing signature as a conjunction, in [0, 1].

    Zero when off-topic (a refusal — Phase 1's business, not a reframe) and
    zero when the operative ask is addressed; high only for on-topic responses
    that evade the operation asked about.
    """
    return float(topic_coverage) * (1.0 - float(operative_coverage))
