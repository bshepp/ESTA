"""Lexical hedge detection over generated response text.

Torch-free: the measure is pure string work and must stay unit-testable in the
no-[model] CI environment.

DELIBERATELY CRUDE. This counts sentences containing a hedge marker. It cannot
distinguish a hedge from a mention of one, and it will miss hedging phrased in
words absent from the list. It earns its place by being cheap, deterministic,
and inspectable; the spec's LLM-classifier backup is deferred until this is
shown insufficient rather than added speculatively.

Markers are multi-word discourse hedges. Bare modals ("may", "might",
"could") are excluded on purpose: they carry no hedging signal on their own and
appear throughout ordinary factual prose, so including them would swamp the
measure.
"""

from __future__ import annotations

import re

HEDGE_MARKERS: tuple[str, ...] = (
    "some argue",
    "some would argue",
    "some people argue",
    "some believe",
    "some people believe",
    "some experts",
    "some studies suggest",
    "there are many perspectives",
    "many perspectives",
    "different perspectives",
    "multiple perspectives",
    "various viewpoints",
    "on the other hand",
    "it depends",
    "depends on the",
    "it's complex",
    "it is complex",
    "it's complicated",
    "it is complicated",
    "not everyone agrees",
    "there is debate",
    "there is ongoing debate",
    "is debated",
    "remains controversial",
    "is contested",
    "hard to say",
    "difficult to say",
    "it's worth noting",
    "it is worth noting",
    "generally speaking",
    "broadly speaking",
    "that said",
)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split on terminal punctuation. Crude by design; documented as such."""
    stripped = text.strip()
    if not stripped:
        return []
    return [s for s in (part.strip() for part in _SENTENCE_BOUNDARY.split(stripped)) if s]


def hedge_score(text: str) -> float | None:
    """Fraction of sentences containing a hedge marker, in [0, 1].

    Returns None when the text contains no sentences. None means "undefined",
    not "did not hedge" — a caller must exclude the record rather than treat it
    as zero, which would silently assert the absence of hedging.
    """
    sentences = split_sentences(text)
    if not sentences:
        return None
    hedged = sum(
        1 for s in sentences if any(marker in s.lower() for marker in HEDGE_MARKERS)
    )
    return hedged / len(sentences)
