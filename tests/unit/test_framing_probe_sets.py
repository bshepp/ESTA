"""Structure tests for the v1b framing probe sets."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROBE_DIR = Path("data/probe_sets")
TOPICS = ["israel-palestine", "abortion", "gun-control", "taiwan-sovereignty"]
VALID_CLASSES = {"two_sided", "one_sided_a", "one_sided_b"}


@pytest.mark.parametrize("topic", TOPICS)
def test_framing_set_is_well_formed(topic) -> None:  # noqa: ANN001
    data = json.loads((PROBE_DIR / f"framing_{topic}.json").read_text(encoding="utf-8"))
    prompts = data["prompts"]
    ids = [p["id"] for p in prompts]
    assert len(ids) == len(set(ids)), "duplicate ids"
    classes = {p["class"] for p in prompts}
    assert classes == VALID_CLASSES, f"missing/extra class in {topic}: {classes}"
    for p in prompts:
        assert p["text"].strip()
        assert p["topic"] == topic
        assert p["class"] in VALID_CLASSES


def test_israel_palestine_has_two_sided_prompts() -> None:
    data = json.loads((PROBE_DIR / "framing_israel-palestine.json").read_text(encoding="utf-8"))
    assert any(p["class"] == "two_sided" for p in data["prompts"])


@pytest.mark.parametrize("topic", TOPICS)
def test_extraction_contrasts_exist_and_are_disjoint_from_validation(topic) -> None:  # noqa: ANN001
    val_text = {
        p["text"].strip()
        for p in json.loads((PROBE_DIR / f"framing_{topic}.json").read_text(encoding="utf-8"))["prompts"]
    }
    for side in ("a", "b", "neutral"):
        lines = [ln.strip() for ln in
                 (PROBE_DIR / "extraction" / f"{topic}_{side}.txt").read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
        assert len(lines) >= 8, f"{topic}_{side}: too few extraction prompts"
        assert not (set(lines) & val_text), f"{topic}_{side}: extraction leaks into validation"
