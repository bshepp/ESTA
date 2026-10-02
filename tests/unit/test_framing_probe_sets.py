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


# --- v1b.1 perturbation data (Israel-Palestine first) ------------------------
# The lean ground truth needs, per two_sided prompt, an authored side-order
# swap pair (A-first / B-first) and authored framing-shifting paraphrases; the
# one_sided classes get paraphrases only (the shared perturbation family).
# Plus an ON-TOPIC descriptive neutral file for the on-topic-baseline experiment.
V1B1_TOPICS = ["israel-palestine"]


@pytest.mark.parametrize("topic", V1B1_TOPICS)
def test_every_two_sided_prompt_has_a_swap_pair_and_paraphrases(topic) -> None:  # noqa: ANN001
    data = json.loads((PROBE_DIR / f"framing_{topic}.json").read_text(encoding="utf-8"))
    for p in data["prompts"]:
        if p["class"] == "two_sided":
            swap = p.get("swap")
            assert isinstance(swap, list) and len(swap) == 2, f"{p['id']}: swap must be [A-first, B-first]"
            assert all(s.strip() and s.strip() != p["text"].strip() for s in swap), f"{p['id']}: swap texts"
            assert swap[0] != swap[1], f"{p['id']}: swap pair must differ"
        if p["class"] in ("two_sided", "one_sided_a", "one_sided_b"):
            paras = p.get("paraphrases")
            assert isinstance(paras, list) and len(paras) >= 1, f"{p['id']}: needs authored paraphrases"
            assert all(s.strip() and s.strip() != p["text"].strip() for s in paras), f"{p['id']}: paraphrases"


@pytest.mark.parametrize("topic", V1B1_TOPICS)
def test_on_topic_neutral_file_exists_and_is_disjoint(topic) -> None:  # noqa: ANN001
    path = PROBE_DIR / "extraction" / f"{topic}_neutral_ontopic.txt"
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 15
    val = {p["text"].strip() for p in
           json.loads((PROBE_DIR / f"framing_{topic}.json").read_text(encoding="utf-8"))["prompts"]}
    a = {ln.strip() for ln in (PROBE_DIR / "extraction" / f"{topic}_a.txt").read_text(encoding="utf-8").splitlines()}
    b = {ln.strip() for ln in (PROBE_DIR / "extraction" / f"{topic}_b.txt").read_text(encoding="utf-8").splitlines()}
    assert not (set(lines) & val), "on-topic neutral leaks into validation"
    assert not (set(lines) & (a | b)), "on-topic neutral duplicates an advocacy line"
