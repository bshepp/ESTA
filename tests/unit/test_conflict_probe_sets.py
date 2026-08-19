"""Structural guards for the conflict-probe validation sets."""

from __future__ import annotations

import json
from pathlib import Path

PROBE_DIR = Path(__file__).resolve().parents[2] / "data" / "probe_sets"


def _load(name: str) -> dict:
    return json.loads((PROBE_DIR / name).read_text(encoding="utf-8"))


class TestConstraintRegion:
    def test_shape_counts_and_unique_ids(self) -> None:
        data = _load("constraint_region.json")
        assert data["category"] == "constraint_region"
        prompts = data["prompts"]
        assert len(prompts) >= 15
        ids = [p["id"] for p in prompts]
        assert len(ids) == len(set(ids))
        for p in prompts:
            assert p["text"].strip() and p["topic"].strip()

    def test_israel_palestine_is_present(self) -> None:
        # The originator's empirical example must be in the positive set.
        blob = json.dumps(_load("constraint_region.json")).lower()
        assert "israel" in blob and "palestin" in blob

    def test_prompts_are_analytical_not_one_word_questions(self) -> None:
        # Reasoning must be able to fire: each ask invites analysis, so it is
        # not a bare recall question. Heuristic: reasonably long.
        for p in _load("constraint_region.json")["prompts"]:
            assert len(p["text"]) >= 40, p["id"]


class TestReasoningControls:
    def test_uncontested_analytical(self) -> None:
        data = _load("uncontested_analytical.json")
        assert data["category"] == "uncontested_analytical"
        prompts = data["prompts"]
        assert len(prompts) >= 15
        ids = [p["id"] for p in prompts]
        assert len(ids) == len(set(ids))
        for p in prompts:
            # analytical: invites reasoning, so not a one-word question
            assert len(p["text"]) >= 40, p["id"]

    def test_direct_recall(self) -> None:
        data = _load("direct_recall.json")
        assert data["category"] == "direct_recall"
        prompts = data["prompts"]
        assert len(prompts) >= 15
        ids = [p["id"] for p in prompts]
        assert len(ids) == len(set(ids))
        for p in prompts:
            assert p["text"].strip()
