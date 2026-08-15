# tests/unit/test_fidelity_probe_sets.py
"""Structural guards for the response-fidelity probe sets.

These enforce the curation rules from the design doc mechanically, so a bad
entry fails CI instead of silently weakening a control class.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROBE_DIR = Path(__file__).resolve().parents[2] / "data" / "probe_sets"


def _load(name: str) -> dict:
    return json.loads((PROBE_DIR / name).read_text(encoding="utf-8"))


def _check_groups(prompt: dict, *, kinds: tuple[str, ...] = ("topic_groups", "operative_groups")) -> None:
    for kind in kinds:
        groups = prompt.get(kind)
        assert groups, f"{prompt['id']}: missing or empty {kind}"
        for group in groups:
            assert group and all(isinstance(t, str) and t.strip() for t in group), (
                f"{prompt['id']}: empty group or blank term in {kind}"
            )


class TestReframePairs:
    def test_counts_and_domains(self) -> None:
        prompts = _load("reframe_pairs.json")["prompts"]
        domains = [p["domain"] for p in prompts]
        assert domains.count("security") >= 35
        assert domains.count("contested") >= 12
        assert set(domains) <= {"security", "contested"}

    def test_ids_unique_and_fields_present(self) -> None:
        prompts = _load("reframe_pairs.json")["prompts"]
        ids = [p["id"] for p in prompts]
        assert len(ids) == len(set(ids))
        for p in prompts:
            assert p["text"].strip() and p["substitute_text"].strip()
            assert p["text"] != p["substitute_text"], p["id"]
            _check_groups(p)

    def test_substitute_shares_topic_but_not_operation(self) -> None:
        """The predicted substitute must be genuinely adjacent: same topic,
        different operation — otherwise convergence cannot distinguish
        reframing from topical similarity (design doc, curation rules)."""
        from esta.fidelity import group_coverage

        prompts = _load("reframe_pairs.json")["prompts"]
        for p in prompts:
            assert group_coverage(p["substitute_text"], p["topic_groups"]) > 0.0, (
                f"{p['id']}: substitute drops the topic entirely"
            )
            assert group_coverage(p["text"], p["operative_groups"]) > 0.0, (
                f"{p['id']}: the borderline ask does not contain its own operative terms"
            )


class TestFidelityControls:
    @pytest.mark.parametrize(
        ("name", "minimum"),
        [
            ("direct_answer_controls.json", 20),
            ("benign_vague.json", 20),
            ("refusal_boundary.json", 12),
        ],
    )
    def test_counts_ids_and_groups(self, name: str, minimum: int) -> None:
        data = _load(name)
        prompts = data["prompts"]
        assert data["category"] == name.removesuffix(".json")
        assert len(prompts) >= minimum
        ids = [p["id"] for p in prompts]
        assert len(ids) == len(set(ids))
        for p in prompts:
            assert p["text"].strip()
            _check_groups(p)

    @pytest.mark.parametrize("name", ["direct_answer_controls.json", "refusal_boundary.json"])
    def test_derived_sets_reference_their_sources(self, name: str) -> None:
        for p in _load(name)["prompts"]:
            assert p.get("derived_from"), p["id"]

    def test_direct_controls_contain_their_own_operative_terms(self) -> None:
        """A direct ask must be scoreable as answered: its own text carries the
        operative vocabulary, so a faithful answer echoing the ask scores low
        distortion. Without this, a control could be unpassable by design."""
        from esta.fidelity import group_coverage

        for p in _load("direct_answer_controls.json")["prompts"]:
            assert group_coverage(p["text"], p["operative_groups"]) > 0.0, p["id"]
