# tests/unit/test_fidelity_probe_sets.py
"""Structural guards for the response-fidelity probe sets.

These enforce the curation rules from the design doc mechanically, so a bad
entry fails CI instead of silently weakening a control class.
"""

from __future__ import annotations

import json
from pathlib import Path

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
