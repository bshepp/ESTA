# tests/unit/test_probe_sets.py
"""Invariants for the Phase 2 probe sets.

These are controls, and a control that quietly loses its properties stops
controlling for anything. The invariants are enforced rather than reviewed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

PROBE_DIR = Path(__file__).resolve().parents[2] / "data" / "probe_sets"
VALIDATION_DIR = Path(__file__).resolve().parents[2] / "data" / "validation_cases"

BINARY_OPENER = re.compile(
    r"^(Is|Are|Does|Do|Did|Was|Were|Can|Has|Have|Should|Would|Will)\b"
)


def _load(name: str, directory: Path = PROBE_DIR) -> dict:
    return json.loads((directory / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def settled() -> dict:
    return _load("binary_settled.json")


@pytest.fixture(scope="module")
def obscure() -> dict:
    return _load("binary_obscure.json")


def test_both_sets_have_fifty_unique_prompts(settled: dict, obscure: dict) -> None:
    for data in (settled, obscure):
        prompts = data["prompts"]
        ids = [p["id"] for p in prompts]
        texts = [p["text"] for p in prompts]
        assert len(prompts) == 50, data["category"]
        assert len(set(ids)) == 50, f"duplicate ids in {data['category']}"
        assert len(set(texts)) == 50, f"duplicate text in {data['category']}"


def test_every_prompt_is_binary_form(settled: dict, obscure: dict) -> None:
    """Form-matching to the positive class is the entire reason these exist."""
    for data in (settled, obscure):
        for p in data["prompts"]:
            assert BINARY_OPENER.match(p["text"]), f"{p['id']} is not yes/no-shaped"


def test_settled_answers_are_balanced(settled: dict) -> None:
    """An all-yes control is passed by a model that always answers yes."""
    answers = [p["expected_answer"] for p in settled["prompts"]]
    assert set(answers) == {"yes", "no"}
    yes = answers.count("yes")
    assert 20 <= yes <= 30, f"yes/no imbalance: {yes} yes of {len(answers)}"


def test_obscure_set_claims_no_ground_truth(obscure: dict) -> None:
    """The answers are genuinely unknown; asserting one would be a fabrication."""
    for p in obscure["prompts"]:
        assert "expected_answer" not in p, p["id"]


def test_derived_from_ids_resolve_to_real_source_prompts(
    settled: dict, obscure: dict
) -> None:
    sources = {
        "binary_settled": _load("factual_unambiguous.json", VALIDATION_DIR),
        "binary_obscure": _load("low_confidence.json", VALIDATION_DIR),
    }
    for data in (settled, obscure):
        source_ids = {p["id"] for p in sources[data["category"]]["prompts"]}
        for p in data["prompts"]:
            assert p["derived_from"] in source_ids, f"{p['id']} -> {p['derived_from']}"


def test_probe_sets_are_not_in_the_calibration_corpus() -> None:
    """calibrate.py globs validation_cases/*.json; these must not leak in."""
    names = {p.name for p in VALIDATION_DIR.glob("*.json")}
    assert "binary_settled.json" not in names
    assert "binary_obscure.json" not in names
