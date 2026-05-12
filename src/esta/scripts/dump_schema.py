"""Dump the EpistemicState JSON Schema to disk.

The on-disk artifact at src/esta/schema/epistemic_state.schema.json is the
canonical contract for downstream consumers that don't load the pydantic model.

Re-run this any time the pydantic model in epistemic_state.py changes. CI runs
test_schema_drift to catch developers who forget; the test fails with a clear
message instructing them to re-run this script.

Usage:
    python -m esta.scripts.dump_schema
    python -m esta.scripts.dump_schema --output /tmp/check.schema.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from esta.schema import EpistemicState

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "schema" / "epistemic_state.schema.json"


def build_schema() -> dict[str, Any]:
    """Return the canonical JSON Schema for EpistemicState."""
    return EpistemicState.model_json_schema()


def render_schema(schema: dict[str, Any]) -> str:
    """Render schema as deterministic, human-readable JSON."""
    return json.dumps(schema, sort_keys=True, indent=2) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_schema(build_schema()), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
