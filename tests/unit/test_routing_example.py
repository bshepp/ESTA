"""Tests for the routing policy in examples/routing_hooks.py.

The example is documentation that can rot; pinning its pure `route()` function
keeps it honest. Loaded by path because examples/ is not an installed package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "routing_hooks.py"
_spec = importlib.util.spec_from_file_location("routing_hooks", _EXAMPLE)
routing_hooks = importlib.util.module_from_spec(_spec)
# Register before exec: the module's @dataclass resolves its string annotations
# (from __future__ import annotations) via sys.modules[__name__].
sys.modules[_spec.name] = routing_hooks
_spec.loader.exec_module(routing_hooks)

route = routing_hooks.route


def _state(*, spikes: int = 0, low_margin: float = 0.0,
           pressure: str = "low", proj_max: float = 0.1) -> dict:
    return {
        "confidence": {"entropy_spike_count": spikes, "low_margin_fraction": low_margin},
        "safety_pressure": {"calibrated_pressure": pressure, "refusal_projection_max": proj_max},
    }


def test_clean_state_proceeds() -> None:
    decision = route(_state())
    assert decision.action == routing_hooks.PROCEED
    assert decision.reasons == []


def test_elevated_pressure_routes_to_review() -> None:
    for label in ("moderate", "high"):
        decision = route(_state(pressure=label, proj_max=1.5))
        assert decision.action == routing_hooks.REVIEW
        assert decision.reasons


def test_low_confidence_requires_sources() -> None:
    decision = route(_state(spikes=10))
    assert decision.action == routing_hooks.GATHER_SOURCES
    decision = route(_state(low_margin=0.5))
    assert decision.action == routing_hooks.GATHER_SOURCES


def test_uncalibrated_proceeds_with_warning() -> None:
    decision = route(_state(pressure="uncalibrated"))
    assert decision.action == routing_hooks.WARN_UNCALIBRATED
    assert decision.reasons


def test_pressure_outranks_low_confidence() -> None:
    # When both fire, the more severe action (review) wins but both reasons surface.
    decision = route(_state(spikes=10, pressure="high", proj_max=2.0))
    assert decision.action == routing_hooks.REVIEW
    assert len(decision.reasons) == 2


@pytest.mark.parametrize("label", ["low", "moderate", "high", "uncalibrated"])
def test_route_handles_every_pressure_label(label: str) -> None:
    # Must not raise for any valid PressureLabel.
    assert route(_state(pressure=label)).action
