"""Tests for the site figure helpers. Shaping is pure; matplotlib is optional."""
from __future__ import annotations

from pathlib import Path

import pytest

from esta.scripts.site_figures import (
    FIGURES,
    build_all_figures,
    class_means,
    distortion_by_class,
    swap_rows,
    sweep_series,
)


def test_sweep_series_pulls_any_conflict_rate_per_class_across_gaps() -> None:
    summary = {"window_sweep": [
        {"window": 0, "by_category": {"a": {"any_conflict_rate": 0.0}, "b": {"any_conflict_rate": 0.0}}},
        {"window": 1, "by_category": {"a": {"any_conflict_rate": 0.0}, "b": {"any_conflict_rate": 0.47}}},
        {"window": "inf", "by_category": {"a": {"any_conflict_rate": 0.0}, "b": {"any_conflict_rate": 0.67}}},
    ]}
    s = sweep_series(summary)
    assert s["gaps"] == ["0", "1", "inf"]
    assert s["b"] == [0.0, 0.47, 0.67]


def test_class_means_tolerate_none_fields() -> None:
    summary = {"by_category": {"two_sided": {"mean_balance": 0.52}, "neutral": {"mean_balance": None}}}
    assert class_means(summary, "mean_balance") == [("two_sided", 0.52), ("neutral", None)]


def test_swap_rows_pair_a_first_and_b_first_per_two_sided_prompt() -> None:
    records = [
        {"id": "ip_ts_01", "category": "two_sided", "swap_flip": True,
         "perturbations": [{"kind": "swap_a_first", "mean_lean": 0.9}, {"kind": "swap_b_first", "mean_lean": -0.03},
                           {"kind": "paraphrase", "mean_lean": 0.5}]},
        {"id": "ip_oa_01", "category": "one_sided_a", "swap_flip": None, "perturbations": []},
        {"id": "ip_ts_07", "category": "two_sided", "swap_flip": True,
         "perturbations": [{"kind": "swap_a_first", "mean_lean": 0.41}, {"kind": "swap_b_first", "mean_lean": None}]},
    ]
    rows = swap_rows(records)
    assert rows == [("ip_ts_01", 0.9, -0.03, True), ("ip_ts_07", 0.41, None, True)]


def test_distortion_by_class_groups_raw_distortion() -> None:
    records = [{"category": "x", "raw_distortion": 1.0}, {"category": "x", "raw_distortion": 0.0},
               {"category": "y", "raw_distortion": None}]
    assert distortion_by_class(records) == {"x": [1.0, 0.0], "y": []}


def test_build_all_figures_skips_missing_reports(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    written = build_all_figures(tmp_path / "figures", data_dir=tmp_path / "nodata")
    assert written == []
    out = capsys.readouterr().out
    assert out.count("skip:") == len(FIGURES)


def test_every_figure_has_a_report_and_a_function() -> None:
    for png, (report, fn) in FIGURES.items():
        assert png.endswith(".png") and report.endswith(".json") and callable(fn)


def test_every_figure_function_runs_on_a_tiny_report(tmp_path: Path) -> None:
    """ALL six wrappers, on tiny reports with None fields -- a renamed matplotlib kwarg
    (boxplot labels -> tick_labels in 3.9) slipped past a smoke test that covered only two."""
    pytest.importorskip("matplotlib")
    from esta.scripts import site_figures as sf

    tiny = {
        "refusal_calibration.png": {"records": [{"category": "c", "refusal_projection_max": 1.0}],
                                    "provenance": {"pressure_low": 0.5, "pressure_moderate": 1.5}},
        "performed_uncertainty.png": {"records": [{"category": "c", "confidence": 0.9, "hedge_score": 0.1},
                                                  {"category": "d", "confidence": None, "hedge_score": None}]},
        "response_fidelity.png": {"records": [{"category": "c", "raw_distortion": 1.0},
                                              {"category": "d", "raw_distortion": None}]},
        "conflict_window_sweep.png": {"summary": {"window_sweep": [
            {"window": 0, "by_category": {"c": {"any_conflict_rate": 0.0}}},
            {"window": "inf", "by_category": {"c": {"any_conflict_rate": None}}}]}},
        "framing_v1b_classes.png": {"summary": {"by_category": {
            "two_sided": {"coactivation_rate": 0.1, "mean_instability": 0.6},
            "neutral": {"coactivation_rate": None, "mean_instability": None}}}},
        "framing_swap_flip.png": {"records": [{"id": "ts1", "category": "two_sided", "swap_flip": True,
                                               "perturbations": [{"kind": "swap_a_first", "mean_lean": 0.5},
                                                                 {"kind": "swap_b_first", "mean_lean": None}]}]},
    }
    assert set(tiny) == set(sf.FIGURES)
    for png, (_, fn) in sf.FIGURES.items():
        fn(tiny[png]).savefig(tmp_path / png)
        assert (tmp_path / png).stat().st_size > 0, png
