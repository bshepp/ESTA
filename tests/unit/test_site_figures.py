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
    for png, spec in FIGURES.items():
        assert png.endswith(".png") and spec["report"][0].endswith(".json") and callable(spec["fn"])


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
    for png, spec in sf.FIGURES.items():
        spec["fn"](tiny[png]).savefig(tmp_path / png)
        assert (tmp_path / png).stat().st_size > 0, png


# --- final-review fixes: real record keys, None annotation, aux calibration ----


def test_projection_by_class_uses_the_dual_use_report_key() -> None:
    from esta.scripts.site_figures import field_by_class

    recs = [{"category": "pair", "projection_max": 20.0}, {"category": "pair", "projection_max": None},
            {"category": "ctrl", "projection_max": 4.0}]
    assert field_by_class(recs, "projection_max") == {"pair": [20.0], "ctrl": [4.0]}


def test_split_na_separates_undefined_points() -> None:
    from esta.scripts.site_figures import split_na

    xs, ok, na = split_na(["0", "1", "inf"], [0.0, None, 0.5])
    assert (xs, ok, na) == (["0", "inf"], [0.0, 0.5], ["1"])


def test_figures_declare_report_candidates_and_optional_aux() -> None:
    for png, spec in FIGURES.items():
        assert png.endswith(".png")
        assert spec["report"] and all(r.endswith(".json") for r in spec["report"])
        assert callable(spec["fn"])
        assert all(a.endswith(".json") for a in spec.get("aux", ()))


def test_figures_plot_real_schema_records_not_placeholders(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from esta.scripts.site_figures import (
        fig_conflict_window_sweep,
        fig_performed_uncertainty,
        fig_refusal_calibration,
    )

    refusal = {"records": [{"category": "pair", "projection_max": 20.0}, {"category": "ctrl", "projection_max": 4.0}],
               "_aux": {"pressure_low": 13.08, "pressure_moderate": 24.22}}
    ax = fig_refusal_calibration(refusal).axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["pair", "ctrl"]   # real classes, not n/a
    assert len(ax.lines) >= 2                                                  # the two band lines drawn

    perf = {"records": [{"category": "settled", "answer_confidence": 0.9, "hedge_score": 0.0},
                        {"category": "obscure", "answer_confidence": 0.7, "hedge_score": 0.3}]}
    fig = fig_performed_uncertainty(perf)
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["obscure", "settled"]

    sweep = {"summary": {"window_sweep": [
        {"window": 0, "by_category": {"c": {"any_conflict_rate": 0.0}}},
        {"window": 1, "by_category": {"c": {"any_conflict_rate": None}}},
        {"window": "inf", "by_category": {"c": {"any_conflict_rate": 0.5}}}]}}
    ax = fig_conflict_window_sweep(sweep).axes[0]
    assert any(t.get_text() == "n/a" for t in ax.texts)                       # None is annotated, not plotted as 0


def test_build_all_figures_picks_first_existing_candidate_and_loads_aux(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    import json

    from esta.scripts import site_figures as sf

    data = tmp_path / "data"
    data.mkdir()
    (data / "dual_use_analysis.json").write_text(json.dumps(
        {"records": [{"category": "pair", "projection_max": 20.0}]}), encoding="utf-8")
    (data / "calibration.json").write_text(json.dumps({"pressure_low": 1.0, "pressure_moderate": 2.0}), encoding="utf-8")
    written = sf.build_all_figures(tmp_path / "figures", data_dir=data)
    assert written == ["refusal_calibration.png"]       # the one figure whose candidates exist
