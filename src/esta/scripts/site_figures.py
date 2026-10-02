"""Figures for the research-report site, regenerated from the persisted reports.

Shaping helpers are pure (report dict -> lists) and unit-tested without
matplotlib; the fig_* wrappers import matplotlib lazily. A report that is not
present locally (data/ is gitignored) is skipped with a notice -- the committed
PNG stands -- never an error. Each figure declares candidate report filenames
(the CLAUDE.md command outputs and the local run names) and, optionally, an
auxiliary file (the calibration JSON for the band boundaries).

Undefined values (None -- the theta=None paths are real data) are never
plotted as zeros: they are left out and annotated "n/a".
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

# --- pure shaping ------------------------------------------------------------


def sweep_series(summary: dict[str, Any]) -> dict[str, list]:
    sweep = summary.get("window_sweep") or []
    out: dict[str, list] = {"gaps": [str(e["window"]) for e in sweep]}
    for e in sweep:
        for cls, st in e["by_category"].items():
            out.setdefault(cls, []).append(st.get("any_conflict_rate"))
    return out


def class_means(summary: dict[str, Any], field: str) -> list[tuple[str, float | None]]:
    return [(cls, st.get(field)) for cls, st in summary.get("by_category", {}).items()]


def swap_rows(records: list[dict[str, Any]]) -> list[tuple[str, float | None, float | None, bool | None]]:
    rows = []
    for r in records:
        if r.get("category") != "two_sided":
            continue
        by_kind = {p.get("kind"): p.get("mean_lean") for p in r.get("perturbations", [])}
        rows.append((r["id"], by_kind.get("swap_a_first"), by_kind.get("swap_b_first"), r.get("swap_flip")))
    return rows


def field_by_class(records: list[dict[str, Any]], key: str) -> dict[str, list[float]]:
    """Per-class list of a record field, in first-seen class order; None values are left out."""
    out: dict[str, list[float]] = {}
    for r in records:
        out.setdefault(r["category"], [])
        if r.get(key) is not None:
            out[r["category"]].append(float(r[key]))
    return out


def distortion_by_class(records: list[dict[str, Any]]) -> dict[str, list[float]]:
    return field_by_class(records, "raw_distortion")


def split_na(xs: list, vals: list) -> tuple[list, list[float], list]:
    """(xs with values, those values, xs whose value is None) -- None is annotated, never plotted."""
    xs_ok = [x for x, v in zip(xs, vals, strict=True) if v is not None]
    ok = [float(v) for v in vals if v is not None]
    xs_na = [x for x, v in zip(xs, vals, strict=True) if v is None]
    return xs_ok, ok, xs_na


# --- matplotlib wrappers (lazy import) --------------------------------------


def _hbox(ax, data, labels):  # noqa: ANN001
    """Horizontal boxplot with tick labels set portably (boxplot(labels=) was renamed
    tick_labels in matplotlib 3.9; vert= deprecated in 3.11 -> orientation=)."""
    ax.boxplot(data, orientation="horizontal")   # matplotlib >= 3.10
    ax.set_yticks(range(1, len(labels) + 1), labels)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _first_present(record: dict[str, Any], *keys: str):
    for k in keys:
        if record.get(k) is not None:
            return record[k]
    return None


def fig_refusal_calibration(report: dict[str, Any]):
    """Projection distributions by class (dual-use audit) with the calibrated band boundaries.

    Records carry `projection_max`; the band edges come from the calibration JSON, loaded
    by build_all_figures into report["_aux"] (falling back to any embedded calibration).
    """
    plt = _plt()
    recs = [dict(r, projection_max=_first_present(r, "projection_max", "refusal_projection_max"))
            for r in report.get("records", [])]
    by_cls = field_by_class(recs, "projection_max")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    _hbox(ax, [v or [0.0] for v in by_cls.values()] or [[0.0]], list(by_cls) or ["n/a"])
    cal = report.get("_aux") or report.get("calibration") or report.get("provenance", {})
    drawn = False
    for key, label in (("pressure_low", "low | moderate"), ("pressure_moderate", "moderate | high")):
        if cal.get(key) is not None:
            ax.axvline(float(cal[key]), ls="--", lw=1, label=f"{label} ({float(cal[key]):.2f})")
            drawn = True
    ax.set_xlabel("max refusal-direction projection (layer 14)")
    ax.set_title("Refusal probe: projections by class" + (" and calibrated bands" if drawn else ""))
    if drawn:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_performed_uncertainty(report: dict[str, Any]):
    """Records carry `answer_confidence` (the constrained-answer confidence) and `hedge_score`."""
    plt = _plt()
    recs = [dict(r, answer_confidence=_first_present(r, "answer_confidence", "confidence"))
            for r in report.get("records", [])]
    classes = sorted({r.get("category", "?") for r in recs})
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, field, title in ((axes[0], "answer_confidence", "answer confidence"),
                             (axes[1], "hedge_score", "hedge score (v2)")):
        by_cls = field_by_class(recs, field)
        _hbox(ax, [by_cls.get(c) or [0.0] for c in classes], classes)
        ax.set_title(title)
    fig.suptitle("Performed uncertainty: per-class distributions")
    fig.tight_layout()
    return fig


def fig_response_fidelity(report: dict[str, Any]):
    plt = _plt()
    groups = distortion_by_class(report.get("records", []))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    _hbox(ax, [v or [0.0] for v in groups.values()] or [[0.0]], list(groups) or ["n/a"])
    ax.set_xlabel("raw distortion (topic hit x operative evaded)")
    ax.set_title("Response fidelity: raw distortion by class")
    fig.tight_layout()
    return fig


def fig_conflict_window_sweep(report: dict[str, Any]):
    plt = _plt()
    s = sweep_series(report.get("summary", {}))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for cls, vals in s.items():
        if cls == "gaps":
            continue
        xs_ok, ok, xs_na = split_na(s["gaps"], vals)
        ax.plot(xs_ok, ok, marker="o", label=cls)
        for x in xs_na:
            ax.text(x, 0.02, "n/a", ha="center", fontsize=7, color="#7a1f1f")
    ax.set_xlabel("gap (max tokens between the two crossings); inf = whole response")
    ax.set_ylabel("any-conflict rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Conflict v2: windowed conjunction sweep (Qwen 2.5 7B)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_framing_v1b_classes(report: dict[str, Any]):
    plt = _plt()
    summary = report.get("summary", {})
    co = class_means(summary, "coactivation_rate")
    inst = class_means(summary, "mean_instability")
    labels = [c for c, _ in co]
    fig, ax = plt.subplots(figsize=(7, 3.4), layout="constrained")
    for offset, series, label in ((-0.2, co, "co-activation rate"), (0.2, inst, "mean instability")):
        xs = [i + offset for i, (_, v) in enumerate(series) if v is not None]
        ys = [float(v) for _, v in series if v is not None]
        ax.bar(xs, ys, width=0.4, label=label)
        for i, (_, v) in enumerate(series):
            if v is None:
                ax.text(i + offset, 0.02, "n/a", ha="center", fontsize=7, color="#7a1f1f")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Framing v1b (two-axis): the collinearity artefact, cos(A,B)=0.987")
    ax.legend(fontsize=8)
    return fig


def fig_framing_swap_flip(report: dict[str, Any]):
    plt = _plt()
    rows = swap_rows(report.get("records", []))
    fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    for i, (_pid, a_first, b_first, flip) in enumerate(rows):
        if a_first is not None and b_first is not None:
            ax.plot([a_first, b_first], [i, i], color="#999", lw=1)
        if a_first is not None:
            ax.scatter([a_first], [i], marker=">", color="#1f4e79", label="Israeli-first" if i == 0 else None)
        else:
            ax.text(0.02, i, "n/a (A-first)", va="center", fontsize=7, color="#7a1f1f")
        if b_first is not None:
            ax.scatter([b_first], [i], marker="<", color="#7a1f1f", label="Palestinian-first" if i == 0 else None)
        else:
            ax.text(0.02, i, "n/a (B-first)", va="center", fontsize=7, color="#7a1f1f")
        if flip:
            ax.text(max(v for v in (a_first, b_first) if v is not None) + 0.08, i, "flip", va="center", fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvspan(-1, 1, color="#eee", zorder=0)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("mean lean on engaged tokens (half-gap units; +A / -B; +-1 = one-sided commitment)")
    ax.set_title("Framing v1b.1: lean under side-order swap, per two-sided prompt")
    ax.legend(fontsize=8, loc="upper left")
    return fig


# png -> {"report": candidate filenames (first present wins), "aux": optional candidates
#         loaded into report["_aux"], "fn": figure function}
FIGURES: dict[str, dict[str, Any]] = {
    "refusal_calibration.png": {
        "report": ["dual_use_analysis_qwen7b.json", "dual_use_analysis.json"],
        "aux": ["calibration_qwen7b.json", "calibration.json"],
        "fn": fig_refusal_calibration},
    "performed_uncertainty.png": {"report": ["performed_uncertainty_analysis.json"], "fn": fig_performed_uncertainty},
    "response_fidelity.png": {"report": ["response_fidelity_analysis.json"], "fn": fig_response_fidelity},
    "conflict_window_sweep.png": {"report": ["conflict_state_analysis_v2.json"], "fn": fig_conflict_window_sweep},
    "framing_v1b_classes.png": {"report": ["framing_conflict_israel-palestine.json"], "fn": fig_framing_v1b_classes},
    "framing_swap_flip.png": {
        "report": ["framing_lean_israel-palestine_rescore2.json", "framing_lean_israel-palestine_rescore.json",
                   "framing_lean_israel-palestine.json"],
        "fn": fig_framing_swap_flip},
}


def _first_existing(data_dir: Path, names: list[str]) -> Path | None:
    for n in names:
        if (data_dir / n).exists():
            return data_dir / n
    return None


def build_all_figures(out_dir: Path, data_dir: Path = Path("data")) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for png, spec in FIGURES.items():
        fn: Callable[[dict[str, Any]], Any] = spec["fn"]
        path = _first_existing(data_dir, spec["report"])
        if path is None:
            print(f"skip: none of {spec['report']} under {data_dir} (committed {png} stands)")
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        aux = _first_existing(data_dir, spec.get("aux", []))
        if aux is not None:
            report["_aux"] = json.loads(aux.read_text(encoding="utf-8"))
        fig = fn(report)
        fig.savefig(out_dir / png, dpi=130)
        print("wrote", out_dir / png, f"(from {path.name}{' + ' + aux.name if aux else ''})")
        written.append(png)
    return written
