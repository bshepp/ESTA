"""Figures for the research-report site, regenerated from the persisted reports.

Shaping helpers are pure (report dict -> lists) and unit-tested without
matplotlib; the fig_* wrappers import matplotlib lazily. A report that is not
present locally (data/ is gitignored) is skipped with a notice -- the committed
PNG stands -- never an error.
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


def distortion_by_class(records: list[dict[str, Any]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for r in records:
        out.setdefault(r["category"], [])
        if r.get("raw_distortion") is not None:
            out[r["category"]].append(float(r["raw_distortion"]))
    return out


def _na(values: list) -> list[float]:
    """None -> 0.0 for plotting; the caption carries the truth (the None paths are real data)."""
    return [0.0 if v is None else float(v) for v in values]


# --- matplotlib wrappers (lazy import) --------------------------------------


def _hbox(ax, data, labels):  # noqa: ANN001
    """Horizontal boxplot with tick labels set portably (boxplot(labels=) was renamed
    tick_labels in matplotlib 3.9; set_yticks works on every supported version)."""
    ax.boxplot(data, vert=False)
    ax.set_yticks(range(1, len(labels) + 1), labels)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_refusal_calibration(report: dict[str, Any]):
    """Projection distributions by class with the calibrated band boundaries (dual-use report)."""
    plt = _plt()
    by_cls: dict[str, list[float]] = {}
    for r in report.get("records", []):
        v = r.get("refusal_projection_max")
        if v is not None:
            by_cls.setdefault(r.get("category", "?"), []).append(float(v))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    _hbox(ax, list(by_cls.values()) or [[0.0]], list(by_cls.keys()) or ["n/a"])
    cal = report.get("calibration") or report.get("provenance", {})
    for key, label in (("pressure_low", "low | moderate"), ("pressure_moderate", "moderate | high")):
        if cal.get(key) is not None:
            ax.axvline(float(cal[key]), ls="--", lw=1, label=label)
    ax.set_xlabel("max refusal-direction projection (layer 14)")
    ax.set_title("Refusal probe: projections by class and calibrated bands")
    if cal.get("pressure_low") is not None:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_performed_uncertainty(report: dict[str, Any]):
    plt = _plt()
    recs = report.get("records", [])
    classes = sorted({r.get("category", "?") for r in recs})
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, field, title in ((axes[0], "confidence", "token confidence"),
                             (axes[1], "hedge_score", "hedge score (v2)")):
        data = [[float(r[field]) for r in recs if r.get("category") == c and r.get(field) is not None]
                for c in classes]
        _hbox(ax, [d or [0.0] for d in data], classes)
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
        ax.plot(s["gaps"], _na(vals), marker="o", label=cls)
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
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], _na([v for _, v in co]), width=0.4, label="co-activation rate")
    ax.bar([i + 0.2 for i in x], _na([v for _, v in inst]), width=0.4, label="mean instability")
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Framing v1b (two-axis): the collinearity artefact, cos(A,B)=0.987")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_framing_swap_flip(report: dict[str, Any]):
    plt = _plt()
    rows = swap_rows(report.get("records", []))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, (_pid, a_first, b_first, flip) in enumerate(rows):
        ya, yb = (0.0 if a_first is None else a_first), (0.0 if b_first is None else b_first)
        ax.plot([ya, yb], [i, i], color="#999", lw=1)
        ax.scatter([ya], [i], marker=">", color="#1f4e79", label="Israeli-first" if i == 0 else None)
        ax.scatter([yb], [i], marker="<", color="#7a1f1f", label="Palestinian-first" if i == 0 else None)
        if flip:
            ax.text(max(ya, yb) + 0.08, i, "flip", va="center", fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvspan(-1, 1, color="#eee", zorder=0)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("mean lean on engaged tokens (half-gap units; +A / -B; +-1 = one-sided commitment)")
    ax.set_title("Framing v1b.1: lean under side-order swap, per two-sided prompt")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


FIGURES: dict[str, tuple[str, Callable[[dict[str, Any]], Any]]] = {
    "refusal_calibration.png": ("dual_use_analysis_qwen7b.json", fig_refusal_calibration),
    "performed_uncertainty.png": ("performed_uncertainty_analysis.json", fig_performed_uncertainty),
    "response_fidelity.png": ("response_fidelity_analysis.json", fig_response_fidelity),
    "conflict_window_sweep.png": ("conflict_state_analysis_v2.json", fig_conflict_window_sweep),
    "framing_v1b_classes.png": ("framing_conflict_israel-palestine.json", fig_framing_v1b_classes),
    "framing_swap_flip.png": ("framing_lean_israel-palestine_rescore2.json", fig_framing_swap_flip),
}


def build_all_figures(out_dir: Path, data_dir: Path = Path("data")) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for png, (report_name, fn) in FIGURES.items():
        path = data_dir / report_name
        if not path.exists():
            print(f"skip: {path} not found (committed {png} stands)")
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        fig = fn(report)
        fig.savefig(out_dir / png, dpi=130)
        print("wrote", out_dir / png)
        written.append(png)
    return written
