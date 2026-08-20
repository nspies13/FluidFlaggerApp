"""
Validate FluidFlagger BMP and CBC models against two reference standards on the
held-out validation sets:

  * expert_review_prediction  -- human expert label (GOLD standard, sparse subset)
  * max_retrospective_prob    -- probability from the independent reference
                                 pipeline shipped in the validation CSV
                                 (SILVER standard, available for every row)

For each panel we score every specimen with the local FluidFlagger real-time and
retrospective classifiers, then evaluate discrimination (auROC, auPRC) and the
operating point at the p>=0.75 contamination threshold (sensitivity, specificity,
PPV, NPV) against each reference standard.

Outputs (written next to the LaTeX source in paper/files/):
  validation_gold_metrics.csv      metrics vs expert review
  validation_silver_metrics.csv    metrics vs retrospective reference (full set)
  validation_dataset_summary.csv   cohort sizes / prevalence
  validation_roc_pr.pdf            2x2 ROC (top) + PR (bottom) for BMP & CBC
  validation_score_dist.pdf        Figure 2: raincloud distributions of scores
                                   and BMP/CBC mixture ratios by outcome
  validation_tables.tex            LaTeX table fragments (\input-able)
"""
from __future__ import annotations

import sys
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.model_loader import load_models_from_dir
from src.inference import make_bmp_predictions, make_cbc_predictions

DATA = REPO / "data"
OUT = REPO / "paper" / "files"
THRESHOLD = 0.75  # contamination operating point from the manuscript
N_BOOT = 2000
RNG_SEED = 13

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _clean(y: np.ndarray, s: np.ndarray):
    """Drop rows where the score is NaN (model not applicable)."""
    m = ~np.isnan(s)
    return y[m].astype(int), s[m]


def bootstrap_ci(y, s, fn, n=N_BOOT, seed=RNG_SEED):
    """Stratified bootstrap percentile CI for a ranking metric."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) < 2 or len(neg) < 2:
        return (np.nan, np.nan)
    vals = []
    for _ in range(n):
        bi = np.concatenate(
            [rng.choice(pos, len(pos), replace=True),
             rng.choice(neg, len(neg), replace=True)]
        )
        try:
            vals.append(fn(y[bi], s[bi]))
        except ValueError:
            continue
    if not vals:
        return (np.nan, np.nan)
    return tuple(np.percentile(vals, [2.5, 97.5]))


def operating_point(y, s, thr=THRESHOLD):
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, sensitivity=sens,
               specificity=spec, ppv=ppv, npv=npv)


def evaluate(y_raw, s_raw, label, thr=THRESHOLD, bootstrap=True):
    y, s = _clean(y_raw, s_raw)
    auroc = roc_auc_score(y, s)
    auprc = average_precision_score(y, s)
    if bootstrap:
        lo_roc, hi_roc = bootstrap_ci(y, s, roc_auc_score)
        lo_pr, hi_pr = bootstrap_ci(y, s, average_precision_score)
    else:
        lo_roc = hi_roc = lo_pr = hi_pr = np.nan
    op = operating_point(y, s, thr)
    return {
        "scorer": label,
        "n": len(y),
        "n_pos": int(y.sum()),
        "auroc": auroc,
        "auroc_lo": lo_roc,
        "auroc_hi": hi_roc,
        "auprc": auprc,
        "auprc_lo": lo_pr,
        "auprc_hi": hi_pr,
        **op,
    }


# ---------------------------------------------------------------------------
# Scoring of the validation sets
# ---------------------------------------------------------------------------


def score_bmp(expert_reviewed_only: bool = False):
    df = pd.read_csv(DATA / "bmp_validation_set.csv")
    if expert_reviewed_only:
        df = df.loc[pd.to_numeric(df["expert_review_prediction"], errors="coerce").notna()].copy()
    expert = pd.to_numeric(df["expert_review_prediction"], errors="coerce").values
    ref_retro = pd.to_numeric(df["max_retrospective_prob"], errors="coerce").values
    ref_rt = pd.to_numeric(df["max_realtime_prob"], errors="coerce").values
    res = make_bmp_predictions(df)
    return {
        "panel": "BMP",
        "expert": expert,
        "ff_real-time": res["max_realtime_prob"].values,
        "ff_retro": res["max_retrospective_prob"].values,
        "ff_mix_ratio": res["max_mix_ratio"].values,
        "ref_retro": ref_retro,
        "ref_real-time": ref_rt,
    }


def score_cbc(expert_reviewed_only: bool = False):
    df = pd.read_csv(DATA / "cbc_validation_set.csv")
    if expert_reviewed_only:
        df = df.loc[df["expert_review_prediction"].notna()].copy()
    exp_raw = df["expert_review_prediction"].astype("string").str.upper()
    expert = np.where(exp_raw.isna(), np.nan,
                      (exp_raw == "T").astype(float))
    ref_retro = pd.to_numeric(df["max_retrospective_prob"], errors="coerce").values
    res = make_cbc_predictions(df)
    return {
        "panel": "CBC",
        "expert": expert,
        "ff_real-time": res["max_realtime_prob"].values,
        "ff_retro": res["max_retrospective_prob"].values,
        "ff_mix_ratio": res.get("mix_ratio_CBC", pd.Series(np.nan, index=res.index)).values,
        "ref_retro": ref_retro,
        "ref_real-time": None,
    }


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------


def gold_metrics(scored):
    """Metrics vs expert review on the expert-reviewed subset."""
    rows = []
    for d in scored:
        panel = d["panel"]
        m = ~np.isnan(d["expert"])
        y = d["expert"][m]
        scorers = [
            ("FluidFlagger real-time", d["ff_real-time"][m]),
            ("FluidFlagger retrospective", d["ff_retro"][m]),
            ("Reference retrospective", d["ref_retro"][m]),
        ]
        if d["ref_real-time"] is not None:
            scorers.insert(2, ("Reference real-time", d["ref_real-time"][m]))
        for name, s in scorers:
            r = evaluate(y, s, name)
            r["panel"] = panel
            rows.append(r)
    return pd.DataFrame(rows)


def silver_metrics(scored):
    """real-time/retrospective FF models vs binarised retrospective reference,
    evaluated on every specimen (the reference covers the full set)."""
    rows = []
    for d in scored:
        panel = d["panel"]
        y = (d["ref_retro"] >= THRESHOLD).astype(float)
        for name, s in [("FluidFlagger real-time", d["ff_real-time"]),
                        ("FluidFlagger retrospective", d["ff_retro"])]:
            r = evaluate(y, s, name, bootstrap=False)
            r["panel"] = panel
            rows.append(r)
    return pd.DataFrame(rows)


def dataset_summary(scored):
    rows = []
    for d in scored:
        exp = d["expert"]
        m = ~np.isnan(exp)
        rows.append({
            "panel": d["panel"],
            "n_specimens": len(exp),
            "n_expert_reviewed": int(m.sum()),
            "n_expert_contaminated": int(np.nansum(exp[m])),
            "n_expert_real": int(m.sum() - np.nansum(exp[m])),
            "expert_prevalence": float(np.nansum(exp[m]) / m.sum()) if m.sum() else np.nan,
            "n_ref_flagged_0.75": int((d["ref_retro"] >= THRESHOLD).sum()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


MODEL_LABELS = {
    "FluidFlagger real-time": "FF real-time",
    "FluidFlagger retrospective": "FF retrospective",
    "Reference retrospective": "Reference retrospective",
    "Reference real-time": "Reference real-time",
}

MODEL_COLORS = {
    "FF real-time": "#0072B2",
    "FF retrospective": "#D55E00",
    "Reference retrospective": "#009E73",
    "Reference real-time": "#6C757D",
}

MODEL_LINETYPES = {
    "FF real-time": "solid",
    "FF retrospective": "dashed",
    "Reference retrospective": "dashdot",
    "Reference real-time": "dotted",
}

EXPERT_COLORS = {
    "Real": "#D55E00",
    "Contaminated": "#0072B2",
}


def _setup_plotnine():
    from plotnine import (
        element_blank,
        element_line,
        element_text,
        theme,
        theme_minimal,
        theme_set,
    )
    from plotnine.themes.elements import margin

    theme_ns = theme_minimal(base_family="Helvetica") + theme(
        text=element_text(family="Helvetica"),
        plot_title=element_text(
            size=16, weight="bold", style="italic", ha="left",
            margin=margin(0, 0, 6, 0)
        ),
        plot_subtitle=element_text(
            size=14, weight="normal", ha="left",
            margin=margin(0, 0, 2, 0)
        ),
        axis_title=element_text(
            size=12, weight="bold", margin=margin(4, 4, 4, 4)
        ),
        axis_title_x=element_text(weight="bold", margin=margin(4, 0, 0, 0)),
        axis_title_y=element_text(weight="bold", margin=margin(0, 4, 0, 0)),
        axis_text_x=element_text(margin=margin(2, 0, 0, 0)),
        axis_text_y=element_text(margin=margin(0, 2, 0, 0)),
        legend_title=element_text(
            weight="bold", style="italic", size=12,
            margin=margin(0, 4, 0, 0)
        ),
        legend_text=element_text(margin=margin(0, 2, 0, 2)),
        axis_line=element_line(color="#333333", size=0.4),
        axis_ticks=element_blank(),
        panel_grid=element_blank(),
        panel_background=element_blank(),
        plot_background=element_blank(),
        legend_key=element_blank(),
        legend_key_size=9,
        legend_key_width=14,
        legend_key_height=8,
        legend_background=element_blank(),
        legend_box_background=element_blank(),
        strip_text=element_text(
            size=12, weight="bold", margin=margin(0, 0, 2, 0)
        ),
        strip_background=element_blank(),
    )
    theme_set(theme_ns)
    return theme_ns


def _save_plotnine(plot, filename: str, width: float, height: float):
    plot.save(OUT / f"{filename}.pdf", width=width, height=height,
              units="in", dpi=300, limitsize=False, verbose=False)
    plot.save(OUT / f"{filename}.png", width=width, height=height,
              units="in", dpi=300, limitsize=False, verbose=False)


def _curve_frames(scored):
    curve_rows = []
    baseline_roc_rows = []
    baseline_pr_rows = []

    for d in scored:
        panel = d["panel"]
        m = ~np.isnan(d["expert"])
        y = d["expert"][m].astype(int)
        prevalence = y.mean()
        curves = [
            ("FluidFlagger real-time", d["ff_real-time"][m]),
            ("FluidFlagger retrospective", d["ff_retro"][m]),
            ("Reference retrospective", d["ref_retro"][m]),
        ]

        baseline_roc_rows.extend([
            {"panel": panel, "curve": "ROC", "x": 0, "y": 0},
            {"panel": panel, "curve": "ROC", "x": 1, "y": 1},
        ])
        baseline_pr_rows.append({
            "panel": panel, "curve": "PR",
            "prevalence": prevalence,
        })

        for name, s in curves:
            model = MODEL_LABELS[name]
            yy, ss = _clean(y, s)

            fpr, tpr, _ = roc_curve(yy, ss)
            curve_rows.extend({
                "panel": panel,
                "curve": "ROC",
                "model": model,
                "x": x,
                "y": yy_,
                "order": j,
            } for j, (x, yy_) in enumerate(zip(fpr, tpr)))

            prec, rec, _ = precision_recall_curve(yy, ss)
            curve_rows.extend({
                "panel": panel,
                "curve": "PR",
                "model": model,
                "x": x,
                "y": yy_,
                "order": j,
            } for j, (x, yy_) in enumerate(zip(rec, prec)))

    curve_df = pd.DataFrame(curve_rows)
    roc_baseline = pd.DataFrame(baseline_roc_rows)
    pr_baseline = pd.DataFrame(baseline_pr_rows)

    curve_order = ["ROC", "PR"]
    panel_order = ["BMP", "CBC"]
    model_order = ["FF real-time", "FF retrospective", "Reference retrospective"]
    facet_order = [
        "BMP ROC",
        "CBC ROC",
        "BMP PR",
        "CBC PR",
    ]
    for df in [curve_df, roc_baseline, pr_baseline]:
        df["curve"] = pd.Categorical(df["curve"], categories=curve_order, ordered=True)
        df["panel"] = pd.Categorical(df["panel"], categories=panel_order, ordered=True)
        df["facet"] = (
            df["panel"].astype(str) + " " +
            df["curve"].astype(str)
        )
        df["facet"] = pd.Categorical(df["facet"], categories=facet_order, ordered=True)
    curve_df["model"] = pd.Categorical(curve_df["model"], categories=model_order, ordered=True)
    return curve_df, roc_baseline, pr_baseline


def _score_distribution_frame(scored):
    rows = []
    for d in scored:
        panel = d["panel"]
        m = ~np.isnan(d["expert"])
        y = d["expert"][m].astype(int)
        for mode, s in [
            ("Real-time", d["ff_real-time"][m]),
            ("Retrospective", d["ff_retro"][m]),
        ]:
            for label_value, label in [(0, "Real"), (1, "Contaminated")]:
                scores = np.clip(s[y == label_value], 0, 1)
                rows.extend({
                    "panel": panel,
                    "mode": mode,
                    "expert_label": label,
                    "score": score,
                } for score in scores)

    dist = pd.DataFrame(rows)
    dist["panel"] = pd.Categorical(dist["panel"], categories=["BMP", "CBC"], ordered=True)
    dist["mode"] = pd.Categorical(
        dist["mode"], categories=["Real-time", "Retrospective"], ordered=True
    )
    dist["expert_label"] = pd.Categorical(
        dist["expert_label"], categories=["Real", "Contaminated"], ordered=True
    )
    return dist


def plot_roc_pr(scored):
    from plotnine import (
        aes,
        coord_cartesian,
        facet_wrap,
        geom_hline,
        geom_path,
        ggplot,
        guide_legend,
        guides,
        labs,
        scale_color_manual,
        scale_linetype_manual,
        scale_x_continuous,
        scale_y_continuous,
        theme,
    )

    _setup_plotnine()
    curve_df, roc_baseline, pr_baseline = _curve_frames(scored)

    p = (
        ggplot(curve_df, aes("x", "y", color="model", linetype="model", group="model"))
        + geom_path(size=0.85)
        + geom_path(
            data=roc_baseline,
            mapping=aes("x", "y", group="panel"),
            inherit_aes=False,
            linetype="dotted",
            color="#777777",
            size=0.55,
        )
        + geom_hline(
            data=pr_baseline,
            mapping=aes(yintercept="prevalence"),
            inherit_aes=False,
            linetype="dotted",
            color="#777777",
            size=0.55,
        )
        + facet_wrap("facet", ncol=2)
        + coord_cartesian(xlim=(0, 1), ylim=(0, 1.02))
        + scale_x_continuous(breaks=[0, 0.25, 0.50, 0.75, 1.00])
        + scale_y_continuous(breaks=[0, 0.25, 0.50, 0.75, 1.00])
        + scale_color_manual(values=MODEL_COLORS)
        + scale_linetype_manual(values=MODEL_LINETYPES)
        + guides(
            color=guide_legend(nrow=1, byrow=True),
            linetype=guide_legend(nrow=1, byrow=True),
        )
        + labs(
            x="FPR (ROC) / Recall (PR)",
            y="Sensitivity (ROC) / Precision (PR)",
            color="Model",
            linetype="Model",
        )
        + theme(
            figure_size=(7.2, 5.8),
            legend_position="bottom",
            legend_direction="horizontal",
            legend_box="horizontal",
            panel_spacing_x=0.08,
            panel_spacing_y=0.03,
        )
    )
    _save_plotnine(p, "validation_roc_pr", width=7.2, height=5.8)


def plot_score_dist(scored):
    """Create Figure 2 as horizontal raincloud plots.

    The top row shows FluidFlagger contamination outputs by expert label.
    The bottom row shows estimated mixture ratios by retrospective classification
    outcome for both BMP and CBC at the manuscript operating threshold.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    score_colours = {"Real": "#D55E00", "Contaminated": "#0072B2"}
    outcome_colours = {
        "True positive": "#0072B2",
        "False positive": "#CC79A7",
        "True negative": "#009E73",
        "False negative": "#D55E00",
    }
    rng = np.random.default_rng(RNG_SEED)
    fig = plt.figure(figsize=(7.45, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, hspace=0.15, wspace=0.20)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]

    def _style_axis(axis):
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#9CA3AF")
        axis.spines[["left", "bottom"]].set_linewidth(0.65)
        axis.tick_params(axis="x", length=3, width=0.65, labelsize=8.5, colors="#4A4A4A")
        axis.tick_params(axis="y", length=0, labelsize=8.2, colors="#4A4A4A", pad=4)
        axis.grid(False)

    def _draw_raincloud(
        axis,
        values,
        positions,
        colours,
        labels,
        *,
        title,
        x_label,
        threshold=False,
        x_max=1.0,
    ):
        """Draw horizontal half-violin, boxplot, and jittered-point rainclouds."""
        for current, position, colour in zip(values, positions, colours):
            current = np.asarray(current, dtype=float)
            current = current[np.isfinite(current)]
            if not len(current):
                continue

            violin = axis.violinplot(
                [current],
                positions=[position],
                vert=False,
                widths=0.72,
                showmedians=False,
                showextrema=False,
            )
            body = violin["bodies"][0]
            body.set_facecolor(colour)
            body.set_edgecolor("#334155")
            body.set_alpha(0.30)
            body.set_linewidth(0.55)
            # Retain the upper half of the horizontal violin as the "cloud".
            body.get_paths()[0].vertices[:, 1] = np.maximum(
                body.get_paths()[0].vertices[:, 1], position
            )

            box = axis.boxplot(
                [current],
                vert=False,
                positions=[position],
                widths=0.14,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#1F2937", "linewidth": 1.0},
                boxprops={"linewidth": 0.55, "edgecolor": "#334155"},
                whiskerprops={"linewidth": 0.55, "color": "#475569"},
                capprops={"linewidth": 0.55, "color": "#475569"},
            )
            box["boxes"][0].set_facecolor(colour)
            box["boxes"][0].set_alpha(0.82)

            plotted = current if len(current) <= 350 else rng.choice(current, 350, replace=False)
            axis.scatter(
                plotted,
                position - rng.uniform(0.12, 0.34, size=len(plotted)),
                s=4,
                color=colour,
                alpha=0.20,
                linewidths=0,
            )

        axis.set_xlim(-0.02, x_max + 0.02)
        if x_max <= 0.5:
            axis.set_xticks(np.arange(0, x_max + 0.001, 0.1), [f"{tick:.2f}" for tick in np.arange(0, x_max + 0.001, 0.1)])
        else:
            axis.set_xticks([0, 0.25, 0.5, THRESHOLD, 1], ["0.00", "0.25", "0.50", "0.75", "1.00"])
        axis.set_ylim(min(positions) - 0.52, max(positions) + 0.54)
        axis.set_yticks(positions, labels)
        axis.set_xlabel(x_label, fontsize=9.5, fontweight="bold", labelpad=5)
        axis.set_title(title, fontsize=11.5, fontweight="bold", pad=8)
        if threshold:
            axis.axvline(THRESHOLD, color="#475569", linestyle=(0, (4, 3)), linewidth=0.85, zorder=0)
        _style_axis(axis)

    def _score_groups(data):
        expert = np.asarray(data["expert"], dtype=float)
        score_groups = []
        for score in (data["ff_real-time"], data["ff_retro"]):
            score = np.asarray(score, dtype=float)
            for label_value in (0, 1):
                mask = np.isfinite(expert) & np.isfinite(score) & (expert == label_value)
                score_groups.append(np.clip(score[mask], 0, 1))
        return score_groups

    def _mixture_groups(data):
        expert = np.asarray(data["expert"], dtype=float)
        score = np.asarray(data["ff_retro"], dtype=float)
        estimate = np.asarray(data["ff_mix_ratio"], dtype=float)
        valid = np.isfinite(expert) & np.isfinite(score) & np.isfinite(estimate)
        expert = expert[valid].astype(int)
        prediction = (score[valid] >= THRESHOLD).astype(int)
        estimate = np.clip(estimate[valid], 0, 1)
        outcome_masks = [
            (expert == 1) & (prediction == 1),
            (expert == 0) & (prediction == 1),
            (expert == 0) & (prediction == 0),
            (expert == 1) & (prediction == 0),
        ]
        return [estimate[mask] for mask in outcome_masks]

    panel_map = {data["panel"]: data for data in scored}
    score_positions = np.array([4.25, 3.25, 1.75, 0.75])
    score_labels = [
        "RT · Real",
        "RT · Contam.",
        "Retro · Real",
        "Retro · Contam.",
    ]
    score_colours_ordered = [
        score_colours["Real"],
        score_colours["Contaminated"],
        score_colours["Real"],
        score_colours["Contaminated"],
    ]
    for axis, panel in zip(axes[:2], ("BMP", "CBC")):
        _draw_raincloud(
            axis,
            _score_groups(panel_map[panel]),
            score_positions,
            score_colours_ordered,
            score_labels,
            title=f"{panel} contamination output",
            x_label="FluidFlagger output",
            threshold=True,
        )

    outcome_positions = np.array([4.0, 3.0, 2.0, 1.0])
    outcome_names = ["True positive", "False positive", "True negative", "False negative"]
    outcome_abbreviations = ["TP", "FP", "TN", "FN"]
    for axis, panel in zip(axes[2:], ("BMP", "CBC")):
        values = _mixture_groups(panel_map[panel])
        labels = [f"{label}  (n={len(current):,})" for label, current in zip(outcome_abbreviations, values)]
        _draw_raincloud(
            axis,
            values,
            outcome_positions,
            [outcome_colours[name] for name in outcome_names],
            labels,
            title=f"{panel} estimated mixture ratio",
            x_label="Estimated mixture ratio",
            x_max=0.50,
        )
        for current, position in zip(values, outcome_positions):
            if not len(current):
                axis.text(0.02, position, "No observations", fontsize=7.5, color="#64748B", va="center")

    fig.legend(
        handles=[Patch(facecolor=score_colours[label], edgecolor="#334155", alpha=0.58, label=label)
                 for label in ["Real", "Contaminated"]],
        title="Expert label",
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.045),
        fontsize=8.7,
        title_fontsize=8.7,
        handlelength=1.2,
        columnspacing=1.4,
    )
    for axis, label in zip(axes, ("A", "B", "C", "D")):
        axis.text(-0.17, 1.065, label, transform=axis.transAxes, fontsize=12, fontweight="bold")

    fig.savefig(OUT / "validation_score_dist.pdf", bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(OUT / "validation_score_dist.png", bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(OUT / "validation_score_dist.jpg", bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(OUT / "validation_score_dist.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# LaTeX fragments
# ---------------------------------------------------------------------------


def _ci(lo, hi):
    if np.isnan(lo) or np.isnan(hi):
        return "--"
    return f"({lo:.3f}--{hi:.3f})"


def _pct(x):
    return "--" if (x is None or np.isnan(x)) else f"{100*x:.1f}"


def write_latex(gold: pd.DataFrame, silver: pd.DataFrame, summ: pd.DataFrame):
    lines = []
    # ---- Gold-standard table ----
    lines += [
        r"% Auto-generated by paper/validate_models.py -- do not edit by hand.",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Validation against expert review (gold standard). Discrimination "
        r"(auROC, auPRC with 95\% bootstrap confidence intervals) and operating-point "
        r"performance at the $p\ge0.75$ contamination threshold. Reference columns are "
        r"the probabilities shipped with the validation set.}",
        r"\label{tab:val-gold}",
        r"\small",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Panel & Model & auROC (95\% CI) & auPRC (95\% CI) & Sens.\ (\%) & "
        r"Spec.\ (\%) & PPV (\%) \\",
        r"\midrule",
    ]
    for panel in ["BMP", "CBC"]:
        sub = gold[gold["panel"] == panel]
        for i, (_, r) in enumerate(sub.iterrows()):
            pcell = panel if i == 0 else ""
            lines.append(
                f"{pcell} & {r['scorer']} & {r['auroc']:.3f} {_ci(r['auroc_lo'], r['auroc_hi'])} "
                f"& {r['auprc']:.3f} {_ci(r['auprc_lo'], r['auprc_hi'])} "
                f"& {_pct(r['sensitivity'])} & {_pct(r['specificity'])} & {_pct(r['ppv'])} \\\\"
            )
        lines.append(r"\midrule" if panel == "BMP" else r"\bottomrule")
    n_bmp = int(summ[summ.panel == "BMP"].n_expert_reviewed.iloc[0])
    p_bmp = int(summ[summ.panel == "BMP"].n_expert_contaminated.iloc[0])
    n_cbc = int(summ[summ.panel == "CBC"].n_expert_reviewed.iloc[0])
    p_cbc = int(summ[summ.panel == "CBC"].n_expert_contaminated.iloc[0])
    lines += [
        r"\end{tabular}",
        r"}%",
        rf"\\[2pt]\footnotesize BMP: $n={n_bmp}$ expert-reviewed specimens "
        rf"({p_bmp} contaminated). CBC: $n={n_cbc}$ ({p_cbc} contaminated).",
        r"\end{table}",
        r"",
    ]
    # ---- Silver-standard table ----
    lines += [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Agreement of the FluidFlagger real-time and retrospective models "
        r"with the retrospective reference probability (silver standard, binarised at "
        r"$0.75$) across the complete validation set.}",
        r"\label{tab:val-silver}",
        r"\small",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Panel & Model & $n$ & Reference-positive & auROC & auPRC \\",
        r"\midrule",
    ]
    for panel in ["BMP", "CBC"]:
        sub = silver[silver["panel"] == panel]
        for i, (_, r) in enumerate(sub.iterrows()):
            pcell = panel if i == 0 else ""
            lines.append(
                f"{pcell} & {r['scorer']} & {r['n']:,} & {r['n_pos']:,} "
                f"& {r['auroc']:.3f} & {r['auprc']:.3f} \\\\"
            )
        lines.append(r"\midrule" if panel == "BMP" else r"\bottomrule")
    lines += [r"\end{tabular}", r"}%", r"\end{table}", r""]
    (OUT / "validation_tables.tex").write_text("\n".join(lines))


# ---------------------------------------------------------------------------


def main():
    global OUT
    parser = argparse.ArgumentParser(description="Validate FluidFlagger models and generate manuscript figures.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT,
        help="Directory for generated metrics, tables, and figures.",
    )
    parser.add_argument(
        "--figure-2-only",
        action="store_true",
        help="Generate only Figure 2 from expert-reviewed specimens.",
    )
    args = parser.parse_args()
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading local models ...", flush=True)
    load_models_from_dir(REPO / "models")
    if args.figure_2_only:
        print("Scoring expert-reviewed BMP specimens for Figure 2 ...", flush=True)
        bmp = score_bmp(expert_reviewed_only=True)
        print("Scoring expert-reviewed CBC specimens for Figure 2 ...", flush=True)
        cbc = score_cbc(expert_reviewed_only=True)
        plot_score_dist([bmp, cbc])
        print(f"Saved Figure 2 to {OUT / 'validation_score_dist.pdf'}", flush=True)
        return

    print("Scoring BMP validation set ...", flush=True)
    bmp = score_bmp()
    print("Scoring CBC validation set ...", flush=True)
    cbc = score_cbc()
    scored = [bmp, cbc]

    summ = dataset_summary(scored)
    gold = gold_metrics(scored)
    silver = silver_metrics(scored)

    summ.to_csv(OUT / "validation_dataset_summary.csv", index=False)
    gold.to_csv(OUT / "validation_gold_metrics.csv", index=False)
    silver.to_csv(OUT / "validation_silver_metrics.csv", index=False)

    print("\n=== Dataset summary ===")
    print(summ.to_string(index=False))
    print("\n=== Gold (vs expert review) ===")
    print(gold[["panel", "scorer", "n", "n_pos", "auroc", "auprc",
                "sensitivity", "specificity", "ppv", "npv"]].to_string(index=False))
    print("\n=== Silver (vs retrospective reference, full set) ===")
    print(silver[["panel", "scorer", "n", "n_pos", "auroc", "auprc"]].to_string(index=False))

    print("\nGenerating plots ...", flush=True)
    plot_roc_pr(scored)
    plot_score_dist(scored)
    write_latex(gold, silver, summ)
    print(f"Wrote outputs to {OUT}")


if __name__ == "__main__":
    main()
