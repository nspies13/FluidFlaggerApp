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
  validation_score_dist.pdf        score by expert label for BMP & CBC
  validation_tables.tex            LaTeX table fragments (\input-able)
"""
from __future__ import annotations

import sys
import warnings
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


def score_bmp():
    df = pd.read_csv(DATA / "bmp_validation_set.csv")
    expert = pd.to_numeric(df["expert_review_prediction"], errors="coerce").values
    ref_retro = pd.to_numeric(df["max_retrospective_prob"], errors="coerce").values
    ref_rt = pd.to_numeric(df["max_realtime_prob"], errors="coerce").values
    res = make_bmp_predictions(df)
    return {
        "panel": "BMP",
        "expert": expert,
        "ff_real-time": res["max_realtime_prob"].values,
        "ff_retro": res["max_retrospective_prob"].values,
        "ref_retro": ref_retro,
        "ref_real-time": ref_rt,
    }


def score_cbc():
    df = pd.read_csv(DATA / "cbc_validation_set.csv")
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
    from plotnine import (
        aes,
        coord_cartesian,
        facet_wrap,
        geom_boxplot,
        geom_hline,
        geom_point,
        geom_violin,
        ggplot,
        guide_legend,
        guides,
        labs,
        position_dodge,
        position_jitterdodge,
        scale_color_manual,
        scale_fill_manual,
        scale_y_continuous,
        theme,
    )

    _setup_plotnine()
    dist = _score_distribution_frame(scored)
    dodge = position_dodge(width=0.74)

    p = (
        ggplot(dist, aes("mode", "score"))
        + geom_violin(
            aes(fill="expert_label"),
            position=dodge,
            width=0.78,
            alpha=0.28,
            trim=False,
            scale="width",
            color="#555555",
            size=0.25,
        )
        + geom_boxplot(
            aes(fill="expert_label"),
            position=dodge,
            width=0.22,
            outlier_shape=None,
            alpha=0.82,
            color="#333333",
            size=0.25,
        )
        + geom_point(
            aes(color="expert_label"),
            position=position_jitterdodge(
                jitter_width=0.2,
                jitter_height=0,
                dodge_width=0.74,
                random_state=RNG_SEED,
            ),
            size=1.5,
            alpha=0.5,
            stroke=0,
            show_legend=False,
        )
        + geom_hline(
            yintercept=THRESHOLD,
            linetype="dashed",
            color="#4A4A4A",
            size=0.55,
        )
        + facet_wrap("panel", nrow=1)
        + scale_y_continuous(
            breaks=[0, 0.25, 0.50, THRESHOLD, 1.0],
            labels=["0.00", "0.25", "0.50", "0.75", "1.00"],
        )
        + coord_cartesian(ylim=(-0.02, 1.02))
        + scale_fill_manual(values=EXPERT_COLORS)
        + scale_color_manual(values=EXPERT_COLORS)
        + guides(fill=guide_legend(nrow=1, byrow=True), color=False)
        + labs(
            x="Feature Set",
            y="Contamination Probability",
            fill="Expert Label",
        )
        + theme(
            figure_size=(7.2, 3.8),
            legend_position="top",
            legend_direction="horizontal",
            panel_spacing=0.08,
        )
    )
    _save_plotnine(p, "validation_score_dist", width=7.2, height=3.8)


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
    print("Loading local models ...", flush=True)
    load_models_from_dir(REPO / "models")
    print("Scoring BMP validation set ...", flush=True)
    bmp = score_bmp()
    print("Scoring CBC validation set ...", flush=True)
    cbc = score_cbc()
    scored = [bmp, cbc]

    summ = dataset_summary(scored)
    gold = gold_metrics(scored)
    silver = silver_metrics(scored)

    OUT.mkdir(parents=True, exist_ok=True)
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
