"""
Validate FluidFlagger BMP and CBC models against two reference standards on the
held-out validation sets:

  * expert_review_prediction  -- human expert label (GOLD standard, sparse subset)
  * max_retrospective_prob    -- probability from the independent reference
                                 pipeline shipped in the validation CSV
                                 (SILVER standard, available for every row)

For each panel we score every specimen with the local FluidFlagger realtime and
retrospective classifiers, then evaluate discrimination (AUROC, AUPRC) and the
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
        "ff_realtime": res["max_realtime_prob"].values,
        "ff_retro": res["max_retrospective_prob"].values,
        "ref_retro": ref_retro,
        "ref_realtime": ref_rt,
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
        "ff_realtime": res["max_realtime_prob"].values,
        "ff_retro": res["max_retrospective_prob"].values,
        "ref_retro": ref_retro,
        "ref_realtime": None,
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
            ("FluidFlagger realtime", d["ff_realtime"][m]),
            ("FluidFlagger retrospective", d["ff_retro"][m]),
            ("Reference retrospective", d["ref_retro"][m]),
        ]
        if d["ref_realtime"] is not None:
            scorers.insert(2, ("Reference realtime", d["ref_realtime"][m]))
        for name, s in scorers:
            r = evaluate(y, s, name)
            r["panel"] = panel
            rows.append(r)
    return pd.DataFrame(rows)


def silver_metrics(scored):
    """Realtime/retrospective FF models vs binarised retrospective reference,
    evaluated on every specimen (the reference covers the full set)."""
    rows = []
    for d in scored:
        panel = d["panel"]
        y = (d["ref_retro"] >= THRESHOLD).astype(float)
        for name, s in [("FluidFlagger realtime", d["ff_realtime"]),
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


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })
    return plt


CURVE_COLORS = {
    "FluidFlagger realtime": "#1f77b4",
    "FluidFlagger retrospective": "#d62728",
    "Reference retrospective": "#7f7f7f",
    "Reference realtime": "#bcbd22",
}


def plot_roc_pr(scored):
    plt = _setup_mpl()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6))
    for col, d in enumerate(scored):
        panel = d["panel"]
        m = ~np.isnan(d["expert"])
        y = d["expert"][m].astype(int)
        prevalence = y.mean()
        curves = [
            ("FluidFlagger realtime", d["ff_realtime"][m]),
            ("FluidFlagger retrospective", d["ff_retro"][m]),
            ("Reference retrospective", d["ref_retro"][m]),
        ]
        ax_roc, ax_pr = axes[0, col], axes[1, col]
        for name, s in curves:
            yy, ss = _clean(y, s)
            fpr, tpr, _ = roc_curve(yy, ss)
            ax_roc.plot(fpr, tpr, color=CURVE_COLORS[name], lw=1.4,
                        label=f"{name} (AUROC {roc_auc_score(yy, ss):.3f})")
            prec, rec, _ = precision_recall_curve(yy, ss)
            ax_pr.plot(rec, prec, color=CURVE_COLORS[name], lw=1.4,
                       label=f"{name} (AUPRC {average_precision_score(yy, ss):.3f})")
        ax_roc.plot([0, 1], [0, 1], ":", color="0.6", lw=1)
        ax_roc.set(xlim=(0, 1), ylim=(0, 1.02),
                   xlabel="1 − specificity", ylabel="Sensitivity",
                   title=f"{panel}: ROC vs expert review")
        ax_roc.legend(loc="lower right", frameon=False)
        ax_pr.axhline(prevalence, ls=":", color="0.6", lw=1)
        ax_pr.set(xlim=(0, 1), ylim=(0, 1.02),
                  xlabel="Recall (sensitivity)", ylabel="Precision (PPV)",
                  title=f"{panel}: precision–recall")
        ax_pr.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "validation_roc_pr.pdf")
    fig.savefig(OUT / "validation_roc_pr.png", dpi=150)
    plt.close(fig)


def plot_score_dist(scored):
    plt = _setup_mpl()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    rng = np.random.default_rng(0)
    for col, d in enumerate(scored):
        ax = axes[col]
        m = ~np.isnan(d["expert"])
        y = d["expert"][m].astype(int)
        for xpos, name, s in [(0, "FF realtime", d["ff_realtime"][m]),
                              (1, "FF retro", d["ff_retro"][m])]:
            for cls, color, off in [(0, "#4c72b0", -0.18), (1, "#c44e52", 0.18)]:
                vals = np.clip(s[y == cls], 1e-4, 1)
                jitter = rng.normal(0, 0.05, size=len(vals))
                ax.scatter(np.full(len(vals), xpos + off) + jitter, vals,
                           s=5, alpha=0.35, color=color, edgecolors="none")
        ax.axhline(THRESHOLD, ls="--", color="0.4", lw=1)
        ax.set_yscale("log")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Realtime", "Retrospective"])
        ax.set_ylim(8e-4, 1.3)
        ax.set_ylabel("Contamination probability")
        ax.set_title(f"{d['panel']}: score by expert label")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", color="#4c72b0", label="Expert: real"),
               Line2D([0], [0], marker="o", ls="", color="#c44e52", label="Expert: contaminated"),
               Line2D([0], [0], ls="--", color="0.4", label=f"Threshold {THRESHOLD}")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "validation_score_dist.pdf", bbox_inches="tight")
    fig.savefig(OUT / "validation_score_dist.png", dpi=150, bbox_inches="tight")
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
        r"(AUROC, AUPRC with 95\% bootstrap confidence intervals) and operating-point "
        r"performance at the $p\ge0.75$ contamination threshold. Reference columns are "
        r"the probabilities shipped with the validation set.}",
        r"\label{tab:val-gold}",
        r"\small",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Panel & Model & AUROC (95\% CI) & AUPRC (95\% CI) & Sens.\ (\%) & "
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
        rf"\\[2pt]\footnotesize BMP: $n={n_bmp}$ expert-reviewed specimens "
        rf"({p_bmp} contaminated). CBC: $n={n_cbc}$ ({p_cbc} contaminated).",
        r"\end{table}",
        r"",
    ]
    # ---- Silver-standard table ----
    lines += [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Agreement of the FluidFlagger realtime and retrospective models "
        r"with the retrospective reference probability (silver standard, binarised at "
        r"$0.75$) across the complete validation set.}",
        r"\label{tab:val-silver}",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Panel & Model & $n$ & Reference-positive & AUROC & AUPRC \\",
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
    lines += [r"\end{tabular}", r"\end{table}", r""]
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
