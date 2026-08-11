"""Exportable HTML and PDF reports for Validate-tab performance results.

The browser dashboard is interactive, while these reports capture the current
server-side operating point (normally the production threshold of 0.75) in a
portable, self-contained format.  They intentionally include only aggregate
validation results, never the uploaded patient-level rows.
"""

from __future__ import annotations

import base64
import datetime as dt
import html
import tempfile
from pathlib import Path
from typing import Any


_BLUE = "#2563eb"
_GREEN = "#059669"
_PURPLE = "#7c3aed"
_SLATE = "#334155"
_MUTED = "#64748b"
_AXIS = "#cbd5e1"
_TICK = "#94a3b8"


def _percent(value: float | None) -> str:
    """Format a rate for a reader-facing report."""
    return "N/A" if value is None else f"{value * 100:.1f}%"


def _decimal(value: float | None) -> str:
    """Format an AUC or threshold for a reader-facing report."""
    return "N/A" if value is None else f"{value:.3f}"


def _count(value: int | float) -> str:
    return f"{int(value):,}"


def _report_metadata(payload: dict[str, Any]) -> list[tuple[str, str]]:
    summary = payload["summary"]
    return [
        ("Prediction probability", str(payload["score_column"])),
        ("Ground-truth label", str(payload["label_column"])),
        ("Evaluable rows", _count(summary["included_rows"])),
        ("Contaminated ground truth", _count(summary["positive_count"])),
        ("Real ground truth", _count(summary["negative_count"])),
        ("Excluded rows", _count(summary["excluded_rows"])),
        ("Decision threshold", _decimal(payload["metrics"]["threshold"])),
        ("Positive prediction", str(payload["threshold_rule"])),
    ]


def _metric_rows(payload: dict[str, Any]) -> list[tuple[str, str]]:
    metrics = payload["metrics"]
    return [
        ("Sensitivity", _percent(metrics["sensitivity"])),
        ("Specificity", _percent(metrics["specificity"])),
        ("PPV", _percent(metrics["ppv"])),
        ("NPV", _percent(metrics["npv"])),
        ("F1", _percent(metrics["f1"])),
    ]


def _style_chart_axes(ax: Any, title: str) -> None:
    """Apply one uncluttered, report-friendly style to validation plots."""
    ax.set_facecolor("#ffffff")
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=3.5,
        width=0.85,
        colors=_TICK,
        labelcolor=_MUTED,
        labelsize=8,
    )
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    for spine_name in ("bottom", "left"):
        spine = ax.spines[spine_name]
        spine.set_color(_AXIS)
        spine.set_linewidth(0.9)
    ax.xaxis.label.set_color(_SLATE)
    ax.yaxis.label.set_color(_SLATE)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.set_title(title, loc="left", color="#0f172a", fontsize=11.5, fontweight="bold", pad=10)


def _style_chart_legend(ax: Any, location: str) -> None:
    """Keep legends compact and legible without a heavy chart frame."""
    legend = ax.legend(
        loc=location,
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        fontsize=7.6,
        handlelength=1.8,
        labelspacing=0.45,
        borderpad=0.55,
    )
    legend_frame = legend.get_frame()
    legend_frame.set_facecolor("#ffffff")
    legend_frame.set_edgecolor("#e2e8f0")
    legend_frame.set_linewidth(0.75)


def _render_chart_images(payload: dict[str, Any], report_dir: Path) -> dict[str, Path]:
    """Render static ROC, PR, and calibration plots for both export formats."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    metrics = payload["metrics"]
    summary = payload["summary"]
    auc = payload["auc"]
    chart_paths = {
        "roc": report_dir / "roc_curve.png",
        "pr": report_dir / "precision_recall_curve.png",
        "calibration": report_dir / "calibration_curve.png",
    }

    def finish(fig, path: Path) -> None:
        fig.tight_layout(pad=1.15)
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # ROC with the threshold-specific 2 x 2 table in the lower-right plot area.
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    roc_fpr = [point["fpr"] for point in payload["roc"]]
    roc_tpr = [point["tpr"] for point in payload["roc"]]
    ax.fill_between(roc_fpr, roc_tpr, 0, color=_BLUE, alpha=0.07, zorder=0)
    ax.plot(
        roc_fpr,
        roc_tpr,
        color=_BLUE,
        linewidth=2.65,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=2,
        label=f"ROC AUC {_decimal(auc['roc'])}",
    )
    ax.plot([0, 1], [0, 1], color=_TICK, linestyle=(0, (4, 4)), linewidth=1.15, zorder=1)
    ax.scatter(
        [1 - metrics["specificity"]],
        [metrics["sensitivity"]],
        s=64,
        color=_BLUE,
        edgecolor="white",
        linewidth=1.5,
        zorder=4,
        label=f"Threshold {_decimal(metrics['threshold'])}",
    )
    ax.set(
        xlabel="False-positive rate",
        ylabel="Sensitivity",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    _style_chart_axes(ax, "ROC curve")
    _style_chart_legend(ax, "upper left")
    confusion_table = ax.table(
        cellText=[
            ["Real", _count(metrics["tn"]), _count(metrics["fp"])],
            ["Contaminated", _count(metrics["fn"]), _count(metrics["tp"])],
        ],
        colLabels=["Truth / predicted", "Real", "Contam."],
        cellLoc="center",
        colLoc="center",
        bbox=[0.45, 0.04, 0.52, 0.27],
        zorder=5,
    )
    confusion_table.auto_set_font_size(False)
    confusion_table.set_fontsize(6.8)
    for (row, column), cell in confusion_table.get_celld().items():
        cell.set_edgecolor("#dbe5f1")
        cell.set_linewidth(0.55)
        if row == 0:
            cell.set_facecolor("#eff6ff")
            cell.set_text_props(weight="bold", color=_SLATE)
        elif column == 0:
            cell.set_facecolor("#f8fafc")
            cell.set_text_props(weight="bold", color=_SLATE)
        else:
            cell.set_facecolor("#ffffff")
            cell.set_text_props(color="#0f172a")
    ax.text(
        0.71,
        0.335,
        f"2 x 2 at threshold {_decimal(metrics['threshold'])}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=_SLATE,
        fontsize=7.2,
        fontweight="bold",
    )
    finish(fig, chart_paths["roc"])

    # Precision-recall curve at the same operating point.
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    pr_recall = [point["recall"] for point in payload["pr"]]
    pr_precision = [point["precision"] for point in payload["pr"]]
    ax.fill_between(pr_recall, pr_precision, summary["prevalence"], color=_GREEN, alpha=0.07, zorder=0)
    ax.plot(
        pr_recall,
        pr_precision,
        color=_GREEN,
        linewidth=2.65,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=2,
        label=f"Average precision {_decimal(auc['pr'])}",
    )
    ax.axhline(
        summary["prevalence"],
        color=_TICK,
        linestyle=(0, (4, 4)),
        linewidth=1.15,
        zorder=1,
        label=f"Prevalence {_percent(summary['prevalence'])}",
    )
    if metrics["ppv"] is not None:
        ax.scatter(
            [metrics["sensitivity"]],
            [metrics["ppv"]],
            s=64,
            color=_GREEN,
            edgecolor="white",
            linewidth=1.5,
            zorder=4,
            label=f"Threshold {_decimal(metrics['threshold'])}",
        )
    ax.set(
        xlabel="Recall (sensitivity)",
        ylabel="Precision (PPV)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    _style_chart_axes(ax, "Precision-recall curve")
    _style_chart_legend(ax, "lower left")
    finish(fig, chart_paths["pr"])

    # Calibration, including an explicit ideal-reference line.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    calibration = payload["calibration"]
    x_values = [point["mean_predicted"] for point in calibration]
    y_values = [point["fraction_positive"] for point in calibration]
    sizes = [max(34, min(140, 18 + point["count"] * 5)) for point in calibration]
    ax.plot(
        [0, 1],
        [0, 1],
        color=_TICK,
        linestyle=(0, (4, 4)),
        linewidth=1.15,
        label="Perfect calibration",
        zorder=1,
    )
    ax.plot(
        x_values,
        y_values,
        color=_PURPLE,
        linewidth=2.5,
        solid_capstyle="round",
        solid_joinstyle="round",
        label="Model",
        zorder=2,
    )
    ax.scatter(x_values, y_values, s=sizes, color=_PURPLE, edgecolor="white", linewidth=1.1, zorder=3)
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed contamination rate",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    _style_chart_axes(ax, "Calibration plot")
    _style_chart_legend(ax, "upper left")
    finish(fig, chart_paths["calibration"])

    return chart_paths


def _image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_html_report(payload: dict[str, Any], chart_paths: dict[str, Path], generated_at: dt.datetime) -> str:
    """Build a standalone report with inline chart images."""
    summary = payload["summary"]
    auc = payload["auc"]
    metrics = payload["metrics"]
    metadata_rows = "".join(
        "<tr><th>{}</th><td>{}</td></tr>".format(html.escape(label), html.escape(value))
        for label, value in _report_metadata(payload)
    )
    metric_cards = "".join(
        "<div class=\"metric\"><span>{}</span><strong>{}</strong></div>".format(
            html.escape(label), html.escape(value)
        )
        for label, value in _metric_rows(payload)
    )
    confusion_rows = "".join(
        "<tr><th>{}</th><td>{}</td><td>{}</td></tr>".format(
            html.escape(label), _count(real), _count(contaminated)
        )
        for label, real, contaminated in (
            ("Real", metrics["tn"], metrics["fp"]),
            ("Contaminated", metrics["fn"], metrics["tp"]),
        )
    )
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>FluidFlagger Validation Report</title>
  <style>
    :root {{ color-scheme: light; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #f1f5f9; color: #1e293b; font: 14px/1.5 Arial, Helvetica, sans-serif; }}
    main {{ max-width: 1040px; margin: 28px auto; padding: 38px; background: #fff; box-shadow: 0 8px 30px rgba(15,23,42,.12); }}
    header {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 28px; padding-bottom: 20px; border-bottom: 3px solid {_BLUE}; }}
    h1 {{ margin: 0; color: #0f172a; font-size: 30px; letter-spacing: -.03em; }}
    h2 {{ margin: 32px 0 12px; color: #0f172a; font-size: 19px; }}
    h3 {{ margin: 0 0 8px; color: #0f172a; font-size: 15px; }}
    p {{ margin: 5px 0; }}
    .eyebrow {{ color: {_BLUE}; font-size: 11px; font-weight: bold; letter-spacing: .12em; text-transform: uppercase; }}
    .meta {{ flex: none; color: {_MUTED}; font-size: 12px; text-align: right; }}
    .metrics {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }}
    .metric {{ padding: 13px; border: 1px solid #dbe5f1; border-top: 3px solid {_BLUE}; border-radius: 8px; background: #fff; }}
    .metric span {{ display: block; color: {_MUTED}; font-size: 11px; font-weight: bold; letter-spacing: .06em; text-transform: uppercase; }}
    .metric strong {{ display: block; margin-top: 3px; color: #0f172a; font-size: 22px; }}
    .details {{ display: grid; grid-template-columns: minmax(280px, 1.1fr) minmax(290px, .9fr); gap: 22px; align-items: start; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 8px 9px; border: 1px solid #dbe5f1; text-align: left; }}
    th {{ color: {_SLATE}; background: #f8fafc; font-weight: bold; }}
    .matrix td {{ text-align: center; font-variant-numeric: tabular-nums; font-weight: bold; }}
    .matrix th {{ text-align: center; }}
    .auc {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 16px; }}
    .auc div {{ padding: 11px 12px; border-radius: 8px; background: #eff6ff; border: 1px solid #bfdbfe; color: #1d4ed8; }}
    .auc span {{ display: block; color: {_MUTED}; font-size: 11px; font-weight: bold; letter-spacing: .06em; text-transform: uppercase; }}
    .auc strong {{ font-size: 21px; }}
    .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }}
    figure {{ margin: 0; padding: 13px; border: 1px solid #dbe5f1; border-radius: 10px; }}
    figure.full {{ grid-column: 1 / -1; }}
    figcaption {{ color: {_MUTED}; font-size: 12px; font-weight: bold; }}
    img {{ display: block; width: 100%; height: auto; margin-top: 6px; }}
    footer {{ margin-top: 28px; padding-top: 12px; border-top: 1px solid #dbe5f1; color: {_MUTED}; font-size: 11px; }}
    @media print {{ body {{ background: #fff; }} main {{ margin: 0; max-width: none; box-shadow: none; }} }}
    @media (max-width: 700px) {{ main {{ margin: 0; padding: 22px; }} header, .details {{ display: block; }} .meta {{ margin-top: 14px; text-align: left; }} .metrics {{ grid-template-columns: repeat(2, 1fr); }} .charts {{ grid-template-columns: 1fr; }} figure.full {{ grid-column: auto; }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <div><div class=\"eyebrow\">FluidFlagger</div><h1>Validation Report</h1><p>Performance summary for reviewed predictions</p></div>
      <div class=\"meta\">Generated {html.escape(generated_at.strftime('%Y-%m-%d %H:%M'))}<br>Threshold rule: {html.escape(str(payload['threshold_rule']))}</div>
    </header>
    <h2>Performance at threshold {_decimal(metrics['threshold'])}</h2>
    <div class=\"metrics\">{metric_cards}</div>
    <h2>Validation cohort</h2>
    <div class=\"details\">
      <table>{metadata_rows}</table>
      <div>
        <div class=\"auc\"><div><span>ROC AUC</span><strong>{_decimal(auc['roc'])}</strong></div><div><span>Average precision</span><strong>{_decimal(auc['pr'])}</strong></div></div>
        <h3>2 x 2 classification table</h3>
        <table class=\"matrix\"><thead><tr><th>Truth / predicted</th><th>Real</th><th>Contaminated</th></tr></thead><tbody>{confusion_rows}</tbody></table>
      </div>
    </div>
    <h2>Discrimination and calibration</h2>
    <div class=\"charts\">
      <figure><figcaption>ROC curve</figcaption><img alt=\"ROC curve with 2 x 2 table\" src=\"{_image_data_uri(chart_paths['roc'])}\"></figure>
      <figure><figcaption>Precision-recall curve</figcaption><img alt=\"Precision-recall curve\" src=\"{_image_data_uri(chart_paths['pr'])}\"></figure>
      <figure class=\"full\"><figcaption>Calibration plot</figcaption><img alt=\"Calibration plot\" src=\"{_image_data_uri(chart_paths['calibration'])}\"></figure>
    </div>
    <footer>FluidFlagger validation reports contain aggregate performance results only. A positive prediction is defined by the stated score rule.</footer>
  </main>
</body>
</html>
"""


def _build_pdf_report(payload: dict[str, Any], chart_paths: dict[str, Path], output_path: Path, generated_at: dt.datetime) -> None:
    """Create the matching polished PDF report using ReportLab."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_RIGHT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    metrics = payload["metrics"]
    auc = payload["auc"]
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="Eyebrow",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor(_BLUE),
        spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor(_MUTED),
    ))
    styles.add(ParagraphStyle(
        name="Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=15,
        spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="Small",
        parent=styles["Normal"],
        fontSize=8.3,
        leading=11,
        textColor=colors.HexColor(_MUTED),
    ))
    styles.add(ParagraphStyle(
        name="MetaRight",
        parent=styles["Normal"],
        alignment=TA_RIGHT,
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor(_MUTED),
    ))

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="FluidFlagger Validation Report",
        author="FluidFlagger",
    )
    story: list[Any] = []
    header_left = [
        Paragraph("FLUIDFLAGGER", styles["Eyebrow"]),
        Paragraph("Validation Report", styles["ReportTitle"]),
        Paragraph("Performance summary for reviewed predictions", styles["ReportSubtitle"]),
    ]
    header_right = Paragraph(
        "Generated {}<br/>Threshold rule: {}".format(
            html.escape(generated_at.strftime("%Y-%m-%d %H:%M")),
            html.escape(str(payload["threshold_rule"])),
        ),
        styles["MetaRight"],
    )
    header = Table([[header_left, header_right]], colWidths=[4.85 * inch, 2.0 * inch], hAlign="LEFT")
    header.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 2, colors.HexColor(_BLUE)),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 13),
    ]))
    story.extend([header, Spacer(1, 8)])

    story.append(Paragraph(f"Performance at threshold {_decimal(metrics['threshold'])}", styles["Section"]))
    metric_labels = _metric_rows(payload)
    metrics_table = Table(
        [[Paragraph(f"<b>{label}</b><br/><font size=15>{value}</font>", styles["Small"]) for label, value in metric_labels]],
        colWidths=[1.37 * inch] * 5,
        hAlign="LEFT",
    )
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#dbe5f1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.75, colors.HexColor("#dbe5f1")),
        ("LINEABOVE", (0, 0), (-1, 0), 2.5, colors.HexColor(_BLUE)),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.extend([metrics_table, Spacer(1, 8)])

    metadata_data = [[Paragraph(f"<b>{html.escape(label)}</b>", styles["Small"]), Paragraph(html.escape(value), styles["Small"])] for label, value in _report_metadata(payload)]
    metadata_table = Table(metadata_data, colWidths=[1.75 * inch, 2.28 * inch], hAlign="LEFT")
    metadata_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8fafc")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dbe5f1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    matrix_data = [
        ["Truth / predicted", "Real", "Contam."],
        ["Real", _count(metrics["tn"]), _count(metrics["fp"])],
        ["Contaminated", _count(metrics["fn"]), _count(metrics["tp"])],
    ]
    matrix_table = Table(matrix_data, colWidths=[1.25 * inch, 0.75 * inch, 0.85 * inch], hAlign="LEFT")
    matrix_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#f8fafc")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dbe5f1")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    auc_table = Table(
        [[Paragraph(f"<b>ROC AUC</b><br/><font size=15>{_decimal(auc['roc'])}</font>", styles["Small"]), Paragraph(f"<b>Average precision</b><br/><font size=15>{_decimal(auc['pr'])}</font>", styles["Small"])]],
        colWidths=[1.43 * inch, 1.43 * inch],
        hAlign="LEFT",
    )
    auc_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#bfdbfe")),
        ("INNERGRID", (0, 0), (-1, -1), 0.75, colors.HexColor("#bfdbfe")),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    analysis_right = [
        auc_table,
        Spacer(1, 9),
        Paragraph("2 x 2 classification table", styles["Small"]),
        Spacer(1, 3),
        matrix_table,
    ]
    details = Table([[metadata_table, analysis_right]], colWidths=[4.05 * inch, 2.85 * inch], hAlign="LEFT")
    details.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")] ))
    story.extend([Paragraph("Validation cohort", styles["Section"]), details])

    story.append(PageBreak())
    story.append(Paragraph("Discrimination and calibration", styles["Section"]))
    roc_image = Image(str(chart_paths["roc"]), width=3.35 * inch, height=2.28 * inch)
    pr_image = Image(str(chart_paths["pr"]), width=3.35 * inch, height=2.28 * inch)
    curve_images = Table([[roc_image, pr_image]], colWidths=[3.43 * inch, 3.43 * inch], hAlign="LEFT")
    curve_images.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.extend([curve_images, Spacer(1, 14)])
    # Keep the calibration figure comfortably above the footer on US Letter.
    story.append(Image(str(chart_paths["calibration"]), width=6.0 * inch, height=3.1 * inch))
    story.extend([
        Spacer(1, 9),
        Paragraph(
            "FluidFlagger validation reports contain aggregate performance results only. "
            "A positive prediction is defined by the stated score rule.",
            styles["Small"],
        ),
    ])

    def add_page_number(canvas, doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#dbe5f1"))
        canvas.line(doc.leftMargin, 0.35 * inch, letter[0] - doc.rightMargin, 0.35 * inch)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor(_MUTED))
        canvas.drawRightString(letter[0] - doc.rightMargin, 0.2 * inch, f"Page {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def build_validation_report_files(
    payload: dict[str, Any],
    output_dir: str | Path | None = None,
) -> tuple[str, str]:
    """Create standalone HTML and PDF downloads for a validation payload.

    The return values are paths suitable for :class:`gradio.DownloadButton`.
    Intermediate chart images share the temporary report directory so the
    downloads remain available throughout the browser session.
    """
    report_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(tempfile.mkdtemp(prefix="fluidflagger_validation_report_"))
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    generated_at = dt.datetime.now().astimezone()
    stamp = generated_at.strftime("%Y%m%d_%H%M%S")
    chart_paths = _render_chart_images(payload, report_dir)
    html_path = report_dir / f"fluidflagger_validation_report_{stamp}.html"
    pdf_path = report_dir / f"fluidflagger_validation_report_{stamp}.pdf"
    html_path.write_text(_build_html_report(payload, chart_paths, generated_at), encoding="utf-8")
    _build_pdf_report(payload, chart_paths, pdf_path, generated_at)
    return str(html_path), str(pdf_path)
