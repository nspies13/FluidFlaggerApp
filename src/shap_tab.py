"""
SHAP tab — global feature importance for a trained FluidFlagger model.

Supports selecting a built-in HuggingFace model from a dropdown or uploading
a custom .joblib file.  Rows with any missing predictor values are dropped
before SHAP computation.
"""

from __future__ import annotations

import re
import tempfile

import gradio as gr
import numpy as np
import pandas as pd

from .features import preprocess_bmp_data, preprocess_cbc_data
from .model_loader import (
    bmp_classification_key,
    bmp_mix_ratio_key,
    cbc_classification_key,
    cbc_mix_ratio_key,
    get_model,
    load_from_file,
)
from .simulate import get_fluid_names

# ---------------------------------------------------------------------------
# Build the HF model catalogue at import time
# ---------------------------------------------------------------------------

_TIMINGS = ("Realtime", "Retrospective")

# Ordered list of (display_label, model_key) for the dropdown
_HF_MODEL_OPTIONS: list[tuple[str, str]] = []
for _fluid in get_fluid_names():
    for _timing in _TIMINGS:
        _HF_MODEL_OPTIONS.append((f"BMP · {_fluid} · {_timing}", bmp_classification_key(_fluid, _timing)))
    _HF_MODEL_OPTIONS.append((f"BMP · {_fluid} · Mix Ratio", bmp_mix_ratio_key(_fluid)))
for _timing in _TIMINGS:
    _HF_MODEL_OPTIONS.append((f"CBC · {_timing}", cbc_classification_key(_timing)))
_HF_MODEL_OPTIONS.append(("CBC · Mix Ratio", cbc_mix_ratio_key()))

_LABEL_TO_KEY: dict[str, str] = {label: key for label, key in _HF_MODEL_OPTIONS}
_DROPDOWN_LABELS: list[str] = [label for label, _ in _HF_MODEL_OPTIONS]

_UPLOAD_SENTINEL = "— Upload custom .joblib —"

# ---------------------------------------------------------------------------
# Feature label helpers
# ---------------------------------------------------------------------------

_ANALYTE_LABEL: dict[str, str] = {
    # BMP
    "sodium":         "Na⁺",
    "chloride":       "Cl⁻",
    "potassium_plas": "K⁺",
    "co2_totl":       "CO₂",
    "bun":            "BUN",
    "creatinine":     "Creatinine",
    "calcium":        "Ca²⁺",
    "glucose":        "Glucose",
    # CBC
    "Hgb": "Hemoglobin",
    "Plt": "Platelets",
    "WBC": "WBC",
}


def _feature_label(name: str) -> str:
    """Convert a raw feature name (e.g. 'sodium_log_delta_prior') to a
    human-readable label (e.g. 'Na⁺ Δ Prior')."""
    if name.endswith("_log_delta_prior"):
        base = name[: -len("_log_delta_prior")]
        return f"{_ANALYTE_LABEL.get(base, base)} Δ Prior"
    if name.endswith("_log_delta_post"):
        base = name[: -len("_log_delta_post")]
        return f"{_ANALYTE_LABEL.get(base, base)} Δ Post"
    return _ANALYTE_LABEL.get(name, name)


def _is_pc_feature(name: str) -> bool:
    return bool(re.search(r"PC\d+", name))


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _compute_shap_png(
    model_label: str,
    custom_model_path: str | None,
    data_path: str,
) -> tuple[str | None, str | None, str]:
    """
    Load model + data, drop incomplete rows, compute global SHAP values,
    filter PC features, select top 10, and save a beeswarm plot.

    Returns (png_path | None, svg_path | None, status_message).
    PNG is used for display (high-DPI); SVG is used for download.
    """
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        return None, None, f"⚠️ Missing dependency: {e}. Run: pip install shap matplotlib"

    # -- Load model ----------------------------------------------------------
    try:
        if model_label == _UPLOAD_SENTINEL:
            if not custom_model_path:
                return None, None, "⚠️ Please upload a custom .joblib file."
            model_dict = load_from_file(custom_model_path)
        else:
            key = _LABEL_TO_KEY[model_label]
            model_dict = get_model(key)
            if model_dict is None:
                return None, None, f"⚠️ Could not download model '{key}' from HuggingFace Hub."
    except Exception as e:
        return None, None, f"⚠️ Could not load model: {e}"

    panel = model_dict.get("panel", "")
    if panel not in ("bmp", "cbc"):
        return None, None, f"⚠️ Unrecognised panel in model: {panel!r}"

    pipeline = model_dict["pipeline"]

    # -- Load & preprocess data ----------------------------------------------
    try:
        raw_df = pd.read_csv(data_path, sep=None, engine="python")
    except Exception as e:
        return None, None, f"⚠️ Could not read dataset: {e}"

    try:
        df = preprocess_bmp_data(raw_df) if panel == "bmp" else preprocess_cbc_data(raw_df)
    except Exception as e:
        return None, None, f"⚠️ Preprocessing failed: {e}"

    # -- Retain rows complete for this model's predictors -------------------
    # A Review export also has probability, metadata, and label columns. Some
    # of those can be empty without preventing a model explanation, so do not
    # use a blanket ``df.dropna()`` here.
    try:
        transformer = pipeline["features"]
        analytes = list(getattr(transformer, "_analytes", []))
        mode = getattr(transformer, "mode", "retrospective")
        required_columns = analytes + [f"{name}_prior" for name in analytes]
        if mode == "retrospective":
            required_columns += [f"{name}_post" for name in analytes]
        missing_columns = [name for name in required_columns if name not in df.columns]
        if missing_columns:
            preview = ", ".join(missing_columns[:4])
            suffix = ", ..." if len(missing_columns) > 4 else ""
            return None, None, f"⚠️ This file is missing model input columns: {preview}{suffix}"
        n_raw = len(df)
        df = df.dropna(subset=required_columns)
        n_dropped = n_raw - len(df)
    except Exception as exc:
        return None, None, f"⚠️ Could not prepare model inputs: {exc}"

    if len(df) == 0:
        return None, None, "⚠️ No complete rows remain after dropping rows with missing values."

    # -- Transform features --------------------------------------------------
    try:
        X = transformer.transform(df)
        feature_names = list(transformer.get_feature_names_out())
    except Exception as e:
        return None, None, f"⚠️ Feature transformation failed: {e}"

    # Safety net: drop any NaNs introduced by the transformer itself
    n_before = len(X)
    X = X.dropna()
    n_dropped += n_before - len(X)

    if len(X) == 0:
        return None, None, "⚠️ No complete rows remain after dropping rows with missing values."

    drop_note = f" ({n_dropped:,} incomplete row{'s' if n_dropped != 1 else ''} dropped)" if n_dropped else ""

    # -- Compute SHAP --------------------------------------------------------
    try:
        estimator = pipeline[-1]
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X)
    except Exception as e:
        return None, None, f"⚠️ SHAP computation failed: {e}"

    # For binary classifiers shap_values is a list [neg_class, pos_class];
    # use the positive (contaminated) class.
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        sv = shap_values

    # -- Filter PCs and select top 10 ----------------------------------------
    non_pc_idx = [i for i, n in enumerate(feature_names) if not _is_pc_feature(n)]
    sv_filtered = sv[:, non_pc_idx]
    X_filtered = X.iloc[:, non_pc_idx]
    names_filtered = [feature_names[i] for i in non_pc_idx]

    mean_abs = np.abs(sv_filtered).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:10]
    sv_plot = sv_filtered[:, top_idx]
    X_plot = X_filtered.iloc[:, top_idx]
    labels_plot = [_feature_label(names_filtered[i]) for i in top_idx]

    # -- Plot ----------------------------------------------------------------
    task = model_dict.get("task", "classification")
    fluid = model_dict.get("fluid", "")
    timing = model_dict.get("type", "")
    title = f"SHAP Summary — {panel.upper()} · {fluid} · {timing}"

    try:
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        })
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            sv_plot, X_plot,
            feature_names=labels_plot,
            show=False,
            plot_size=None,
        )
        fig = plt.gcf()
        ax = fig.axes[0]
        ax.set_xlabel("SHAP Value (Impact on Model Output)", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.grid(False)
        ax.set_title(title, fontsize=12, fontweight="bold", fontstyle="italic", pad=14)
        fig.tight_layout()

        tmp_png = tempfile.mktemp(suffix=".png")
        tmp_svg = tempfile.mktemp(suffix=".svg")
        fig.savefig(tmp_png, bbox_inches="tight", dpi=600)
        fig.savefig(tmp_svg, bbox_inches="tight", format="svg", dpi=600)
        plt.close(fig)
    except Exception as e:
        plt.close("all")
        return None, None, f"⚠️ Plot generation failed: {e}"

    return tmp_png, tmp_svg, f"✓ SHAP plot computed on {len(X):,} rows{drop_note}."


# ---------------------------------------------------------------------------
# Gradio section
# ---------------------------------------------------------------------------

def build_shap_section() -> None:
    """Build the Explain UI in the current Gradio layout context."""
    with gr.Row():
        # ── Left sidebar ───────────────────────────────────────────
        with gr.Column(scale=1, min_width=220):
            gr.HTML('<p class="ff-section-title">Model</p>', elem_classes="ff-th")
            model_dropdown = gr.Dropdown(
                choices=_DROPDOWN_LABELS + [_UPLOAD_SENTINEL],
                value=_DROPDOWN_LABELS[0],
                label="Select model",
                interactive=True,
            )
            custom_model_upload = gr.File(
                label="Upload .joblib file",
                file_types=[".joblib"],
                visible=False,
            )

            gr.HTML('<hr class="ff-divider">')
            gr.HTML('<p class="ff-section-title">Dataset</p>', elem_classes="ff-th")
            data_upload = gr.File(
                label="Upload CSV",
                file_types=[".csv", ".tsv"],
            )

            gr.HTML('<hr class="ff-divider">')
            run_btn = gr.Button(
                "📊  Generate SHAP Plot",
                variant="primary", size="lg",
                interactive=False,
            )

        # ── Right main area ────────────────────────────────────────
        with gr.Column(scale=3):
            empty_state = gr.HTML(
                '<div class="ff-empty-state">'
                '<div style="font-size:1.75rem;margin-bottom:10px">📊</div>'
                '<p>Select a <strong>model</strong> and upload a <strong>dataset CSV</strong>'
                ' to generate a global SHAP summary plot</p>'
                '</div>',
                visible=True,
            )
            status_msg = gr.Markdown("", elem_classes="ff-status")
            plot_image = gr.Image(
                label="SHAP Summary Plot",
                visible=False,
                interactive=False,
                elem_classes="ff-shap-plot",
            )
            download_btn = gr.DownloadButton(
                "⬇  Download Plot (SVG)",
                visible=False, variant="secondary",
                elem_id="shap-download-btn",
            )

    # ── Show/hide custom upload when sentinel selected ─────────────
    def _on_model_change(label, data_f):
        show_upload = label == _UPLOAD_SENTINEL
        # Ready if a HF model is selected, OR upload sentinel + file present
        ready = bool(data_f) and (not show_upload or False)
        return gr.update(visible=show_upload), gr.update(interactive=ready)

    model_dropdown.change(
        _on_model_change,
        inputs=[model_dropdown, data_upload],
        outputs=[custom_model_upload, run_btn],
    )

    # ── Enable run button when data + valid model are present ──────
    def _toggle_run(label, custom_f, data_f):
        if not data_f:
            return gr.update(interactive=False)
        if label == _UPLOAD_SENTINEL and not custom_f:
            return gr.update(interactive=False)
        return gr.update(interactive=True)

    for trigger in (model_dropdown, custom_model_upload, data_upload):
        trigger.change(
            _toggle_run,
            inputs=[model_dropdown, custom_model_upload, data_upload],
            outputs=run_btn,
        )

    # ── Generate callback ──────────────────────────────────────────
    def _generate(label, custom_f, data_f):
        if not data_f:
            return (
                "⚠️ Please upload a dataset CSV.",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
            )
        png_path, svg_path, msg = _compute_shap_png(label, custom_f, data_f)
        if png_path is None:
            return (
                msg,
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
            )
        return (
            msg,
            gr.update(value=png_path, visible=True),
            gr.update(value=svg_path, visible=True),
            gr.update(visible=False),
        )

    _outputs = [status_msg, plot_image, download_btn, empty_state]

    run_btn.click(
        lambda: (
            "⏳ Computing SHAP values — this may take a moment…",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        ),
        outputs=_outputs,
        queue=False,
    ).then(
        _generate,
        inputs=[model_dropdown, custom_model_upload, data_upload],
        outputs=_outputs,
    )


def build_validation_shap_plot(validation_file: gr.File) -> None:
    """Build Validate's inline SHAP controls and plot.

    The reviewed-predictions upload already contains the model inputs needed
    for explanation, so this compact variant deliberately reuses that file
    instead of asking the user to upload a second dataset.  It is intended to
    sit immediately below the calibration chart in the Validate results flow.
    """
    gr.HTML(
        '<div class="ff-validation-shap-header">'
        '<div>'
        '<p class="ff-section-title">Feature importance</p>'
        '<h3>Understand this validation result</h3>'
        '<p>Generate a SHAP summary for the selected model using the same reviewed file.</p>'
        '</div>'
        '</div>'
    )
    with gr.Row(elem_classes="ff-validation-shap-row"):
        with gr.Column(
            scale=1,
            min_width=220,
            elem_classes="ff-validation-shap-toolbar",
        ):
            model_dropdown = gr.Dropdown(
                choices=_DROPDOWN_LABELS + [_UPLOAD_SENTINEL],
                value=_DROPDOWN_LABELS[0],
                label="Model to explain",
                interactive=True,
            )
            custom_model_upload = gr.File(
                label="Upload .joblib model",
                file_types=[".joblib"],
                visible=False,
            )
            run_btn = gr.Button(
                "📊  Generate SHAP Plot",
                variant="primary",
                size="lg",
                interactive=False,
            )

        with gr.Column(scale=3, elem_classes="ff-validation-shap-output"):
            empty_state = gr.HTML(
                '<div class="ff-empty-state">'
                '<div style="font-size:1.5rem;margin-bottom:8px">✨</div>'
                '<p>Select the model used for this validation file, then generate '
                'a <strong>SHAP feature-importance plot</strong>.</p>'
                '</div>',
                visible=True,
            )
            status_msg = gr.Markdown("", elem_classes="ff-status")
            plot_image = gr.Image(
                label="SHAP Feature Importance",
                visible=False,
                interactive=False,
                elem_classes="ff-shap-plot",
            )
            download_btn = gr.DownloadButton(
                "⬇  Download SHAP Plot (SVG)",
                visible=False,
                variant="secondary",
                elem_id="validation-shap-download-btn",
            )

    def _show_custom_model(label):
        return gr.update(visible=label == _UPLOAD_SENTINEL)

    def _toggle_run(label, custom_f, validation_f):
        if not validation_f:
            return gr.update(interactive=False)
        if label == _UPLOAD_SENTINEL and not custom_f:
            return gr.update(interactive=False)
        return gr.update(interactive=True)

    def _clear_plot():
        """Avoid showing an explanation generated for an older file/model."""
        return (
            "",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
        )

    model_dropdown.change(
        _show_custom_model,
        inputs=model_dropdown,
        outputs=custom_model_upload,
    )
    for trigger in (model_dropdown, custom_model_upload, validation_file):
        trigger.change(
            _toggle_run,
            inputs=[model_dropdown, custom_model_upload, validation_file],
            outputs=run_btn,
        )

    def _generate(label, custom_f, validation_f):
        if not validation_f:
            return (
                "⚠️ Upload and validate a reviewed predictions file first.",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
            )
        png_path, svg_path, msg = _compute_shap_png(label, custom_f, validation_f)
        if png_path is None:
            return (
                msg,
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
            )
        return (
            msg,
            gr.update(value=png_path, visible=True),
            gr.update(value=svg_path, visible=True),
            gr.update(visible=False),
        )

    outputs = [status_msg, plot_image, download_btn, empty_state]
    for trigger in (model_dropdown, custom_model_upload, validation_file):
        trigger.change(_clear_plot, outputs=outputs)

    run_btn.click(
        lambda: (
            "⏳ Computing SHAP values - this may take a moment...",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        ),
        outputs=outputs,
        queue=False,
    ).then(
        _generate,
        inputs=[model_dropdown, custom_model_upload, validation_file],
        outputs=outputs,
    )


def build_shap_tab() -> None:
    """Build the legacy standalone Explain tab."""
    with gr.Tab("🔍  Explain"):
        build_shap_section()
