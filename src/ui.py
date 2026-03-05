"""
Gradio Blocks UI — three tabs mirroring the Shiny prediction_app.R:
  1. Predict  — file upload (wide/long CSV) or manual entry → predictions table + download
  2. Train    — upload training data → train models → download models
  3. Review   — step through predictions and label each row
"""

from __future__ import annotations

import io
import json
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import gradio as gr
import pandas as pd

from .features import (
    BMP_ANALYTES,
    CBC_ANALYTES,
    preprocess_bmp_data,
    preprocess_cbc_data,
)
from .model_loader import (
    cache_model,
    load_from_file,
    model_key as _model_key,
)
from .simulate import get_fluid_names

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_FLUIDS = get_fluid_names()

BMP_MANUAL_FIELDS = [
    ("sodium",         "Sodium (mEq/L)",       135.0, 145.0),
    ("chloride",       "Chloride (mEq/L)",      98.0, 106.0),
    ("potassium_plas", "Potassium (mEq/L)",      3.5,   5.0),
    ("co2_totl",       "CO₂/Bicarb (mEq/L)",   22.0,  29.0),
    ("bun",            "BUN (mg/dL)",            7.0,  20.0),
    ("creatinine",     "Creatinine (mg/dL)",     0.6,   1.2),
    ("calcium",        "Calcium (mg/dL)",        8.5,  10.2),
    ("glucose",        "Glucose (mg/dL)",       70.0, 100.0),
]

CBC_MANUAL_FIELDS = [
    ("Hgb", "Hemoglobin (g/dL)", 12.0, 17.5),
    ("WBC", "WBC (×10³/µL)",      4.5,  11.0),
    ("Plt", "Platelets (×10³/µL)", 150.0, 400.0),
]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _read_uploaded_csv(file_path: str | None) -> Optional[pd.DataFrame]:
    if not file_path:
        return None
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise gr.Error(f"Could not read file: {e}")


def _save_temp_csv(df: pd.DataFrame) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)
    return tmp.name

# ---------------------------------------------------------------------------
# Tab 1 – Predict
# ---------------------------------------------------------------------------

def _run_bmp_prediction(
    file_path: str | None,
    input_format: str,
    selected_fluids: list[str],
    *manual_values,
) -> tuple[pd.DataFrame | None, str | None, str]:
    """Core BMP prediction handler. Returns (result_df, csv_path, status_msg)."""
    from .inference import make_bmp_predictions

    if file_path:
        df = _read_uploaded_csv(file_path)
        if input_format == "Long (one row per analyte)":
            df = preprocess_bmp_data(df)
        else:
            df = preprocess_bmp_data(df)
    else:
        # Build single-row DataFrame from manual inputs
        row = {analyte: val for (analyte, *_), val in zip(BMP_MANUAL_FIELDS, manual_values)}
        df = pd.DataFrame([row])

    fluids = selected_fluids if selected_fluids else ALL_FLUIDS
    try:
        result = make_bmp_predictions(df, selected_fluids=fluids)
    except RuntimeError as e:
        return None, None, f"⚠️ {e}"
    except Exception as e:
        return None, None, f"Error: {traceback.format_exc()}"

    csv_path = _save_temp_csv(result)
    return result, csv_path, f"✓ {len(result)} row(s) predicted."


def _run_cbc_prediction(
    file_path: str | None,
    input_format: str,
    *manual_values,
) -> tuple[pd.DataFrame | None, str | None, str]:
    from .inference import make_cbc_predictions

    if file_path:
        df = _read_uploaded_csv(file_path)
        if input_format == "Long (one row per analyte)":
            df = preprocess_cbc_data(df)
        else:
            df = preprocess_cbc_data(df)
    else:
        row = {analyte: val for (analyte, *_), val in zip(CBC_MANUAL_FIELDS, manual_values)}
        df = pd.DataFrame([row])

    try:
        result = make_cbc_predictions(df)
    except RuntimeError as e:
        return None, None, f"⚠️ {e}"
    except Exception as e:
        return None, None, f"Error: {traceback.format_exc()}"

    csv_path = _save_temp_csv(result)
    return result, csv_path, f"✓ {len(result)} row(s) predicted."


def build_predict_tab() -> None:
    with gr.Tab("Predict"):
        gr.Markdown("## Contamination Prediction\nUpload a CSV or enter values manually.")

        panel = gr.Radio(["BMP", "CBC"], value="BMP", label="Panel", interactive=True)
        input_format = gr.Radio(
            ["Wide (one row per draw)", "Long (one row per analyte)"],
            value="Wide (one row per draw)",
            label="Input Format",
            interactive=True,
        )
        file_upload = gr.File(label="Upload CSV (optional)", file_types=[".csv"])

        # --- BMP manual entry ---
        with gr.Group(visible=True) as bmp_manual:
            gr.Markdown("**Manual entry** (single draw — leave file empty above)")
            bmp_inputs = []
            with gr.Row():
                for analyte, label, lo, hi in BMP_MANUAL_FIELDS[:4]:
                    inp = gr.Number(label=label, value=None, minimum=0)
                    bmp_inputs.append(inp)
            with gr.Row():
                for analyte, label, lo, hi in BMP_MANUAL_FIELDS[4:]:
                    inp = gr.Number(label=label, value=None, minimum=0)
                    bmp_inputs.append(inp)

            gr.Markdown("**Fluid filter** (uncheck to exclude fluids from aggregate results)")
            fluid_checkboxes = gr.CheckboxGroup(
                choices=ALL_FLUIDS,
                value=ALL_FLUIDS,
                label="Fluids to include",
            )

        # --- CBC manual entry ---
        with gr.Group(visible=False) as cbc_manual:
            gr.Markdown("**Manual entry** (single draw — leave file empty above)")
            cbc_inputs = []
            with gr.Row():
                for analyte, label, lo, hi in CBC_MANUAL_FIELDS:
                    inp = gr.Number(label=label, value=None, minimum=0)
                    cbc_inputs.append(inp)

        run_btn = gr.Button("Run Prediction", variant="primary")
        status_msg = gr.Markdown("")
        results_table = gr.DataFrame(label="Results", interactive=False, wrap=True)
        download_btn = gr.File(label="Download results CSV", visible=False)

        # Show/hide panel-specific groups
        def _toggle_panel(p):
            return gr.update(visible=p == "BMP"), gr.update(visible=p == "CBC")

        panel.change(_toggle_panel, inputs=panel, outputs=[bmp_manual, cbc_manual])

        # Run prediction
        def _predict(p, fmt, file, fluids, *vals):
            n_bmp = len(bmp_inputs)
            bmp_vals = vals[:n_bmp]
            cbc_vals = vals[n_bmp:]
            if p == "BMP":
                df, csv_path, msg = _run_bmp_prediction(file, fmt, fluids, *bmp_vals)
            else:
                df, csv_path, msg = _run_cbc_prediction(file, fmt, *cbc_vals)

            if df is None:
                return msg, gr.update(value=None), gr.update(visible=False)
            return msg, gr.update(value=df), gr.update(value=csv_path, visible=True)

        run_btn.click(
            _predict,
            inputs=[panel, input_format, file_upload, fluid_checkboxes] + bmp_inputs + cbc_inputs,
            outputs=[status_msg, results_table, download_btn],
        )

# ---------------------------------------------------------------------------
# Tab 2 – Train
# ---------------------------------------------------------------------------

def _run_training(
    panel: str,
    template_file: str | None,
    fluids_file: str | None,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str | None]:
    """Train models and return (status_message, zip_path_or_none)."""
    import zipfile
    from .train import train_bmp_models, train_cbc_models, save_models

    if not template_file:
        return "⚠️ Please upload a training template CSV.", None

    try:
        template_df = pd.read_csv(template_file)
    except Exception as e:
        return f"Could not read template: {e}", None

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            if panel == "BMP":
                progress(0.1, desc="Loading fluid concentrations…")
                if fluids_file:
                    fluids_df = pd.read_csv(fluids_file, sep=None, engine="python")
                else:
                    from .simulate import get_fluid_concentrations
                    fluids_df = get_fluid_concentrations()

                progress(0.2, desc="Training BMP models (this may take several minutes)…")
                models = train_bmp_models(template_df, fluids_df)
            else:
                progress(0.2, desc="Training CBC models…")
                models = train_cbc_models(template_df)

            progress(0.8, desc="Saving models…")
            paths = save_models(models, tmpdir)

            # Zip all .joblib files for download
            zip_path = tempfile.mktemp(suffix=".zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in paths:
                    zf.write(p, arcname=Path(p).name)

            # Also load into cache so they can be used immediately
            for m in models:
                cache_model(_model_key(m), m)

            progress(1.0, desc="Done.")
            return f"✓ Trained {len(models)} models. Download the zip below.", zip_path

        except Exception:
            return f"Training failed:\n{traceback.format_exc()}", None


def build_train_tab() -> None:
    with gr.Tab("Train"):
        gr.Markdown(
            "## Train Custom Models\n"
            "Upload a wide-format CSV (one row per draw, analyte values as columns).\n"
            "Prior/post columns (`sodium_prior`, `sodium_post`, etc.) are required for retrospective models."
        )

        panel = gr.Radio(["BMP", "CBC"], value="BMP", label="Panel", interactive=True)
        template_file = gr.File(label="Training template CSV", file_types=[".csv"])

        with gr.Group() as bmp_train_extras:
            fluids_file = gr.File(
                label="Fluid concentrations TSV (optional — uses built-in defaults if omitted)",
                file_types=[".tsv", ".csv"],
            )

        panel.change(
            lambda p: gr.update(visible=p == "BMP"),
            inputs=panel,
            outputs=bmp_train_extras,
        )

        train_btn = gr.Button("Train Models", variant="primary")
        train_status = gr.Markdown("")
        model_download = gr.File(label="Download trained models (zip)", visible=False)

        def _train(p, tmpl, fluids, progress=gr.Progress()):
            msg, zip_path = _run_training(p, tmpl, fluids, progress)
            if zip_path:
                return msg, gr.update(value=zip_path, visible=True)
            return msg, gr.update(visible=False)

        train_btn.click(
            _train,
            inputs=[panel, template_file, fluids_file],
            outputs=[train_status, model_download],
        )

        # --- Upload custom models ---
        gr.Markdown("---\n### Load Custom Models\nUpload individual `.joblib` model files to use in the Predict tab.")
        model_upload = gr.File(
            label="Upload .joblib model file(s)",
            file_types=[".joblib"],
            file_count="multiple",
        )
        load_btn = gr.Button("Load Models")
        load_status = gr.Markdown("")

        def _load_models(files):
            if not files:
                return "No files selected."
            loaded = []
            for f in files:
                try:
                    m = load_from_file(f)
                    key = _model_key(m)
                    cache_model(key, m)
                    loaded.append(key)
                except Exception as e:
                    return f"Failed to load {Path(f).name}: {e}"
            return f"✓ Loaded: {', '.join(loaded)}"

        load_btn.click(_load_models, inputs=model_upload, outputs=load_status)

# ---------------------------------------------------------------------------
# Tab 3 – Review
# ---------------------------------------------------------------------------

def build_review_tab() -> None:
    with gr.Tab("Review"):
        gr.Markdown(
            "## Label Predictions\n"
            "Upload a predictions CSV, then step through rows and label each one."
        )

        review_file = gr.File(label="Upload predictions CSV", file_types=[".csv"])
        load_review_btn = gr.Button("Load File")
        review_status = gr.Markdown("")

        # Navigation state (stored as JSON in a hidden textbox)
        state = gr.State({"df": None, "labels": [], "idx": 0})

        current_row = gr.DataFrame(label="Current row", interactive=False)
        row_counter = gr.Markdown("")

        with gr.Row():
            prev_btn = gr.Button("← Previous")
            next_btn = gr.Button("Next →")

        with gr.Row():
            real_btn = gr.Button("✓ Real", variant="secondary")
            equiv_btn = gr.Button("~ Equivocal", variant="secondary")
            contam_btn = gr.Button("✗ Contaminated", variant="secondary")

        download_labels_btn = gr.File(label="Download labels CSV", visible=False)

        # -- Load file --
        def _load_file(file_path, st):
            if not file_path:
                return st, "No file uploaded.", None, ""
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                return st, f"Error: {e}", None, ""
            new_state = {"df": df.to_dict(orient="records"), "labels": [None] * len(df), "idx": 0}
            row_df = pd.DataFrame([df.iloc[0]])
            counter = f"Row 1 / {len(df)}"
            return new_state, f"✓ Loaded {len(df)} rows.", row_df, counter

        load_review_btn.click(
            _load_file,
            inputs=[review_file, state],
            outputs=[state, review_status, current_row, row_counter],
        )

        # -- Navigation --
        def _go(st, delta):
            if st["df"] is None:
                return st, None, ""
            df = pd.DataFrame(st["df"])
            idx = max(0, min(len(df) - 1, st["idx"] + delta))
            st = {**st, "idx": idx}
            row_df = pd.DataFrame([df.iloc[idx]])
            counter = f"Row {idx + 1} / {len(df)}  |  Labeled: {sum(l is not None for l in st['labels'])}"
            return st, row_df, counter

        prev_btn.click(_go, inputs=[state, gr.Number(value=-1, visible=False)], outputs=[state, current_row, row_counter])
        next_btn.click(_go, inputs=[state, gr.Number(value=1, visible=False)], outputs=[state, current_row, row_counter])

        # -- Labelling --
        def _label(st, label_value):
            if st["df"] is None:
                return st, "", gr.update(visible=False)
            labels = list(st["labels"])
            labels[st["idx"]] = label_value
            st = {**st, "labels": labels}
            n_labeled = sum(l is not None for l in labels)
            counter = f"Row {st['idx'] + 1} / {len(labels)}  |  Labeled: {n_labeled}"

            # Build download CSV
            df = pd.DataFrame(st["df"])
            df["human_label"] = labels
            csv_path = _save_temp_csv(df)
            return st, counter, gr.update(value=csv_path, visible=True)

        real_btn.click(
            lambda st: _label(st, "Real"),
            inputs=state, outputs=[state, row_counter, download_labels_btn],
        )
        equiv_btn.click(
            lambda st: _label(st, "Equivocal"),
            inputs=state, outputs=[state, row_counter, download_labels_btn],
        )
        contam_btn.click(
            lambda st: _label(st, "Contaminated"),
            inputs=state, outputs=[state, row_counter, download_labels_btn],
        )

# ---------------------------------------------------------------------------
# Assemble full UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="FluidFlagger", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# FluidFlagger\n"
            "**IV-fluid contamination detection for BMP and CBC laboratory results.**\n\n"
            "Upload lab data or enter values manually to detect potential IV-fluid contamination."
        )
        build_predict_tab()
        build_train_tab()
        build_review_tab()
    return demo
