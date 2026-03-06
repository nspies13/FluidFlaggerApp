"""
Gradio Blocks UI — three tabs mirroring the Shiny prediction_app.R:
  1. Predict  — file upload (wide/long CSV) or manual entry → predictions table + download
  2. Train    — upload training data → train models → download models
  3. Review   — step through predictions and label each row
"""

from __future__ import annotations

import datetime
import io
import tempfile
import traceback
import warnings
from pathlib import Path
from typing import Optional

import gradio as gr
import pandas as pd

from .features import (
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

# ---------------------------------------------------------------------------
# Review table helpers
# ---------------------------------------------------------------------------

_BMP_FIELDS_REVIEW = [
    ("sodium",         "Na"),
    ("chloride",       "Cl"),
    ("potassium_plas", "K"),
    ("co2_totl",       "CO₂"),
    ("bun",            "BUN"),
    ("creatinine",     "Cr"),
    ("calcium",        "Ca"),
    ("glucose",        "Glu"),
]

_CBC_FIELDS_REVIEW = [
    ("Hgb", "Hgb"),
    ("WBC", "WBC"),
    ("Plt", "Plt"),
]


def _build_review_html(row_dict: dict) -> str:
    """Build a styled prior/current/post table for a single review row (all inline styles)."""
    if "sodium" in row_dict:
        fields = _BMP_FIELDS_REVIEW
    elif "Hgb" in row_dict:
        fields = _CBC_FIELDS_REVIEW
    else:
        return "<p style='color:#94a3b8;font-size:0.875rem'>No analyte columns detected.</p>"

    analytes = [col for col, _ in fields]
    abbrevs  = [abbr for _, abbr in fields]

    def _val(col):
        v = row_dict.get(col)
        if v is None:
            return None
        try:
            f = float(v)
            return None if pd.isna(f) else f
        except (TypeError, ValueError):
            return None

    def _fmt(v):
        if v is None:
            return "—"
        return f"{v:g}"

    current_vals = [_val(col) for col in analytes]
    prior_vals   = [_val(f"{col}_prior") for col in analytes]
    post_vals    = [_val(f"{col}_post")  for col in analytes]

    has_prior = any(v is not None for v in prior_vals)
    has_post  = any(v is not None for v in post_vals)

    S_TABLE  = "border-collapse:collapse;font-size:0.875rem;font-family:inherit;width:100%;border:none"
    S_TH     = "text-align:center;padding:8px 16px;background:#f8fafc;font-weight:600;font-style:italic;color:#475569;font-size:0.8125rem;white-space:nowrap;border:none"
    S_LBL    = "text-align:right;padding:8px 6px 8px 6px;font-weight:600;font-style:italic;font-size:0.6875rem;text-transform:uppercase;letter-spacing:0.06em;white-space:nowrap;min-width:60px;border:none"
    S_TD_P   = "text-align:center;padding:8px 16px;background:#f5f3ff;color:#4b5563;white-space:nowrap;border:none"
    S_TD_O   = "text-align:center;padding:8px 16px;background:#f0fdf4;color:#4b5563;white-space:nowrap;border:none"
    S_TD_C   = "text-align:center;padding:0;white-space:nowrap;border:none"
    # Header row
    th_cells = "".join(f'<th style="{S_TH}">{abbr}</th>' for abbr in abbrevs)
    html = f'<tr><th style="{S_LBL}"></th>{th_cells}</tr>'

    # Prior row
    if has_prior:
        cells = "".join(f'<td style="{S_TD_P}">{_fmt(v)}</td>' for v in prior_vals)
        html += f'<tr><td style="{S_LBL};color:#818cf8">prior</td>{cells}</tr>'

    S_ARROW_UP   = "display:block;font-size:0.6rem;line-height:1;color:#22d3ee;height:11px;text-align:center"
    S_ARROW_DN   = "display:block;font-size:0.6rem;line-height:1;color:#f472b6;height:11px;text-align:center"
    S_ARROW_NONE = "display:block;height:11px"
    S_NUM        = "display:block;font-size:0.875rem;font-weight:700;line-height:1.3;text-align:center;color:#1e40af"

    # Current row with arrows
    def _arrow_cell(curr, prior, post):
        if curr is None:
            return f'<td style="{S_TD_C}"><div style="background:#f0f9ff;padding:2px 14px;text-align:center"><span style="{S_NUM}">—</span></div></td>'
        # top arrow: prior → current
        if prior is not None and curr > prior:
            top = f'<div style="{S_ARROW_UP}">&#9650;</div>'
        elif prior is not None and curr < prior:
            top = f'<div style="{S_ARROW_DN}">&#9660;</div>'
        else:
            top = f'<div style="{S_ARROW_NONE}"></div>'
        # bottom arrow: current → post
        if post is not None and post > curr:
            bot = f'<div style="{S_ARROW_UP}">&#9650;</div>'
        elif post is not None and post < curr:
            bot = f'<div style="{S_ARROW_DN}">&#9660;</div>'
        else:
            bot = f'<div style="{S_ARROW_NONE}"></div>'
        return (
            f'<td style="{S_TD_C}">'
            f'<div style="background:#f0f9ff;padding:2px 14px;text-align:center">'
            f'{top}'
            f'<div style="{S_NUM}">{_fmt(curr)}</div>'
            f'{bot}'
            f'</div></td>'
        )

    curr_cells = "".join(
        _arrow_cell(c, p, o)
        for c, p, o in zip(current_vals, prior_vals, post_vals)
    )
    html += f'<tr><td style="{S_LBL};color:#38bdf8">current</td>{curr_cells}</tr>'

    # Post row
    if has_post:
        cells = "".join(f'<td style="{S_TD_O}">{_fmt(v)}</td>' for v in post_vals)
        html += f'<tr><td style="{S_LBL};color:#34d399">post</td>{cells}</tr>'

    return f'<div style="overflow-x:auto;margin:4px 0 12px 0;border:none"><table style="{S_TABLE}">{html}</table></div>'




BMP_MANUAL_FIELDS = [
    ("sodium",         "Sodium",      "mEq/L",  135.0, 145.0),
    ("chloride",       "Chloride",    "mEq/L",   98.0, 106.0),
    ("potassium_plas", "Potassium",   "mEq/L",    3.5,   5.0),
    ("co2_totl",       "CO₂",        "mEq/L",   22.0,  29.0),
    ("bun",            "BUN",         "mg/dL",    7.0,  20.0),
    ("creatinine",     "Creatinine",  "mg/dL",    0.6,   1.2),
    ("calcium",        "Calcium",     "mg/dL",    8.5,  10.2),
    ("glucose",        "Glucose",     "mg/dL",   70.0, 100.0),
]

CBC_MANUAL_FIELDS = [
    ("Hgb", "Hemoglobin", "g/dL",     12.0, 17.5),
    ("WBC", "WBC",        "×10³/µL",   4.5, 11.0),
    ("Plt", "Platelets",  "×10³/µL", 150.0, 400.0),
]

_INSTRUCTIONS_HTML = """
<div class="ff-instructions">
  <div class="ff-instr-grid">
    <div class="ff-instr-card">
      <div class="ff-instr-icon">📁</div>
      <h4>Upload a CSV</h4>
      <p>Upload a <strong>wide-format</strong> CSV (one row per draw, analyte values as columns) or a <strong>long-format</strong> CSV (one row per analyte, with a <code>TASK_ASSAY</code> column). Format is detected automatically. Include <code>_prior</code> and <code>_post</code> columns to enable retrospective models.</p>
    </div>
    <div class="ff-instr-card">
      <div class="ff-instr-icon">✏️</div>
      <h4>Manual Entry</h4>
      <p>Enter values for a single draw across three timepoints: the <strong>prior draw</strong>, the <strong>current draw</strong>, and the <strong>post draw</strong>. Prior and post values are optional but enable retrospective models.</p>
    </div>
    <div class="ff-instr-card">
      <div class="ff-instr-icon">🧪</div>
      <h4>Panels</h4>
      <p><strong>BMP</strong> — Na, Cl, K, CO₂, BUN, Creatinine, Ca, Glucose. Tests against any combination of IV fluids.<br><br>
         <strong>CBC</strong> — Hemoglobin, WBC, Platelets. Detects dilutional contamination.</p>
    </div>
    <div class="ff-instr-card">
      <div class="ff-instr-icon">📊</div>
      <h4>Output</h4>
      <p>Each row in the results shows a <strong>contamination probability</strong> per fluid and timing (realtime vs. retrospective), plus estimated mix ratio for flagged draws.</p>
    </div>
  </div>
</div>
"""

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


def _preview_csv(file_path):
    """Return (gr.DataFrame update, gr.Markdown update) for a CSV file upload preview."""
    if not file_path:
        return gr.update(visible=False), gr.update(value="", visible=False)
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return gr.update(visible=False), gr.update(value=f"⚠️ Could not read file: {e}", visible=True)
    n_rows, n_cols = df.shape
    return (
        gr.update(value=df.head(10), visible=True),
        gr.update(value=f"**{n_rows:,} rows × {n_cols} columns**", visible=True),
    )

# ---------------------------------------------------------------------------
# Tab 1 – Predict
# ---------------------------------------------------------------------------

def _run_bmp_prediction(
    mode: str,
    file_path: str | None,
    selected_fluids: list[str],
    *manual_values,  # 8 prior + 8 current + 8 post
) -> tuple[pd.DataFrame | None, str | None, str]:
    """Core BMP prediction handler. Returns (result_df, csv_path, status_msg)."""
    from .inference import make_bmp_predictions

    if mode == "Upload File":
        if not file_path:
            return None, None, "⚠️ Please upload a CSV file."
        df = preprocess_bmp_data(_read_uploaded_csv(file_path))
    else:
        n = len(BMP_MANUAL_FIELDS)
        prior_vals, current_vals, post_vals = manual_values[:n], manual_values[n:2*n], manual_values[2*n:3*n]
        row = {}
        for (analyte, *_), curr, prior, post in zip(BMP_MANUAL_FIELDS, current_vals, prior_vals, post_vals):
            row[analyte] = curr
            if prior is not None:
                row[f"{analyte}_prior"] = prior
            if post is not None:
                row[f"{analyte}_post"] = post
        df = pd.DataFrame([row])

    fluids = selected_fluids if selected_fluids else ALL_FLUIDS
    try:
        result = make_bmp_predictions(df, selected_fluids=fluids)
    except RuntimeError as e:
        return None, None, f"⚠️ {e}"
    except Exception:
        return None, None, f"Error: {traceback.format_exc()}"

    csv_path = _save_temp_csv(result)
    return result, csv_path, f"✓ {len(result)} row(s) predicted."


def _run_cbc_prediction(
    mode: str,
    file_path: str | None,
    *manual_values,  # 3 prior + 3 current + 3 post
) -> tuple[pd.DataFrame | None, str | None, str]:
    from .inference import make_cbc_predictions

    if mode == "Upload File":
        if not file_path:
            return None, None, "⚠️ Please upload a CSV file."
        df = preprocess_cbc_data(_read_uploaded_csv(file_path))
    else:
        n = len(CBC_MANUAL_FIELDS)
        prior_vals, current_vals, post_vals = manual_values[:n], manual_values[n:2*n], manual_values[2*n:3*n]
        row = {}
        for (analyte, *_), curr, prior, post in zip(CBC_MANUAL_FIELDS, current_vals, prior_vals, post_vals):
            row[analyte] = curr
            if prior is not None:
                row[f"{analyte}_prior"] = prior
            if post is not None:
                row[f"{analyte}_post"] = post
        df = pd.DataFrame([row])

    try:
        result = make_cbc_predictions(df)
    except RuntimeError as e:
        return None, None, f"⚠️ {e}"
    except Exception:
        return None, None, f"Error: {traceback.format_exc()}"

    csv_path = _save_temp_csv(result)
    return result, csv_path, f"✓ {len(result)} row(s) predicted."


def _make_manual_grid(fields: list) -> tuple[list, list, list]:
    """Render a Prior/Current/Post grid for the given fields. Returns (prior, current, post) input lists."""
    abbrevs = [f[1] for f in fields]
    units   = [f[2] for f in fields]
    n = len(fields)

    header_cells = "".join(
        f'<div class="ff-gh-cell"><span class="ff-gh-abbrev">{a}</span>'
        f'<span class="ff-gh-unit">{u}</span></div>'
        for a, u in zip(abbrevs, units)
    )
    gr.HTML(
        f'<div class="ff-grid-header">'
        f'  <div class="ff-gh-spacer"></div>'
        f'  {header_cells}'
        f'</div>'
    )

    prior, current, post = [], [], []
    for row_label, store, cls in [
        ("Prior",   prior,   "ff-grid-row ff-row-prior"),
        ("Current", current, "ff-grid-row ff-row-current"),
        ("Post",    post,    "ff-grid-row ff-row-post"),
    ]:
        with gr.Row(elem_classes=cls):
            gr.HTML(f'<div class="ff-row-label">{row_label}</div>')
            for _ in range(n):
                store.append(gr.Number(value=None, show_label=False, min_width=72, container=False))

    return prior, current, post


def build_predict_tab() -> None:
    with gr.Tab("🔬  Predict"):
        with gr.Row():
            # ── Left sidebar ───────────────────────────────────────────
            with gr.Column(scale=1, min_width=260):
                panel = gr.Radio(
                    ["BMP", "CBC"], value="BMP", label="Panel",
                    interactive=True, elem_classes="ff-top-radio",
                )

                input_mode = gr.Radio(
                    ["Upload File", "Manual Entry"],
                    value="Upload File", label="Prediction Input",
                    interactive=True, elem_classes="ff-top-radio",
                )

                with gr.Group(visible=True) as upload_section:
                    file_upload = gr.File(
                        label="Upload Wide- or Long-Form CSV",
                        file_types=[".csv", ".tsv"],
                        elem_classes="ff-compact-file",
                    )
                    format_badge = gr.HTML("")

                with gr.Group(visible=True) as upload_fluid_section:
                    fluid_checkboxes_upload = gr.CheckboxGroup(
                        choices=ALL_FLUIDS, value=ALL_FLUIDS,
                        label="Fluids to Include",
                        elem_classes="ff-fluid-filter",
                    )

                with gr.Accordion("Custom Models (Optional)", open=False):
                    custom_model_upload = gr.File(
                        label="Upload .joblib model file(s)",
                        file_types=[".joblib"],
                        file_count="multiple",
                    )
                    custom_model_status = gr.Markdown("", elem_classes="ff-status")

                gr.HTML('<hr class="ff-divider">')
                gr.HTML('<p class="ff-section-title">Actions</p>', elem_classes="ff-th")
                run_btn = gr.Button("▶  Run Predictions", variant="primary", size="lg")
                download_btn = gr.DownloadButton(
                    "⬇  Download Predictions",
                    visible=False, variant="secondary",
                    elem_id="predict-download-btn",
                )
                status_msg = gr.Markdown("", elem_classes="ff-status")

            # ── Right main area ────────────────────────────────────────
            with gr.Column(scale=3):
                data_preview = gr.DataFrame(
                    label="Data Preview (first 10 rows)",
                    interactive=False, wrap=False, visible=False,
                    elem_classes="ff-preview-table",
                )
                preview_info = gr.Markdown("", visible=False, elem_classes="ff-preview-info")

                instructions = gr.HTML(_INSTRUCTIONS_HTML, visible=True)

                with gr.Group(visible=False) as bmp_manual_section:
                    bmp_prior, bmp_current, bmp_post = _make_manual_grid(BMP_MANUAL_FIELDS)
                    gr.HTML('<hr class="ff-divider">')
                    fluid_checkboxes = gr.CheckboxGroup(
                        choices=ALL_FLUIDS, value=ALL_FLUIDS, label="Fluid Filter",
                        elem_classes="ff-fluid-filter",
                    )

                with gr.Group(visible=False) as cbc_manual_section:
                    cbc_prior, cbc_current, cbc_post = _make_manual_grid(CBC_MANUAL_FIELDS)

                results_table = gr.DataFrame(
                    label="Results", interactive=False, wrap=True, visible=False,
                )

        # ── Format-detection badge helper ──────────────────────────────
        def _format_badge(file_path):
            if not file_path:
                return ""
            try:
                df = pd.read_csv(file_path, nrows=2, sep=None, engine="python")
                cols_upper = {c.upper() for c in df.columns}
                is_long = "TASK_ASSAY" in cols_upper
                fmt = "long" if is_long else "wide"
                bg = "#e0f2fe" if fmt == "wide" else "#fef9c3"
                border = "#7dd3fc" if fmt == "wide" else "#fde047"
                fg = "#0369a1" if fmt == "wide" else "#854d0e"
                return (
                    f'<span style="display:inline-block;padding:3px 10px;border-radius:9999px;'
                    f'font-size:0.8rem;font-weight:500;background:{bg};color:{fg};'
                    f'border:1px solid {border}">Detected format: {fmt}</span>'
                )
            except Exception:
                return ""

        def _on_upload(f):
            badge = _format_badge(f)
            preview, info = _preview_csv(f)
            return badge, preview, info

        file_upload.change(
            _on_upload,
            inputs=file_upload,
            outputs=[format_badge, data_preview, preview_info],
        )

        # ── Custom model auto-load ─────────────────────────────────────
        def _auto_load_models(files):
            if not files:
                return ""
            loaded = []
            for f in files:
                try:
                    m = load_from_file(f)
                    key = _model_key(m)
                    cache_model(key, m)
                    loaded.append(key)
                except Exception as e:
                    return f"⚠️ Failed to load {Path(f).name}: {e}"
            return f"✓ Loaded: {', '.join(loaded)}"

        custom_model_upload.change(
            _auto_load_models, inputs=custom_model_upload, outputs=custom_model_status,
        )

        # ── Toggle callbacks ───────────────────────────────────────────
        def _toggle_mode(mode, p):
            is_upload = mode == "Upload File"
            is_bmp = p == "BMP"
            return (
                gr.update(visible=is_upload),                      # upload_section
                gr.update(visible=is_upload and is_bmp),           # upload_fluid_section
                gr.update(visible=not is_upload and is_bmp),       # bmp_manual_section
                gr.update(visible=not is_upload and not is_bmp),   # cbc_manual_section
            )

        for trigger, inputs in [
            (input_mode, [input_mode, panel]),
            (panel,      [input_mode, panel]),
        ]:
            trigger.change(
                _toggle_mode, inputs=inputs,
                outputs=[upload_section, upload_fluid_section, bmp_manual_section, cbc_manual_section],
            )

        # ── Prediction callback ────────────────────────────────────────
        all_manual = bmp_prior + bmp_current + bmp_post + cbc_prior + cbc_current + cbc_post
        n_bmp3 = len(BMP_MANUAL_FIELDS) * 3
        n_cbc3 = len(CBC_MANUAL_FIELDS) * 3

        def _predict(p, mode, file_path, fluids_manual, fluids_upload, *vals):
            fluids = fluids_manual if mode == "Manual Entry" else fluids_upload
            bmp_vals = vals[:n_bmp3]
            cbc_vals = vals[n_bmp3:n_bmp3 + n_cbc3]
            if p == "BMP":
                df, csv_path, msg = _run_bmp_prediction(mode, file_path, fluids, *bmp_vals)
            else:
                df, csv_path, msg = _run_cbc_prediction(mode, file_path, *cbc_vals)

            if df is None:
                return msg, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
            return (
                msg,
                gr.update(value=df, visible=True),
                gr.update(value=csv_path, visible=True),
                gr.update(visible=False),
            )

        _predict_inputs = [
            panel, input_mode, file_upload,
            fluid_checkboxes, fluid_checkboxes_upload,
        ] + all_manual
        _predict_outputs = [status_msg, results_table, download_btn, instructions]

        run_btn.click(
            lambda: (
                "⏳ Running predictions — this may take a moment…",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            ),
            outputs=_predict_outputs,
            queue=False,
        ).then(
            _predict,
            inputs=_predict_inputs,
            outputs=_predict_outputs,
        ).then(
            None,
            js="""() => {
                const wrap = document.querySelector('#predict-download-btn');
                if (wrap) {
                    const a = wrap.querySelector('a');
                    if (a && a.href) setTimeout(() => a.click(), 300);
                }
            }""",
        )

# ---------------------------------------------------------------------------
# Tab 2 – Train
# ---------------------------------------------------------------------------

def _run_training(
    panel: str,
    template_file: str | None,
    fluids_file: str | None,
    progress: gr.Progress = gr.Progress(),
):
    """Train models, yielding (status_message, zip_update, metrics_update) as training progresses."""
    import zipfile
    from .train import train_bmp_fluid, train_cbc_models, save_models, save_cv_metrics

    _hidden = (gr.update(visible=False), gr.update(visible=False))

    if not template_file:
        yield "⚠️ Please upload a training template CSV.", *_hidden
        return

    try:
        template_df = pd.read_csv(template_file)
    except Exception as e:
        yield f"⚠️ Could not read template: {e}", *_hidden
        return

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            if panel == "BMP":
                from .simulate import get_fluid_concentrations
                from .train import _available_cores
                if fluids_file:
                    fluids_df = pd.read_csv(fluids_file, sep=None, engine="python")
                else:
                    fluids_df = get_fluid_concentrations()

                n_total = len(fluids_df)
                n_inner = max(1, _available_cores() // 3)
                all_rows = list(fluids_df.iterrows())
                models = []

                for i, (_, fluid_row) in enumerate(all_rows):
                    fluid_name = fluid_row["fluid"]
                    remaining = n_total - i - 1
                    suffix = f"{remaining} more to go" if remaining > 0 else "last one!"
                    yield (
                        f"⏳ Training **{fluid_name}** — retrospective, realtime & mix ratio"
                        f" ({i + 1} / {n_total}, {suffix}). This may take a while…",
                        *_hidden,
                    )
                    progress(i / n_total, desc=f"Training {fluid_name} ({i+1}/{n_total})…")
                    n_rows = len(template_df)
                    models.extend(train_bmp_fluid(template_df, fluid_row, n_rows,
                                                  n_inner_jobs=n_inner))

            else:
                yield "⏳ Training **CBC** — retrospective, realtime & mix ratio. This may take a while…", *_hidden
                progress(0.1, desc="Training CBC models…")
                models = train_cbc_models(template_df)

            yield "⏳ Saving models…", *_hidden
            progress(0.9, desc="Saving models…")
            paths = save_models(models, tmpdir)

            zip_path = tempfile.mktemp(suffix=".zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in paths:
                    zf.write(p, arcname=Path(p).name)

            metrics_csv_path = tempfile.mktemp(suffix=".csv")
            save_cv_metrics(models, metrics_csv_path)

            for m in models:
                cache_model(_model_key(m), m)

            progress(1.0, desc="Done.")
            yield (
                f"✓ Trained {len(models)} models successfully. Download below.",
                gr.update(value=zip_path, visible=True),
                gr.update(value=metrics_csv_path, visible=True),
            )

    except Exception:
        yield f"⚠️ Training failed:\n{traceback.format_exc()}", *_hidden


def build_train_tab() -> None:
    with gr.Tab("⚙️  Train"):
        with gr.Row():
            # ── Left sidebar ───────────────────────────────────────────
            with gr.Column(scale=1, min_width=260):
                panel = gr.Radio(["BMP", "CBC"], value="BMP", label="Panel",
                                 interactive=True, elem_classes="ff-top-radio")

                gr.HTML('<hr class="ff-divider">')
                gr.HTML('<p class="ff-section-title">Training Data</p>', elem_classes="ff-th")
                template_file = gr.File(label="Training template CSV", file_types=[".csv"])

                with gr.Group() as bmp_train_extras:
                    fluids_file = gr.File(
                        label="Fluid concentrations TSV (optional)",
                        file_types=[".tsv", ".csv"],
                    )

                panel.change(
                    lambda p: gr.update(visible=p == "BMP"),
                    inputs=panel,
                    outputs=bmp_train_extras,
                )

                gr.HTML('<hr class="ff-divider">')
                gr.HTML('<p class="ff-section-title">Actions</p>', elem_classes="ff-th")
                train_btn = gr.Button("🚀  Train Models", variant="primary", size="lg")
                model_download = gr.DownloadButton("⬇  Download Models (zip)", visible=False, variant="secondary")
                metrics_download = gr.DownloadButton("⬇  Download CV Metrics CSV", visible=False, variant="secondary")
                train_status = gr.Markdown("", elem_classes="ff-status")

                gr.HTML('<hr class="ff-divider">')
                gr.HTML('<p class="ff-section-title">Load Existing Models</p>', elem_classes="ff-th")
                model_upload = gr.File(
                    label="Upload .joblib model file(s)",
                    file_types=[".joblib"],
                    file_count="multiple",
                )
                load_btn = gr.Button("📂  Load Models", variant="secondary")
                load_status = gr.Markdown("", elem_classes="ff-status")

            # ── Right main area ────────────────────────────────────────
            with gr.Column(scale=3):
                gr.Markdown(
                    "Upload a wide-format CSV (one row per draw). "
                    "Prior/post columns (`sodium_prior`, `sodium_post`, …) are required for retrospective models."
                )
                template_preview = gr.DataFrame(
                    label="Template Preview (first 10 rows)",
                    interactive=False, wrap=False, visible=False,
                )
                template_info = gr.Markdown("", visible=False)
                fluids_preview = gr.DataFrame(
                    label="Fluids Preview (first 10 rows)",
                    interactive=False, wrap=False, visible=False,
                )
                fluids_info = gr.Markdown("", visible=False)

        template_file.change(_preview_csv, inputs=template_file, outputs=[template_preview, template_info])
        fluids_file.change(_preview_csv, inputs=fluids_file, outputs=[fluids_preview, fluids_info])

        def _train(p, tmpl, fluids, progress=gr.Progress()):
            yield from _run_training(p, tmpl, fluids, progress)

        _train_outputs = [train_status, model_download, metrics_download]

        train_btn.click(
            lambda: (
                "⏳ Starting training…",
                gr.update(visible=False),
                gr.update(visible=False),
            ),
            outputs=_train_outputs,
            queue=False,
        ).then(
            _train,
            inputs=[panel, template_file, fluids_file],
            outputs=_train_outputs,
        )

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
    with gr.Tab("🏷️  Review"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<p class="ff-section-title">Load Predictions</p>', elem_classes="ff-th")
                review_file = gr.File(label="Upload predictions CSV", file_types=[".csv"])
                review_preview = gr.DataFrame(
                    label="Preview (first 10 rows)",
                    interactive=False, wrap=False, visible=False,
                )
                review_preview_info = gr.Markdown("", visible=False)
                load_review_btn = gr.Button("📂  Load File", variant="secondary")
                review_status = gr.Markdown("", elem_classes="ff-status")
                reviewer_name = gr.Textbox(
                    placeholder="Your name (optional)",
                    label="Reviewer name",
                    max_lines=1,
                )
                download_labels_btn = gr.DownloadButton(
                    "⬇  Download Labels", visible=False, variant="secondary"
                )

            with gr.Column(scale=2):
                gr.HTML('<p class="ff-section-title">Label Each Row</p>', elem_classes="ff-th")
                state = gr.State({"df": None, "labels": [], "timestamps": [], "reviewers": [], "idx": 0})
                row_counter = gr.Markdown("", elem_classes="ff-counter")
                current_row = gr.HTML("<p style='color:#94a3b8;font-size:0.875rem;padding:8px 0'>Load a file to begin reviewing.</p>")

                with gr.Row():
                    prev_btn = gr.Button("← Previous", elem_classes="btn-nav", variant="secondary")
                    next_btn = gr.Button("Next →", elem_classes="btn-nav", variant="secondary")

                with gr.Row():
                    real_btn   = gr.Button("Real",         elem_classes="btn-real",   variant="secondary")
                    equiv_btn  = gr.Button("Equivocal",    elem_classes="btn-equiv",  variant="secondary")
                    contam_btn = gr.Button("Contaminated", elem_classes="btn-contam", variant="secondary")

        # -- Load file --
        def _load_file(file_path, st):
            if not file_path:
                return st, "No file uploaded.", gr.update(), ""
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                return st, f"Error: {e}", gr.update(), ""
            n = len(df)
            new_state = {
                "df": df.to_dict(orient="records"),
                "labels": [None] * n,
                "timestamps": [None] * n,
                "reviewers": [None] * n,
                "idx": 0,
            }
            row_html = _build_review_html(df.iloc[0].to_dict())
            counter = f"Reviewing 1 of {n}"
            return new_state, "Review file loaded.", gr.update(value=row_html), counter

        _load_outputs = [state, review_status, current_row, row_counter]
        _load_inputs  = [review_file, state]

        load_review_btn.click(_load_file, inputs=_load_inputs, outputs=_load_outputs)
        review_file.change(_load_file, inputs=_load_inputs, outputs=_load_outputs)
        review_file.change(_preview_csv, inputs=review_file, outputs=[review_preview, review_preview_info])

        # -- Navigation --
        def _go(st, delta):
            if st["df"] is None:
                return st, gr.update(), ""
            records = st["df"]
            idx = max(0, min(len(records) - 1, st["idx"] + delta))
            st = {**st, "idx": idx}
            row_html = _build_review_html(records[idx])
            n_labeled = sum(l is not None for l in st["labels"])
            counter = f"Reviewing {idx + 1} of {len(records)}  ·  Labeled: {n_labeled}"
            return st, gr.update(value=row_html), counter

        prev_btn.click(_go, inputs=[state, gr.Number(value=-1, visible=False)], outputs=[state, current_row, row_counter])
        next_btn.click(_go, inputs=[state, gr.Number(value=1, visible=False)], outputs=[state, current_row, row_counter])

        # -- Labelling --
        def _label(st, label_value, reviewer):
            if st["df"] is None:
                return st, "", gr.update(visible=False), gr.update()
            idx = st["idx"]
            records = st["df"]
            labels = list(st["labels"])
            timestamps = list(st["timestamps"])
            reviewers = list(st["reviewers"])

            labels[idx] = label_value
            timestamps[idx] = datetime.datetime.now().isoformat()
            reviewers[idx] = reviewer or ""

            # Advance to next unlabeled row, or just next row
            next_idx = idx + 1
            if next_idx >= len(records):
                next_idx = idx  # stay on last row

            st = {**st, "labels": labels, "timestamps": timestamps, "reviewers": reviewers, "idx": next_idx}
            n_labeled = sum(l is not None for l in labels)
            counter = f"Reviewing {next_idx + 1} of {len(records)}  ·  Labeled: {n_labeled}"

            # Build download CSV
            df = pd.DataFrame(records)
            df["human_label"] = labels
            df["label_timestamp"] = timestamps
            df["reviewer"] = reviewers
            csv_path = _save_temp_csv(df)

            row_html = _build_review_html(records[next_idx])
            return st, counter, gr.update(value=csv_path, visible=True), gr.update(value=row_html)

        for btn, lbl in [(real_btn, "Real"), (equiv_btn, "Equivocal"), (contam_btn, "Contaminated")]:
            btn.click(
                lambda st, rev, lv=lbl: _label(st, lv, rev),
                inputs=[state, reviewer_name],
                outputs=[state, row_counter, download_labels_btn, current_row],
            )

# ---------------------------------------------------------------------------
# Tab 4 – Self Test
# ---------------------------------------------------------------------------

def build_self_test_tab() -> None:
    from .self_test import build_answer_html, format_score, generate_case, init_db, log_case
    import uuid

    init_db()

    _PLACEHOLDER = (
        "<p style='color:#94a3b8;font-size:0.875rem;padding:8px 0'>"
        "Click <strong>New Case</strong> to begin.</p>"
    )

    with gr.Tab("🎯  Self Test"):
        with gr.Row():
            # ── Left column: controls + score ─────────────────────────
            with gr.Column(scale=1, min_width=200):
                panel_radio = gr.Radio(
                    ["CBC", "BMP"], value="CBC", label="Panel",
                    interactive=True, elem_classes="ff-top-radio",
                )
                new_case_btn = gr.Button("🎲  New Case", variant="primary", size="lg")
                gr.HTML('<hr class="ff-divider">')
                gr.HTML('<p class="ff-section-title">Score</p>', elem_classes="ff-th")
                score_md = gr.Markdown("No cases attempted yet.")

            # ── Right column: case display + buttons ──────────────────
            with gr.Column(scale=2):
                gr.HTML('<p class="ff-section-title">Current Case</p>', elem_classes="ff-th")

                state = gr.State({
                    "case": None,      # dict from generate_case()
                    "revealed": False,
                    "correct": 0,
                    "total": 0,
                    "session_id": None,  # UUID assigned on first New Case
                })

                case_html   = gr.HTML(_PLACEHOLDER, elem_classes="ff-case-display")
                answer_area = gr.HTML("", elem_classes="ff-case-display")

                with gr.Row(elem_classes="ff-btn-row"):
                    real_btn   = gr.Button("Real",         elem_classes="btn-real",   variant="secondary", scale=1, min_width=80)
                    contam_btn = gr.Button("Contaminated", elem_classes="btn-contam", variant="secondary", scale=1, min_width=80)

                name_input = gr.Textbox(
                    placeholder="Your name (optional)",
                    label="Name",
                    max_lines=1,
                )

        # ── Callbacks ────────────────────────────────────────────────

        def _new_case(panel, st):
            session_id = st["session_id"] or str(uuid.uuid4())
            case = generate_case(panel)
            if case["error"]:
                return (
                    st,
                    f"<p style='color:#b91c1c'>{case['error']}</p>",
                    "",
                    format_score(st["correct"], st["total"]),
                )
            new_st = {**st, "case": case, "revealed": False, "session_id": session_id}
            row_html = _build_review_html(case["row_dict"])
            return (
                new_st,
                row_html,
                "",
                format_score(st["correct"], st["total"]),
            )

        _case_outputs = [state, case_html, answer_area, score_md]

        new_case_btn.click(_new_case, inputs=[panel_radio, state], outputs=_case_outputs)

        def _guess(guess: str, st: dict, name: str):
            if st["case"] is None:
                return st, "", gr.update()

            case     = st["case"]
            revealed = st["revealed"]

            correct = st["correct"]
            total   = st["total"]

            if not revealed:
                total += 1
                if (guess == "Contaminated") == case["contaminated"]:
                    correct += 1
                log_case(case, guess, name or "", st.get("session_id", ""))

            new_st = {**st, "revealed": True, "correct": correct, "total": total}

            answer_html = build_answer_html(case, guess)
            score_text  = format_score(correct, total)
            return new_st, answer_html, score_text

        _guess_outputs = [state, answer_area, score_md]

        for btn, lbl in [
            (real_btn,   "Real"),
            (contam_btn, "Contaminated"),
        ]:
            btn.click(
                lambda st, name, lv=lbl: _guess(lv, st, name),
                inputs=[state, name_input],
                outputs=_guess_outputs,
            )


# ---------------------------------------------------------------------------
# Theme & CSS
# ---------------------------------------------------------------------------

_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#f0f4f8",
    body_background_fill_dark="#0f172a",
    block_background_fill="#ffffff",
    block_background_fill_dark="#1e293b",
    block_border_width="1px",
    block_border_color="#e2e8f0",
    block_border_color_dark="#334155",
    block_label_background_fill="#f8fafc",
    block_label_background_fill_dark="#1e293b",
    block_label_text_color="#475569",
    block_label_text_color_dark="#94a3b8",
    block_shadow="0 1px 3px 0 rgb(0 0 0 / 0.08), 0 1px 2px -1px rgb(0 0 0 / 0.08)",
    block_title_text_weight="600",
    input_background_fill="#f8fafc",
    input_background_fill_dark="#0f172a",
    input_border_color="#e2e8f0",
    input_border_color_dark="#334155",
    button_primary_background_fill="linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%)",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#f1f5f9",
    button_secondary_background_fill_hover="#e2e8f0",
    button_secondary_text_color="#1e293b",
    button_secondary_border_color="#e2e8f0",
)

_CSS = """
/* ── Page chrome ───────────────────────────────────────────── */
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }

/* ── App header ────────────────────────────────────────────── */
.ff-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 8px;
    color: white;
}
.ff-header h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: white !important;
    margin: 0 0 6px 0 !important;
    letter-spacing: -0.02em;
}
.ff-header p { color: #bfdbfe !important; font-size: 1rem; margin: 0; }

/* ── Section cards ─────────────────────────────────────────── */
.ff-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
}
.ff-section-title {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #64748b !important;
    margin: 0 !important;
}
/* ── Section title HTML blocks: remove gap before next component ── */
.ff-th {
    margin-bottom: calc(-1 * var(--layout-gap, 8px)) !important;
    padding-bottom: 0 !important;
    line-height: 1 !important;
}
.ff-divider { border: none; border-top: 1px solid #e2e8f0; margin: 20px 0; }

/* ── Status pills ──────────────────────────────────────────── */
.ff-status p {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.875rem;
    font-weight: 500;
    background: #f0fdf4;
    color: #166534;
    border: 1px solid #bbf7d0;
}
.ff-status p:empty { display: none; }

/* ── Labelling buttons ─────────────────────────────────────── */
.btn-real button   { background: #dcfce7 !important; color: #15803d !important; border: 1px solid #86efac !important; font-weight: 600 !important; }
.btn-equiv button  { background: #fef9c3 !important; color: #854d0e !important; border: 1px solid #fde047 !important; font-weight: 600 !important; }
.btn-contam button { background: #fee2e2 !important; color: #b91c1c !important; border: 1px solid #fca5a5 !important; font-weight: 600 !important; }

/* ── Nav buttons ───────────────────────────────────────────── */
.btn-nav button { font-size: 0.875rem !important; padding: 8px 18px !important; }

/* ── Tab bar ───────────────────────────────────────────────── */
.tab-nav button { font-weight: 500; font-size: 0.9375rem; padding: 10px 20px; }
.tab-nav button.selected { font-weight: 700; }

/* ── Row counter ───────────────────────────────────────────── */
.ff-counter p {
    font-size: 0.8125rem;
    color: #64748b;
    font-variant-numeric: tabular-nums;
}

/* ── Top radio labels styled as section titles ─────────────── */
.ff-top-radio > .wrap > legend,
.ff-top-radio > label,
.ff-top-radio legend {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #64748b !important;
    margin-bottom: 6px !important;
    padding: 0 !important;
}

/* ── Manual entry grid ─────────────────────────────────────── */
.ff-grid-header {
    display: flex;
    align-items: flex-end;
    gap: var(--layout-gap, 8px);
    padding: 4px 20px 2px 4px;
}
.ff-gh-spacer { flex: 0 0 72px; }
.ff-gh-cell {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 72px;
}
.ff-gh-abbrev {
    font-size: 0.8125rem;
    font-weight: 600;
    color: var(--body-text-color, #334155);
}
.ff-gh-unit {
    font-size: 0.6875rem;
    color: var(--body-text-color, #94a3b8);
    opacity: 0.55;
    white-space: nowrap;
}
.ff-grid-row {
    align-items: center !important;
    gap: var(--layout-gap, 8px) !important;
}
.ff-row-label {
    flex: 0 0 72px !important;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    text-align: right;
    padding-right: 8px;
    align-self: center;
}
.ff-row-prior   .ff-row-label { color: #818cf8; }
.ff-row-current .ff-row-label { color: #38bdf8; }
.ff-row-post    .ff-row-label { color: #34d399; }
.ff-row-prior   input { border-color: #e0e7ff !important; background: #f5f3ff !important; color: #1e293b !important; }
.ff-row-current input { border-color: #bae6fd !important; background: #f0f9ff !important; color: #1e293b !important; }
.ff-row-post    input { border-color: #a7f3d0 !important; background: #f0fdf4 !important; color: #1e293b !important; }
.ff-grid-row { padding-right: 16px !important; }


/* ── Self-test case display: strip block chrome ────────────── */
.ff-case-display,
.ff-case-display > .block,
.ff-case-display .block {
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    background: transparent !important;
}

/* ── Self-test button row: don't wrap ───────────────────────── */
.ff-btn-row { flex-wrap: nowrap !important; }
.ff-btn-row button { min-width: 0 !important; }

/* ── Compact file upload ───────────────────────────────────── */
.ff-compact-file .upload-container,
.ff-compact-file [data-testid="file-upload"],
.ff-compact-file .file-preview {
    min-height: 60px !important;
    max-height: 60px !important;
    padding: 8px 12px !important;
}
.ff-compact-file .upload-container .icon-wrap { display: none !important; }
.ff-compact-file .upload-container p,
.ff-compact-file .upload-container span {
    font-size: 0.8rem !important;
    line-height: 1.3 !important;
}

/* ── Instructions ──────────────────────────────────────────── */
.ff-instructions { padding: 8px 0; }
.ff-instr-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
}
.ff-instr-card {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px;
    padding: 20px 22px;
    color-scheme: light;
}
.ff-instr-icon { font-size: 1.5rem; margin-bottom: 8px; }
.ff-instr-card h4 {
    font-size: 0.9375rem !important;
    font-weight: 600 !important;
    color: #1e293b !important;
    margin: 0 0 8px !important;
}
.ff-instr-card p,
.ff-instr-card p strong,
.ff-instr-card p br {
    font-size: 0.875rem !important;
    color: #475569 !important;
    line-height: 1.55;
    margin: 0;
}
.ff-instr-card p strong { font-weight: 600 !important; }
.ff-instr-card code {
    font-size: 0.8125rem !important;
    background: #dde3ec !important;
    color: #334155 !important;
    border-radius: 4px;
    padding: 1px 5px;
}
"""

# ---------------------------------------------------------------------------
# Assemble full UI
# ---------------------------------------------------------------------------

def build_ui(on_load=None) -> gr.Blocks:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The parameters have been moved", category=UserWarning)
        _blocks = gr.Blocks(title="FluidFlagger", theme=_THEME, css=_CSS)
    with _blocks as demo:
        gr.HTML("""
        <div class="ff-header">
            <h1>🧪 FluidFlagger</h1>
            <p>IV-fluid contamination detection for BMP and CBC laboratory results</p>
        </div>
        """)
        build_predict_tab()
        build_train_tab()
        build_review_tab()
        build_self_test_tab()
        if on_load is not None:
            demo.load(on_load)
    return demo
