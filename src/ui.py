"""FluidFlagger's Gradio Blocks UI.

The app provides Predict, Train, Review, Validate, and Self Test tabs.
Validate consumes reviewed prediction exports and includes the former Explain
(SHAP) tools in its lower section.
"""

from __future__ import annotations

import datetime
import html
import io
import json
import os
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
from .validation import (
    ValidationDataError,
    build_validation_payload,
    default_label_column,
    default_score_column,
    find_label_columns,
    find_score_columns,
)


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


def _build_review_html(row_dict: dict, mode: str = "Retrospective") -> str:
    """Build a styled prior/current/post table for a single review row (all inline styles)."""
    if "sodium" in row_dict:
        fields = _BMP_FIELDS_REVIEW
    elif "Hgb" in row_dict:
        fields = _CBC_FIELDS_REVIEW
    else:
        return "<p class='ff-muted-text'>No analyte columns detected.</p>"

    show_prior = mode in ("Retrospective", "Real-time")
    show_post  = mode == "Retrospective"

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
    prior_vals   = [_val(f"{col}_prior") for col in analytes] if show_prior else [None] * len(analytes)
    post_vals    = [_val(f"{col}_post")  for col in analytes] if show_post  else [None] * len(analytes)

    has_prior = any(v is not None for v in prior_vals)
    has_post  = any(v is not None for v in post_vals)

    # Header row
    th_cells = "".join(f'<th class="ff-rv-th-col">{abbr}</th>' for abbr in abbrevs)
    html = f'<tr><th class="ff-rv-th-lbl"></th>{th_cells}</tr>'

    # Prior row
    if has_prior:
        cells = "".join(f'<td class="ff-rv-td-prior">{_fmt(v)}</td>' for v in prior_vals)
        html += f'<tr><td class="ff-rv-th-lbl ff-rv-lbl-prior">prior</td>{cells}</tr>'

    # Current row with arrows
    def _arrow_cell(curr, prior, post):
        if curr is None:
            return '<td class="ff-rv-td-curr"><div class="ff-rv-curr-inner"><span class="ff-rv-num">—</span></div></td>'
        # top arrow: prior → current
        if show_prior and prior is not None and curr > prior:
            top = '<div class="ff-rv-arrow-up">&#9650;</div>'
        elif show_prior and prior is not None and curr < prior:
            top = '<div class="ff-rv-arrow-dn">&#9660;</div>'
        else:
            top = '<div class="ff-rv-arrow-none"></div>'
        # bottom arrow: current → post
        if show_post and post is not None and post > curr:
            bot = '<div class="ff-rv-arrow-up">&#9650;</div>'
        elif show_post and post is not None and post < curr:
            bot = '<div class="ff-rv-arrow-dn">&#9660;</div>'
        else:
            bot = '<div class="ff-rv-arrow-none"></div>'
        return (
            '<td class="ff-rv-td-curr">'
            '<div class="ff-rv-curr-inner">'
            f'{top}'
            f'<div class="ff-rv-num">{_fmt(curr)}</div>'
            f'{bot}'
            '</div></td>'
        )

    curr_cells = "".join(
        _arrow_cell(c, p, o)
        for c, p, o in zip(current_vals, prior_vals, post_vals)
    )
    html += f'<tr><td class="ff-rv-th-lbl ff-rv-lbl-curr">current</td>{curr_cells}</tr>'

    # Post row
    if has_post:
        cells = "".join(f'<td class="ff-rv-td-post">{_fmt(v)}</td>' for v in post_vals)
        html += f'<tr><td class="ff-rv-th-lbl ff-rv-lbl-post">post</td>{cells}</tr>'

    return f'<div class="ff-rv-wrap"><div class="ff-rv-card"><table class="ff-rv-table">{html}</table></div></div>'




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

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _err(msg: str) -> str:
    """Wrap a message in a red error pill for gr.Markdown status fields."""
    return f'<span class="ff-err">{msg}</span>'


def _counter_html(idx: int, total: int, n_labeled: int) -> str:
    """Build styled progress counter HTML for the Review tab."""
    pct = n_labeled / total * 100 if total > 0 else 0
    return (
        f'<div class="ff-counter-wrap">'
        f'<div class="ff-counter-text">'
        f'Reviewing <strong>{idx + 1}</strong> of {total}'
        f'&ensp;·&ensp;Labeled: <strong>{n_labeled}</strong>'
        f'</div>'
        f'<div class="ff-counter-track">'
        f'<div class="ff-counter-fill" style="width:{pct:.1f}%"></div>'
        f'</div>'
        f'</div>'
    )


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


def _read_uploaded_delimited_file(file_path: str | None) -> Optional[pd.DataFrame]:
    """Read a comma- or tab-delimited upload, inferring its delimiter."""
    if not file_path:
        return None
    try:
        return pd.read_csv(file_path, sep=None, engine="python")
    except Exception as exc:
        raise ValidationDataError(f"Could not read file: {exc}") from exc


def _validation_dashboard_html(payload: dict) -> str:
    """Wrap a JSON-safe performance payload for the interactive HTML renderer."""
    payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    escaped_payload = html.escape(payload_json, quote=True)
    return (
        '<div class="ff-validation-dashboard" '
        f'data-payload="{escaped_payload}"></div>'
    )

# ---------------------------------------------------------------------------
# Tab 1 – Predict
# ---------------------------------------------------------------------------

def _run_bmp_prediction(
    mode: str,
    file_path: str | None,
    selected_fluids: list[str],
    *manual_values,  # 8 prior + 8 current + 8 post
) -> tuple[pd.DataFrame | None, str]:
    """Core BMP prediction handler. Returns (result_df, status_msg)."""
    from .inference import make_bmp_predictions

    if mode == "Upload":
        if not file_path:
            return None, "⚠️ Please upload a CSV file."
        df = preprocess_bmp_data(_read_uploaded_csv(file_path))
        found = [c for c in BMP_ANALYTES if c in df.columns]
        if not found:
            return None, (
                "⚠️ No BMP analyte columns found. This looks like it may be a CBC file. "
                "Expected columns such as: " + ", ".join(BMP_ANALYTES[:4]) + ", …"
            )
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
        return None, f"⚠️ {e}"
    except Exception:
        return None, f"Error: {traceback.format_exc()}"

    return result, f"✓ {len(result)} row(s) predicted."


def _run_cbc_prediction(
    mode: str,
    file_path: str | None,
    *manual_values,  # 3 prior + 3 current + 3 post
) -> tuple[pd.DataFrame | None, str]:
    from .inference import make_cbc_predictions

    if mode == "Upload":
        if not file_path:
            return None, "⚠️ Please upload a CSV file."
        df = preprocess_cbc_data(_read_uploaded_csv(file_path))
        found = [c for c in CBC_ANALYTES if c in df.columns]
        if not found:
            return None, (
                "⚠️ No CBC analyte columns found. This looks like it may be a BMP file. "
                "Expected columns such as: " + ", ".join(CBC_ANALYTES)
            )
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
        return None, f"⚠️ {e}"
    except Exception:
        return None, f"Error: {traceback.format_exc()}"

    return result, f"✓ {len(result)} row(s) predicted."


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
            with gr.Column(scale=1, min_width=220):
                panel = gr.Radio(
                    ["BMP", "CBC"], value="BMP", label="Panel",
                    interactive=True, elem_classes="ff-top-radio",
                )

                input_mode = gr.Radio(
                    ["Upload", "Manual"],
                    value="Upload", label="Input",
                    interactive=True, elem_classes="ff-top-radio",
                )

                with gr.Group(visible=True) as upload_section:
                    file_upload = gr.File(
                        label="Upload Wide- or Long-Form CSV",
                        file_types=[".csv", ".tsv"],
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
                        label="Upload model folder",
                        file_count="directory",
                    )
                    custom_model_status = gr.Markdown("", elem_classes="ff-status")

            # ── Right main area ────────────────────────────────────────
            with gr.Column(scale=3):
                empty_state = gr.HTML(
                    '<div class="ff-empty-state">'
                    '<div style="font-size:1.75rem;margin-bottom:10px">📁</div>'
                    '<p>Upload a CSV or switch to <strong>Manual</strong> to get started</p>'
                    '</div>',
                    visible=True,
                )
                data_preview = gr.DataFrame(
                    label="Data Preview (first 10 rows)",
                    interactive=False, wrap=False, visible=False,
                    elem_classes="ff-preview-table",
                )
                preview_info = gr.Markdown("", visible=False, elem_classes="ff-preview-info")

                run_btn = gr.Button("▶  Run Predictions", variant="primary", size="lg", visible=False)
                download_btn = gr.DownloadButton(
                    "⬇  Download Predictions",
                    visible=False, variant="primary",
                    elem_id="predict-download-btn",
                )
                status_msg = gr.Markdown("", elem_classes="ff-status")

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
                fmt_cls = "ff-fmt-wide" if fmt == "wide" else "ff-fmt-long"
                return f'<span class="ff-fmt-badge {fmt_cls}">Detected format: {fmt}</span>'
            except Exception:
                return ""

        def _on_upload(f):
            badge = _format_badge(f)
            preview, info = _preview_csv(f)
            return badge, preview, info, gr.update(visible=False)

        file_upload.change(
            _on_upload,
            inputs=file_upload,
            outputs=[format_badge, data_preview, preview_info, download_btn],
        )

        # ── Custom model auto-load ─────────────────────────────────────
        def _auto_load_models(files):
            if not files:
                return ""
            loaded = []
            joblib_files = [f for f in files if Path(f).suffix == ".joblib"]
            if not joblib_files:
                return "⚠️ No .joblib files found in the uploaded folder."
            for f in joblib_files:
                try:
                    m = load_from_file(f)
                    key = _model_key(m)
                    cache_model(key, m)
                    loaded.append(key)
                except Exception as e:
                    return f"⚠️ Failed to load {Path(f).name}: {e}"
            return f"✓ Loaded {len(loaded)} models: {', '.join(loaded)}"

        custom_model_upload.change(
            _auto_load_models, inputs=custom_model_upload, outputs=custom_model_status,
        )

        # ── Toggle callbacks ───────────────────────────────────────────
        def _toggle_mode(mode, p, f):
            is_upload = mode == "Upload"
            is_bmp = p == "BMP"
            show_run = not is_upload or bool(f)
            show_empty = is_upload and not bool(f)
            return (
                gr.update(visible=is_upload),                      # upload_section
                gr.update(visible=is_upload and is_bmp),           # upload_fluid_section
                gr.update(visible=not is_upload and is_bmp),       # bmp_manual_section
                gr.update(visible=not is_upload and not is_bmp),   # cbc_manual_section
                gr.update(visible=show_run),                       # run_btn
                gr.update(visible=show_empty),                     # empty_state
            )

        for trigger, inputs in [
            (input_mode,  [input_mode, panel, file_upload]),
            (panel,       [input_mode, panel, file_upload]),
            (file_upload, [input_mode, panel, file_upload]),
        ]:
            trigger.change(
                _toggle_mode, inputs=inputs,
                outputs=[upload_section, upload_fluid_section, bmp_manual_section, cbc_manual_section, run_btn, empty_state],
            )

        # ── Prediction callback ────────────────────────────────────────
        all_manual = bmp_prior + bmp_current + bmp_post + cbc_prior + cbc_current + cbc_post
        n_bmp3 = len(BMP_MANUAL_FIELDS) * 3
        n_cbc3 = len(CBC_MANUAL_FIELDS) * 3

        def _predict(p, mode, file_path, fluids_manual, fluids_upload, *vals):
            fluids = fluids_manual if mode == "Manual" else fluids_upload
            bmp_vals = vals[:n_bmp3]
            cbc_vals = vals[n_bmp3:n_bmp3 + n_cbc3]
            if p == "BMP":
                df, msg = _run_bmp_prediction(mode, file_path, fluids, *bmp_vals)
            else:
                df, msg = _run_cbc_prediction(mode, file_path, *cbc_vals)

            if df is None:
                return _err(msg), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

            import datetime as _dt
            ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = str(Path(tempfile.gettempdir()) / f"{p.lower()}_predictions_{ts}.csv")
            df.to_csv(csv_path, index=False)
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
        _predict_outputs = [status_msg, results_table, download_btn, run_btn]

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
                    if (a && a.href) {
                        const tmp = document.createElement('a');
                        tmp.href = a.href;
                        tmp.download = a.download || '';
                        document.body.appendChild(tmp);
                        setTimeout(() => { tmp.click(); tmp.remove(); }, 300);
                    }
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
    fmt: str = "joblib",
    progress: gr.Progress = gr.Progress(),
):
    """Train models, yielding (status_message, zip_update, metrics_update) as training progresses."""
    import zipfile
    from .train import train_bmp_fluid, train_cbc_models, save_models, save_cv_metrics

    _hidden = (gr.update(visible=False), gr.update(visible=False))

    if not template_file:
        yield _err("⚠️ Please upload a training template CSV."), *_hidden
        return

    try:
        template_df = pd.read_csv(template_file)
    except Exception as e:
        yield _err(f"⚠️ Could not read template: {e}"), *_hidden
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
                n_cores = _available_cores()
                n_inner = max(1, (n_cores - 1) // 3)
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
                    fluid_models = train_bmp_fluid(template_df, fluid_row, n_rows,
                                                   n_inner_jobs=n_inner)
                    for m in fluid_models:
                        cache_model(_model_key(m), m)
                    models.extend(fluid_models)

            else:
                yield "⏳ Training **CBC** — retrospective, realtime & mix ratio. This may take a while…", *_hidden
                progress(0.1, desc="Training CBC models…")
                models = train_cbc_models(template_df)
                for m in models:
                    cache_model(_model_key(m), m)

            yield "⏳ Saving models…", *_hidden
            progress(0.9, desc="Saving models…")
            paths = save_models(models, tmpdir, fmt=fmt)

            import time
            folder_name = f"{panel}_models_{int(time.time())}"
            zip_path = tempfile.mktemp(suffix=".zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in paths:
                    zf.write(p, arcname=f"{folder_name}/{Path(p).name}")

            metrics_csv_path = tempfile.mktemp(suffix=".csv")
            save_cv_metrics(models, metrics_csv_path)

            progress(1.0, desc="Done.")
            yield (
                f"✓ Trained {len(models)} models successfully. Download below.",
                gr.update(value=zip_path, visible=True),
                gr.update(value=metrics_csv_path, visible=True),
            )

    except Exception:
        yield _err(f"⚠️ Training failed:\n{traceback.format_exc()}"), *_hidden


def build_train_tab() -> None:
    with gr.Tab("⚙️  Train"):
        with gr.Row():
            # ── Left sidebar ───────────────────────────────────────────
            with gr.Column(scale=1, min_width=220):
                panel = gr.Radio(["BMP", "CBC"], value="BMP", label="Panel",
                                 interactive=True, elem_classes="ff-top-radio")

                gr.HTML('<hr class="ff-divider">')
                with gr.Group():
                    gr.HTML('<p class="ff-section-title" style="padding:8px 4px 2px">Training Data</p>')
                    template_file = gr.File(
                        label="Training template CSV", file_types=[".csv"],
                    )
                    with gr.Group(visible=True) as bmp_train_extras:
                        fluids_file = gr.File(
                            label="Fluid concentrations TSV (optional)",
                            file_types=[".tsv", ".csv"],
                        )

                gr.HTML('<hr class="ff-divider">')
                gr.HTML('<p class="ff-section-title" style="padding:8px 4px 2px">Export Format</p>')
                fmt_radio = gr.Radio(
                    [".joblib", ".pkl"], value=".joblib",
                    label="Model file format",
                    interactive=True, elem_classes="ff-top-radio",
                )

                panel.change(
                    lambda p: gr.update(visible=p == "BMP"),
                    inputs=panel,
                    outputs=bmp_train_extras,
                )

            # ── Right main area ────────────────────────────────────────
            with gr.Column(scale=3):
                train_empty_state = gr.HTML(
                    '<div class="ff-empty-state">'
                    '<div style="font-size:1.75rem;margin-bottom:10px">📂</div>'
                    '<p>Upload a <strong>training template CSV</strong> in the sidebar to get started</p>'
                    '</div>',
                    visible=True,
                )
                gr.HTML(
                    '<p class="ff-hint">'
                    'Upload a <strong>wide-format CSV</strong> (one row per draw). '
                    'Prior/post columns (<code>sodium_prior</code>, <code>sodium_post</code>, …) '
                    'are required for retrospective models.'
                    '</p>'
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

                train_btn = gr.Button("🚀  Train Models", variant="primary", size="lg", visible=False)
                model_download = gr.DownloadButton(
                    "⬇  Download Models (zip)", visible=False, variant="primary",
                    elem_id="train-download-btn",
                )
                metrics_download = gr.DownloadButton("⬇  Download CV Metrics CSV", visible=False, variant="secondary")
                train_status = gr.Markdown("", elem_classes="ff-status")

        def _on_template_upload(f):
            preview, info = _preview_csv(f)
            has_file = bool(f)
            return preview, info, gr.update(visible=has_file), gr.update(visible=False), gr.update(visible=not has_file)

        template_file.change(
            _on_template_upload,
            inputs=template_file,
            outputs=[template_preview, template_info, train_btn, model_download, train_empty_state],
        )
        fluids_file.change(_preview_csv, inputs=fluids_file, outputs=[fluids_preview, fluids_info])

        def _train(p, tmpl, fluids, fmt, progress=gr.Progress()):
            yield from _run_training(p, tmpl, fluids, fmt.lstrip("."), progress)

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
            inputs=[panel, template_file, fluids_file, fmt_radio],
            outputs=_train_outputs,
        ).then(
            None,
            js="""() => {
                const wrap = document.querySelector('#train-download-btn');
                if (wrap) {
                    const a = wrap.querySelector('a');
                    if (a && a.href) {
                        const tmp = document.createElement('a');
                        tmp.href = a.href;
                        tmp.download = a.download || '';
                        document.body.appendChild(tmp);
                        setTimeout(() => {
                            tmp.click();
                            tmp.remove();
                            setTimeout(() => window.location.reload(), 500);
                        }, 300);
                    }
                }
            }""",
        )


# ---------------------------------------------------------------------------
# Tab 3 – Review
# ---------------------------------------------------------------------------

def build_review_tab() -> None:
    with gr.Tab("🏷️  Review"):
        with gr.Row():
            with gr.Column(scale=1):
                reviewer_name = gr.Textbox(
                    placeholder="Your name (optional)",
                    label="Reviewer",
                    max_lines=1,
                )
                gr.HTML('<hr class="ff-divider">')
                with gr.Group():
                    gr.HTML('<p class="ff-section-title">Load Predictions</p>', elem_classes="ff-th")
                    review_file = gr.File(label="Upload predictions CSV", file_types=[".csv"])
                review_status = gr.Markdown("", elem_classes="ff-status")
                download_labels_btn = gr.DownloadButton(
                    "⬇  Download Labels", visible=False, variant="secondary"
                )

            with gr.Column(scale=2):
                gr.HTML('<p class="ff-section-title">Label Each Row</p>', elem_classes="ff-th")
                state = gr.State({"df": None, "labels": [], "timestamps": [], "reviewers": [], "idx": 0})
                row_counter = gr.HTML("")
                current_row = gr.HTML("<p class='ff-muted-text' style='padding:8px 0'>Load a file to begin reviewing.</p>")

                with gr.Row():
                    prev_btn = gr.Button("← Prev", elem_classes="btn-nav", variant="secondary", size="sm")
                    next_btn = gr.Button("Next →", elem_classes="btn-nav", variant="secondary", size="sm")

                with gr.Row():
                    real_btn   = gr.Button("Real",         elem_classes="btn-real",   variant="secondary")
                    equiv_btn  = gr.Button("Equivocal",    elem_classes="btn-equiv",  variant="secondary")
                    contam_btn = gr.Button("Contaminated", elem_classes="btn-contam", variant="secondary")

        # -- Load file --
        def _load_file(file_path, st):
            if not file_path:
                return st, _err("No file uploaded."), gr.update(), ""
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                return st, _err(f"Error: {e}"), gr.update(), ""
            n = len(df)
            new_state = {
                "df": df.to_dict(orient="records"),
                "labels": [None] * n,
                "timestamps": [None] * n,
                "reviewers": [None] * n,
                "idx": 0,
            }
            row_html = _build_review_html(df.iloc[0].to_dict())
            return new_state, f"✓ {n:,} rows loaded.", gr.update(value=row_html), _counter_html(0, n, 0)

        _load_outputs = [state, review_status, current_row, row_counter]
        _load_inputs  = [review_file, state]

        review_file.change(_load_file, inputs=_load_inputs, outputs=_load_outputs)

        # -- Navigation --
        def _go(st, delta):
            if st["df"] is None:
                return st, gr.update(), ""
            records = st["df"]
            idx = max(0, min(len(records) - 1, st["idx"] + delta))
            st = {**st, "idx": idx}
            row_html = _build_review_html(records[idx])
            n_labeled = sum(l is not None for l in st["labels"])
            return st, gr.update(value=row_html), _counter_html(idx, len(records), n_labeled)

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

            next_idx = idx + 1
            if next_idx >= len(records):
                next_idx = idx

            st = {**st, "labels": labels, "timestamps": timestamps, "reviewers": reviewers, "idx": next_idx}
            n_labeled = sum(l is not None for l in labels)

            df = pd.DataFrame(records)
            df["human_label"] = labels
            df["label_timestamp"] = timestamps
            df["reviewer"] = reviewers
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            name_part = f"_{reviewer.strip()}" if reviewer and reviewer.strip() else ""
            fname = f"reviewed_results_{ts}{name_part}.csv"
            tmp_dir = tempfile.mkdtemp()
            csv_path = os.path.join(tmp_dir, fname)
            df.to_csv(csv_path, index=False)

            row_html = _build_review_html(records[next_idx])
            return st, _counter_html(next_idx, len(records), n_labeled), gr.update(value=csv_path, visible=True), gr.update(value=row_html)

        for btn, lbl in [(real_btn, "Real"), (equiv_btn, "Equivocal"), (contam_btn, "Contaminated")]:
            btn.click(
                lambda st, rev, lv=lbl: _label(st, lv, rev),
                inputs=[state, reviewer_name],
                outputs=[state, row_counter, download_labels_btn, current_row],
            )

# ---------------------------------------------------------------------------
# Tab 4 – Validate
# ---------------------------------------------------------------------------

def build_validate_tab() -> None:
    """Build interactive performance validation plus the embedded Explain tools."""
    from .shap_tab import build_shap_section

    with gr.Tab("✅  Validate"):
        with gr.Row():
            # ── Left sidebar ───────────────────────────────────────────
            with gr.Column(scale=1, min_width=240):
                gr.HTML('<p class="ff-section-title">Validation file</p>', elem_classes="ff-th")
                validation_file = gr.File(
                    label="Upload reviewed predictions CSV",
                    file_types=[".csv", ".tsv"],
                )
                gr.HTML(
                    '<p class="ff-hint">Use the file downloaded from <strong>Review</strong>. '
                    'It should contain a probability column (for example, '
                    '<code>max_retrospective_prob</code>) and a ground-truth '
                    '<code>human_label</code>.</p>'
                )

                with gr.Group(visible=False) as validation_controls:
                    gr.HTML('<hr class="ff-divider">')
                    gr.HTML('<p class="ff-section-title">Analysis settings</p>', elem_classes="ff-th")
                    label_column = gr.Dropdown(
                        choices=[],
                        label="Ground-truth label",
                        interactive=True,
                        allow_custom_value=True,
                    )
                    score_column = gr.Dropdown(
                        choices=[],
                        label="Prediction probability",
                        interactive=True,
                        allow_custom_value=True,
                    )
                    equivocal_policy = gr.Radio(
                        [
                            "Exclude equivocal labels",
                            "Treat equivocal as contaminated",
                            "Treat equivocal as real",
                        ],
                        value="Exclude equivocal labels",
                        label="Equivocal labels",
                        interactive=True,
                        elem_classes="ff-top-radio",
                    )

            # ── Right main area ────────────────────────────────────────
            with gr.Column(scale=3):
                validate_empty_state = gr.HTML(
                    '<div class="ff-empty-state">'
                    '<div style="font-size:1.75rem;margin-bottom:10px">📈</div>'
                    '<p>Upload a <strong>reviewed predictions CSV</strong> to evaluate model performance</p>'
                    '</div>',
                    visible=True,
                )
                validation_status = gr.Markdown("", elem_classes="ff-status")
                validation_dashboard = gr.HTML(
                    "",
                    visible=False,
                    elem_id="validate-performance-dashboard",
                    elem_classes="ff-validation-dashboard-wrap",
                    apply_default_css=False,
                    js_on_load=_VALIDATION_DASHBOARD_JS,
                )
                validation_preview_info = gr.Markdown("", visible=False, elem_classes="ff-preview-info")
                validation_preview = gr.DataFrame(
                    label="Data Preview (first 10 rows)",
                    interactive=False,
                    wrap=False,
                    visible=False,
                    elem_classes="ff-preview-table",
                )

        def _load_validation_file(file_path, policy):
            """Inspect the file, choose sensible columns, and generate the first report."""
            if not file_path:
                return (
                    gr.update(choices=[], value=None),
                    gr.update(choices=[], value=None),
                    gr.update(visible=False),
                    "",
                    gr.update(visible=False),
                    gr.update(value="", visible=False),
                    gr.update(value="", visible=False),
                    gr.update(visible=True),
                )

            try:
                df = _read_uploaded_delimited_file(file_path)
                labels = find_label_columns(df)
                scores = find_score_columns(df)
                selected_label = default_label_column(labels)
                selected_score = default_score_column(scores)

                if not labels:
                    raise ValidationDataError(
                        "Could not identify a ground-truth label column. "
                        "Expected a column such as human_label, ground_truth, or label."
                    )
                if not scores:
                    raise ValidationDataError(
                        "Could not identify a probability column. "
                        "Expected a 0–1 column such as max_retrospective_prob or prob_*."
                    )

                payload = build_validation_payload(
                    df,
                    score_column=selected_score,
                    label_column=selected_label,
                    equivocal_policy=policy,
                )
                summary = payload["summary"]
                status = (
                    f"✓ {summary['included_rows']:,} evaluable rows loaded "
                    f"({summary['positive_count']:,} contaminated; "
                    f"{summary['excluded_rows']:,} excluded)."
                )
                return (
                    gr.update(choices=labels, value=selected_label),
                    gr.update(choices=scores, value=selected_score),
                    gr.update(visible=True),
                    status,
                    gr.update(value=df.head(10), visible=True),
                    gr.update(value=f"**{len(df):,} rows × {len(df.columns)} columns**", visible=True),
                    gr.update(value=_validation_dashboard_html(payload), visible=True),
                    gr.update(visible=False),
                )
            except (ValidationDataError, ValueError) as exc:
                return (
                    gr.update(choices=[], value=None),
                    gr.update(choices=[], value=None),
                    gr.update(visible=False),
                    _err(f"⚠️ {exc}"),
                    gr.update(visible=False),
                    gr.update(value="", visible=False),
                    gr.update(value="", visible=False),
                    gr.update(visible=True),
                )
            except Exception:
                return (
                    gr.update(choices=[], value=None),
                    gr.update(choices=[], value=None),
                    gr.update(visible=False),
                    _err(f"Could not prepare validation: {traceback.format_exc()}"),
                    gr.update(visible=False),
                    gr.update(value="", visible=False),
                    gr.update(value="", visible=False),
                    gr.update(visible=True),
                )

        def _refresh_validation(file_path, selected_label, selected_score, policy):
            if not file_path or not selected_label or not selected_score:
                return "", gr.update(value="", visible=False)
            try:
                df = _read_uploaded_delimited_file(file_path)
                payload = build_validation_payload(
                    df,
                    score_column=selected_score,
                    label_column=selected_label,
                    equivocal_policy=policy,
                )
                summary = payload["summary"]
                return (
                    f"✓ Updated using {summary['included_rows']:,} evaluable rows "
                    f"({summary['excluded_rows']:,} excluded).",
                    gr.update(value=_validation_dashboard_html(payload), visible=True),
                )
            except (ValidationDataError, ValueError) as exc:
                return _err(f"⚠️ {exc}"), gr.update(value="", visible=False)
            except Exception:
                return (
                    _err(f"Could not update validation: {traceback.format_exc()}"),
                    gr.update(value="", visible=False),
                )

        _load_outputs = [
            label_column,
            score_column,
            validation_controls,
            validation_status,
            validation_preview,
            validation_preview_info,
            validation_dashboard,
            validate_empty_state,
        ]
        validation_file.change(
            _load_validation_file,
            inputs=[validation_file, equivocal_policy],
            outputs=_load_outputs,
        )

        _refresh_inputs = [validation_file, label_column, score_column, equivocal_policy]
        _refresh_outputs = [validation_status, validation_dashboard]
        for trigger in (label_column, score_column, equivocal_policy):
            trigger.change(
                _refresh_validation,
                inputs=_refresh_inputs,
                outputs=_refresh_outputs,
            )

        gr.HTML('<hr class="ff-divider">')
        with gr.Accordion("🔍  Explain model behavior", open=False, elem_classes="ff-explain-accordion"):
            gr.HTML(
                '<p class="ff-hint">Generate a SHAP summary plot for a FluidFlagger model. '
                'This replaces the former standalone Explain tab.</p>'
            )
            build_shap_section()

# ---------------------------------------------------------------------------
# Tab 5 – Self Test
# ---------------------------------------------------------------------------

def build_self_test_tab(demo=None) -> None:
    from .self_test import build_answer_html, format_score, generate_case, init_db, log_case
    import uuid

    init_db()

    with gr.Tab("🎯  Self Test"):
        with gr.Row():
            # ── Left column: controls + score ─────────────────────────
            with gr.Column(scale=1, min_width=200):
                panel_radio = gr.Radio(
                    ["CBC", "BMP"], value="BMP", label="Panel",
                    interactive=True, elem_classes="ff-top-radio",
                )
                mode_radio = gr.Radio(
                    ["Retrospective", "Real-time", "Current Only", "Random"],
                    value="Random", label="Mode",
                    interactive=True, elem_classes="ff-top-radio ff-mode-radio",
                )
                new_case_btn = gr.Button("🎲  New Case", variant="primary", size="lg")
                name_input = gr.Textbox(
                    placeholder="Your name (optional)",
                    label="Name",
                    max_lines=1,
                )
                consent_cb = gr.Checkbox(
                    value=True,
                    label="Include my responses in academic research projects",
                    interactive=True,
                )
                consent_note = gr.Markdown("", visible=False, elem_classes="ff-muted-text")
                gr.HTML('<hr class="ff-divider">')
                gr.HTML('<p class="ff-section-title">Score</p>', elem_classes="ff-th")
                score_md = gr.HTML(format_score(0, 0))

            # ── Right column: case display + buttons ──────────────────
            with gr.Column(scale=2):
                gr.HTML('<p class="ff-section-title">Current Case</p>', elem_classes="ff-th")

                _initial_case = generate_case("BMP")
                state = gr.State({
                    "case": _initial_case,
                    "revealed": False,
                    "correct": 0,
                    "total": 0,
                    "session_id": None,
                    "display_mode": "Retrospective",
                })

                case_html = gr.HTML(
                    _build_review_html(_initial_case["row_dict"], "Retrospective"),
                    elem_classes="ff-case-display",
                )

                with gr.Row(elem_classes="ff-btn-row"):
                    real_btn   = gr.Button("Real",         elem_classes="btn-real   ff-guess-btn", variant="secondary", scale=1)
                    contam_btn = gr.Button("Contaminated", elem_classes="btn-contam ff-guess-btn", variant="secondary", scale=1)

                next_case_btn = gr.Button("Next Case →", variant="secondary", visible=False, size="lg")

                answer_area = gr.HTML("", elem_classes="ff-case-display")

        # ── Callbacks ────────────────────────────────────────────────

        import random as _random
        _CONCRETE_MODES = ["Retrospective", "Real-time", "Current Only"]

        def _new_case(panel, mode, st):
            session_id = st["session_id"] or str(uuid.uuid4())
            case = generate_case(panel)
            if case["error"]:
                return (
                    st,
                    _err(case['error']),
                    "",
                    format_score(st["correct"], st["total"]),
                    gr.update(visible=False),
                )
            display_mode = _random.choice(_CONCRETE_MODES) if mode == "Random" else mode
            new_st = {**st, "case": case, "revealed": False, "session_id": session_id, "display_mode": display_mode}
            row_html = _build_review_html(case["row_dict"], display_mode)
            return (
                new_st,
                row_html,
                "",
                format_score(st["correct"], st["total"]),
                gr.update(visible=False),
            )

        _case_outputs = [state, case_html, answer_area, score_md, next_case_btn]

        new_case_btn.click(_new_case, inputs=[panel_radio, mode_radio, state], outputs=_case_outputs)
        next_case_btn.click(_new_case, inputs=[panel_radio, mode_radio, state], outputs=_case_outputs)

        consent_cb.change(
            lambda v: gr.update(visible=not v, value="Your responses will not be included."),
            inputs=consent_cb,
            outputs=consent_note,
        )

        def _guess(guess: str, st: dict, name: str, consent: bool):
            if st["case"] is None:
                return st, "", gr.update(), gr.update(visible=False)

            case     = st["case"]
            revealed = st["revealed"]
            correct  = st["correct"]
            total    = st["total"]

            if not revealed:
                total += 1
                if (guess == "Contaminated") == case["contaminated"]:
                    correct += 1
                if consent:
                    log_case(case, guess, name or "", st.get("session_id", ""), st.get("display_mode", "Retrospective"))

            new_st = {**st, "revealed": True, "correct": correct, "total": total}

            answer_html = build_answer_html(case, guess)
            score_text  = format_score(correct, total)
            return new_st, answer_html, score_text, gr.update(visible=True)

        _guess_outputs = [state, answer_area, score_md, next_case_btn]

        for btn, lbl in [
            (real_btn,   "Real"),
            (contam_btn, "Contaminated"),
        ]:
            btn.click(
                lambda st, name, consent, lv=lbl: _guess(lv, st, name, consent),
                inputs=[state, name_input, consent_cb],
                outputs=_guess_outputs,
            )

        if demo is not None:
            demo.load(
                lambda st: _new_case("BMP", "Random", st)[:-1],
                inputs=[state],
                outputs=_case_outputs[:-1],
            )


# ---------------------------------------------------------------------------
# Theme & CSS
# ---------------------------------------------------------------------------

_CSS = (Path(__file__).parent / "styles.css").read_text()
_VALIDATION_DASHBOARD_JS = (Path(__file__).parent / "validation_dashboard.js").read_text()


# ---------------------------------------------------------------------------
# Assemble full UI
# ---------------------------------------------------------------------------

def build_ui(on_load=None) -> gr.Blocks:
    _blocks = gr.Blocks(title="FluidFlagger")
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
        build_validate_tab()
        build_self_test_tab(demo)
        if on_load is not None:
            demo.load(on_load)
    return demo
