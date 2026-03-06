"""
Self-Test tab — presents randomised CBC / BMP trios and lets the user
guess whether each specimen is contaminated, then reveals the ground truth.

CBC  — dilution-only contamination, mix ratio ~ Uniform(0.05, 0.25)
BMP  — one of NS / LR / SW(=Water) / D5NS / D5W, same mix ratio range
"""

from __future__ import annotations

import datetime
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .features import BMP_ANALYTES, CBC_ANALYTES
from .simulate import get_fluid_concentrations

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SELF_TEST_DIR = Path(__file__).parent.parent / "data" / "self_test_templates"

# BMP fluid pool for self-test (relative frequencies must sum to 1)
_BMP_FLUIDS  = ["NS",  "LR",  "Water", "D5NS", "D5W"]
_BMP_WEIGHTS = [0.3,   0.3,   0.2,     0.1,    0.1 ]

# Display name overrides (Water → SW for clinical familiarity)
_FLUID_DISPLAY = {"Water": "SW"}


def _display(fluid: str) -> str:
    return _FLUID_DISPLAY.get(fluid, fluid)


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

def load_self_test_template(panel: str) -> Optional[pd.DataFrame]:
    """Return the first matching self-test CSV for the given panel, or None."""
    prefix = "cbc" if panel == "CBC" else "bmp"
    matches = sorted(_SELF_TEST_DIR.glob(f"{prefix}_*.csv"))
    if not matches:
        return None
    return pd.read_csv(matches[0])


# ---------------------------------------------------------------------------
# Case generation
# ---------------------------------------------------------------------------

def generate_case(panel: str, rng: Optional[np.random.Generator] = None) -> dict:
    """
    Randomly select one row from the self-test template and optionally
    simulate contamination.

    Returns a dict:
      row_dict     – {col: value} ready for _build_review_html
      contaminated – bool
      fluid        – str or None  (BMP only; display name already applied)
      mix_ratio    – float or None
      error        – str or None
    """
    if rng is None:
        rng = np.random.default_rng()

    df = load_self_test_template(panel)
    if df is None:
        return {
            "row_dict": {}, "contaminated": False,
            "fluid": None, "mix_ratio": None,
            "error": f"No {panel} self-test template found in data/self_test_templates/.",
        }

    row = df.iloc[int(rng.integers(0, len(df)))].copy()

    contaminated = bool(rng.random() < 0.5)
    mix_ratio: Optional[float] = round(float(rng.uniform(0.05, 0.25)), 2) if contaminated else None
    fluid: Optional[str] = None

    if contaminated:
        if panel == "CBC":
            for col in CBC_ANALYTES:
                if col in row.index:
                    row[col] = float(row[col]) * (1.0 - mix_ratio)
            # Round to match clinical precision
            for col in ["Hgb", "WBC"]:
                if col in row.index:
                    row[col] = round(float(row[col]), 1)
            if "Plt" in row.index:
                row["Plt"] = round(float(row["Plt"]))

        else:  # BMP
            fluid_key = str(rng.choice(_BMP_FLUIDS, p=_BMP_WEIGHTS))
            fluid = _display(fluid_key)
            fluids_df = get_fluid_concentrations()
            fluid_row = fluids_df[fluids_df["fluid"] == fluid_key].iloc[0]
            for col in BMP_ANALYTES:
                if col in row.index and col in fluid_row.index:
                    row[col] = (1.0 - mix_ratio) * float(row[col]) + mix_ratio * float(fluid_row[col])
            # Round to clinical precision
            for col in ["sodium", "chloride", "co2_totl", "bun", "glucose"]:
                if col in row.index:
                    row[col] = round(float(row[col]))
            for col in ["potassium_plas", "calcium"]:
                if col in row.index:
                    row[col] = round(float(row[col]), 1)
            if "creatinine" in row.index:
                row["creatinine"] = round(float(row["creatinine"]), 2)

    return {
        "row_dict": row.to_dict(),
        "contaminated": contaminated,
        "fluid": fluid,
        "mix_ratio": mix_ratio,
        "panel": panel,
        "error": None,
    }


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def build_answer_html(case: dict, guess: Optional[str]) -> str:
    """Render the ground-truth reveal panel."""
    contaminated = case["contaminated"]
    fluid        = case.get("fluid")
    mix_ratio    = case.get("mix_ratio")

    # ── Verdict ──────────────────────────────────────────────────────────────
    if guess is not None:
        correct = (guess == "Contaminated") == contaminated
        if correct:
            verdict_icon   = "✓"
            verdict_text   = "Correct"
            verdict_color  = "#15803d"
            verdict_bg     = "#dcfce7"
            verdict_border = "#86efac"
        else:
            verdict_icon   = "✗"
            verdict_text   = "Incorrect"
            verdict_color  = "#b91c1c"
            verdict_bg     = "#fee2e2"
            verdict_border = "#fca5a5"
        verdict_html = (
            f'<div style="display:flex;align-items:center;gap:12px;padding:14px 20px;'
            f'background:{verdict_bg};border:1px solid {verdict_border};border-radius:12px;'
            f'margin-bottom:10px">'
            f'<span style="font-size:1.75rem;line-height:1;color:{verdict_color}">{verdict_icon}</span>'
            f'<span style="font-size:1.125rem;font-weight:700;color:{verdict_color}">{verdict_text}</span>'
            f'</div>'
        )
    else:
        verdict_html = ""

    # ── Ground truth card ────────────────────────────────────────────────────
    if contaminated:
        gt_label_color = "#9f1239"
        gt_bg          = "#fff1f2"
        gt_border      = "#fda4af"
        gt_status      = "CONTAMINATED"
        gt_status_color = "#b91c1c"

        badge_bg     = "#fecdd3"
        badge_color  = "#9f1239"
        badges = []
        if fluid:
            badges.append(
                f'<span style="background:{badge_bg};color:{badge_color};padding:3px 11px;'
                f'border-radius:9999px;font-size:0.8125rem;font-weight:600">{fluid}</span>'
            )
        if mix_ratio is not None:
            badges.append(
                f'<span style="background:{badge_bg};color:{badge_color};padding:3px 11px;'
                f'border-radius:9999px;font-size:0.8125rem;font-weight:600">{mix_ratio:.0%} mix</span>'
            )
        badges_html = (
            f'<div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap">{"".join(badges)}</div>'
            if badges else ""
        )
    else:
        gt_label_color  = "#166534"
        gt_bg           = "#f0fdf4"
        gt_border       = "#86efac"
        gt_status       = "CLEAN"
        gt_status_color = "#15803d"
        badges_html     = ""

    truth_html = (
        f'<div style="padding:14px 18px;background:{gt_bg};border:1px solid {gt_border};border-radius:12px">'
        f'<div style="font-size:0.6875rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;'
        f'color:{gt_label_color};margin-bottom:6px">Ground Truth</div>'
        f'<div style="font-size:1.0625rem;font-weight:700;color:{gt_status_color}">{gt_status}</div>'
        f'{badges_html}'
        f'</div>'
    )

    return (
        f'<div style="margin-top:14px;animation:ff-fade-in 0.2s ease">'
        f'{verdict_html}{truth_html}'
        f'</div>'
    )


_DB_PATH = Path(__file__).parent.parent / "data" / "self_test_log.db"

_BMP_ANALYTES = ["sodium", "chloride", "potassium_plas", "co2_totl",
                 "bun", "creatinine", "calcium", "glucose"]
_CBC_ANALYTES = ["Hgb", "Plt", "WBC"]
_ALL_ANALYTES = _BMP_ANALYTES + _CBC_ANALYTES

_analyte_cols = ",\n    ".join(
    f"{col}_prior REAL, {col}_current REAL, {col}_post REAL"
    for col in _ALL_ANALYTES
)

_CREATE_TABLE = f"""
CREATE TABLE IF NOT EXISTS results (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT    NOT NULL,
    timestamp     TEXT    NOT NULL,
    name          TEXT,
    panel         TEXT    NOT NULL,
    contaminated  INTEGER NOT NULL,
    fluid         TEXT,
    mix_ratio     REAL,
    guess         TEXT    NOT NULL,
    correct       INTEGER NOT NULL,
    {_analyte_cols}
)
"""


def init_db() -> None:
    """Create the results table if it doesn't exist (safe to call repeatedly)."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(_CREATE_TABLE)
        conn.commit()


def log_case(case: dict, guess: str, name: str, session_id: str) -> None:
    """Append one result row to the SQLite database."""
    row_dict = case["row_dict"]

    record: dict = {
        "session_id":   session_id,
        "timestamp":    datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "name":         name,
        "panel":        case.get("panel", ""),
        "contaminated": int(case["contaminated"]),
        "fluid":        case.get("fluid") or "",
        "mix_ratio":    case.get("mix_ratio"),
        "guess":        guess,
        "correct":      int((guess == "Contaminated") == case["contaminated"]),
    }
    for col in _ALL_ANALYTES:
        record[f"{col}_prior"]   = row_dict.get(f"{col}_prior")
        record[f"{col}_current"] = row_dict.get(col)
        record[f"{col}_post"]    = row_dict.get(f"{col}_post")

    cols         = ", ".join(record.keys())
    placeholders = ", ".join(f":{k}" for k in record.keys())
    with sqlite3.connect(_DB_PATH, timeout=10) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"INSERT INTO results ({cols}) VALUES ({placeholders})", record)
        conn.commit()


def format_score(correct: int, total: int) -> str:
    if total == 0:
        return "<p style='color:#94a3b8;font-size:0.875rem;margin:0;padding:4px 0'>No cases attempted yet.</p>"
    pct = correct / total * 100
    if pct >= 70:
        color, bg, border = "#15803d", "#f0fdf4", "#bbf7d0"
    elif pct >= 50:
        color, bg, border = "#92400e", "#fffbeb", "#fde68a"
    else:
        color, bg, border = "#b91c1c", "#fef2f2", "#fca5a5"
    bar_w = min(100.0, pct)
    return (
        f'<div style="padding:10px 12px;background:{bg};border:1px solid {border};border-radius:10px;margin-top:4px">'
        f'<div style="font-size:1.5rem;font-weight:700;color:{color};line-height:1;text-align:center">{correct}/{total}</div>'
        f'<div style="font-size:0.75rem;font-weight:600;color:{color};text-align:center;margin-top:2px">{pct:.0f}% correct</div>'
        f'<div style="background:#e2e8f0;border-radius:3px;height:4px;margin-top:8px;overflow:hidden">'
        f'<div style="background:{color};width:{bar_w:.1f}%;height:100%;border-radius:3px"></div>'
        f'</div></div>'
    )
