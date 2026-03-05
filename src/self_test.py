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

    if contaminated:
        if fluid:
            truth_str = f"CONTAMINATED — {fluid} at {mix_ratio:.0%} mix ratio"
        else:
            truth_str = f"CONTAMINATED — {mix_ratio:.0%} dilution"
        truth_color  = "#b91c1c"
        truth_bg     = "#fee2e2"
        truth_border = "#fca5a5"
    else:
        truth_str    = "CLEAN — no contamination"
        truth_color  = "#15803d"
        truth_bg     = "#dcfce7"
        truth_border = "#86efac"

    correctness_html = ""
    if guess is not None:
        if (guess == "Contaminated") == contaminated:
            correctness_html = (
                '<div style="margin-bottom:10px;padding:8px 14px;background:#dcfce7;'
                'border:1px solid #86efac;border-radius:8px;color:#15803d;font-weight:600">'
                "✓ Correct!</div>"
            )
        else:
            correctness_html = (
                '<div style="margin-bottom:10px;padding:8px 14px;background:#fee2e2;'
                'border:1px solid #fca5a5;border-radius:8px;color:#b91c1c;font-weight:600">'
                "✗ Incorrect</div>"
            )

    truth_html = (
        f'<div style="padding:10px 14px;background:{truth_bg};border:1px solid {truth_border};'
        f'border-radius:8px;color:{truth_color};font-weight:600;font-size:0.9rem">'
        f"Ground truth: {truth_str}</div>"
    )
    return f'<div style="margin-top:12px">{correctness_html}{truth_html}</div>'


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
        return "No cases attempted yet."
    pct = correct / total * 100
    return f"**{correct} / {total}** correct ({pct:.0f}%)"
