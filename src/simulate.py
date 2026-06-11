"""
Contamination simulation for BMP and CBC training data generation.

Ports the simulation logic from train_bmp_models.R and train_cbc_models.R.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .features import BMP_ANALYTES, CBC_ANALYTES

# ---------------------------------------------------------------------------
# Fluid concentrations
# ---------------------------------------------------------------------------

_DEFAULT_FLUIDS_PATH = Path(__file__).parent.parent / "data" / "fluid_concentrations.tsv"


def load_fluid_concentrations(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load fluid concentrations table (fluid, sodium, chloride, ...)."""
    p = Path(path) if path else _DEFAULT_FLUIDS_PATH
    return pd.read_csv(p, sep="\t")


# Cached at module level on first use
_FLUID_CONCENTRATIONS: Optional[pd.DataFrame] = None


def get_fluid_concentrations() -> pd.DataFrame:
    global _FLUID_CONCENTRATIONS
    if _FLUID_CONCENTRATIONS is None:
        _FLUID_CONCENTRATIONS = load_fluid_concentrations()
    return _FLUID_CONCENTRATIONS


def get_fluid_names() -> list[str]:
    return get_fluid_concentrations()["fluid"].tolist()

# ---------------------------------------------------------------------------
# BMP simulation
# ---------------------------------------------------------------------------

def _make_mix_ratios_bmp(n: int, fluid_name: str, rng: np.random.Generator) -> np.ndarray:
    """
    Generate BMP mix ratios using Beta(1, 5) + minimum_significant_contamination.
    Mirrors R: rbeta(n, 1, 5) + min_contam, capped at 0.8.
    """
    min_contam = 0.05 if "D5" in fluid_name else 0.15
    ratios = rng.beta(1, 5, size=n) + min_contam
    return np.minimum(ratios, 0.8)


def simulate_bmp_contamination(
    patient_df: pd.DataFrame,
    fluid_row: pd.Series,
    mix_ratios: np.ndarray,
) -> pd.DataFrame:
    """
    Simulate BMP contamination: (1 - ratio) * patient + ratio * fluid.
    Mirrors R simulateContaminationRow.
    """
    result = patient_df.copy()
    fluid_name = fluid_row["fluid"]

    for col in BMP_ANALYTES:
        if col in result.columns and col in fluid_row.index:
            result[col] = (1 - mix_ratios) * result[col] + mix_ratios * fluid_row[col]

    # Round to match clinical reporting precision
    for col in ["sodium", "chloride", "co2_totl", "bun", "glucose"]:
        if col in result.columns:
            result[col] = result[col].round(0)
    for col in ["potassium_plas", "calcium"]:
        if col in result.columns:
            result[col] = result[col].round(1)
    if "creatinine" in result.columns:
        result["creatinine"] = result["creatinine"].round(2)

    result["mix_ratio"] = mix_ratios
    result["label"] = fluid_name
    return result


def make_binary_training_data_bmp(
    template_df: pd.DataFrame,
    fluid_row: pd.Series,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Create 50/50 clean vs. contaminated BMP training data for one fluid.
    Requires template_df to have all 24 columns (8 current + 8 prior + 8 post).
    """
    all_cols = (
        BMP_ANALYTES
        + [f"{c}_prior" for c in BMP_ANALYTES]
        + [f"{c}_post" for c in BMP_ANALYTES]
    )
    df = template_df.copy()
    for col in all_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[c for c in all_cols if c in df.columns])
    df = df[(df[all_cols] > 0).all(axis=1)]

    n = len(df)
    n_uncontam = n // 2
    n_contam = n - n_uncontam

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    idx_uncontam = idx[:n_uncontam]
    idx_contam = idx[n_uncontam:]

    clean = df.iloc[idx_uncontam].copy()
    clean["mix_ratio"] = np.nan
    clean["label"] = "Patient"
    clean["target"] = 0

    contam_input = df.iloc[idx_contam].copy()
    mix_ratios = _make_mix_ratios_bmp(n_contam, fluid_row["fluid"], rng)
    # Trim to actual count in case beta produced fewer
    mix_ratios = mix_ratios[:n_contam]

    contam = simulate_bmp_contamination(contam_input, fluid_row, mix_ratios)
    contam["target"] = 1

    combined = pd.concat([clean, contam], ignore_index=True)
    combined["target"] = combined["target"].astype(int)
    return combined

# ---------------------------------------------------------------------------
# CBC simulation
# ---------------------------------------------------------------------------

def make_binary_training_data_cbc(
    template_df: pd.DataFrame,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Create 50/50 clean vs. contaminated CBC training data.
    CBC contamination = dilution: values * (1 - mix_ratio).
    Mix ratios from Uniform(0.1, 0.8).
    """
    all_cols = (
        CBC_ANALYTES
        + [f"{c}_prior" for c in CBC_ANALYTES]
        + [f"{c}_post" for c in CBC_ANALYTES]
    )
    df = template_df.copy()
    for col in all_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[c for c in all_cols if c in df.columns])
    df = df[(df[[c for c in all_cols if c in df.columns]] > 0).all(axis=1)]

    n = len(df)
    n_uncontam = n // 2
    n_contam = n - n_uncontam

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    idx_uncontam = idx[:n_uncontam]
    idx_contam = idx[n_uncontam:]

    clean = df.iloc[idx_uncontam].copy()
    clean["mix_ratio"] = 0.0
    clean["target"] = 0

    contam = df.iloc[idx_contam].copy()
    mix_ratios = rng.uniform(0.1, 0.8, size=n_contam)
    # Dilute only the current draw; prior and post values are left unmodified so
    # they reflect the patient's true physiology (a single contaminated draw).
    # Diluting prior/post by the same factor would preserve every ratio and
    # strip all contamination signal from the log-delta features.
    for col in CBC_ANALYTES:
        if col in contam.columns:
            contam[col] = contam[col] * (1 - mix_ratios)
    contam["mix_ratio"] = mix_ratios
    contam["target"] = 1

    combined = pd.concat([clean, contam], ignore_index=True)
    combined["target"] = combined["target"].astype(int)
    return combined
