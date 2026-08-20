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

# Regression models use a balanced simulation set instead of the classifier's
# prevalence-oriented binary simulation.  The inclusive grid has 51 levels
# (0.00, 0.01, ..., 0.50), yielding 51,000 cases per fluid at the default.
REGRESSION_MIX_RATIO_MIN = 0.0
REGRESSION_MIX_RATIO_MAX = 0.5
REGRESSION_MIX_RATIO_STEP = 0.01
REGRESSION_CASES_PER_RATIO = 1_000


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


def make_regression_mix_ratio_grid(
    cases_per_ratio: int = REGRESSION_CASES_PER_RATIO,
    *,
    minimum: float = REGRESSION_MIX_RATIO_MIN,
    maximum: float = REGRESSION_MIX_RATIO_MAX,
    step: float = REGRESSION_MIX_RATIO_STEP,
) -> np.ndarray:
    """Return a balanced, inclusive mixture-ratio grid for regression training."""
    if not isinstance(cases_per_ratio, (int, np.integer)) or cases_per_ratio < 1:
        raise ValueError("cases_per_ratio must be a positive integer")
    if step <= 0 or maximum < minimum:
        raise ValueError("mixture-ratio bounds must be ordered and step must be positive")

    levels = np.arange(minimum, maximum + step / 2, step)
    levels = np.round(levels, 10)
    if not np.isclose(levels[-1], maximum):
        raise ValueError("maximum must fall on the requested mixture-ratio grid")
    return np.repeat(levels, cases_per_ratio)

# ---------------------------------------------------------------------------
# BMP simulation
# ---------------------------------------------------------------------------

def _min_contam_for_fluid(fluid_name: str) -> float:
    """Minimum significant contamination fraction for a fluid.

    Dextrose-containing fluids (D5*) perturb glucose so strongly that even small
    admixtures are detectable, so they use a lower floor than other fluids.
    """
    return 0.05 if "D5" in fluid_name else 0.15


def _make_mix_ratios_bmp(n: int, fluid_name: str, rng: np.random.Generator) -> np.ndarray:
    """
    Generate BMP mix ratios using Beta(1, 5) + minimum_significant_contamination.
    Mirrors R: rbeta(n, 1, 5) + min_contam, capped at 0.8.
    """
    ratios = rng.beta(1, 5, size=n) + _min_contam_for_fluid(fluid_name)
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


def make_regression_training_data_bmp(
    template_df: pd.DataFrame,
    fluid_row: pd.Series,
    seed: int = 123,
    cases_per_ratio: int = REGRESSION_CASES_PER_RATIO,
) -> pd.DataFrame:
    """Create balanced BMP regression data for one IV fluid.

    Every mixture ratio from 0.00 through 0.50, inclusive and in 0.01 steps,
    receives the same number of simulated specimens.  The baseline specimen is
    sampled from complete patient rows; sampling with replacement preserves the
    requested balance when a user-provided template has fewer than 51,000 rows.
    """
    all_cols = (
        BMP_ANALYTES
        + [f"{c}_prior" for c in BMP_ANALYTES]
        + [f"{c}_post" for c in BMP_ANALYTES]
    )
    missing = [col for col in all_cols if col not in template_df.columns]
    if missing:
        raise ValueError(f"BMP regression template is missing required columns: {', '.join(missing)}")

    df = template_df.loc[:, all_cols].copy()
    for col in all_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=all_cols)
    df = df[(df[all_cols] > 0).all(axis=1)]
    if df.empty:
        raise ValueError("BMP regression template has no complete, positive rows")

    mix_ratios = make_regression_mix_ratio_grid(cases_per_ratio)
    rng = np.random.default_rng(seed)
    row_indices = rng.choice(len(df), size=len(mix_ratios), replace=len(df) < len(mix_ratios))
    input_sample = df.iloc[row_indices].reset_index(drop=True)
    return simulate_bmp_contamination(input_sample, fluid_row, mix_ratios)


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


def simulate_cbc_contamination(
    patient_df: pd.DataFrame,
    mix_ratios: np.ndarray,
) -> pd.DataFrame:
    """Simulate CBC dilution while retaining the prior and post patient values."""
    mix_ratios = np.asarray(mix_ratios, dtype=float)
    if len(patient_df) != len(mix_ratios):
        raise ValueError("patient_df and mix_ratios must have the same length")

    result = patient_df.copy()
    for col in CBC_ANALYTES:
        if col in result.columns:
            result[col] = result[col] * (1 - mix_ratios)
    result["mix_ratio"] = mix_ratios
    return result


def make_regression_training_data_cbc(
    template_df: pd.DataFrame,
    seed: int = 123,
    cases_per_ratio: int = REGRESSION_CASES_PER_RATIO,
) -> pd.DataFrame:
    """Create balanced CBC dilution-regression data on the 0.00-0.50 grid."""
    all_cols = (
        CBC_ANALYTES
        + [f"{c}_prior" for c in CBC_ANALYTES]
        + [f"{c}_post" for c in CBC_ANALYTES]
    )
    missing = [col for col in all_cols if col not in template_df.columns]
    if missing:
        raise ValueError(f"CBC regression template is missing required columns: {', '.join(missing)}")

    df = template_df.loc[:, all_cols].copy()
    for col in all_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=all_cols)
    df = df[(df[all_cols] > 0).all(axis=1)]
    if df.empty:
        raise ValueError("CBC regression template has no complete, positive rows")

    mix_ratios = make_regression_mix_ratio_grid(cases_per_ratio)
    rng = np.random.default_rng(seed)
    row_indices = rng.choice(len(df), size=len(mix_ratios), replace=len(df) < len(mix_ratios))
    input_sample = df.iloc[row_indices].reset_index(drop=True)
    return simulate_cbc_contamination(input_sample, mix_ratios)

def make_binary_training_data_cbc(
    template_df: pd.DataFrame,
    seed: int = 123,
    fluids_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Create 50/50 clean vs. contaminated CBC training data.
    CBC contamination = dilution: values * (1 - mix_ratio).

    Mix ratios are drawn from the same distribution as the BMP simulation:
    a contaminating fluid is sampled per contaminated row from the fluid table,
    and its mix ratio is Beta(1, 5) + min_contam, capped at 0.8. IV fluids carry
    no cellular components, so the fluid identity affects only the minimum
    contamination floor (lower for dextrose-containing fluids); the dilution
    itself is fluid-independent.
    """
    if fluids_df is None:
        fluids_df = get_fluid_concentrations()
    fluid_names = fluids_df["fluid"].tolist()
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
    sampled_fluids = rng.choice(fluid_names, size=n_contam)
    min_contam = np.array([_min_contam_for_fluid(f) for f in sampled_fluids])
    mix_ratios = np.minimum(rng.beta(1, 5, size=n_contam) + min_contam, 0.8)
    # Dilute only the current draw; prior and post values are left unmodified so
    # they reflect the patient's true physiology (a single contaminated draw).
    # Diluting prior/post by the same factor would preserve every ratio and
    # strip all contamination signal from the log-delta features.
    contam = simulate_cbc_contamination(contam, mix_ratios)
    contam["target"] = 1

    combined = pd.concat([clean, contam], ignore_index=True)
    combined["target"] = combined["target"].astype(int)
    return combined
