"""
Feature engineering for BMP and CBC contamination detection.

Ports the logic from bmp_helpers.R and cbc_helpers.R:
- Assay name synonym mapping with fuzzy fallback
- Long → wide pivoting with prior/post window computation
- sklearn-compatible feature transformers (log-delta ratios + PCA)
"""

from __future__ import annotations

import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BMP_ANALYTES = [
    "sodium", "chloride", "potassium_plas", "co2_totl",
    "bun", "creatinine", "calcium", "glucose",
]

CBC_ANALYTES = ["Hgb", "Plt", "WBC"]

BMP_SYNONYMS: dict[str, list[str]] = {
    "sodium":         ["sodium", "na", "na+", "sod"],
    "chloride":       ["chloride", "cl", "cl-", "chl", "chlor"],
    "potassium_plas": ["potassium", "potassiumplasma", "potassiumplas",
                       "potassium_plas", "k", "k+", "potas"],
    "co2_totl":       ["co2", "co2totl", "co2total", "tco2", "bicarb",
                       "bicarbonate", "hco3", "hco3-", "carbondioxide"],
    "bun":            ["bun", "bloodureanitrogen", "ureanitrogen", "urean", "urea"],
    "creatinine":     ["creatinine", "creat", "cr", "creatinin"],
    "calcium":        ["calcium", "ca", "ca2+", "ca++", "cal"],
    "glucose":        ["glucose", "glu", "gluc", "bloodglucose", "serumglucose"],
}

CBC_SYNONYMS: dict[str, list[str]] = {
    "Hgb": ["hgb", "hb", "hemoglobin", "haemoglobin"],
    "WBC": ["wbc", "whitebloodcell", "whitebloodcells", "wcc",
            "leukocyte", "leukocytes"],
    "Plt": ["plt", "platelet", "platelets", "thrombocyte", "thrombocytes"],
}

# Pre-build lookup dicts: normalized_label → canonical_name
def _build_lookup(synonyms: dict[str, list[str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canon, aliases in synonyms.items():
        for alias in [canon] + aliases:
            lookup[_normalize(alias)] = canon
    return lookup

BMP_SYNONYM_LOOKUP = _build_lookup(BMP_SYNONYMS)
CBC_SYNONYM_LOOKUP = _build_lookup(CBC_SYNONYMS)

# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize(label: str) -> str:
    """Lowercase and strip all non-alphanumeric characters (mirrors R normalize_assay_label)."""
    return re.sub(r"[^a-z0-9]+", "", str(label).lower())


def _edit_distance(a: str, b: str) -> int:
    """Simple edit distance (Levenshtein) for fuzzy matching."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
    return dp[n]


def _map_assay_name(label: str, lookup: dict[str, str], targets: list[str]) -> str:
    """Map a single assay label to a canonical name, with fuzzy fallback."""
    norm = _normalize(label)
    if norm in lookup:
        return lookup[norm]

    # Fuzzy: accept if edit distance ≤ 25% of input length (min 1)
    max_dist = max(1, int(len(norm) * 0.25))
    best_dist, best_target = None, None
    for t in targets:
        d = _edit_distance(norm, _normalize(t))
        if best_dist is None or d < best_dist:
            best_dist, best_target = d, t
    if best_dist is not None and best_dist <= max_dist:
        return best_target
    return label  # return original if no match


def map_bmp_assay_names(labels: list[str]) -> list[str]:
    targets = list(BMP_SYNONYMS.keys())
    return [_map_assay_name(l, BMP_SYNONYM_LOOKUP, targets) for l in labels]


def map_cbc_assay_names(labels: list[str]) -> list[str]:
    targets = list(CBC_SYNONYMS.keys())
    return [_map_assay_name(l, CBC_SYNONYM_LOOKUP, targets) for l in labels]


def _clean_column_name(name: str) -> str:
    """Mirrors janitor::make_clean_names: lowercase, replace non-alnum runs with _."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip())
    s = s.strip("_").lower()
    return s


def map_bmp_wide_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename BMP wide-format DataFrame columns to canonical analyte names."""
    df = df.copy()
    new_names = {}
    for col in df.columns:
        cleaned = _clean_column_name(col)
        suffix = ""
        base = cleaned
        if base.endswith("_prior"):
            suffix = "_prior"
            base = base[: -len("_prior")]
        elif base.endswith("_post"):
            suffix = "_post"
            base = base[: -len("_post")]
        mapped = _map_assay_name(base, BMP_SYNONYM_LOOKUP, list(BMP_SYNONYMS.keys()))
        if mapped in BMP_SYNONYMS:
            new_names[col] = mapped + suffix
        else:
            new_names[col] = col
    df.rename(columns=new_names, inplace=True)
    return df


def map_cbc_wide_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename CBC wide-format DataFrame columns to canonical analyte names."""
    df = df.copy()
    new_names = {}
    for col in df.columns:
        cleaned = _clean_column_name(col)
        suffix = ""
        base = cleaned
        if base.endswith("_prior"):
            suffix = "_prior"
            base = base[: -len("_prior")]
        elif base.endswith("_post"):
            suffix = "_post"
            base = base[: -len("_post")]
        # CBC analytes are mixed-case; try against lowercase synonyms
        norm = _normalize(base)
        mapped_norm = CBC_SYNONYM_LOOKUP.get(norm)
        if mapped_norm:
            new_names[col] = mapped_norm + suffix
        else:
            new_names[col] = col
    df.rename(columns=new_names, inplace=True)
    return df

# ---------------------------------------------------------------------------
# Long → wide conversion
# ---------------------------------------------------------------------------

def _parse_datetime(series: pd.Series) -> pd.Series:
    """Parse timestamps supporting multiple formats."""
    formats = [
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
        "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
        "%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %I:%M %p",
    ]
    result = pd.NaT
    for fmt in formats:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors="coerce", utc=True)
            if parsed.notna().any():
                # Fill in where still NaT
                if result is pd.NaT:
                    result = parsed
                else:
                    result = result.fillna(parsed)
        except Exception:
            continue
    if result is pd.NaT:
        result = pd.to_datetime(series, infer_datetime_format=True, errors="coerce", utc=True)
    return result


def rectangularize_long(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long-format lab data (PATIENT_ID, DRAWN_DT_TM, TASK_ASSAY, RESULT_VALUE)
    to wide format with one row per patient/timestamp.
    """
    df = df_long.copy()
    # Strip comparison operators and parse numeric
    df["RESULT_VALUE"] = (
        df["RESULT_VALUE"].astype(str)
        .str.replace(r"[<>]", "", regex=True)
    )
    df["RESULT_VALUE"] = pd.to_numeric(df["RESULT_VALUE"], errors="coerce")
    df = df.sort_values("DRAWN_DT_TM")

    wide = df.pivot_table(
        index=["PATIENT_ID", "DRAWN_DT_TM"],
        columns="TASK_ASSAY",
        values="RESULT_VALUE",
        aggfunc="last",
    ).reset_index()
    wide.columns.name = None
    # Forward-fill analyte values within each patient/timestamp group
    analyte_cols = [c for c in wide.columns if c not in ("PATIENT_ID", "DRAWN_DT_TM")]
    wide[analyte_cols] = wide.groupby("PATIENT_ID")[analyte_cols].ffill()
    wide = wide.drop_duplicates(subset=["PATIENT_ID", "DRAWN_DT_TM"])
    return wide


def add_pre_post_bmp(df: pd.DataFrame, lookback_hours: float = 48.0) -> pd.DataFrame:
    """Add _prior and _post columns for BMP analytes within lookback_hours window."""
    df = df.copy().sort_values(["PATIENT_ID", "DRAWN_DT_TM"])
    analytes = [c for c in BMP_ANALYTES if c in df.columns]

    for col in analytes:
        # prior: previous value within lookback window
        def _prior(group: pd.DataFrame) -> pd.Series:
            hours_diff = group["DRAWN_DT_TM"].diff().dt.total_seconds() / 3600
            prior_vals = group[col].shift(1)
            return prior_vals.where(hours_diff < lookback_hours, other=np.nan)

        # post: next value within lookback window
        def _post(group: pd.DataFrame) -> pd.Series:
            hours_diff = (-group["DRAWN_DT_TM"].diff(-1)).dt.total_seconds() / 3600
            post_vals = group[col].shift(-1)
            return post_vals.where(hours_diff < lookback_hours, other=np.nan)

        df[f"{col}_prior"] = df.groupby("PATIENT_ID", group_keys=False).apply(_prior)
        df[f"{col}_post"] = df.groupby("PATIENT_ID", group_keys=False).apply(_post)

    return df


def add_pre_post_cbc(df: pd.DataFrame, collection_interval: float = 48.0) -> pd.DataFrame:
    """Add _prior and _post columns for CBC analytes within collection_interval hours."""
    df = df.copy().sort_values(["PATIENT_ID", "DRAWN_DT_TM"])
    analytes = [c for c in CBC_ANALYTES if c in df.columns]

    for col in analytes:
        def _prior(group: pd.DataFrame) -> pd.Series:
            hours_diff = group["DRAWN_DT_TM"].diff().dt.total_seconds() / 3600
            prior_vals = group[col].shift(1)
            return prior_vals.where(hours_diff < collection_interval, other=np.nan)

        def _post(group: pd.DataFrame) -> pd.Series:
            hours_diff = (-group["DRAWN_DT_TM"].diff(-1)).dt.total_seconds() / 3600
            post_vals = group[col].shift(-1)
            return post_vals.where(hours_diff < collection_interval, other=np.nan)

        df[f"{col}_prior"] = df.groupby("PATIENT_ID", group_keys=False).apply(_prior)
        df[f"{col}_post"] = df.groupby("PATIENT_ID", group_keys=False).apply(_post)

    return df

# ---------------------------------------------------------------------------
# Full preprocessing pipelines
# ---------------------------------------------------------------------------

def preprocess_bmp_data(df: pd.DataFrame, lookback_hours: float = 48.0) -> pd.DataFrame:
    """
    Auto-detect wide vs. long format and preprocess BMP data.
    Wide: rename columns to canonical names.
    Long: pivot, add prior/post, rename.
    """
    cols_upper = {c.upper() for c in df.columns}
    is_long = "TASK_ASSAY" in cols_upper and (
        "RESULT_VALUE" in cols_upper or "RESULT_VALUE_NUMERIC" in cols_upper
    )

    if not is_long:
        return map_bmp_wide_names(df)

    df = df.copy()
    df.columns = [c.upper() for c in df.columns]
    if "RESULT_VALUE" not in df.columns and "RESULT_VALUE_NUMERIC" in df.columns:
        df = df.rename(columns={"RESULT_VALUE_NUMERIC": "RESULT_VALUE"})

    df["TASK_ASSAY"] = map_bmp_assay_names(df["TASK_ASSAY"].tolist())
    df["RESULT_VALUE"] = df["RESULT_VALUE"].astype(str)
    df["DRAWN_DT_TM"] = _parse_datetime(df["DRAWN_DT_TM"])
    df = df.dropna(subset=["DRAWN_DT_TM"])

    wide = rectangularize_long(df)
    return add_pre_post_bmp(wide, lookback_hours=lookback_hours)


def preprocess_cbc_data(df: pd.DataFrame, collection_interval: float = 48.0) -> pd.DataFrame:
    """
    Auto-detect wide vs. long format and preprocess CBC data.
    """
    cols_upper = {c.upper() for c in df.columns}
    is_long = "TASK_ASSAY" in cols_upper and (
        "RESULT_VALUE" in cols_upper or "RESULT_VALUE_NUMERIC" in cols_upper
    )

    if not is_long:
        return map_cbc_wide_names(df)

    df = df.copy()
    df.columns = [c.upper() for c in df.columns]
    if "RESULT_VALUE" not in df.columns and "RESULT_VALUE_NUMERIC" in df.columns:
        df = df.rename(columns={"RESULT_VALUE_NUMERIC": "RESULT_VALUE"})

    df["TASK_ASSAY"] = map_cbc_assay_names(df["TASK_ASSAY"].tolist())
    df["RESULT_VALUE"] = pd.to_numeric(
        df["RESULT_VALUE"].astype(str).str.replace(r"[<>]", "", regex=True),
        errors="coerce",
    )
    df["DRAWN_DT_TM"] = _parse_datetime(df["DRAWN_DT_TM"])
    df = df.dropna(subset=["DRAWN_DT_TM"])

    wide = (
        df.sort_values("DRAWN_DT_TM")
        .pivot_table(
            index=["PATIENT_ID", "DRAWN_DT_TM"],
            columns="TASK_ASSAY",
            values="RESULT_VALUE",
            aggfunc="last",
        )
        .reset_index()
    )
    wide.columns.name = None
    wide = wide.drop_duplicates()
    return add_pre_post_cbc(wide, collection_interval=collection_interval)

# ---------------------------------------------------------------------------
# sklearn Feature Transformers
# ---------------------------------------------------------------------------

def _log_delta_prior(current: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """log(0.01 + current / max(prior, 0.1))  — mirrors R formula."""
    safe_prior = np.where(prior == 0, 0.1, prior)
    return np.log(0.01 + current / safe_prior)


def _log_delta_post(current: np.ndarray, post: np.ndarray) -> np.ndarray:
    """log(0.01 + post / max(current, 0.1))  — mirrors R formula."""
    safe_current = np.where(current == 0, 0.1, current)
    return np.log(0.01 + post / safe_current)


def _pca_pipeline(n_components: int = 3) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=n_components))])


class BMPFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Transform BMP DataFrame rows into model-ready feature matrix.

    mode='realtime':
        Uses current + prior values only.
        Features: 8 current + 8 prior_deltas + 3 all_PC + 3 prior_PC = 22

    mode='retrospective':
        Uses current + prior + post values.
        Features: 8 current + 8 prior_deltas + 8 post_deltas + 3 prior_PC
                  + 3 post_PC + 3 all_PC = 33
    """

    def __init__(self, mode: str = "retrospective", n_components: int = 3):
        self.mode = mode
        self.n_components = n_components

    def _compute_deltas(self, X: pd.DataFrame):
        prior_deltas = np.column_stack([
            _log_delta_prior(
                X[col].values.astype(float),
                X[f"{col}_prior"].values.astype(float),
            )
            for col in BMP_ANALYTES
        ])
        if self.mode == "retrospective":
            post_deltas = np.column_stack([
                _log_delta_post(
                    X[col].values.astype(float),
                    X[f"{col}_post"].values.astype(float),
                )
                for col in BMP_ANALYTES
            ])
            return prior_deltas, post_deltas
        return prior_deltas, None

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        current = X[BMP_ANALYTES].values.astype(float)
        prior_deltas, post_deltas = self._compute_deltas(X)

        if self.mode == "retrospective":
            all_deltas = np.hstack([prior_deltas, post_deltas])
            self.prior_pca_ = _pca_pipeline(self.n_components).fit(prior_deltas)
            self.post_pca_ = _pca_pipeline(self.n_components).fit(post_deltas)
            self.all_pca_ = _pca_pipeline(self.n_components).fit(all_deltas)
        else:
            all_predictors = np.hstack([current, prior_deltas])
            self.all_pca_ = _pca_pipeline(self.n_components).fit(all_predictors)
            self.prior_pca_ = _pca_pipeline(self.n_components).fit(prior_deltas)
        return self

    def transform(self, X: pd.DataFrame):
        X = pd.DataFrame(X)
        current = X[BMP_ANALYTES].values.astype(float)
        prior_deltas, post_deltas = self._compute_deltas(X)

        if self.mode == "retrospective":
            all_deltas = np.hstack([prior_deltas, post_deltas])
            prior_pcs = self.prior_pca_.transform(prior_deltas)
            post_pcs = self.post_pca_.transform(post_deltas)
            all_pcs = self.all_pca_.transform(all_deltas)
            return np.hstack([current, prior_deltas, post_deltas,
                               prior_pcs, post_pcs, all_pcs])
        else:
            all_predictors = np.hstack([current, prior_deltas])
            all_pcs = self.all_pca_.transform(all_predictors)
            prior_pcs = self.prior_pca_.transform(prior_deltas)
            return np.hstack([current, prior_deltas, all_pcs, prior_pcs])

    def get_feature_names_out(self, input_features=None):
        if self.mode == "retrospective":
            names = (
                BMP_ANALYTES
                + [f"{c}_log_delta_prior" for c in BMP_ANALYTES]
                + [f"{c}_log_delta_post" for c in BMP_ANALYTES]
                + [f"prior_PC{i+1}" for i in range(self.n_components)]
                + [f"post_PC{i+1}" for i in range(self.n_components)]
                + [f"all_PC{i+1}" for i in range(self.n_components)]
            )
        else:
            names = (
                BMP_ANALYTES
                + [f"{c}_log_delta_prior" for c in BMP_ANALYTES]
                + [f"all_PC{i+1}" for i in range(self.n_components)]
                + [f"prior_PC{i+1}" for i in range(self.n_components)]
            )
        return np.array(names)


class CBCFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Transform CBC DataFrame rows into model-ready feature matrix.

    mode='realtime':
        Features: 3 current + 3 prior_deltas + 3 all_PC = 9

    mode='retrospective':
        Features: 3 current + 3 prior_deltas + 3 post_deltas
                  + 3 prior_PC + 3 post_PC + 3 all_PC = 18
    """

    def __init__(self, mode: str = "retrospective", n_components: int = 3):
        self.mode = mode
        self.n_components = n_components

    def _compute_deltas(self, X: pd.DataFrame):
        prior_deltas = np.column_stack([
            _log_delta_prior(
                X[col].values.astype(float),
                X[f"{col}_prior"].values.astype(float),
            )
            for col in CBC_ANALYTES
        ])
        if self.mode == "retrospective":
            post_deltas = np.column_stack([
                _log_delta_post(
                    X[col].values.astype(float),
                    X[f"{col}_post"].values.astype(float),
                )
                for col in CBC_ANALYTES
            ])
            return prior_deltas, post_deltas
        return prior_deltas, None

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        current = X[CBC_ANALYTES].values.astype(float)
        prior_deltas, post_deltas = self._compute_deltas(X)

        if self.mode == "retrospective":
            all_deltas = np.hstack([prior_deltas, post_deltas])
            self.prior_pca_ = _pca_pipeline(self.n_components).fit(prior_deltas)
            self.post_pca_ = _pca_pipeline(self.n_components).fit(post_deltas)
            self.all_pca_ = _pca_pipeline(self.n_components).fit(all_deltas)
        else:
            all_predictors = np.hstack([current, prior_deltas])
            self.all_pca_ = _pca_pipeline(self.n_components).fit(all_predictors)
        return self

    def transform(self, X: pd.DataFrame):
        X = pd.DataFrame(X)
        current = X[CBC_ANALYTES].values.astype(float)
        prior_deltas, post_deltas = self._compute_deltas(X)

        if self.mode == "retrospective":
            all_deltas = np.hstack([prior_deltas, post_deltas])
            prior_pcs = self.prior_pca_.transform(prior_deltas)
            post_pcs = self.post_pca_.transform(post_deltas)
            all_pcs = self.all_pca_.transform(all_deltas)
            return np.hstack([current, prior_deltas, post_deltas,
                               prior_pcs, post_pcs, all_pcs])
        else:
            all_predictors = np.hstack([current, prior_deltas])
            all_pcs = self.all_pca_.transform(all_predictors)
            return np.hstack([current, prior_deltas, all_pcs])

    def get_feature_names_out(self, input_features=None):
        if self.mode == "retrospective":
            names = (
                CBC_ANALYTES
                + [f"{c}_log_delta_prior" for c in CBC_ANALYTES]
                + [f"{c}_log_delta_post" for c in CBC_ANALYTES]
                + [f"prior_PC{i+1}" for i in range(self.n_components)]
                + [f"post_PC{i+1}" for i in range(self.n_components)]
                + [f"all_PC{i+1}" for i in range(self.n_components)]
            )
        else:
            names = (
                CBC_ANALYTES
                + [f"{c}_log_delta_prior" for c in CBC_ANALYTES]
                + [f"all_PC{i+1}" for i in range(self.n_components)]
            )
        return np.array(names)
