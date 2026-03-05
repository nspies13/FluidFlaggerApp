"""
Tests for feature engineering (features.py).

Validates:
- Assay name synonym mapping
- Log-delta formulas
- Wide/long preprocessing
- Feature transformer shapes
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import (
    BMP_ANALYTES,
    CBC_ANALYTES,
    BMPFeatureTransformer,
    CBCFeatureTransformer,
    map_bmp_assay_names,
    map_bmp_wide_names,
    map_cbc_assay_names,
    map_cbc_wide_names,
    preprocess_bmp_data,
    preprocess_cbc_data,
    _log_delta_prior,
    _log_delta_post,
)

DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Assay name mapping
# ---------------------------------------------------------------------------

def test_bmp_exact_synonyms():
    assert map_bmp_assay_names(["sodium"])[0] == "sodium"
    assert map_bmp_assay_names(["Na"])[0] == "sodium"
    assert map_bmp_assay_names(["K"])[0] == "potassium_plas"
    assert map_bmp_assay_names(["CO2"])[0] == "co2_totl"
    assert map_bmp_assay_names(["bicarb"])[0] == "co2_totl"
    assert map_bmp_assay_names(["Creatinine"])[0] == "creatinine"
    assert map_bmp_assay_names(["CA"])[0] == "calcium"
    assert map_bmp_assay_names(["gluc"])[0] == "glucose"


def test_bmp_fuzzy_mapping():
    # "creatinin" is 1 edit away from "creatinine"
    result = map_bmp_assay_names(["creatinin"])
    assert result[0] == "creatinine"


def test_bmp_unknown_passthrough():
    result = map_bmp_assay_names(["ALT"])
    assert result[0] == "ALT"  # no match → original returned


def test_cbc_exact_synonyms():
    assert map_cbc_assay_names(["hemoglobin"])[0] == "Hgb"
    assert map_cbc_assay_names(["wbc"])[0] == "WBC"
    assert map_cbc_assay_names(["platelet"])[0] == "Plt"


def test_map_bmp_wide_names():
    df = pd.DataFrame(columns=["Na", "Cl", "K", "CO2", "BUN", "Creat", "Ca", "Glu",
                                "Na_prior", "Na_post"])
    renamed = map_bmp_wide_names(df)
    assert "sodium" in renamed.columns
    assert "chloride" in renamed.columns
    assert "potassium_plas" in renamed.columns
    assert "sodium_prior" in renamed.columns
    assert "sodium_post" in renamed.columns


def test_map_cbc_wide_names():
    df = pd.DataFrame(columns=["hemoglobin", "WBC", "platelet", "Hgb_prior"])
    renamed = map_cbc_wide_names(df)
    assert "Hgb" in renamed.columns
    assert "WBC" in renamed.columns
    assert "Plt" in renamed.columns
    assert "Hgb_prior" in renamed.columns

# ---------------------------------------------------------------------------
# Log-delta formulas
# ---------------------------------------------------------------------------

def test_log_delta_prior_formula():
    current = np.array([138.0])
    prior = np.array([140.0])
    result = _log_delta_prior(current, prior)
    expected = np.log(0.01 + 138.0 / 140.0)
    np.testing.assert_allclose(result, [expected], rtol=1e-6)


def test_log_delta_prior_zero_prior():
    # prior=0 should use 0.1 as denominator
    current = np.array([5.0])
    prior = np.array([0.0])
    result = _log_delta_prior(current, prior)
    expected = np.log(0.01 + 5.0 / 0.1)
    np.testing.assert_allclose(result, [expected], rtol=1e-6)


def test_log_delta_post_formula():
    current = np.array([100.0])
    post = np.array([95.0])
    result = _log_delta_post(current, post)
    expected = np.log(0.01 + 95.0 / 100.0)
    np.testing.assert_allclose(result, [expected], rtol=1e-6)


def test_log_delta_post_zero_current():
    current = np.array([0.0])
    post = np.array([5.0])
    result = _log_delta_post(current, post)
    expected = np.log(0.01 + 5.0 / 0.1)
    np.testing.assert_allclose(result, [expected], rtol=1e-6)

# ---------------------------------------------------------------------------
# BMP feature transformer shapes
# ---------------------------------------------------------------------------

def _make_bmp_df(n=20, with_post=True):
    rng = np.random.default_rng(42)
    rows = {}
    for col in BMP_ANALYTES:
        rows[col] = rng.uniform(50, 200, n)
        rows[f"{col}_prior"] = rng.uniform(50, 200, n)
        if with_post:
            rows[f"{col}_post"] = rng.uniform(50, 200, n)
    return pd.DataFrame(rows)


def test_bmp_retro_transformer_shape():
    df = _make_bmp_df(n=30)
    t = BMPFeatureTransformer(mode="retrospective")
    t.fit(df)
    out = t.transform(df)
    # 8 current + 8 prior_delta + 8 post_delta + 3 prior_PC + 3 post_PC + 3 all_PC = 33
    assert out.shape == (30, 33), f"Expected (30, 33), got {out.shape}"


def test_bmp_realtime_transformer_shape():
    df = _make_bmp_df(n=30, with_post=False)
    t = BMPFeatureTransformer(mode="realtime")
    t.fit(df)
    out = t.transform(df)
    # 8 current + 8 prior_delta + 3 all_PC + 3 prior_PC = 22
    assert out.shape == (30, 22), f"Expected (30, 22), got {out.shape}"


def test_bmp_transformer_no_nans_on_valid_input():
    df = _make_bmp_df(n=20)
    t = BMPFeatureTransformer(mode="retrospective")
    t.fit(df)
    out = t.transform(df)
    assert not np.any(np.isnan(out)), "Unexpected NaNs in transformer output"

# ---------------------------------------------------------------------------
# CBC feature transformer shapes
# ---------------------------------------------------------------------------

def _make_cbc_df(n=20, with_post=True):
    rng = np.random.default_rng(42)
    rows = {}
    for col in CBC_ANALYTES:
        rows[col] = rng.uniform(1, 20, n)
        rows[f"{col}_prior"] = rng.uniform(1, 20, n)
        if with_post:
            rows[f"{col}_post"] = rng.uniform(1, 20, n)
    return pd.DataFrame(rows)


def test_cbc_retro_transformer_shape():
    df = _make_cbc_df(n=20)
    t = CBCFeatureTransformer(mode="retrospective")
    t.fit(df)
    out = t.transform(df)
    # 3 current + 3 prior_delta + 3 post_delta + 3 prior_PC + 3 post_PC + 3 all_PC = 18
    assert out.shape == (20, 18), f"Expected (20, 18), got {out.shape}"


def test_cbc_realtime_transformer_shape():
    df = _make_cbc_df(n=20, with_post=False)
    t = CBCFeatureTransformer(mode="realtime")
    t.fit(df)
    out = t.transform(df)
    # 3 current + 3 prior_delta + 3 all_PC = 9
    assert out.shape == (20, 9), f"Expected (20, 9), got {out.shape}"

# ---------------------------------------------------------------------------
# Wide CSV preprocessing
# ---------------------------------------------------------------------------

def test_preprocess_bmp_wide():
    csv_path = DATA_DIR / "bmp_test_wide.csv"
    if not csv_path.exists():
        pytest.skip("bmp_test_wide.csv not present")
    df = pd.read_csv(csv_path)
    result = preprocess_bmp_data(df)
    for col in BMP_ANALYTES:
        assert col in result.columns, f"Missing column: {col}"


def test_preprocess_cbc_wide():
    csv_path = DATA_DIR / "cbc_test_wide.csv"
    if not csv_path.exists():
        pytest.skip("cbc_test_wide.csv not present")
    df = pd.read_csv(csv_path)
    result = preprocess_cbc_data(df)
    for col in CBC_ANALYTES:
        assert col in result.columns, f"Missing column: {col}"


def test_preprocess_bmp_long():
    csv_path = DATA_DIR / "bmp_test_long.csv"
    if not csv_path.exists():
        pytest.skip("bmp_test_long.csv not present")
    df = pd.read_csv(csv_path)
    result = preprocess_bmp_data(df)
    for col in BMP_ANALYTES:
        assert col in result.columns, f"Missing column: {col}"
