"""
Tests for inference.py.

These tests use locally trained minimal models (not HF Hub) to verify that
the prediction pipeline works end-to-end without network access.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import BMP_ANALYTES, CBC_ANALYTES
from src.inference import label_pred_class, make_bmp_predictions, make_cbc_predictions
from src.model_loader import cache_model, clear_cache, model_key
from src.train import train_bmp_models, train_cbc_models

DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# label_pred_class
# ---------------------------------------------------------------------------

def test_label_contaminated():
    # Spec: contaminated when p >= 0.50.
    assert label_pred_class(0.5) == "Contaminated"
    assert label_pred_class(0.76) == "Contaminated"
    assert label_pred_class(1.0) == "Contaminated"


def test_label_real():
    assert label_pred_class(0.0) == "Real"
    assert label_pred_class(0.49) == "Real"


def test_label_none():
    assert label_pred_class(None) is None
    assert label_pred_class(float("nan")) is None

# ---------------------------------------------------------------------------
# Fixtures: tiny trained models loaded into cache
# ---------------------------------------------------------------------------

def _make_synthetic_bmp_df(n=60):
    rng = np.random.default_rng(0)
    data = {}
    for col in BMP_ANALYTES:
        data[col] = rng.uniform(50, 200, n)
        data[f"{col}_prior"] = rng.uniform(50, 200, n)
        data[f"{col}_post"] = rng.uniform(50, 200, n)
    return pd.DataFrame(data)


def _make_synthetic_cbc_df(n=60):
    rng = np.random.default_rng(0)
    data = {}
    for col in CBC_ANALYTES:
        data[col] = rng.uniform(1, 20, n)
        data[f"{col}_prior"] = rng.uniform(1, 20, n)
        data[f"{col}_post"] = rng.uniform(1, 20, n)
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def bmp_models_in_cache():
    """Train tiny BMP models for NS only and load into cache."""
    clear_cache()
    from src.simulate import load_fluid_concentrations
    fluids = load_fluid_concentrations()
    ns_fluids = fluids[fluids["fluid"] == "NS"]
    df = _make_synthetic_bmp_df(n=60)
    models = train_bmp_models(df, fluids_df=ns_fluids, regression_cases_per_ratio=2)
    for m in models:
        cache_model(model_key(m), m)
    yield models
    clear_cache()


@pytest.fixture(scope="module")
def cbc_models_in_cache():
    """Train tiny CBC models and load into cache."""
    clear_cache()
    df = _make_synthetic_cbc_df(n=60)
    models = train_cbc_models(df, regression_cases_per_ratio=2)
    for m in models:
        cache_model(model_key(m), m)
    yield models
    clear_cache()

# ---------------------------------------------------------------------------
# BMP prediction tests
# ---------------------------------------------------------------------------

def test_bmp_predictions_returns_dataframe(bmp_models_in_cache):
    df = _make_synthetic_bmp_df(n=10)
    result = make_bmp_predictions(df, selected_fluids=["NS"])
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 10


def test_bmp_predictions_has_prob_columns(bmp_models_in_cache):
    df = _make_synthetic_bmp_df(n=5)
    result = make_bmp_predictions(df, selected_fluids=["NS"])
    prob_cols = [c for c in result.columns if c.startswith("prob_")]
    assert len(prob_cols) >= 1, "Expected at least one prob_ column"


def test_bmp_predictions_has_pred_columns(bmp_models_in_cache):
    df = _make_synthetic_bmp_df(n=5)
    result = make_bmp_predictions(df, selected_fluids=["NS"])
    pred_cols = [c for c in result.columns if c.startswith("pred_")]
    assert len(pred_cols) >= 1


def test_bmp_pred_labels_valid(bmp_models_in_cache):
    df = _make_synthetic_bmp_df(n=10)
    result = make_bmp_predictions(df, selected_fluids=["NS"])
    pred_cols = [c for c in result.columns if c.startswith("pred_")]
    valid_labels = {"Real", "Contaminated", None}
    for col in pred_cols:
        for val in result[col]:
            assert val in valid_labels, f"Unexpected label: {val}"


def test_bmp_mix_ratios_are_bounded(bmp_models_in_cache):
    result = make_bmp_predictions(_make_synthetic_bmp_df(n=10), selected_fluids=["NS"])
    mix_columns = [column for column in result if column.startswith("mix_ratio_")]
    assert mix_columns
    for column in mix_columns:
        assert result[column].dropna().between(0.0, 0.50).all()


def test_bmp_na_rows_give_na_predictions(bmp_models_in_cache):
    df = _make_synthetic_bmp_df(n=3)
    # Set one row's prior values to NaN → realtime should still work, retro → NA
    df.loc[0, [f"{c}_prior" for c in BMP_ANALYTES]] = np.nan
    result = make_bmp_predictions(df, selected_fluids=["NS"])
    retro_pred_cols = [c for c in result.columns if "Retrospective" in c and c.startswith("pred_")]
    if retro_pred_cols:
        assert pd.isna(result.loc[0, retro_pred_cols[0]]) or result.loc[0, retro_pred_cols[0]] is None


def test_bmp_no_models_raises(monkeypatch):
    clear_cache()
    monkeypatch.setattr("src.model_loader._cache", {})
    # patch hf download to always fail
    import src.model_loader as ml
    monkeypatch.setattr(ml, "_download_from_hub", lambda key: None)
    df = _make_synthetic_bmp_df(n=2)
    with pytest.raises(RuntimeError, match="No BMP models"):
        make_bmp_predictions(df)

# ---------------------------------------------------------------------------
# CBC prediction tests
# ---------------------------------------------------------------------------

def test_cbc_predictions_returns_dataframe(cbc_models_in_cache):
    df = _make_synthetic_cbc_df(n=10)
    result = make_cbc_predictions(df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 10


def test_cbc_predictions_has_expected_columns(cbc_models_in_cache):
    df = _make_synthetic_cbc_df(n=5)
    result = make_cbc_predictions(df)
    assert any("prob_CBC" in c for c in result.columns)
    assert any("pred_CBC" in c for c in result.columns)


def test_cbc_mix_ratio_is_bounded(cbc_models_in_cache):
    result = make_cbc_predictions(_make_synthetic_cbc_df(n=10))
    assert result["mix_ratio_CBC"].dropna().between(0.0, 0.50).all()


def test_cbc_no_models_raises(monkeypatch):
    clear_cache()
    import src.model_loader as ml
    monkeypatch.setattr(ml, "_download_from_hub", lambda key: None)
    df = _make_synthetic_cbc_df(n=2)
    with pytest.raises(RuntimeError, match="No CBC models"):
        make_cbc_predictions(df)
