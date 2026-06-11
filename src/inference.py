"""
BMP and CBC prediction functions.

Ports makeBmpPredictions() from bmp_helpers.R and
makeCbcPredictions() from cbc_helpers.R.

Predicted labels use an equivocal zone between 0.5 and 0.75

"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .features import BMP_ANALYTES, CBC_ANALYTES
from .model_loader import (
    bmp_classification_key,
    bmp_mix_ratio_key,
    cbc_classification_key,
    cbc_mix_ratio_key,
    get_model,
)

# ---------------------------------------------------------------------------
# Prediction labelling
# ---------------------------------------------------------------------------

_CONTAM_THRESHOLD = 0.75
_EQUIVOCAL_LOW = 0.5 


def label_pred_class(prob: float) -> str:
    """Convert a contamination probability to a class label."""
    if prob is None or (isinstance(prob, float) and np.isnan(prob)):
        return None
    if prob > _CONTAM_THRESHOLD:
        return "Contaminated"
    if prob >= _EQUIVOCAL_LOW:
        return "Equivocal"
    return "Real"


def _predict_proba_1(pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return P(class=1) for each row. Works for both classifiers and sklearn Pipelines."""
    proba = pipeline.predict_proba(X)
    if proba.ndim == 2:
        clf = pipeline[-1] if hasattr(pipeline, "__getitem__") else pipeline
        classes = getattr(clf, "classes_", [0, 1])
        idx = list(classes).index(1) if 1 in list(classes) else 1
        return proba[:, idx]
    return proba


def _chunk_indices(n: int, chunk_size: int = 10_000):
    for start in range(0, n, chunk_size):
        yield slice(start, min(start + chunk_size, n))


def _predict_chunked(
    pipeline,
    X: pd.DataFrame,
    na_mask: np.ndarray,
    n: int,
    chunk_size: int,
    predict_proba: bool,
) -> np.ndarray:
    """Run chunked inference over valid (non-NA) rows and return a full-length array."""
    out = np.full(n, np.nan)
    valid_idx = np.where(~na_mask)[0]
    if len(valid_idx) == 0:
        return out
    X_valid = X.iloc[valid_idx]
    chunks = []
    for sl in _chunk_indices(len(X_valid), chunk_size):
        batch = X_valid.iloc[sl]
        chunks.append(_predict_proba_1(pipeline, batch) if predict_proba else pipeline.predict(batch))
    out[valid_idx] = np.concatenate(chunks).round(3)
    return out

# ---------------------------------------------------------------------------
# BMP predictions
# ---------------------------------------------------------------------------

def make_bmp_predictions(
    df: pd.DataFrame,
    selected_fluids: Optional[list[str]] = None,
    timings: tuple[str, ...] = ("Realtime", "Retrospective"),
    chunk_size: int = 10_000,
) -> pd.DataFrame:
    """
    Run all available BMP models against df, returning a result DataFrame
    with probability, class, and aggregate columns appended.

    selected_fluids: restrict to these fluid names (default: all available)
    """
    from .simulate import get_fluid_names
    all_fluids = get_fluid_names()
    fluids = selected_fluids if selected_fluids else all_fluids

    # Work out which input columns are available
    realtime_cols = BMP_ANALYTES + [f"{c}_prior" for c in BMP_ANALYTES]
    retro_cols = realtime_cols + [f"{c}_post" for c in BMP_ANALYTES]

    available_realtime = [c for c in realtime_cols if c in df.columns]
    available_retro = [c for c in retro_cols if c in df.columns]

    # Count missing required values per row (no post = no retro)
    rt_required = BMP_ANALYTES + [f"{c}_prior" for c in BMP_ANALYTES]
    retro_required = rt_required + [f"{c}_post" for c in BMP_ANALYTES]

    num_na_realtime = df[[c for c in rt_required if c in df.columns]].isna().any(axis=1).astype(int)
    num_na_retro = df[[c for c in retro_required if c in df.columns]].isna().any(axis=1).astype(int)

    prob_cols: dict[str, np.ndarray] = {}
    pred_cols: dict[str, np.ndarray] = {}
    mix_cols: dict[str, np.ndarray] = {}

    for fluid in fluids:
        for timing in timings:
            key = bmp_classification_key(fluid, timing)
            model_dict = get_model(key)
            if model_dict is None:
                continue

            pipeline = model_dict["pipeline"]
            col_name = f"{fluid}_{timing}"

            if timing == "Realtime":
                X = df[[c for c in realtime_cols if c in df.columns]].copy()
                na_mask = num_na_realtime.values.astype(bool)
            else:
                X = df[[c for c in retro_cols if c in df.columns]].copy()
                na_mask = num_na_retro.values.astype(bool)

            probs = _predict_chunked(pipeline, X, na_mask, len(df), chunk_size, predict_proba=True)
            preds = np.full(len(df), None, dtype=object)
            valid_idx = np.where(~np.isnan(probs))[0]
            preds[valid_idx] = [label_pred_class(p) for p in probs[valid_idx]]

            prob_cols[f"prob_{col_name}"] = probs
            pred_cols[f"pred_{col_name}"] = preds

        # Mix ratio (retrospective only)
        mix_key = bmp_mix_ratio_key(fluid)
        mix_dict = get_model(mix_key)
        if mix_dict is not None:
            pipeline = mix_dict["pipeline"]
            X = df[[c for c in retro_cols if c in df.columns]].copy()
            na_mask = num_na_retro.values.astype(bool)
            mix_cols[f"mix_ratio_{fluid}"] = _predict_chunked(
                pipeline, X, na_mask, len(df), chunk_size, predict_proba=False
            )

    if not prob_cols and not pred_cols:
        raise RuntimeError(
            "No BMP models are available. "
            "Please upload custom models or ensure the default model repository is accessible."
        )

    result = df.copy()
    for k, v in prob_cols.items():
        result[k] = v
    for k, v in pred_cols.items():
        result[k] = v
    for k, v in mix_cols.items():
        result[k] = v

    # Aggregate columns: any fluid flagged as Contaminated with and without LR
    rt_pred_cols = [c for c in pred_cols if "_Realtime" in c]
    retro_pred_cols = [c for c in pred_cols if "_Retrospective" in c]
    rt_no_lr = [c for c in rt_pred_cols if "_LR_" not in c]
    retro_no_lr = [c for c in retro_pred_cols if "_LR_" not in c]

    if rt_pred_cols:
        result["any_realtime_pred"] = result[rt_no_lr].apply(
            lambda row: any(v == "Contaminated" for v in row if v is not None), axis=1
        )
        result["any_realtime_pred_with_LR"] = result[rt_pred_cols].apply(
            lambda row: any(v == "Contaminated" for v in row if v is not None), axis=1
        )
    if retro_pred_cols:
        result["any_retrospective_pred"] = result[retro_no_lr].apply(
            lambda row: any(v == "Contaminated" for v in row if v is not None), axis=1
        )
        result["any_retrospective_pred_with_LR"] = result[retro_pred_cols].apply(
            lambda row: any(v == "Contaminated" for v in row if v is not None), axis=1
        )

    # Max prob across fluids with and without LR
    rt_prob_cols = [c for c in prob_cols if "_Realtime" in c]
    retro_prob_cols = [c for c in prob_cols if "_Retrospective" in c]
    rt_prob_no_lr = [c for c in rt_prob_cols if "_LR_" not in c]
    retro_prob_no_lr = [c for c in retro_prob_cols if "_LR_" not in c]

    if rt_prob_no_lr:
        result["max_realtime_prob"] = result[rt_prob_no_lr].max(axis=1)
        _rt = result[rt_prob_no_lr]
        result["max_prob_fluid_realtime"] = _rt[_rt.notna().any(axis=1)].idxmax(axis=1).str.split("_").str[1]
    if rt_prob_cols:
        result["max_realtime_prob_with_LR"] = result[rt_prob_cols].max(axis=1)
    if retro_prob_no_lr:
        result["max_retrospective_prob"] = result[retro_prob_no_lr].max(axis=1)
        _retro = result[retro_prob_no_lr]
        result["max_prob_fluid_retrospective"] = _retro[_retro.notna().any(axis=1)].idxmax(axis=1).str.split("_").str[1]
    if retro_prob_cols:
        result["max_retrospective_prob_with_LR"] = result[retro_prob_cols].max(axis=1)

    if mix_cols:
        mix_no_lr = {k: v for k, v in mix_cols.items() if "_LR" not in k}
        if mix_no_lr:
            result["max_mix_ratio"] = np.column_stack(list(mix_no_lr.values())).max(axis=1)
        result["max_mix_ratio_with_LR"] = np.column_stack(list(mix_cols.values())).max(axis=1)

    return result

# ---------------------------------------------------------------------------
# CBC predictions
# ---------------------------------------------------------------------------

def make_cbc_predictions(
    df: pd.DataFrame,
    timings: tuple[str, ...] = ("Realtime", "Retrospective"),
    chunk_size: int = 10_000,
) -> pd.DataFrame:
    """
    Run all available CBC models against df.
    """
    realtime_cols = CBC_ANALYTES + [f"{c}_prior" for c in CBC_ANALYTES]
    retro_cols = realtime_cols + [f"{c}_post" for c in CBC_ANALYTES]

    rt_required = CBC_ANALYTES + [f"{c}_prior" for c in CBC_ANALYTES]
    retro_required = rt_required + [f"{c}_post" for c in CBC_ANALYTES]

    num_na_realtime = df[[c for c in rt_required if c in df.columns]].isna().any(axis=1).astype(int)
    num_na_retro = df[[c for c in retro_required if c in df.columns]].isna().any(axis=1).astype(int)

    prob_cols: dict[str, np.ndarray] = {}
    pred_cols: dict[str, np.ndarray] = {}

    for timing in timings:
        key = cbc_classification_key(timing)
        model_dict = get_model(key)
        if model_dict is None:
            continue

        pipeline = model_dict["pipeline"]

        if timing == "Realtime":
            X = df[[c for c in realtime_cols if c in df.columns]].copy()
            na_mask = num_na_realtime.values.astype(bool)
        else:
            X = df[[c for c in retro_cols if c in df.columns]].copy()
            na_mask = num_na_retro.values.astype(bool)

        probs = _predict_chunked(pipeline, X, na_mask, len(df), chunk_size, predict_proba=True)
        preds = np.full(len(df), None, dtype=object)
        valid_idx = np.where(~np.isnan(probs))[0]
        preds[valid_idx] = [label_pred_class(p) for p in probs[valid_idx]]

        prob_cols[f"prob_CBC_{timing}"] = probs
        pred_cols[f"pred_CBC_{timing}"] = preds

    # Mix ratio
    mix_dict = get_model(cbc_mix_ratio_key())
    mix_ratio: Optional[np.ndarray] = None
    if mix_dict is not None:
        pipeline = mix_dict["pipeline"]
        X = df[[c for c in retro_cols if c in df.columns]].copy()
        na_mask = num_na_retro.values.astype(bool)
        mix_ratio = _predict_chunked(
            pipeline, X, na_mask, len(df), chunk_size, predict_proba=False
        )

    if not prob_cols and not pred_cols:
        raise RuntimeError(
            "No CBC models are available. "
            "Please upload custom models or ensure the default model repository is accessible."
        )

    result = df.copy()
    for k, v in prob_cols.items():
        result[k] = v
    for k, v in pred_cols.items():
        result[k] = v
    if mix_ratio is not None:
        result["mix_ratio_CBC"] = mix_ratio

    # Aggregate
    rt_pred = [c for c in pred_cols if "_Realtime" in c]
    retro_pred = [c for c in pred_cols if "_Retrospective" in c]

    if rt_pred:
        result["any_realtime_pred"] = result[rt_pred].apply(
            lambda row: any(v == "Contaminated" for v in row if v is not None), axis=1
        )
    if retro_pred:
        result["any_retrospective_pred"] = result[retro_pred].apply(
            lambda row: any(v == "Contaminated" for v in row if v is not None), axis=1
        )

    rt_prob = [c for c in prob_cols if "_Realtime" in c]
    retro_prob = [c for c in prob_cols if "_Retrospective" in c]
    if rt_prob:
        result["max_realtime_prob"] = result[rt_prob].max(axis=1)
    if retro_prob:
        result["max_retrospective_prob"] = result[retro_prob].max(axis=1)

    return result
