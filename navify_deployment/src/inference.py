"""
BMP-only prediction functions for the Navify deployment image.

Realtime models run on current + prior BMP values. Retrospective classifiers and
mix-ratio regressors run only when complete post-specimen values are supplied.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .features import BMP_ANALYTES
from .model_loader import BMP_FLUIDS, bmp_classification_key, bmp_mix_ratio_key, get_model

_CONTAM_THRESHOLD = 0.25


def label_pred_class(prob: float) -> str | None:
    if prob is None or (isinstance(prob, float) and np.isnan(prob)):
        return None
    if prob >= _CONTAM_THRESHOLD:
        return "Contaminated"
    return "Real"


def _predict_proba_1(pipeline, x: pd.DataFrame) -> np.ndarray:
    proba = pipeline.predict_proba(x)
    if proba.ndim == 2:
        clf = pipeline[-1] if hasattr(pipeline, "__getitem__") else pipeline
        classes = list(getattr(clf, "classes_", [0, 1]))
        idx = classes.index(1) if 1 in classes else 1
        return proba[:, idx]
    return proba


def _chunk_indices(n: int, chunk_size: int):
    for start in range(0, n, chunk_size):
        yield slice(start, min(start + chunk_size, n))


def _predict_chunked(
    pipeline,
    x: pd.DataFrame,
    na_mask: np.ndarray,
    n: int,
    chunk_size: int,
    predict_proba: bool,
) -> np.ndarray:
    out = np.full(n, np.nan)
    valid_idx = np.where(~na_mask)[0]
    if len(valid_idx) == 0:
        return out

    chunks = []
    x_valid = x.iloc[valid_idx]
    for sl in _chunk_indices(len(x_valid), chunk_size):
        batch = x_valid.iloc[sl]
        pred = _predict_proba_1(pipeline, batch) if predict_proba else pipeline.predict(batch)
        chunks.append(pred)
    out[valid_idx] = np.concatenate(chunks).round(3)
    return out


def make_bmp_predictions(
    df: pd.DataFrame,
    selected_fluids: Optional[list[str]] = None,
    timings: tuple[str, ...] = ("Realtime", "Retrospective"),
    chunk_size: int = 10_000,
) -> pd.DataFrame:
    """Run loaded BMP models and append prediction, mix-ratio, and aggregate columns."""
    fluids = selected_fluids if selected_fluids else list(BMP_FLUIDS)
    timings = tuple(timings)

    realtime_cols = BMP_ANALYTES + [f"{col}_prior" for col in BMP_ANALYTES]
    retro_cols = realtime_cols + [f"{col}_post" for col in BMP_ANALYTES]

    rt_missing = [col for col in realtime_cols if col not in df.columns]
    if rt_missing:
        raise ValueError(f"Missing realtime input columns: {', '.join(rt_missing)}")

    if "Retrospective" in timings:
        retro_missing = [col for col in retro_cols if col not in df.columns]
        if retro_missing:
            raise ValueError(f"Missing retrospective input columns: {', '.join(retro_missing)}")

    num_na_realtime = df[realtime_cols].isna().any(axis=1).to_numpy(dtype=bool)
    num_na_retro = (
        df[retro_cols].isna().any(axis=1).to_numpy(dtype=bool)
        if "Retrospective" in timings
        else np.ones(len(df), dtype=bool)
    )

    prob_cols: dict[str, np.ndarray] = {}
    pred_cols: dict[str, np.ndarray] = {}
    mix_cols: dict[str, np.ndarray] = {}

    for fluid in fluids:
        for timing in timings:
            key = bmp_classification_key(fluid, timing)
            model_dict = get_model(key)
            if model_dict is None:
                raise RuntimeError(f"Required model is not loaded: {key}")

            pipeline = model_dict["pipeline"]
            if timing == "Realtime":
                x = df[realtime_cols].copy()
                na_mask = num_na_realtime
            else:
                x = df[retro_cols].copy()
                na_mask = num_na_retro

            probs = _predict_chunked(
                pipeline, x, na_mask, len(df), chunk_size, predict_proba=True
            )
            preds = np.full(len(df), None, dtype=object)
            valid_idx = np.where(~np.isnan(probs))[0]
            preds[valid_idx] = [label_pred_class(prob) for prob in probs[valid_idx]]

            col_name = f"{fluid}_{timing}"
            prob_cols[f"prob_{col_name}"] = probs
            pred_cols[f"pred_{col_name}"] = preds

        if "Retrospective" in timings:
            mix_key = bmp_mix_ratio_key(fluid)
            mix_dict = get_model(mix_key)
            if mix_dict is None:
                raise RuntimeError(f"Required model is not loaded: {mix_key}")
            mix_cols[f"mix_ratio_{fluid}"] = _predict_chunked(
                mix_dict["pipeline"],
                df[retro_cols].copy(),
                num_na_retro,
                len(df),
                chunk_size,
                predict_proba=False,
            )

    result = df.copy()
    for key, values in prob_cols.items():
        result[key] = values
    for key, values in pred_cols.items():
        result[key] = values
    for key, values in mix_cols.items():
        result[key] = values

    rt_pred_cols = [col for col in pred_cols if "_Realtime" in col]
    retro_pred_cols = [col for col in pred_cols if "_Retrospective" in col]
    rt_no_lr = [col for col in rt_pred_cols if "_LR_" not in col]
    retro_no_lr = [col for col in retro_pred_cols if "_LR_" not in col]

    if rt_pred_cols:
        result["any_realtime_pred"] = result[rt_no_lr].apply(
            lambda row: any(value == "Contaminated" for value in row if value is not None),
            axis=1,
        )
        result["any_realtime_pred_with_LR"] = result[rt_pred_cols].apply(
            lambda row: any(value == "Contaminated" for value in row if value is not None),
            axis=1,
        )
    if retro_pred_cols:
        result["any_retrospective_pred"] = result[retro_no_lr].apply(
            lambda row: any(value == "Contaminated" for value in row if value is not None),
            axis=1,
        )
        result["any_retrospective_pred_with_LR"] = result[retro_pred_cols].apply(
            lambda row: any(value == "Contaminated" for value in row if value is not None),
            axis=1,
        )

    rt_prob_cols = [col for col in prob_cols if "_Realtime" in col]
    retro_prob_cols = [col for col in prob_cols if "_Retrospective" in col]
    rt_prob_no_lr = [col for col in rt_prob_cols if "_LR_" not in col]
    retro_prob_no_lr = [col for col in retro_prob_cols if "_LR_" not in col]

    if rt_prob_no_lr:
        result["max_realtime_prob"] = result[rt_prob_no_lr].max(axis=1)
        rt_frame = result[rt_prob_no_lr]
        result["max_prob_fluid_realtime"] = (
            rt_frame.idxmax(axis=1).str.split("_").str[1]
        )
    if rt_prob_cols:
        result["max_realtime_prob_with_LR"] = result[rt_prob_cols].max(axis=1)
    if retro_prob_no_lr:
        result["max_retrospective_prob"] = result[retro_prob_no_lr].max(axis=1)
        retro_frame = result[retro_prob_no_lr]
        result["max_prob_fluid_retrospective"] = (
            retro_frame.idxmax(axis=1).str.split("_").str[1]
        )
    if retro_prob_cols:
        result["max_retrospective_prob_with_LR"] = result[retro_prob_cols].max(axis=1)

    if mix_cols:
        mix_no_lr = {key: values for key, values in mix_cols.items() if "_LR" not in key}
        if mix_no_lr:
            result["max_mix_ratio"] = np.column_stack(list(mix_no_lr.values())).max(axis=1)
        result["max_mix_ratio_with_LR"] = np.column_stack(list(mix_cols.values())).max(axis=1)

    return result
