"""
Training pipelines for BMP and CBC contamination detection models.

Ports train_bmp_models.R and train_cbc_models.R.

Each trained model is a joblib-serialized dict:
    {
        "pipeline": sklearn Pipeline (feature transformer + classifier/regressor),
        "type": "Realtime" | "Retrospective",
        "fluid": str,      # BMP: fluid name, CBC: "CBC"
        "panel": "bmp" | "cbc",
        "task": "classification" | "mix_ratio",
    }
"""

from __future__ import annotations

import io
import os
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

from .features import (
    BMP_ANALYTES,
    CBC_ANALYTES,
    BMPFeatureTransformer,
    CBCFeatureTransformer,
)
from .simulate import (
    get_fluid_concentrations,
    make_binary_training_data_bmp,
    make_binary_training_data_cbc,
)

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

_RF_THRESHOLD = 200_000  # use RF below, LightGBM above


def _make_classifier(n_rows: int):
    if n_rows < _RF_THRESHOLD:
        return RandomForestClassifier(n_estimators=500, random_state=123, n_jobs=-1)
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=1000, max_depth=10, learning_rate=0.3,
            min_child_samples=32, reg_lambda=0.1, random_state=123,
            verbose=-1,
        )
    except ImportError:
        warnings.warn("lightgbm not installed; falling back to RandomForest.")
        return RandomForestClassifier(n_estimators=500, random_state=123, n_jobs=-1)


def _make_regressor(n_rows: int):
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=1000, learning_rate=0.5, random_state=123,
            verbose=-1,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(n_estimators=500, random_state=123)

# ---------------------------------------------------------------------------
# BMP training
# ---------------------------------------------------------------------------

def train_bmp_models(
    template_df: pd.DataFrame,
    fluids_df: Optional[pd.DataFrame] = None,
    seed: int = 123,
) -> list[dict]:
    """
    Train all BMP contamination models: 9 fluids × 2 timings (Realtime, Retrospective).

    Returns a list of model dicts compatible with model_loader.py.
    """
    if fluids_df is None:
        fluids_df = get_fluid_concentrations()

    all_required = (
        BMP_ANALYTES
        + [f"{c}_prior" for c in BMP_ANALYTES]
        + [f"{c}_post" for c in BMP_ANALYTES]
    )
    template = template_df.copy()
    for col in all_required:
        if col in template.columns:
            template[col] = pd.to_numeric(template[col], errors="coerce")
    template = template.dropna(subset=[c for c in all_required if c in template.columns])
    template = template[(template[[c for c in all_required if c in template.columns]] > 0).all(axis=1)]

    n_rows = len(template)
    models = []

    for _, fluid_row in fluids_df.iterrows():
        fluid_name = fluid_row["fluid"]
        print(f"Training BMP models for fluid: {fluid_name} ({n_rows} rows)")

        train_df = make_binary_training_data_bmp(template, fluid_row, seed=seed)
        y = train_df["target"].values

        retro_cols = (
            BMP_ANALYTES
            + [f"{c}_prior" for c in BMP_ANALYTES]
            + [f"{c}_post" for c in BMP_ANALYTES]
        )
        realtime_cols = BMP_ANALYTES + [f"{c}_prior" for c in BMP_ANALYTES]

        X_retro = train_df[retro_cols]
        X_realtime = train_df[realtime_cols]

        # Retrospective model
        retro_pipe = Pipeline([
            ("features", BMPFeatureTransformer(mode="retrospective")),
            ("clf", _make_classifier(n_rows)),
        ])
        retro_pipe.fit(X_retro, y)
        models.append({
            "pipeline": retro_pipe,
            "type": "Retrospective",
            "fluid": fluid_name,
            "panel": "bmp",
            "task": "classification",
        })

        # Realtime model
        realtime_pipe = Pipeline([
            ("features", BMPFeatureTransformer(mode="realtime")),
            ("clf", _make_classifier(n_rows)),
        ])
        realtime_pipe.fit(X_realtime, y)
        models.append({
            "pipeline": realtime_pipe,
            "type": "Realtime",
            "fluid": fluid_name,
            "panel": "bmp",
            "task": "classification",
        })

        # Mix ratio regression (retrospective features)
        contam_mask = train_df["target"] == 1
        if contam_mask.sum() > 10:
            X_mix = train_df.loc[contam_mask, retro_cols]
            y_mix = train_df.loc[contam_mask, "mix_ratio"].fillna(0).values
            mix_pipe = Pipeline([
                ("features", BMPFeatureTransformer(mode="retrospective")),
                ("reg", _make_regressor(n_rows)),
            ])
            mix_pipe.fit(X_mix, y_mix)
            models.append({
                "pipeline": mix_pipe,
                "type": "Retrospective",
                "fluid": fluid_name,
                "panel": "bmp",
                "task": "mix_ratio",
            })

    return models

# ---------------------------------------------------------------------------
# CBC training
# ---------------------------------------------------------------------------

def train_cbc_models(
    template_df: pd.DataFrame,
    seed: int = 123,
    train_mix: bool = True,
) -> list[dict]:
    """
    Train CBC contamination models: Realtime + Retrospective + optional mix ratio.

    Returns a list of model dicts compatible with model_loader.py.
    """
    all_cols = (
        CBC_ANALYTES
        + [f"{c}_prior" for c in CBC_ANALYTES]
        + [f"{c}_post" for c in CBC_ANALYTES]
    )
    template = template_df.copy()
    for col in all_cols:
        if col in template.columns:
            template[col] = pd.to_numeric(template[col], errors="coerce")
    template = template.dropna(subset=[c for c in all_cols if c in template.columns])
    template = template[(template[[c for c in all_cols if c in template.columns]] > 0).all(axis=1)]

    n_rows = len(template)
    print(f"Training CBC models ({n_rows} rows)")

    train_df = make_binary_training_data_cbc(template, seed=seed)
    y = train_df["target"].values

    retro_cols = (
        CBC_ANALYTES
        + [f"{c}_prior" for c in CBC_ANALYTES]
        + [f"{c}_post" for c in CBC_ANALYTES]
    )
    realtime_cols = CBC_ANALYTES + [f"{c}_prior" for c in CBC_ANALYTES]

    X_retro = train_df[retro_cols]
    X_realtime = train_df[realtime_cols]

    models = []

    retro_pipe = Pipeline([
        ("features", CBCFeatureTransformer(mode="retrospective")),
        ("clf", _make_classifier(n_rows)),
    ])
    retro_pipe.fit(X_retro, y)
    models.append({
        "pipeline": retro_pipe,
        "type": "Retrospective",
        "fluid": "CBC",
        "panel": "cbc",
        "task": "classification",
    })

    realtime_pipe = Pipeline([
        ("features", CBCFeatureTransformer(mode="realtime")),
        ("clf", _make_classifier(n_rows)),
    ])
    realtime_pipe.fit(X_realtime, y)
    models.append({
        "pipeline": realtime_pipe,
        "type": "Realtime",
        "fluid": "CBC",
        "panel": "cbc",
        "task": "classification",
    })

    if train_mix:
        contam_mask = train_df["target"] == 1
        if contam_mask.sum() > 10:
            X_mix = train_df.loc[contam_mask, retro_cols]
            y_mix = train_df.loc[contam_mask, "mix_ratio"].values
            mix_pipe = Pipeline([
                ("features", CBCFeatureTransformer(mode="retrospective")),
                ("reg", _make_regressor(n_rows)),
            ])
            mix_pipe.fit(X_mix, y_mix)
            models.append({
                "pipeline": mix_pipe,
                "type": "Retrospective",
                "fluid": "CBC",
                "panel": "cbc",
                "task": "mix_ratio",
            })

    return models

# ---------------------------------------------------------------------------
# Save / upload helpers
# ---------------------------------------------------------------------------

def model_key(model_dict: dict) -> str:
    """Canonical filename stem for a model, e.g. 'bmp_NS_Realtime'."""
    panel = model_dict["panel"]
    fluid = model_dict["fluid"]
    typ = model_dict["type"]
    task = model_dict["task"]
    if task == "mix_ratio":
        return f"{panel}_{fluid}_mix_ratio"
    return f"{panel}_{fluid}_{typ}"


def save_models(models: list[dict], output_dir: str | Path, compress: int = 3) -> list[Path]:
    """Save model dicts as joblib files. Returns list of saved paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for m in models:
        key = model_key(m)
        path = output_dir / f"{key}.joblib"
        joblib.dump(m, path, compress=compress)
        print(f"Saved: {path}")
        paths.append(path)
    return paths


def upload_models_to_hub(
    model_paths: list[Path],
    repo_id: str,
    token: Optional[str] = None,
) -> None:
    """Upload saved model files to a HuggingFace Hub model repository."""
    from huggingface_hub import HfApi
    api = HfApi()
    token = token or os.environ.get("HF_TOKEN")

    # Ensure repo exists
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)
    except Exception as e:
        print(f"Warning: could not create repo ({e}); assuming it exists.")

    for path in model_paths:
        print(f"Uploading {path.name} to {repo_id} ...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
    print("Upload complete.")
