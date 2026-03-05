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

import os
import warnings
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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


def _classifier_learning_rate(n_rows: int) -> float:
    """Scale learning rate down as dataset grows: more data → finer steps."""
    import math
    lr = 0.3 * math.sqrt(_RF_THRESHOLD / max(n_rows, _RF_THRESHOLD))
    return round(max(0.05, min(0.5, lr)), 4)


def _tune_lgbm_classifier(X, y, n_rows: int, n_trials: int) -> "LGBMClassifier":
    """
    Bayesian HPO for LGBMClassifier scored with log loss.
    Returns an unfitted LGBMClassifier with the best hyperparameters.
    """
    import optuna
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    base_lr = _classifier_learning_rate(n_rows)
    lr_lo = max(0.02, base_lr * 0.3)
    lr_hi = min(0.9, base_lr * 3.0)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 2000),
            "max_depth":         trial.suggest_int("max_depth", 4, 12),
            "learning_rate":     trial.suggest_float("learning_rate", lr_lo, lr_hi, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 300),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 123,
            "verbose": -1,
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
        scores = cross_val_score(
            LGBMClassifier(**params), X, y,
            cv=cv, scoring="neg_log_loss",
        )
        return scores.mean()  # maximize (neg log loss → minimize log loss)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=123),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  Best log-loss: {-study.best_value:.4f}  params: {study.best_params}")
    return LGBMClassifier(**study.best_params, random_state=123, verbose=-1)


def _tune_lgbm_regressor(X, y, n_trials: int) -> "LGBMRegressor":
    """
    Bayesian HPO for LGBMRegressor with pseudo-Huber objective.
    The CV metric is neg-MAE (a robust proxy); the model trains with objective='huber'.
    Returns an unfitted LGBMRegressor with the best hyperparameters.
    """
    import optuna
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import KFold, cross_val_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 2000),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.1, 0.95, log=True),
            "num_leaves":       trial.suggest_int("num_leaves", 20, 200),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "objective":        "huber",
            "alpha":            trial.suggest_float("alpha", 0.05, 0.5),
            "random_state": 123,
            "verbose": -1,
        }
        cv = KFold(n_splits=3, shuffle=True, random_state=123)
        scores = cross_val_score(
            LGBMRegressor(**params), X, y,
            cv=cv, scoring="neg_mean_absolute_error",
        )
        return scores.mean()  # maximize neg-MAE

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=123),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  Best MAE: {-study.best_value:.4f}  params: {study.best_params}")
    return LGBMRegressor(**study.best_params, random_state=123, verbose=-1)


def _make_classifier(n_rows: int, X=None, y=None, n_trials: int = 0):
    """
    Build a classifier. If X/y are provided and n_rows >= RF_THRESHOLD and
    optuna is available, runs Bayesian HPO (n_trials). Otherwise uses defaults.
    """
    if n_rows < _RF_THRESHOLD:
        return RandomForestClassifier(n_estimators=500, random_state=123, n_jobs=-1)
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        warnings.warn("lightgbm not installed; falling back to RandomForest.")
        return RandomForestClassifier(n_estimators=500, random_state=123, n_jobs=-1)

    if n_trials > 0 and X is not None and y is not None:
        try:
            return _tune_lgbm_classifier(X, y, n_rows, n_trials)
        except ImportError:
            pass  # optuna not available

    lr = _classifier_learning_rate(n_rows)
    return LGBMClassifier(
        n_estimators=1000, max_depth=10, learning_rate=lr,
        min_child_samples=32, reg_lambda=0.1, random_state=123,
        verbose=-1,
    )


def _make_regressor(X=None, y=None, n_trials: int = 0):
    """
    Build a regressor. If X/y are provided and optuna is available, runs
    Bayesian HPO with pseudo-Huber objective (n_trials). Otherwise uses defaults.
    """
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(n_estimators=500, random_state=123)

    if n_trials > 0 and X is not None and y is not None:
        try:
            return _tune_lgbm_regressor(X, y, n_trials)
        except ImportError:
            pass  # optuna not available

    return LGBMRegressor(
        n_estimators=1000, learning_rate=0.8, objective="huber",
        alpha=0.1, random_state=123, verbose=-1,
    )

# ---------------------------------------------------------------------------
# BMP training
# ---------------------------------------------------------------------------

def train_bmp_models(
    template_df: pd.DataFrame,
    fluids_df: Optional[pd.DataFrame] = None,
    seed: int = 123,
    n_trials: int = 30,
) -> list[dict]:
    """
    Train all BMP contamination models: 9 fluids × 2 timings (Realtime, Retrospective).

    n_trials: Bayesian HPO trials per model (0 = skip HPO, use defaults).
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

        # Pre-transform once for HPO (transformer has no hyperparams to tune)
        retro_transformer = BMPFeatureTransformer(mode="retrospective")
        X_retro_t = retro_transformer.fit_transform(X_retro, y)

        realtime_transformer = BMPFeatureTransformer(mode="realtime")
        X_realtime_t = realtime_transformer.fit_transform(X_realtime, y)

        # Retrospective model
        print(f"  Retrospective classifier ({n_trials} HPO trials)")
        retro_clf = _make_classifier(n_rows, X=X_retro_t, y=y, n_trials=n_trials)
        retro_pipe = Pipeline([
            ("features", BMPFeatureTransformer(mode="retrospective")),
            ("clf", retro_clf),
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
        print(f"  Realtime classifier ({n_trials} HPO trials)")
        realtime_clf = _make_classifier(n_rows, X=X_realtime_t, y=y, n_trials=n_trials)
        realtime_pipe = Pipeline([
            ("features", BMPFeatureTransformer(mode="realtime")),
            ("clf", realtime_clf),
        ])
        realtime_pipe.fit(X_realtime, y)
        models.append({
            "pipeline": realtime_pipe,
            "type": "Realtime",
            "fluid": fluid_name,
            "panel": "bmp",
            "task": "classification",
        })

        # Mix ratio regression (retrospective features, contaminated rows only)
        contam_mask = train_df["target"] == 1
        if contam_mask.sum() > 10:
            X_mix = train_df.loc[contam_mask, retro_cols]
            y_mix = train_df.loc[contam_mask, "mix_ratio"].fillna(0).values
            mix_transformer = BMPFeatureTransformer(mode="retrospective")
            X_mix_t = mix_transformer.fit_transform(X_mix, y_mix)
            print(f"  Mix ratio regressor ({n_trials} HPO trials)")
            mix_reg = _make_regressor(X=X_mix_t, y=y_mix, n_trials=n_trials)
            mix_pipe = Pipeline([
                ("features", BMPFeatureTransformer(mode="retrospective")),
                ("reg", mix_reg),
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
    n_trials: int = 30,
) -> list[dict]:
    """
    Train CBC contamination models: Realtime + Retrospective + optional mix ratio.

    n_trials: Bayesian HPO trials per model (0 = skip HPO, use defaults).
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

    # Pre-transform once for HPO
    retro_transformer = CBCFeatureTransformer(mode="retrospective")
    X_retro_t = retro_transformer.fit_transform(X_retro, y)

    realtime_transformer = CBCFeatureTransformer(mode="realtime")
    X_realtime_t = realtime_transformer.fit_transform(X_realtime, y)

    models = []

    print(f"  Retrospective classifier ({n_trials} HPO trials)")
    retro_clf = _make_classifier(n_rows, X=X_retro_t, y=y, n_trials=n_trials)
    retro_pipe = Pipeline([
        ("features", CBCFeatureTransformer(mode="retrospective")),
        ("clf", retro_clf),
    ])
    retro_pipe.fit(X_retro, y)
    models.append({
        "pipeline": retro_pipe,
        "type": "Retrospective",
        "fluid": "CBC",
        "panel": "cbc",
        "task": "classification",
    })

    print(f"  Realtime classifier ({n_trials} HPO trials)")
    realtime_clf = _make_classifier(n_rows, X=X_realtime_t, y=y, n_trials=n_trials)
    realtime_pipe = Pipeline([
        ("features", CBCFeatureTransformer(mode="realtime")),
        ("clf", realtime_clf),
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
            mix_transformer = CBCFeatureTransformer(mode="retrospective")
            X_mix_t = mix_transformer.fit_transform(X_mix, y_mix)
            print(f"  Mix ratio regressor ({n_trials} HPO trials)")
            mix_reg = _make_regressor(X=X_mix_t, y=y_mix, n_trials=n_trials)
            mix_pipe = Pipeline([
                ("features", CBCFeatureTransformer(mode="retrospective")),
                ("reg", mix_reg),
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
