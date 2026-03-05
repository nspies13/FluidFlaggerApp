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



_CLASSIFIER_GRID = {
    "n_estimators":      [1000],
    "max_depth":         [8, 16],
    "learning_rate":     [0.1, 0.3],
    "reg_lambda":        [0.1, 1.0],
}

_REGRESSOR_GRID = {
    "n_estimators":      [1000],
    "max_depth":         [8, 16],
    "learning_rate":     [0.1, 0.3],
    "alpha":             [0.1, 0.3],
}


def _tune_lgbm_classifier(X, y) -> tuple:
    """
    Parallel grid search for LGBMClassifier scored with log loss + auROC.
    Returns (unfitted LGBMClassifier with best params, cv_metrics dict).
    """
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
    search = GridSearchCV(
        LGBMClassifier(random_state=123, verbose=-1),
        _CLASSIFIER_GRID,
        scoring={"log_loss": "neg_log_loss", "roc_auc": "roc_auc"},
        refit="log_loss",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X, y)
    best_idx = search.best_index_
    cv_log_loss = -search.cv_results_["mean_test_log_loss"][best_idx]
    cv_auroc    =  search.cv_results_["mean_test_roc_auc"][best_idx]
    best = search.best_params_
    print(f"  Best log-loss: {cv_log_loss:.4f}  auROC: {cv_auroc:.4f}  params: {best}")
    metrics = {"cv_log_loss": round(cv_log_loss, 4), "cv_auroc": round(cv_auroc, 4)}
    return LGBMClassifier(**best, random_state=123, verbose=-1), metrics


def _tune_lgbm_regressor(X, y) -> tuple:
    """
    Parallel grid search for LGBMRegressor with pseudo-Huber objective.
    Returns (unfitted LGBMRegressor with best params, cv_metrics dict).
    """
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import GridSearchCV, KFold

    cv = KFold(n_splits=3, shuffle=True, random_state=123)
    search = GridSearchCV(
        LGBMRegressor(objective="huber", random_state=123, verbose=-1),
        _REGRESSOR_GRID,
        scoring="neg_mean_absolute_error",
        refit=True,
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X, y)
    cv_mae = -search.best_score_
    best = search.best_params_
    print(f"  Best MAE: {cv_mae:.4f}  params: {best}")
    metrics = {"cv_mae": round(cv_mae, 4)}
    return LGBMRegressor(**best, objective="huber", random_state=123, verbose=-1), metrics


def _make_classifier(n_rows: int, X, y) -> tuple:
    """
    Build a classifier via parallel grid search. Returns (clf, cv_metrics dict).
    Falls back to RandomForest if lightgbm is unavailable (no grid search for RF).
    """
    if n_rows < _RF_THRESHOLD:
        from sklearn.model_selection import cross_validate, StratifiedKFold
        clf = RandomForestClassifier(n_estimators=500, random_state=123, n_jobs=-1)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
        results = cross_validate(clf, X, y, cv=cv,
                                 scoring={"log_loss": "neg_log_loss", "roc_auc": "roc_auc"},
                                 n_jobs=-1)
        metrics = {
            "cv_log_loss": round(-results["test_log_loss"].mean(), 4),
            "cv_auroc":    round( results["test_roc_auc"].mean(),  4),
        }
        return clf, metrics
    import importlib.util
    if importlib.util.find_spec("lightgbm") is None:
        warnings.warn("lightgbm not installed; falling back to RandomForest.")
        from sklearn.model_selection import cross_validate, StratifiedKFold
        clf = RandomForestClassifier(n_estimators=500, random_state=123, n_jobs=-1)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
        results = cross_validate(clf, X, y, cv=cv,
                                 scoring={"log_loss": "neg_log_loss", "roc_auc": "roc_auc"},
                                 n_jobs=-1)
        metrics = {
            "cv_log_loss": round(-results["test_log_loss"].mean(), 4),
            "cv_auroc":    round( results["test_roc_auc"].mean(),  4),
        }
        return clf, metrics

    return _tune_lgbm_classifier(X, y)


def _make_regressor(X, y) -> tuple:
    """
    Build a regressor via parallel grid search. Returns (reg, cv_metrics dict).
    Falls back to GradientBoostingRegressor if lightgbm is unavailable.
    """
    import importlib.util
    if importlib.util.find_spec("lightgbm") is None:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_validate, KFold
        reg = GradientBoostingRegressor(n_estimators=500, random_state=123)
        cv_kf = KFold(n_splits=3, shuffle=True, random_state=123)
        results = cross_validate(reg, X, y, cv=cv_kf, scoring="neg_mean_absolute_error", n_jobs=-1)
        return reg, {"cv_mae": round(-results["test_score"].mean(), 4)}

    return _tune_lgbm_regressor(X, y)

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
    Uses parallel grid search (GridSearchCV) for hyperparameter tuning.
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
        print("  Retrospective classifier (grid search)")
        retro_clf, retro_metrics = _make_classifier(n_rows, X_retro_t, y)
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
            "cv_metrics": retro_metrics,
        })

        # Realtime model
        print("  Realtime classifier (grid search)")
        realtime_clf, realtime_metrics = _make_classifier(n_rows, X_realtime_t, y)
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
            "cv_metrics": realtime_metrics,
        })

        # Mix ratio regression (retrospective features, contaminated rows only)
        contam_mask = train_df["target"] == 1
        if contam_mask.sum() > 10:
            X_mix = train_df.loc[contam_mask, retro_cols]
            y_mix = train_df.loc[contam_mask, "mix_ratio"].fillna(0).values
            mix_transformer = BMPFeatureTransformer(mode="retrospective")
            X_mix_t = mix_transformer.fit_transform(X_mix, y_mix)
            print("  Mix ratio regressor (grid search)")
            mix_reg, mix_metrics = _make_regressor(X_mix_t, y_mix)
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
                "cv_metrics": mix_metrics,
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
    Uses parallel grid search (GridSearchCV) for hyperparameter tuning.
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

    print("  Retrospective classifier (grid search)")
    retro_clf, retro_metrics = _make_classifier(n_rows, X_retro_t, y)
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
        "cv_metrics": retro_metrics,
    })

    print("  Realtime classifier (grid search)")
    realtime_clf, realtime_metrics = _make_classifier(n_rows, X_realtime_t, y)
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
        "cv_metrics": realtime_metrics,
    })

    if train_mix:
        contam_mask = train_df["target"] == 1
        if contam_mask.sum() > 10:
            X_mix = train_df.loc[contam_mask, retro_cols]
            y_mix = train_df.loc[contam_mask, "mix_ratio"].values
            mix_transformer = CBCFeatureTransformer(mode="retrospective")
            X_mix_t = mix_transformer.fit_transform(X_mix, y_mix)
            print("  Mix ratio regressor (grid search)")
            mix_reg, mix_metrics = _make_regressor(X_mix_t, y_mix)
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
                "cv_metrics": mix_metrics,
            })

    return models

# ---------------------------------------------------------------------------
# Save / upload / CLI
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


def save_cv_metrics(models: list[dict], output_path: str | Path) -> Path:
    """Save a cross-validation performance summary CSV for all trained models."""
    rows = []
    for m in models:
        metrics = m.get("cv_metrics", {})
        row = {
            "panel": m["panel"],
            "fluid": m["fluid"],
            "type": m["type"],
            "task": m["task"],
        }
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    path = Path(output_path)
    df.to_csv(path, index=False)
    return path


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


def main():
    """CLI entry point.

    Usage:
        python -m src.train --panel bmp --template data/bmp_template.csv
        python -m src.train --panel cbc --template data/cbc_template.csv --upload
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train FluidFlagger models")
    parser.add_argument("--panel", choices=["bmp", "cbc"], required=True)
    parser.add_argument("--template", required=True, help="Wide-format training CSV")
    parser.add_argument("--fluids", default=None, help="BMP fluid concentrations TSV (uses built-in if omitted)")
    parser.add_argument("--output", default="models/", help="Output directory for .joblib files")
    parser.add_argument("--upload", action="store_true", help="Upload models to HF Hub after training")
    parser.add_argument("--repo", default=None, help="HF Hub model repo ID (overrides HF_MODEL_REPO env var)")
    args = parser.parse_args()

    template_df = pd.read_csv(args.template)
    print(f"Loaded template: {len(template_df)} rows")

    if args.panel == "bmp":
        fluids_df = pd.read_csv(args.fluids, sep=None, engine="python") if args.fluids else None
        models = train_bmp_models(template_df, fluids_df)
    else:
        models = train_cbc_models(template_df)

    paths = save_models(models, args.output)
    print(f"Saved {len(paths)} model files to {args.output}")

    metrics_path = Path(args.output) / "cross_validation_performance_summary.csv"
    save_cv_metrics(models, metrics_path)
    print(f"Saved CV metrics: {metrics_path}")

    if args.upload:
        from .model_loader import HF_REPO_ID
        repo = args.repo or HF_REPO_ID
        upload_models_to_hub(paths, repo_id=repo)


if __name__ == "__main__":
    main()
