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


def _available_cores() -> int:
    """Return CPU cores available for training: all physical cores minus one."""
    return max(1, (os.cpu_count() or 1) - 1)


_LR_THRESHOLD = 500_000  # learning rate: 0.1 above, 0.3 below


def _learning_rate(n_rows: int) -> float:
    return 0.1 if n_rows > _LR_THRESHOLD else 0.3


def _make_classifier(n_rows: int, X, y, n_jobs: int = 1) -> tuple:
    """
    Fit a classifier with fixed hyperparameters and evaluate via 5-fold CV.
    Returns (unfitted clf with fixed params, cv_metrics dict).
    Uses RandomForest below _RF_THRESHOLD rows, LightGBM above.
    Falls back to RandomForest if lightgbm is unavailable.
    """
    from sklearn.model_selection import cross_validate, StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    import importlib.util
    use_lgbm = n_rows >= _RF_THRESHOLD and importlib.util.find_spec("lightgbm") is not None
    if n_rows >= _RF_THRESHOLD and not use_lgbm:
        warnings.warn("lightgbm not installed; falling back to RandomForest.")

    if use_lgbm:
        from lightgbm import LGBMClassifier
        lr = _learning_rate(n_rows)
        clf = LGBMClassifier(
            n_estimators=1000, max_depth=16, learning_rate=lr,
            reg_lambda=1, random_state=123, verbose=-1,
        )
        print(f"  LGBMClassifier  lr={lr}  n_rows={n_rows}")
    else:
        clf = RandomForestClassifier(
            n_estimators=1000, max_depth=16, random_state=123, n_jobs=n_jobs,
        )
        print(f"  RandomForestClassifier  n_rows={n_rows}")

    results = cross_validate(clf, X, y, cv=cv,
                             scoring={"log_loss": "neg_log_loss", "roc_auc": "roc_auc"},
                             n_jobs=n_jobs)
    metrics = {
        "cv_log_loss": round(-results["test_log_loss"].mean(), 4),
        "cv_auroc":    round( results["test_roc_auc"].mean(),  4),
    }
    print(f"  CV log-loss: {metrics['cv_log_loss']}  auROC: {metrics['cv_auroc']}")
    return clf, metrics


def _make_regressor(n_rows: int, X, y, n_jobs: int = 1) -> tuple:
    """
    Fit a regressor with fixed hyperparameters and evaluate via 5-fold CV.
    Returns (unfitted reg with fixed params, cv_metrics dict).
    Falls back to GradientBoostingRegressor if lightgbm is unavailable.
    """
    from sklearn.model_selection import cross_validate, KFold
    cv = KFold(n_splits=5, shuffle=True, random_state=123)

    import importlib.util
    if importlib.util.find_spec("lightgbm") is None:
        from sklearn.ensemble import GradientBoostingRegressor
        reg = GradientBoostingRegressor(
            n_estimators=1000, max_depth=16, learning_rate=_learning_rate(n_rows),
            alpha=0.1, random_state=123,
        )
        results = cross_validate(reg, X, y, cv=cv,
                                 scoring="neg_mean_absolute_error", n_jobs=n_jobs)
        cv_mae = round(-results["test_score"].mean(), 4)
        print(f"  GradientBoostingRegressor  CV MAE: {cv_mae}")
        return reg, {"cv_mae": cv_mae}

    from lightgbm import LGBMRegressor
    lr = _learning_rate(n_rows)
    reg = LGBMRegressor(
        n_estimators=1000, max_depth=16, learning_rate=lr,
        alpha=0.1, objective="huber", random_state=123, verbose=-1,
    )
    print(f"  LGBMRegressor  lr={lr}  n_rows={n_rows}")
    results = cross_validate(reg, X, y, cv=cv,
                             scoring="neg_mean_absolute_error", n_jobs=n_jobs)
    cv_mae = round(-results["test_score"].mean(), 4)
    print(f"  CV MAE: {cv_mae}")
    return reg, {"cv_mae": cv_mae}


# ---------------------------------------------------------------------------
# Per-model training helpers (top-level so joblib can pickle them)
# ---------------------------------------------------------------------------

def _train_bmp_clf(mode: str, fluid_name: str, X_raw, X_t, y, n_rows: int, n_jobs: int) -> dict:
    """Train one BMP classification pipeline (realtime or retrospective)."""
    model_type = "Realtime" if mode == "realtime" else "Retrospective"
    print(f"  [{fluid_name}] {model_type} classifier")
    clf, metrics = _make_classifier(n_rows, X_t, y, n_jobs=n_jobs)
    pipe = Pipeline([("features", BMPFeatureTransformer(mode=mode)), ("clf", clf)])
    pipe.fit(X_raw, y)
    return {"pipeline": pipe, "type": model_type, "fluid": fluid_name,
            "panel": "bmp", "task": "classification", "cv_metrics": metrics}


def _train_bmp_mix(fluid_name: str, X_mix, X_mix_t, y_mix, n_rows: int, n_jobs: int) -> dict:
    """Train one BMP mix-ratio regression pipeline."""
    print(f"  [{fluid_name}] Mix ratio regressor")
    reg, metrics = _make_regressor(n_rows, X_mix_t, y_mix, n_jobs=n_jobs)
    pipe = Pipeline([("features", BMPFeatureTransformer(mode="retrospective")), ("reg", reg)])
    pipe.fit(X_mix, y_mix)
    return {"pipeline": pipe, "type": "Retrospective", "fluid": fluid_name,
            "panel": "bmp", "task": "mix_ratio", "cv_metrics": metrics}


def _train_cbc_clf(mode: str, X_raw, X_t, y, n_rows: int, n_jobs: int) -> dict:
    """Train one CBC classification pipeline (realtime or retrospective)."""
    model_type = "Realtime" if mode == "realtime" else "Retrospective"
    print(f"  [CBC] {model_type} classifier")
    clf, metrics = _make_classifier(n_rows, X_t, y, n_jobs=n_jobs)
    pipe = Pipeline([("features", CBCFeatureTransformer(mode=mode)), ("clf", clf)])
    pipe.fit(X_raw, y)
    return {"pipeline": pipe, "type": model_type, "fluid": "CBC",
            "panel": "cbc", "task": "classification", "cv_metrics": metrics}


def _train_cbc_mix(X_mix, X_mix_t, y_mix, n_rows: int, n_jobs: int) -> dict:
    """Train one CBC mix-ratio regression pipeline."""
    print("  [CBC] Mix ratio regressor")
    reg, metrics = _make_regressor(n_rows, X_mix_t, y_mix, n_jobs=n_jobs)
    pipe = Pipeline([("features", CBCFeatureTransformer(mode="retrospective")), ("reg", reg)])
    pipe.fit(X_mix, y_mix)
    return {"pipeline": pipe, "type": "Retrospective", "fluid": "CBC",
            "panel": "cbc", "task": "mix_ratio", "cv_metrics": metrics}

# ---------------------------------------------------------------------------
# BMP training
# ---------------------------------------------------------------------------

def train_bmp_fluid(
    template: pd.DataFrame,
    fluid_row: pd.Series,
    n_rows: int,
    seed: int = 123,
    n_inner_jobs: Optional[int] = None,
) -> list[dict]:
    """Train all models (retrospective clf, realtime clf, mix ratio reg) for one BMP fluid.

    Returns a list of model dicts. Called per-fluid from train_bmp_models and from the UI
    training loop so that progress can be reported between fluids.
    """
    if n_inner_jobs is None:
        n_inner_jobs = max(1, _available_cores() // 3)

    fluid_name = fluid_row["fluid"]
    print(f"Training BMP models for fluid: {fluid_name} ({n_rows} rows)")

    retro_cols = (
        BMP_ANALYTES
        + [f"{c}_prior" for c in BMP_ANALYTES]
        + [f"{c}_post" for c in BMP_ANALYTES]
    )
    realtime_cols = BMP_ANALYTES + [f"{c}_prior" for c in BMP_ANALYTES]

    train_df = make_binary_training_data_bmp(template, fluid_row, seed=seed)
    y = train_df["target"].values

    X_retro = train_df[retro_cols]
    X_realtime = train_df[realtime_cols]

    retro_transformer = BMPFeatureTransformer(mode="retrospective")
    X_retro_t = retro_transformer.fit_transform(X_retro, y)

    realtime_transformer = BMPFeatureTransformer(mode="realtime")
    X_realtime_t = realtime_transformer.fit_transform(X_realtime, y)

    tasks = [
        joblib.delayed(_train_bmp_clf)("retrospective", fluid_name, X_retro, X_retro_t, y, n_rows, n_inner_jobs),
        joblib.delayed(_train_bmp_clf)("realtime", fluid_name, X_realtime, X_realtime_t, y, n_rows, n_inner_jobs),
    ]

    contam_mask = train_df["target"] == 1
    if contam_mask.sum() > 10:
        X_mix = train_df.loc[contam_mask, retro_cols]
        y_mix = train_df.loc[contam_mask, "mix_ratio"].fillna(0).values
        mix_transformer = BMPFeatureTransformer(mode="retrospective")
        X_mix_t = mix_transformer.fit_transform(X_mix, y_mix)
        tasks.append(joblib.delayed(_train_bmp_mix)(fluid_name, X_mix, X_mix_t, y_mix, n_rows, n_inner_jobs))

    N = _available_cores()
    return joblib.Parallel(n_jobs=min(len(tasks), N))(tasks)


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

    N = _available_cores()
    n_inner = max(1, N // 3)
    print(f"Available cores: {N}  |  parallel models per fluid: 3  |  inner n_jobs: {n_inner}")

    models = []
    for _, fluid_row in fluids_df.iterrows():
        models.extend(train_bmp_fluid(template, fluid_row, n_rows, seed=seed, n_inner_jobs=n_inner))
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

    # Divide available cores across the 3 model types trained in parallel.
    N = _available_cores()
    n_parallel = 3
    n_inner = max(1, N // n_parallel)
    print(f"Available cores: {N}  |  parallel models: {n_parallel}  |  inner n_jobs: {n_inner}")

    tasks = [
        joblib.delayed(_train_cbc_clf)("retrospective", X_retro, X_retro_t, y, n_rows, n_inner),
        joblib.delayed(_train_cbc_clf)("realtime", X_realtime, X_realtime_t, y, n_rows, n_inner),
    ]

    if train_mix:
        contam_mask = train_df["target"] == 1
        if contam_mask.sum() > 10:
            X_mix = train_df.loc[contam_mask, retro_cols]
            y_mix = train_df.loc[contam_mask, "mix_ratio"].values
            mix_transformer = CBCFeatureTransformer(mode="retrospective")
            X_mix_t = mix_transformer.fit_transform(X_mix, y_mix)
            tasks.append(joblib.delayed(_train_cbc_mix)(X_mix, X_mix_t, y_mix, n_rows, n_inner))

    return joblib.Parallel(n_jobs=min(len(tasks), N))(tasks)

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
