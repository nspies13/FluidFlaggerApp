"""
Retrain CBC realtime, retrospective, and mix-ratio models on the full
cbc_training_template.csv and save them to models/.

Loads only the analyte + prior + post columns the CBC pipeline needs (memory
lean) and reuses src.train.train_cbc_models so the simulation/feature/HPO logic
is identical to the production training path.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from src.features import CBC_ANALYTES
from src.train import train_cbc_models, save_models, save_cv_metrics

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "data" / "training_templates" / "cbc_training_template.csv"
OUTDIR = REPO / "models"

USECOLS = (
    CBC_ANALYTES
    + [f"{c}_prior" for c in CBC_ANALYTES]
    + [f"{c}_post" for c in CBC_ANALYTES]
)


def main() -> None:
    t0 = time.time()
    print(f"Loading template (cols={USECOLS}) ...", flush=True)
    template_df = pd.read_csv(TEMPLATE, usecols=USECOLS)
    print(f"  loaded {len(template_df):,} rows in {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    models = train_cbc_models(template_df, seed=123, train_mix=True)
    print(f"Training finished in {(time.time()-t1)/60:.1f} min", flush=True)

    for m in models:
        print(f"  {m['panel']}_{m['fluid']}_{m['type']}/{m['task']}: "
              f"{m.get('cv_metrics')}", flush=True)

    paths = save_models(models, OUTDIR)
    print(f"Saved {len(paths)} model files to {OUTDIR}", flush=True)
    save_cv_metrics(models, OUTDIR / "cbc_retrain_cv_summary.csv")
    print(f"Total wall time: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
