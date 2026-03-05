"""
CLI entry point for training models and optionally uploading to HF Hub.

Usage:
    python -m scripts.train_and_upload --panel bmp --template data/bmp_template.csv
    python -m scripts.train_and_upload --panel cbc --template data/cbc_template.csv --upload
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import save_models, train_bmp_models, train_cbc_models, upload_models_to_hub


def main():
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

    if args.upload:
        from src.model_loader import HF_REPO_ID
        repo = args.repo or HF_REPO_ID
        upload_models_to_hub(paths, repo_id=repo)


if __name__ == "__main__":
    main()
