---
title: FluidFlagger
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.36.0"
app_file: app.py
pinned: false
---

# FluidFlagger

**IV-fluid contamination detection for BMP and CBC laboratory results.**

Detects when blood specimens have been contaminated with intravenous fluids
(Normal Saline, Lactated Ringer's, Dextrose solutions, etc.) using machine
learning models trained on simulated contamination data.

## Features

- **Predict** — upload a CSV (wide or long format) or enter values manually
- **Train** — train custom models on your own institution's data
- **Review** — step through predictions and add human labels for QA

## Panels

| Panel | Analytes |
|-------|----------|
| BMP | Sodium, Chloride, Potassium, CO₂, BUN, Creatinine, Calcium, Glucose |
| CBC | Hemoglobin, WBC, Platelets |

## API

The app exposes a REST API at `/api/`:

```
GET  /api/health
POST /api/bmp/predict          — JSON body → predictions
POST /api/bmp/predict_stream   — newline-delimited JSON
POST /api/cbc/predict
POST /api/cbc/predict_stream
```

Example:
```bash
curl -X POST https://nspies13-fluidflagger.hf.space/api/bmp/predict \
  -H "Content-Type: application/json" \
  -d '[{"sodium": 154, "chloride": 154, "potassium_plas": 2.1, "co2_totl": 18,
        "bun": 5, "creatinine": 0.6, "calcium": 8.2, "glucose": 98}]'
```

## Input Format

**Wide format** (one row per blood draw):

| sodium | chloride | potassium_plas | co2_totl | bun | creatinine | calcium | glucose |
|--------|----------|----------------|----------|-----|------------|---------|---------|
| 138 | 102 | 4.1 | 24 | 12 | 0.9 | 9.2 | 95 |

Prior and post values (`sodium_prior`, `sodium_post`, etc.) enable retrospective models.

**Long format** (one row per analyte result) requires columns:
`PATIENT_ID`, `DRAWN_DT_TM`, `TASK_ASSAY`, `RESULT_VALUE`

## Training Custom Models

1. Prepare a wide-format CSV with analyte columns (+ `_prior`/`_post` for retrospective)
2. Upload in the **Train** tab or run the CLI:

```bash
python -m scripts.train_and_upload \
  --panel bmp \
  --template my_training_data.csv \
  --output models/ \
  --upload \
  --repo yourname/fluidflagger-models
```

## Deployment

This Space is automatically redeployed on every push to `main` via GitHub Actions.
Model files are stored separately in the
[fluidflagger-models](https://huggingface.co/nspies13/fluidflagger-models) HF Hub
model repository and downloaded lazily on first use.
