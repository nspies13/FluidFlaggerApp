---
title: FluidFlagger
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# FluidFlagger

**IV-fluid contamination detection for BMP and CBC laboratory results.**

Detects when blood specimens have been contaminated with intravenous fluids
(Normal Saline, Lactated Ringer's, Dextrose solutions, etc.) using machine
learning models trained on simulated contamination data.

## Web UI

The Gradio interface has five tabs:

- **Predict** — upload a CSV (wide or long format) or enter values manually
- **Train** — train custom models on your own institution's data
- **Review** — step through predictions and add human labels for QA
- **Validate** — upload reviewed predictions to explore Sens, Spec, PPV, NPV,
  F1, ROC/PR AUCs, an interactive threshold-dependent 2×2 table, and calibration;
  its lower section includes the model Explain (SHAP) tools
- **Self Test** — practice identifying contaminated specimens with simulated cases

## Panels

| Panel | Analytes |
|-------|----------|
| BMP | Sodium, Chloride, Potassium, CO₂, BUN, Creatinine, Calcium, Glucose |
| CBC | Hemoglobin, WBC, Platelets |

---

## Docker

The multi-target Dockerfile produces three images:

| Target | Purpose | Port |
|--------|---------|------|
| `inference` | Standalone prediction API (for navify Algorithm Suite) | 8080 |
| `nomodel` | Gradio web UI (no models baked in) | 7860 |
| `train` | Model training job | — |

### Building & running the inference container

```bash
# Build
docker build --target inference -t fluidflagger-inference .

# Run
docker run -d -p 8080:8080 --name ff fluidflagger-inference

# Push to Docker Hub
docker tag fluidflagger-inference nspies13/fluidflagger-inference:1.0.0
docker push nspies13/fluidflagger-inference:1.0.0
```

### Health probes

```bash
# Liveness — always returns 200 if the process is running
curl http://localhost:8080/health/live
# → {"status": "ok"}

# Readiness — returns 200 after models are loaded, 503 before
curl http://localhost:8080/health/ready
# → {"status": "ok"}
```

---

## Prediction API

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | BMP prediction (primary calculation endpoint) |
| POST | `/api/bmp/predict` | BMP prediction (alias) |
| POST | `/api/cbc/predict` | CBC prediction |
| GET | `/health/live` | Liveness probe |
| GET | `/health/ready` | Readiness probe |

All prediction endpoints accept `application/json` and return `application/json`.
They also accept `text/csv` and `text/tab-separated-values` and will respond in the same format.

### BMP — JSON input

Send a single object or an array of objects. Each object represents one blood draw.

**Current specimen fields** (always required):

| Field | Description | Example |
|-------|-------------|---------|
| `sodium` | Sodium (mEq/L) | 140 |
| `chloride` | Chloride (mEq/L) | 102 |
| `potassium_plas` | Potassium (mEq/L) | 4.1 |
| `co2_totl` | Total CO₂ (mEq/L) | 24 |
| `bun` | Blood Urea Nitrogen (mg/dL) | 15 |
| `creatinine` | Creatinine (mg/dL) | 1.0 |
| `calcium` | Calcium (mg/dL) | 9.2 |
| `glucose` | Glucose (mg/dL) | 95 |

**Prior specimen fields** (required for Realtime predictions) — the most recent previous result for each analyte, suffixed with `_prior` (e.g. `sodium_prior`, `chloride_prior`).

**Post specimen fields** (required for Retrospective predictions) — the next result collected after the specimen in question, suffixed with `_post` (e.g. `sodium_post`, `chloride_post`). The retrospective model uses current + prior + post values and is more accurate than realtime.

#### Example request

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '[
    {
      "sodium": 148, "chloride": 144, "potassium_plas": 2.7, "co2_totl": 16,
      "bun": 40, "creatinine": 2.42, "calcium": 6.9, "glucose": 181,
      "sodium_prior": 132, "chloride_prior": 83, "potassium_plas_prior": 4.5,
      "co2_totl_prior": 24, "bun_prior": 49, "creatinine_prior": 3.62,
      "calcium_prior": 10.3, "glucose_prior": 135,
      "sodium_post": 135, "chloride_post": 94, "potassium_plas_post": 3.4,
      "co2_totl_post": 28, "bun_post": 32, "creatinine_post": 1.75,
      "calcium_post": 9.0, "glucose_post": 133
    }
  ]'
```

#### Example response

The response echoes all input fields plus prediction columns for each fluid type:

```json
[
  {
    "sodium": 148,
    "chloride": 144,
    "potassium_plas": 2.7,
    "...": "... (all input fields echoed back) ...",

    "prob_NS_Realtime": 0.982,
    "pred_NS_Realtime": "Contaminated",
    "prob_NS_Retrospective": 0.995,
    "pred_NS_Retrospective": "Contaminated",
    "mix_ratio_NS": 0.43,

    "prob_LR_Realtime": 0.871,
    "pred_LR_Realtime": "Contaminated",
    "prob_LR_Retrospective": 0.912,
    "pred_LR_Retrospective": "Contaminated",
    "mix_ratio_LR": 0.38,

    "prob_D5W_Realtime": 0.654,
    "pred_D5W_Realtime": "Equivocal",

    "...": "... (one prob/pred pair per fluid × timing) ...",

    "any_realtime_pred": true,
    "any_realtime_pred_with_LR": true,
    "any_retrospective_pred": true,
    "any_retrospective_pred_with_LR": true,
    "max_realtime_prob": 0.982,
    "max_realtime_prob_with_LR": 0.982,
    "max_retrospective_prob": 0.995,
    "max_retrospective_prob_with_LR": 0.995,
    "max_prob_fluid_realtime": "NS",
    "max_prob_fluid_retrospective": "NS",
    "max_mix_ratio": 0.43,
    "max_mix_ratio_with_LR": 0.43
  }
]
```

**Key output fields:**

| Field pattern | Description |
|---------------|-------------|
| `prob_{fluid}_{timing}` | Contamination probability (0–1) for a specific fluid and timing |
| `pred_{fluid}_{timing}` | Classification: `"Contaminated"` (>0.75), `"Equivocal"` (0.5–0.75), or `"Real"` (<0.5) |
| `mix_ratio_{fluid}` | Estimated IV-fluid mix ratio (retrospective only) |
| `any_realtime_pred` | `true` if any fluid (excluding LR) flagged as Contaminated in realtime |
| `any_retrospective_pred` | Same for retrospective |
| `max_realtime_prob` | Highest contamination probability across all fluids (excluding LR) |
| `max_prob_fluid_realtime` | Which fluid had the highest realtime probability |

The `_with_LR` variants include Lactated Ringer's in the aggregation.

### CBC — JSON input

Same structure, but with CBC analytes:

| Field | Description | Example |
|-------|-------------|---------|
| `Hgb` | Hemoglobin (g/dL) | 14.0 |
| `Plt` | Platelets (×10³/µL) | 250 |
| `WBC` | White Blood Cells (×10³/µL) | 7.5 |

**Prior fields** (required for Realtime): `Hgb_prior`, `Plt_prior`, `WBC_prior`

**Post fields** (required for Retrospective): `Hgb_post`, `Plt_post`, `WBC_post`

```bash
curl -X POST http://localhost:8080/api/cbc/predict \
  -H "Content-Type: application/json" \
  -d '[{"Hgb": 14.0, "Plt": 250, "WBC": 7.5,
        "Hgb_prior": 13.8, "Plt_prior": 245, "WBC_prior": 7.2}]'
```

### Error handling

| HTTP Status | Meaning |
|-------------|---------|
| 200 | Success — JSON array of results |
| 400 | Bad request — malformed JSON, missing fields, or unparseable input |
| 503 | Models not loaded yet (retry after readiness probe returns 200) |

---

## Input Format (CSV)

**Wide format** — one row per blood draw, analyte values as columns:

| sodium | chloride | potassium_plas | co2_totl | bun | creatinine | calcium | glucose |
|--------|----------|----------------|----------|-----|------------|---------|---------|
| 138 | 102 | 4.1 | 24 | 12 | 0.9 | 9.2 | 95 |

Add `_prior` columns (e.g. `sodium_prior`) to enable Realtime predictions.
Add `_post` columns (e.g. `sodium_post`) to also enable Retrospective predictions.

**Long format** — one row per analyte result, requires columns:
`PATIENT_ID`, `DRAWN_DT_TM`, `TASK_ASSAY`, `RESULT_VALUE`

---

## Training Custom Models

1. Prepare a wide-format CSV with analyte columns (+ `_prior`/`_post` for retrospective)
2. Upload in the **Train** tab or run the CLI:

```bash
python -m src.train \
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
