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

FluidFlagger is a Gradio application for detecting possible intravenous-fluid
contamination in basic metabolic panel (BMP) and complete blood count (CBC)
results. It supports prediction, local model training, expert review, performance
validation, and case-based self-testing in one interface.

[Open the hosted FluidFlagger app](https://huggingface.co/spaces/nickspies/FluidFlagger)

## What the app supports

| Panel | Current-result columns |
|---|---|
| BMP | `sodium`, `chloride`, `potassium_plas`, `co2_totl`, `bun`, `creatinine`, `calcium`, `glucose` |
| CBC | `Hgb`, `Plt`, `WBC` |

FluidFlagger provides two prediction timings:

- **Realtime** uses the current results and a complete set of `_prior` results.
- **Retrospective** uses the current, `_prior`, and `_post` results. Mixture-ratio
  estimates are available only for retrospective predictions.

Classification is binary. An output of **0.25 or greater** is labeled
`Contaminated`; a lower output is labeled `Real`. Estimated mixture ratios are
limited to the range 0.00–0.50.

## How to use the app

### Predict

1. Select **BMP** or **CBC**.
2. Choose **Upload** to analyze a comma-delimited CSV, or **Manual** to enter one
   set of results directly.
3. For BMP predictions, select the IV fluids to evaluate. The built-in options
   include NS, LR, half-normal saline, water, and several dextrose-containing
   fluids.
4. Select **Run Predictions**.
5. Review the table and download the prediction CSV.

The **Custom Models** section accepts a folder of `.joblib` models. Model files
loaded there are used by the running application until it restarts or they are
replaced.

Include every required current and prior column to receive realtime results, and
every post column to receive retrospective classifications and mixture-ratio
estimates. Rows containing missing values for a timing receive blank results for
that timing.

### Train

1. Select the BMP or CBC panel.
2. Upload a wide-format training CSV containing complete current, `_prior`, and
   `_post` columns.
3. For BMP, optionally upload a custom fluid-concentration CSV or TSV. Otherwise,
   the built-in concentrations are used.
4. Select the model export format. Use `.joblib` if the models will be loaded
   through the Predict tab.
5. Select **Train Models**, then download the model archive and five-fold
   cross-validation metrics CSV.

Training can be computationally intensive and may take a substantial amount of
time, particularly for the full set of BMP fluids.

### Review

1. Upload the CSV downloaded from Predict.
2. Optionally enter the reviewer's name.
3. Move through the rows and label each specimen **Real** or **Contaminated**.
4. Download the reviewed file.

The exported CSV keeps all prediction fields and adds `human_label`,
`label_timestamp`, and `reviewer` columns.

### Validate

1. Upload the CSV downloaded from Review. Validate accepts comma- or
   tab-delimited files.
2. Confirm the detected ground-truth and prediction-output columns. The usual
   defaults are `human_label` and `max_retrospective_prob`.
3. Review Sens, Spec, PPV, NPV, and F1 at the default 0.25 threshold.
4. Drag the ROC operating point or use the threshold slider. The metrics, ROC and
   precision-recall markers, and 2×2 classification table update together.
5. Inspect the calibration plot and download the formatted HTML or PDF report.
6. To create a SHAP feature-importance plot, select the matching model below the
   calibration plot and choose **Generate SHAP Plot**. The plot is not generated
   automatically.

Validation requires a probability-like output column with values from 0 to 1 and
at least one `Real` and one `Contaminated` ground-truth label. The report follows
the most recently committed threshold selection.

### Self Test

1. Select BMP or CBC and a display mode: Retrospective, Real-time, Current Only,
   or Random.
2. Classify each simulated case as **Real** or **Contaminated**.
3. Review the revealed answer and track the running score.

## Preparing prediction files

### Wide format

Wide files contain one specimen per row. Use the canonical current-result columns
shown above, then add the same column names with `_prior` and `_post` suffixes.
For example, a retrospective BMP file includes `sodium`, `sodium_prior`, and
`sodium_post`, along with the equivalent columns for the other seven analytes.

Examples:

- [BMP wide-format CSV](data/bmp_test_wide.csv)
- [CBC wide-format CSV](data/cbc_test_wide.csv)

### Long format

Long files contain one analyte result per row and require:

| Column | Contents |
|---|---|
| `PATIENT_ID` | Patient or encounter identifier |
| `DRAWN_DT_TM` | Collection timestamp |
| `TASK_ASSAY` | Analyte name |
| `RESULT_VALUE` | Numeric result; `RESULT_VALUE_NUMERIC` is also accepted |

The app pivots long data to one row per collection time and derives prior and post
results from draws for the same patient within 48 hours.

Examples:

- [BMP long-format CSV](data/bmp_test_long.csv)
- [CBC long-format CSV](data/cbc_test_long.csv)

### Main prediction outputs

| Field pattern | Meaning |
|---|---|
| `prob_{fluid}_{timing}` | BMP contamination output for a fluid and timing |
| `pred_{fluid}_{timing}` | Binary BMP label at the 0.25 threshold |
| `mix_ratio_{fluid}` | Retrospective BMP mixture-ratio estimate |
| `prob_CBC_{timing}` | CBC contamination output for a timing |
| `pred_CBC_{timing}` | Binary CBC label at the 0.25 threshold |
| `mix_ratio_CBC` | Retrospective CBC mixture-ratio estimate |
| `max_*` | Highest BMP output or mixture ratio across evaluated fluids |
| `any_*_pred` | Whether any evaluated BMP fluid was labeled contaminated |

BMP aggregate fields without `_with_LR` exclude Lactated Ringer's; fields ending
in `_with_LR` include it.

## Run the complete app with Docker

### Requirements

- Docker Desktop or Docker Engine with BuildKit/buildx
- Internet access during the build so the public model bundle can be downloaded

No environment file or external database is required.

### 1. Get the source

```bash
git clone https://github.com/nickspies/FluidFlaggerApp.git
cd FluidFlaggerApp
```

### 2. Build the full application image

```bash
docker buildx build --load --target hf -t fluidflagger:latest .
```

The `hf` target packages the web application, supporting data, and the current
public model bundle into the image.

### 3. Start the container

```bash
docker run --rm \
  --name fluidflagger \
  -p 7860:7860 \
  fluidflagger:latest
```

Open [http://localhost:7860](http://localhost:7860) in a browser. Stop the
foreground container with `Ctrl+C`.

### 4. Check the running app

From another terminal:

```bash
curl --fail http://localhost:7860/api/health
```

Expected response:

```json
{"status":"ok"}
```

Interactive API documentation is available at
[http://localhost:7860/docs](http://localhost:7860/docs). The app exposes JSON
prediction endpoints at `/api/bmp/predict` and `/api/cbc/predict`, plus
newline-delimited JSON variants ending in `_stream`.

### Run in the background

```bash
docker run -d \
  --name fluidflagger \
  -p 7860:7860 \
  fluidflagger:latest

docker logs -f fluidflagger
```

Stop and remove the background container with:

```bash
docker stop fluidflagger
docker rm fluidflagger
```

### Use a different host port

If port 7860 is already occupied, map another host port to container port 7860:

```bash
docker run --rm --name fluidflagger -p 7861:7860 fluidflagger:latest
```

Then open [http://localhost:7861](http://localhost:7861).

Self-test history is stored inside the container and is removed when a container
started with `--rm` stops. Prediction files, reports, model archives, and plots
are downloaded through the browser and remain on the user's computer.
