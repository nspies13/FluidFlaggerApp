"""
Standalone FastAPI server for navify Algorithm Suite deployment.

Supports:
  - Single-item JSON (POST application/json)
  - Batch CSV or TSV (POST text/csv or text/tab-separated-values)
  - Health/readiness probes (GET /health/ready, GET /health/live)

The calculation endpoint is POST /predict (configurable in navify registration UI).
"""

from __future__ import annotations

import csv
import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from .features import preprocess_bmp_data, preprocess_cbc_data
from .inference import make_bmp_predictions, make_cbc_predictions
from .model_loader import (
    bmp_classification_key,
    bmp_mix_ratio_key,
    cbc_classification_key,
    cbc_mix_ratio_key,
    get_model,
    load_models_from_dir,
)
from .simulate import get_fluid_names

logger = logging.getLogger("fluidflagger")

# ---------------------------------------------------------------------------
# Model readiness state
# ---------------------------------------------------------------------------

_models_ready = False


def _prefetch_models() -> bool:
    """Download all models into the in-process cache. Returns True if any loaded."""
    all_fluids = get_fluid_names()
    timings = ("Realtime", "Retrospective")
    keys = (
        [bmp_classification_key(f, t) for f in all_fluids for t in timings]
        + [bmp_mix_ratio_key(f) for f in all_fluids]
        + [cbc_classification_key(t) for t in timings]
        + [cbc_mix_ratio_key()]
    )
    ok = 0
    for key in keys:
        try:
            if get_model(key) is not None:
                ok += 1
        except Exception as e:
            logger.warning("Failed to load model %s: %s", key, e)
    logger.info("Model prefetch complete: %d models loaded.", ok)
    return ok > 0


# ---------------------------------------------------------------------------
# Lifespan: load models at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _models_ready
    # Load local models first (if mounted), then fall back to HF Hub
    local_dir = Path("/app/models")
    if local_dir.is_dir():
        n = len(load_models_from_dir(local_dir))
        logger.info("Loaded %d local model(s).", n)
        if n > 0:
            _models_ready = True
    if not _models_ready:
        _models_ready = _prefetch_models()
    yield


app = FastAPI(
    title="FluidFlagger API",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Health / readiness probes (navify requirement)
# ---------------------------------------------------------------------------


@app.get("/health/live")
async def liveness():
    """Liveness probe — always OK if the process is running."""
    return {"status": "ok"}


@app.get("/health/ready")
async def readiness():
    """Readiness probe — OK only after models have been loaded."""
    if not _models_ready:
        raise HTTPException(status_code=503, detail="Models not yet loaded")
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df_to_json(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to JSON-serialisable list of records."""
    return df.where(pd.notna(df), None).to_dict(orient="records")


def _df_to_csv(df: pd.DataFrame, delimiter: str = ",") -> str:
    """Convert DataFrame to CSV or TSV string."""
    return df.to_csv(index=False, sep=delimiter)


def _detect_delimiter(content_type: str, body: bytes) -> Optional[str]:
    """Determine CSV delimiter from Content-Type or sniff the body."""
    ct = content_type.lower()
    if "tab-separated" in ct or "tsv" in ct:
        return "\t"
    if "csv" in ct:
        return ","
    # Sniff from the first line
    first_line = body.split(b"\n", 1)[0].decode("utf-8", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(first_line, delimiters=",\t")
        return dialect.delimiter
    except csv.Error:
        return None


def _parse_request(content_type: str, body: bytes) -> tuple[pd.DataFrame, str]:
    """
    Parse the request body into a DataFrame.

    Returns (df, format) where format is "json", "csv", or "tsv".
    """
    ct = content_type.lower()
    if "json" in ct or body.lstrip().startswith(b"{") or body.lstrip().startswith(b"["):
        import json
        payload = json.loads(body)
        if isinstance(payload, dict):
            payload = [payload]
        return pd.DataFrame(payload), "json"

    delimiter = _detect_delimiter(content_type, body)
    if delimiter is None:
        raise ValueError(
            "Cannot determine input format. Use Content-Type: application/json, "
            "text/csv, or text/tab-separated-values."
        )
    fmt = "tsv" if delimiter == "\t" else "csv"
    df = pd.read_csv(io.BytesIO(body), sep=delimiter)
    return df, fmt


# ---------------------------------------------------------------------------
# Calculation endpoints
# ---------------------------------------------------------------------------

@app.post("/predict")
@app.post("/api/bmp/predict")
async def bmp_predict(request: Request):
    """
    BMP prediction endpoint (default calculation endpoint).

    Accepts:
      - application/json: single object or array of objects
      - text/csv: CSV with header row
      - text/tab-separated-values: TSV with header row
    Returns results in the same format as the input.
    """
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty request body")

    content_type = request.headers.get("content-type", "application/json")
    try:
        df, fmt = _parse_request(content_type, body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parse error: {e}")

    try:
        df = preprocess_bmp_data(df)
        result = make_bmp_predictions(df)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return _format_response(result, fmt)


@app.post("/api/cbc/predict")
async def cbc_predict(request: Request):
    """CBC prediction endpoint. Same input/output format handling as BMP."""
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty request body")

    content_type = request.headers.get("content-type", "application/json")
    try:
        df, fmt = _parse_request(content_type, body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parse error: {e}")

    try:
        df = preprocess_cbc_data(df)
        result = make_cbc_predictions(df)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return _format_response(result, fmt)


def _format_response(result: pd.DataFrame, fmt: str) -> Response:
    """Return the result DataFrame in the same format as the input."""
    if fmt == "json":
        return JSONResponse(content=_df_to_json(result))
    elif fmt == "tsv":
        return Response(
            content=_df_to_csv(result, delimiter="\t"),
            media_type="text/tab-separated-values",
        )
    else:  # csv
        return Response(
            content=_df_to_csv(result, delimiter=","),
            media_type="text/csv",
        )
