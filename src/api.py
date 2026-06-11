"""
FastAPI sub-application exposing BMP and CBC prediction endpoints.

Mirrors the REST API surface from run_bmp.R and run_cbc.R:
  GET  /api/health
  POST /api/bmp/predict          — JSON array or object → predictions JSON
  POST /api/bmp/predict_stream   — newline-delimited JSON input
  POST /api/cbc/predict
  POST /api/cbc/predict_stream

Mounted inside the Gradio app in app.py via gr.mount_gradio_app.
"""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from .features import preprocess_bmp_data, preprocess_cbc_data
from .inference import make_bmp_predictions, make_cbc_predictions

api = FastAPI(title="FluidFlagger API", version="1.0.0")

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@api.get("/api/health")
async def health():
    return {"status": "ok"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df_to_json(df: pd.DataFrame) -> Any:
    """Convert DataFrame to JSON-serialisable list of records."""
    return df.where(pd.notna(df), None).to_dict(orient="records")


def _parse_body_to_df(body: bytes) -> pd.DataFrame:
    """Parse a JSON request body (object or array) into a DataFrame."""
    import json
    payload = json.loads(body)
    if isinstance(payload, dict):
        payload = [payload]
    return pd.DataFrame(payload)


def _parse_ndjson_body(body: bytes) -> list[pd.DataFrame]:
    """Parse newline-delimited JSON body into a list of single-row DataFrames."""
    import json
    frames = []
    for line in body.decode().splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            payload = [payload]
        frames.append(pd.DataFrame(payload))
    return frames

# ---------------------------------------------------------------------------
# BMP endpoints
# ---------------------------------------------------------------------------

@api.post("/api/bmp/predict")
async def bmp_predict(request: Request):
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty request body")
    try:
        df = _parse_body_to_df(body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"JSON parse error: {e}")

    try:
        df = preprocess_bmp_data(df)
        result = make_bmp_predictions(df)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content=_df_to_json(result))


@api.post("/api/bmp/predict_stream")
async def bmp_predict_stream(request: Request):
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty request body")
    try:
        frames = _parse_ndjson_body(body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"NDJSON parse error: {e}")

    results = []
    for frame in frames:
        try:
            frame = preprocess_bmp_data(frame)
            preds = make_bmp_predictions(frame)
            results.extend(_df_to_json(preds))
        except Exception as e:
            results.append({"error": str(e)})

    return JSONResponse(content=results)

# ---------------------------------------------------------------------------
# CBC endpoints
# ---------------------------------------------------------------------------

@api.post("/api/cbc/predict")
async def cbc_predict(request: Request):
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty request body")
    try:
        df = _parse_body_to_df(body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"JSON parse error: {e}")

    try:
        df = preprocess_cbc_data(df)
        result = make_cbc_predictions(df)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content=_df_to_json(result))


@api.post("/api/cbc/predict_stream")
async def cbc_predict_stream(request: Request):
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty request body")
    try:
        frames = _parse_ndjson_body(body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"NDJSON parse error: {e}")

    results = []
    for frame in frames:
        try:
            frame = preprocess_cbc_data(frame)
            preds = make_cbc_predictions(frame)
            results.extend(_df_to_json(preds))
        except Exception as e:
            results.append({"error": str(e)})

    return JSONResponse(content=results)
