"""
Navify Algorithm Suite BMP calculation API.

The hosted algorithm contract is intentionally narrow:
  - POST /predict with application/json only
  - GET /health/live
  - GET /health/ready
  - baked local BMP models only
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from .features import BMP_ANALYTES, preprocess_bmp_data
from .inference import make_bmp_predictions
from .model_loader import (
    BMP_FLUIDS,
    EXPECTED_BMP_MODEL_COUNT,
    clear_cache,
    load_required_models_from_dir,
    missing_model_keys,
)

logger = logging.getLogger("fluidflagger.navify")

MODEL_DIR_ENV = "MODEL_DIR"
CURRENT_FIELDS = tuple(BMP_ANALYTES)
PRIOR_FIELDS = tuple(f"{col}_prior" for col in BMP_ANALYTES)
POST_FIELDS = tuple(f"{col}_post" for col in BMP_ANALYTES)
REQUIRED_REALTIME_FIELDS = CURRENT_FIELDS + PRIOR_FIELDS
NUMERIC_FIELDS = REQUIRED_REALTIME_FIELDS + POST_FIELDS
NULLABLE_RETROSPECTIVE_RESPONSE_FIELDS = (
    POST_FIELDS
    + tuple(f"prob_{fluid}_Retrospective" for fluid in BMP_FLUIDS)
    + tuple(f"pred_{fluid}_Retrospective" for fluid in BMP_FLUIDS)
    + tuple(f"mix_ratio_{fluid}" for fluid in BMP_FLUIDS)
    + (
        "any_retrospective_pred",
        "any_retrospective_pred_with_LR",
        "max_retrospective_prob",
        "max_prob_fluid_retrospective",
        "max_retrospective_prob_with_LR",
        "max_mix_ratio",
        "max_mix_ratio_with_LR",
    )
)

_models_ready = False
_model_load_error: str | None = None


def default_model_dir() -> Path:
    env_path = os.environ.get(MODEL_DIR_ENV)
    if env_path:
        return Path(env_path)
    docker_path = Path("/app/models")
    if docker_path.is_dir():
        return docker_path
    return Path(__file__).resolve().parents[1] / "models"


def load_models_for_app(model_dir: str | Path | None = None) -> bool:
    """Load all baked BMP models and update readiness state."""
    global _models_ready, _model_load_error
    clear_cache()
    try:
        loaded = load_required_models_from_dir(model_dir or default_model_dir())
    except Exception as exc:
        _models_ready = False
        _model_load_error = str(exc)
        logger.exception("BMP model loading failed")
        return False

    _models_ready = True
    _model_load_error = None
    logger.info("Loaded %d required BMP models.", len(loaded))
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models_for_app()
    yield


app = FastAPI(
    title="FluidFlagger BMP Navify API",
    version="1.0.3",
    lifespan=lifespan,
)


@app.get("/health/live")
async def liveness():
    return {"status": "ok"}


@app.get("/health/ready")
async def readiness():
    if not _models_ready:
        detail = "Models not yet loaded"
        if _model_load_error:
            detail = f"{detail}: {_model_load_error}"
        raise HTTPException(status_code=503, detail=detail)
    return {"status": "ok", "models_loaded": EXPECTED_BMP_MODEL_COUNT}


def _require_json_content_type(request: Request) -> None:
    content_type = request.headers.get("content-type", "")
    if "application/json" not in content_type.lower():
        raise HTTPException(
            status_code=400,
            detail="Content-Type must be application/json",
        )


async def _parse_json_payload(request: Request) -> list[dict[str, Any]]:
    _require_json_content_type(request)
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Malformed JSON: {exc}") from exc

    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list):
        if len(payload) != 1:
            raise HTTPException(
                status_code=400,
                detail="Payload array must contain exactly one BMP record",
            )
        records = payload
    else:
        raise HTTPException(status_code=400, detail="Payload must be an object or array")

    if not all(isinstance(item, dict) for item in records):
        raise HTTPException(status_code=400, detail="Every payload item must be an object")
    return records


def _coerce_and_validate(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    df = preprocess_bmp_data(df)

    missing = [field for field in REQUIRED_REALTIME_FIELDS if field not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required BMP fields: {', '.join(missing)}",
        )

    present_post = [field for field in POST_FIELDS if field in df.columns]
    if present_post and len(present_post) != len(POST_FIELDS):
        missing_post = [field for field in POST_FIELDS if field not in df.columns]
        raise HTTPException(
            status_code=400,
            detail=(
                "Post specimen fields must be supplied as a complete set; "
                f"missing: {', '.join(missing_post)}"
            ),
        )

    fields_to_check = list(REQUIRED_REALTIME_FIELDS)
    has_complete_post = len(present_post) == len(POST_FIELDS)
    if has_complete_post:
        fields_to_check.extend(POST_FIELDS)

    for field in fields_to_check:
        converted = pd.to_numeric(df[field], errors="coerce")
        bad_mask = converted.isna()
        if bad_mask.any():
            bad_rows = ", ".join(str(idx) for idx in df.index[bad_mask].tolist()[:5])
            raise HTTPException(
                status_code=400,
                detail=f"Field {field} must be numeric and non-null; bad row index(es): {bad_rows}",
            )
        df[field] = converted.astype(float)

    return df, has_complete_post


def _record_from_df(
    df: pd.DataFrame,
    *,
    fill_unavailable_retrospective: bool,
) -> dict[str, Any]:
    records = df.where(pd.notna(df), None).to_dict(orient="records")
    if len(records) != 1:
        raise HTTPException(
            status_code=400,
            detail="Payload must contain exactly one BMP record",
        )
    record = jsonable_encoder(records[0])
    if fill_unavailable_retrospective:
        for field in NULLABLE_RETROSPECTIVE_RESPONSE_FIELDS:
            record.setdefault(field, None)
    return record


@app.post("/predict")
async def predict(request: Request):
    if not _models_ready:
        missing = missing_model_keys()
        detail = "Models not yet loaded"
        if missing:
            detail = f"{detail}; missing {len(missing)} model(s)"
        raise HTTPException(status_code=503, detail=detail)

    records = await _parse_json_payload(request)
    df = pd.DataFrame(records)
    df, has_complete_post = _coerce_and_validate(df)
    timings = ("Realtime", "Retrospective") if has_complete_post else ("Realtime",)

    try:
        result = make_bmp_predictions(df, timings=timings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return JSONResponse(
        content=_record_from_df(
            result,
            fill_unavailable_retrospective=not has_complete_post,
        )
    )
