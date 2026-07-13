from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import api_server  # noqa: E402
from src.model_loader import clear_cache  # noqa: E402

MODELS_DIR = ROOT / "models"


BASE_REALTIME = {
    "sodium": 148,
    "chloride": 144,
    "potassium_plas": 2.7,
    "co2_totl": 16,
    "bun": 40,
    "creatinine": 2.42,
    "calcium": 6.9,
    "glucose": 181,
    "sodium_prior": 132,
    "chloride_prior": 83,
    "potassium_plas_prior": 4.5,
    "co2_totl_prior": 24,
    "bun_prior": 49,
    "creatinine_prior": 3.62,
    "calcium_prior": 10.3,
    "glucose_prior": 135,
}

POST_FIELDS = {
    "sodium_post": 135,
    "chloride_post": 94,
    "potassium_plas_post": 3.4,
    "co2_totl_post": 28,
    "bun_post": 32,
    "creatinine_post": 1.75,
    "calcium_post": 9.0,
    "glucose_post": 133,
}


@pytest.fixture()
def loaded_client():
    assert api_server.load_models_for_app(MODELS_DIR)
    client = TestClient(api_server.app, raise_server_exceptions=False)
    yield client
    clear_cache()
    api_server._models_ready = False
    api_server._model_load_error = None


def test_readiness_fails_before_model_load():
    clear_cache()
    api_server._models_ready = False
    api_server._model_load_error = None
    client = TestClient(api_server.app, raise_server_exceptions=False)

    response = client.get("/health/ready")

    assert response.status_code == 503


def test_readiness_succeeds_after_all_models_load(loaded_client):
    response = loaded_client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["models_loaded"] == 27


def test_valid_realtime_request_returns_realtime_predictions(loaded_client):
    response = loaded_client.post("/predict", json=[BASE_REALTIME])

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert "prob_NS_Realtime" in body[0]
    assert "pred_NS_Realtime" in body[0]
    assert "prob_NS_Retrospective" not in body[0]
    assert "mix_ratio_NS" not in body[0]


def test_valid_retrospective_request_returns_retro_and_mix_predictions(loaded_client):
    payload = copy.deepcopy(BASE_REALTIME)
    payload.update(POST_FIELDS)

    response = loaded_client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert "prob_NS_Realtime" in body[0]
    assert "prob_NS_Retrospective" in body[0]
    assert "mix_ratio_NS" in body[0]


def test_partial_post_fields_return_400(loaded_client):
    payload = copy.deepcopy(BASE_REALTIME)
    payload["sodium_post"] = 135

    response = loaded_client.post("/predict", json=[payload])

    assert response.status_code == 400
    assert "Post specimen fields" in response.json()["detail"]


def test_missing_required_field_returns_400(loaded_client):
    payload = copy.deepcopy(BASE_REALTIME)
    del payload["sodium_prior"]

    response = loaded_client.post("/predict", json=[payload])

    assert response.status_code == 400
    assert "Missing required BMP fields" in response.json()["detail"]


def test_nonnumeric_bmp_value_returns_400(loaded_client):
    payload = copy.deepcopy(BASE_REALTIME)
    payload["sodium"] = "not-a-number"

    response = loaded_client.post("/predict", json=[payload])

    assert response.status_code == 400
    assert "must be numeric" in response.json()["detail"]


def test_non_json_content_type_returns_400(loaded_client):
    response = loaded_client.post(
        "/predict",
        content="sodium,chloride\n148,144\n",
        headers={"Content-Type": "text/csv"},
    )

    assert response.status_code == 400
    assert "application/json" in response.json()["detail"]
