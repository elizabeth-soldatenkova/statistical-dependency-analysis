import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app, MODEL_PATH, MODEL_NAME
from src.npv_prediction.data import FEATURE_COLUMNS
from src.npv_prediction.preprocessing import preprocess_dataset


TEST_INPUT = {
    "Feature 1": "P50P50",
    "Feature 2": "P90P90",
    "Feature 3": "P10P50",
    "Feature 4": "P50",
    "Feature 5": "P90",
    "Feature 6": "P10",
    "Feature 7": "P50P10",
    "Feature 8": "P10P10",
    "Feature 9": "P90P50",
    "Feature 10": "P50P90"
}


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model"] == MODEL_NAME


def test_prediction_endpoint(client):
    response = client.post(
        "/predict",
        json=TEST_INPUT
    )

    assert response.status_code == 200

    result = response.json()

    assert "predicted_npv" in result
    assert isinstance(result["predicted_npv"], (int, float))
    assert result["model"] == MODEL_NAME


def test_api_prediction_matches_direct_prediction(client):
    model = joblib.load(MODEL_PATH)

    input_df = pd.DataFrame(
        [TEST_INPUT],
        columns=FEATURE_COLUMNS
    )

    processed_data = preprocess_dataset(input_df)

    direct_prediction = float(
        model.predict(processed_data)[0]
    )

    response = client.post(
        "/predict",
        json=TEST_INPUT
    )

    assert response.status_code == 200

    api_prediction = response.json()["predicted_npv"]

    assert api_prediction == pytest.approx(
        direct_prediction,
        rel=1e-9,
        abs=1e-9
    )


def test_prediction_rejects_missing_feature(client):
    invalid_input = TEST_INPUT.copy()

    del invalid_input["Feature 10"]

    response = client.post(
        "/predict",
        json=invalid_input
    )

    assert response.status_code == 422


def test_prediction_rejects_extra_feature(client):
    invalid_input = TEST_INPUT.copy()

    invalid_input["Feature 11"] = "P50"

    response = client.post(
        "/predict",
        json=invalid_input
    )

    assert response.status_code == 422
