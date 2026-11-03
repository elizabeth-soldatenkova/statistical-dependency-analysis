from contextlib import asynccontextmanager

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException

from src.npv_prediction.data import (
    FEATURE_COLUMNS,
    PROJECT_ROOT
)

from src.npv_prediction.preprocessing import (
    preprocess_dataset
)

from app.schemas import NPVRequest, NPVResponse


MODEL_NAME = "lightgbm_tuned"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / f"{MODEL_NAME}.joblib"
)

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    print(f"Model loaded successfully: {MODEL_NAME}")

    yield

    model = None


app = FastAPI(
    title="NPV Prediction API",
    description=(
        "REST API for predicting Net Present Value "
        "of oil and gas investment projects."
    ),
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health_check():

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    return {
        "status": "healthy",
        "model": MODEL_NAME
    }


@app.post(
    "/predict",
    response_model=NPVResponse
)
def predict_npv(request: NPVRequest):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    try:
        input_data = request.model_dump(
            by_alias=True
        )

        input_df = pd.DataFrame(
            [input_data],
            columns=FEATURE_COLUMNS
        )


        processed_data = preprocess_dataset(
            input_df
        )

        processed_data = processed_data[
            FEATURE_COLUMNS
        ]

        prediction = model.predict(
            processed_data
        )

        predicted_npv = float(prediction[0])

        return NPVResponse(
            predicted_npv=predicted_npv,
            model=MODEL_NAME
        )

    except (ValueError, TypeError) as error:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid feature values: {error}"
        ) from error
