from pathlib import Path

import joblib
import pandas as pd

from src.npv_prediction.data import (
    FEATURE_COLUMNS,
    PROJECT_ROOT
)

from src.npv_prediction.preprocessing import (
    preprocess_dataset,
    split_features_target
)


MODELS_DIR = PROJECT_ROOT / "models"


def load_model(model_name):

    model_path = MODELS_DIR / f"{model_name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    return joblib.load(model_path)


def predict_npv(model_name, new_data):
 
    if not isinstance(new_data, pd.DataFrame):
        raise TypeError("new_data must be a pandas DataFrame.")

    missing_columns = [
        column for column in FEATURE_COLUMNS
        if column not in new_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing feature columns: {missing_columns}"
        )

    X = new_data[FEATURE_COLUMNS].copy()

    X = preprocess_dataset(X)

    model = load_model(model_name)

    if model_name == "mlp":

        scaler_path = MODELS_DIR / "mlp_scaler.joblib"

        if not scaler_path.exists():
            raise FileNotFoundError(
                f"MLP scaler not found: {scaler_path}"
            )

        scaler = joblib.load(scaler_path)

        X = scaler.transform(X)

    predictions = model.predict(X)

    return predictions
