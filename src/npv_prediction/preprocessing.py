import pandas as pd
from src.npv_prediction.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.npv_prediction.data import FEATURE_COLUMNS, TARGET_COLUMN


DATA_ENCODING = {
    "P90P90": 1,
    "P90P50": 2,
    "P50P90": 3,
    "P90P10": 4,
    "P50P50": 5,
    "P10P90": 6,
    "P90": 7,
    "P50P10": 8,
    "P10P50": 9,
    "P50": 10,
    "P10P10": 11,
    "P10": 12
}


def preprocess_dataset(df):

    result = df.copy()

    result = result.replace("X", 0)

    result = result.replace(DATA_ENCODING)

    for column in FEATURE_COLUMNS:
        result[column] = pd.to_numeric(
            result[column],
            errors="raise"
        )

    return result


def split_features_target(df):

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    X = X.astype("float64")
    y = y.astype("float64")

    return X, y
