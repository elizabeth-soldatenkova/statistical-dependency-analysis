from pathlib import Path

import joblib

from src.npv_prediction.data import (
    load_datasets,
    validate_datasets
)

from src.npv_prediction.preprocessing import (
    preprocess_dataset,
    split_features_target
)

from src.npv_prediction.train import (
    train_linear_regression,
    train_random_forest,
    tune_random_forest,
    train_xgboost,
    tune_xgboost,
    train_lightgbm,
    tune_lightgbm,
    train_mlp
)

from src.npv_prediction.evaluate import evaluate_models

#1-project paths

PROJECT_ROOT = Path(__file__).resolve().parent

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

#2-load datasets

print("Loading datasets...")

df_train, df_control = load_datasets()

#3-validate datasets

validate_datasets(df_train, df_control)

#4-preprocess datasets

print("\nPreprocessing datasets...")

df_train = preprocess_dataset(df_train)
df_control = preprocess_dataset(df_control)

#5-split features and target

X_train, y_train = split_features_target(df_train)
X_control, y_control = split_features_target(df_control)

print("\nChecking feature data types:")
print(X_train.dtypes)

print("\nChecking for missing values:")
print(X_train.isna().sum())

print("\nChecking target data type:")
print(y_train.dtype)

# 6-train models

print("\nTraining Linear Regression...")

linear_model = train_linear_regression(
    X_train,
    y_train
)


print("\nTraining Random Forest...")

rf_model = train_random_forest(
    X_train,
    y_train
)


print("\nTuning Random Forest...")

best_rf = tune_random_forest(
    X_train,
    y_train
)


print("\nTraining XGBoost...")

xgb_model = train_xgboost(
    X_train,
    y_train
)


print("\nTuning XGBoost...")

best_xgb = tune_xgboost(
    X_train,
    y_train
)


print("\nTraining LightGBM...")

lgb_model = train_lightgbm(
    X_train,
    y_train
)


print("\nTuning LightGBM...")

best_lgb = tune_lightgbm(
    X_train,
    y_train
)


print("\nTraining MLP...")

mlp_model, scaler = train_mlp(
    X_train,
    y_train
)

# 7-save trained models

models = {
    "linear_regression": linear_model,
    "random_forest": rf_model,
    "random_forest_tuned": best_rf,
    "xgboost": xgb_model,
    "xgboost_tuned": best_xgb,
    "lightgbm": lgb_model,
    "lightgbm_tuned": best_lgb,
    "mlp": mlp_model
}


print("\nSaving models...")

for model_name, model in models.items():

    model_path = MODELS_DIR / f"{model_name}.joblib"

    joblib.dump(model, model_path)

    print(f"Saved: {model_path.name}")


#save the scaler used by MLP
joblib.dump(
    scaler,
    MODELS_DIR / "mlp_scaler.joblib"
)

#8-evaluate models on control dataset

print("\nEvaluating models...")

results = evaluate_models(
    models=models,
    X_control=X_control,
    y_control=y_control,
    scaler=scaler
)

# 9-save evaluation results

results_path = REPORTS_DIR / "model_evaluation.csv"

results.to_csv(
    results_path,
    index=False
)


print("\nEvaluation results:")
print(results.to_string(index=False))

print(f"\nResults saved to: {results_path}")

print("\nTraining pipeline completed successfully.")
