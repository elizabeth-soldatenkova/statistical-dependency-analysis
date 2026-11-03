import pandas as pd

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


def evaluate_model(model, X_control, y_control):

    predictions = model.predict(X_control)

    metrics = {
        "R2": r2_score(y_control, predictions),
        "MAE": mean_absolute_error(y_control, predictions),
        "RMSE": mean_squared_error(
            y_control,
            predictions
        ) ** 0.5
    }

    return metrics


def evaluate_models(models, X_control, y_control, scaler=None):

    results = []

    for model_name, model in models.items():

        if model_name == "mlp":
            if scaler is None:
                raise ValueError(
                    "Scaler is required to evaluate MLP."
                )

            X = scaler.transform(X_control)

        else:
            X = X_control

        metrics = evaluate_model(
            model,
            X,
            y_control
        )

        results.append({
            "Model": model_name,
            **metrics
        })

    return pd.DataFrame(results)
