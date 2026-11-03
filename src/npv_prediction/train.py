import numpy as np

from scipy.stats import randint, uniform

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor



class MyLinearRegression:
    def fit(self, X, y):

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        X = np.c_[np.ones(X.shape[0]), X]

        self.coef_ = np.linalg.pinv(X.T @ X) @ X.T @ y

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)

        X = np.c_[np.ones(X.shape[0]), X]

        return X @ self.coef_


def train_linear_regression(X_train, y_train):

    model = MyLinearRegression()

    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train):
    """
    Train baseline Random Forest model.
    """

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def tune_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        random_state=42
    )

    param_grid = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 4, 5, 6, 10],
        "max_features": ["sqrt", "log2"]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print("Random Forest best parameters:")
    print(grid_search.best_params_)

    print("Random Forest best CV R2:")
    print(grid_search.best_score_)

    return grid_search.best_estimator_


def train_xgboost(X_train, y_train):
    """
    Train baseline XGBoost model.
    """

    model = XGBRegressor(
        n_estimators=618,
        learning_rate=0.031448,
        max_depth=3,
        subsample=0.7586,
        colsample_bytree=0.74,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def tune_xgboost(X_train, y_train):

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1
    )

    param_dist = {
        "n_estimators": randint(100, 700),
        "max_depth": randint(3, 12),
        "learning_rate": uniform(0.01, 0.15),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 10),
        "gamma": uniform(0, 0.5),
        "reg_alpha": uniform(0, 1),
        "reg_lambda": uniform(0.5, 5)
    }

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=50,
        scoring="r2",
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        return_train_score=True
    )

    random_search.fit(X_train, y_train)

    print("XGBoost best parameters:")
    print(random_search.best_params_)

    print("XGBoost best CV R2:")
    print(random_search.best_score_)

    return random_search.best_estimator_


def train_lightgbm(X_train, y_train):
    """
    Train baseline LightGBM model.
    """

    model = LGBMRegressor(
        n_estimators=476,
        learning_rate=0.06344,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def tune_lightgbm(X_train, y_train):
    model = LGBMRegressor(
        random_state=42,
        verbose=-1
    )

    param_dist = {
        "n_estimators": randint(100, 700),
        "learning_rate": uniform(0.01, 0.15),
        "num_leaves": randint(15, 120),
        "max_depth": randint(3, 15),
        "min_child_samples": randint(5, 50),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_alpha": uniform(0, 1),
        "reg_lambda": uniform(0.5, 5)
    }

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=50,
        scoring="r2",
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        return_train_score=True
    )

    random_search.fit(X_train, y_train)

    print("LightGBM best parameters:")
    print(random_search.best_params_)

    print("LightGBM best CV R2:")
    print(random_search.best_score_)

    return random_search.best_estimator_


def train_mlp(X_train, y_train):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        alpha=0.0005,
        learning_rate_init=0.001,
        max_iter=1500,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    return model, scaler
