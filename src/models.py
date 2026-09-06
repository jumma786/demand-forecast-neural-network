"""Four models on identical features and identical splits."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def seasonal_naive(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Last week's same weekday. The bar every model must clear."""
    return test["lag_7"].to_numpy()


def build_models(seed: int = 42) -> dict:
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "mlp_small": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(32,), max_iter=3000,
                         early_stopping=True, n_iter_no_change=25,
                         random_state=seed)),
        "mlp_deep": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=3000,
                         early_stopping=True, n_iter_no_change=25,
                         learning_rate_init=0.003, random_state=seed)),
        "gradient_boosting": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, random_state=seed),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape_pct": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    }


def run_all(train: pd.DataFrame, test: pd.DataFrame, features: list,
            target: str = "demand_ml_d", seed: int = 42) -> pd.DataFrame:
    Xtr, ytr = train[features], train[target].to_numpy()
    Xte, yte = test[features], test[target].to_numpy()

    rows = [{"model": "seasonal_naive", **evaluate(yte, seasonal_naive(train, test))}]
    for name, model in build_models(seed).items():
        model.fit(Xtr, ytr)
        rows.append({"model": name, **evaluate(yte, model.predict(Xte))})
    return (pd.DataFrame(rows).sort_values("mae").reset_index(drop=True))
