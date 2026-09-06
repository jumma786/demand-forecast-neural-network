"""Synthetic daily demand series with a fully declared generating process.

Written out explicitly so a reader can see exactly what signal the models are
being asked to recover, and can judge for themselves whether a result is
impressive or merely a model rediscovering something obvious.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def make_series(n_days: int = 2190, seed: int = 42) -> pd.DataFrame:
    """Six years of daily demand.

    level + weekly seasonality + annual seasonality + temperature response
    + slow trend + autocorrelated noise
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)

    trend = 0.010 * t
    weekly = 6.0 * np.sin(2 * np.pi * (idx.dayofweek.values) / 7.0)
    annual = 22.0 * np.sin(2 * np.pi * (idx.dayofyear.values - 105) / 365.25)

    # temperature: seasonal with weather noise; demand responds non-linearly
    temp = (11.0 + 9.0 * np.sin(2 * np.pi * (idx.dayofyear.values - 105) / 365.25)
            + rng.normal(0, 2.6, n_days))
    heat_response = 1.6 * np.clip(temp - 19.0, 0, None) ** 1.35

    # autocorrelated residual: today's error partly carries into tomorrow
    noise = np.zeros(n_days)
    for i in range(1, n_days):
        noise[i] = 0.55 * noise[i - 1] + rng.normal(0, 4.2)

    demand = 300.0 + trend + weekly + annual + heat_response + noise
    return pd.DataFrame({"date": idx, "demand_ml_d": demand,
                         "temp_c": temp}).set_index("date")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag and calendar features. Every feature uses only past information."""
    out = df.copy()
    for lag in (1, 2, 3, 7, 14, 365):
        out[f"lag_{lag}"] = out["demand_ml_d"].shift(lag)
    for win in (7, 28):
        # shift(1) first: today's value must never enter its own rolling mean
        out[f"roll_mean_{win}"] = out["demand_ml_d"].shift(1).rolling(win).mean()
        out[f"roll_std_{win}"] = out["demand_ml_d"].shift(1).rolling(win).std()
    out["dow"] = out.index.dayofweek
    out["month"] = out.index.month
    out["doy_sin"] = np.sin(2 * np.pi * out.index.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out.index.dayofyear / 365.25)
    out["temp_lag_1"] = out["temp_c"].shift(1)
    out = out.drop(columns=["temp_c"])  # same-day temp would be unavailable
    return out.dropna()


FEATURES = [c for c in
            ["lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_365",
             "roll_mean_7", "roll_std_7", "roll_mean_28", "roll_std_28",
             "dow", "month", "doy_sin", "doy_cos", "temp_lag_1"]]


def chronological_split(df: pd.DataFrame, test_days: int = 365):
    """Time-ordered split. No shuffling, no random sampling."""
    train = df.iloc[:-test_days]
    test = df.iloc[-test_days:]
    return train, test
