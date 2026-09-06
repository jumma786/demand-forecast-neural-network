"""Real GB electricity demand from NESO (National Energy System Operator).

Source: NESO Historic Demand Data, half-hourly settlement periods.
https://www.neso.energy/data-portal/historic-demand-data
Licence: NESO Open Data Licence.

Files are not committed (see .gitignore); `python -m src.download` fetches them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
YEARS = (2019, 2020, 2021, 2022, 2023, 2024)


def load_half_hourly(years=YEARS) -> pd.DataFrame:
    frames = []
    for y in years:
        f = RAW / f"demanddata_{y}.csv"
        if not f.exists():
            raise FileNotFoundError(
                f"{f} missing. Run: python -m src.download")
        df = pd.read_csv(f)
        df.columns = [c.strip().upper() for c in df.columns]
        frames.append(df[["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND"]])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["SETTLEMENT_DATE"], format="mixed",
                                 dayfirst=True)
    out["ND"] = pd.to_numeric(out["ND"], errors="coerce")
    return out.dropna(subset=["ND"])


def to_daily(hh: pd.DataFrame) -> pd.DataFrame:
    """Daily mean National Demand (MW).

    Days without a full 46-50 settlement periods are dropped rather than
    averaged over a partial day, which would look like a demand dip that
    never happened. Clock-change days legitimately have 46 or 50.
    """
    g = hh.groupby("date")["ND"]
    daily = g.mean().to_frame("demand_mw")
    daily["periods"] = g.size()
    complete = daily[(daily["periods"] >= 46) & (daily["periods"] <= 50)]
    return complete.drop(columns=["periods"]).sort_index()


def make_series() -> pd.DataFrame:
    return to_daily(load_half_hourly())


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag and calendar features. Every feature uses only past information.

    No weather covariate: the NESO demand file carries no temperature, and
    temperature is the largest known omitted driver of electricity demand.
    Stated here rather than quietly ignored.
    """
    out = df.copy()
    for lag in (1, 2, 3, 7, 14, 365):
        out[f"lag_{lag}"] = out["demand_mw"].shift(lag)
    for win in (7, 28):
        # shift(1) first: today's value must never enter its own rolling mean
        out[f"roll_mean_{win}"] = out["demand_mw"].shift(1).rolling(win).mean()
        out[f"roll_std_{win}"] = out["demand_mw"].shift(1).rolling(win).std()
    out["dow"] = out.index.dayofweek
    out["month"] = out.index.month
    out["doy_sin"] = np.sin(2 * np.pi * out.index.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out.index.dayofyear / 365.25)
    out["is_weekend"] = (out.index.dayofweek >= 5).astype(int)
    return out.dropna()


FEATURES = ["lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_365",
            "roll_mean_7", "roll_std_7", "roll_mean_28", "roll_std_28",
            "dow", "month", "doy_sin", "doy_cos", "is_weekend"]

TARGET = "demand_mw"


def chronological_split(df: pd.DataFrame, test_days: int = 365):
    """Time-ordered split. No shuffling, no random sampling."""
    return df.iloc[:-test_days], df.iloc[-test_days:]
