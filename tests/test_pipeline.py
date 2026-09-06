import numpy as np
import pandas as pd
import pytest
from src.data import make_series, add_features, chronological_split, FEATURES
from src.models import build_models, evaluate, seasonal_naive, run_all


@pytest.fixture(scope="module")
def frame():
    return add_features(make_series(n_days=1200, seed=42))


def test_series_is_reproducible():
    assert make_series(seed=1).equals(make_series(seed=1))


def test_split_is_chronological_and_disjoint(frame):
    train, test = chronological_split(frame, test_days=180)
    assert train.index.max() < test.index.min(), "test must follow train in time"
    assert len(set(train.index) & set(test.index)) == 0


def test_no_same_day_temperature_leak(frame):
    """Same-day temperature would not be known at forecast time."""
    assert "temp_c" not in frame.columns
    assert "temp_lag_1" in frame.columns


def test_rolling_features_exclude_current_day():
    """roll_mean_7 on day t must not contain day t's own demand."""
    df = add_features(make_series(n_days=900, seed=3))
    row = df.iloc[100]
    raw = make_series(n_days=900, seed=3)["demand_ml_d"]
    pos = raw.index.get_loc(df.index[100])
    expected = raw.iloc[pos - 7:pos].mean()
    assert np.isclose(row["roll_mean_7"], expected), "rolling window leaked today"


def test_lag_features_align_correctly():
    df = add_features(make_series(n_days=900, seed=3))
    raw = make_series(n_days=900, seed=3)["demand_ml_d"]
    pos = raw.index.get_loc(df.index[50])
    assert np.isclose(df.iloc[50]["lag_1"], raw.iloc[pos - 1])
    assert np.isclose(df.iloc[50]["lag_7"], raw.iloc[pos - 7])


def test_no_nans_after_feature_build(frame):
    assert not frame[FEATURES].isna().any().any()


def test_evaluate_perfect_prediction_is_zero_error():
    y = np.array([1.0, 2.0, 3.0])
    m = evaluate(y, y.copy())
    assert m["mae"] == 0 and m["rmse"] == 0


def test_seasonal_naive_returns_lag_7(frame):
    train, test = chronological_split(frame, test_days=100)
    assert np.allclose(seasonal_naive(train, test), test["lag_7"].to_numpy())


def test_models_build_and_fit(frame):
    train, test = chronological_split(frame, test_days=100)
    for name, model in build_models(seed=0).items():
        model.fit(train[FEATURES], train["demand_ml_d"])
        pred = model.predict(test[FEATURES])
        assert pred.shape == (100,), f"{name} returned wrong shape"
        assert np.isfinite(pred).all(), f"{name} produced non-finite predictions"


def test_deterministic_models_beat_a_constant_mean(frame):
    """Sanity floor for the models that should always clear it."""
    train, test = chronological_split(frame, test_days=120)
    res = run_all(train, test, FEATURES, seed=0).set_index("model")
    const = evaluate(test["demand_ml_d"].to_numpy(),
                     np.full(len(test), train["demand_ml_d"].mean()))
    for m in ("ridge", "gradient_boosting", "seasonal_naive"):
        assert res.loc[m, "mae"] < const["mae"], f"{m} failed the sanity floor"


def test_mlp_is_data_hungry_a_documented_limitation(frame):
    """Documents a real weakness rather than hiding it.

    On this reduced fixture (~700 training rows) the MLPs perform WORSE than
    simply predicting the training mean, while ridge and gradient boosting
    comfortably beat it. On the full six-year series the MLPs do clear the
    seasonal-naive baseline. The gap is sample size, not implementation.

    This test asserts the weakness so that if a future change makes the MLP
    competitive on small samples, the test fails and the README gets corrected.
    """
    train, test = chronological_split(frame, test_days=120)
    res = run_all(train, test, FEATURES, seed=0).set_index("model")
    const = evaluate(test["demand_ml_d"].to_numpy(),
                     np.full(len(test), train["demand_ml_d"].mean()))
    assert res.loc["mlp_small", "mae"] > const["mae"], (
        "MLP now beats a constant mean on small data; update the README claim")


def test_mlp_is_the_only_seed_sensitive_model(frame):
    """Ridge and boosting are deterministic here; the MLPs are not."""
    train, test = chronological_split(frame, test_days=120)
    a = run_all(train, test, FEATURES, seed=0).set_index("model")["mae"]
    b = run_all(train, test, FEATURES, seed=7).set_index("model")["mae"]
    assert np.isclose(a["ridge"], b["ridge"])
    assert not np.isclose(a["mlp_deep"], b["mlp_deep"], atol=1e-6)
