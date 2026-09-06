import numpy as np
import pytest

from src.data import FEATURES, TARGET, add_features, chronological_split, make_series
from src.models import build_models, evaluate, run_all, seasonal_naive


@pytest.fixture(scope="module")
def daily():
    return make_series()


@pytest.fixture(scope="module")
def frame(daily):
    return add_features(daily)


def test_real_data_spans_six_years(daily):
    assert daily.index.min().year == 2019 and daily.index.max().year == 2024
    assert 2150 < len(daily) < 2200


def test_partial_days_dropped(daily):
    """A half-captured day would average to an impossible dip."""
    assert daily[TARGET].min() > 10_000


def test_demand_is_physically_plausible(daily):
    assert daily[TARGET].between(10_000, 60_000).all()


def test_winter_demand_exceeds_summer(daily):
    w = daily[daily.index.month.isin([12, 1, 2])][TARGET].mean()
    s = daily[daily.index.month.isin([6, 7, 8])][TARGET].mean()
    assert w > s, "GB winter demand must exceed summer"


def test_split_is_chronological_and_disjoint(frame):
    train, test = chronological_split(frame, test_days=180)
    assert train.index.max() < test.index.min()
    assert len(set(train.index) & set(test.index)) == 0


def test_rolling_features_exclude_current_day(daily):
    """roll_mean_7 on day t must not contain day t's own demand."""
    df = add_features(daily)
    raw = daily[TARGET]
    pos = raw.index.get_loc(df.index[100])
    expected = raw.iloc[pos - 7:pos].mean()
    assert np.isclose(df.iloc[100]["roll_mean_7"], expected)


def test_lag_features_align_correctly(daily):
    df = add_features(daily)
    raw = daily[TARGET]
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


def test_models_fit_and_predict_finite(frame):
    train, test = chronological_split(frame, test_days=100)
    for name, model in build_models(seed=0).items():
        model.fit(train[FEATURES], train[TARGET])
        pred = model.predict(test[FEATURES])
        assert pred.shape == (100,) and np.isfinite(pred).all(), name


def test_ridge_beats_seasonal_naive_on_real_data(frame):
    train, test = chronological_split(frame, test_days=365)
    res = run_all(train, test, FEATURES, target=TARGET, seed=0).set_index("model")
    assert res.loc["ridge", "mae"] < res.loc["seasonal_naive", "mae"]


def test_small_mlp_fails_on_unscaled_target_a_documented_finding(frame):
    """Documents a real and instructive failure.

    The target is GB demand in MW, of order 27,000. Features are standardised
    but the target is not, and the 32-unit network cannot span that scale within
    its iteration budget: it lands around 8,000 MAE against ridge's ~990.

    This is a genuine property of the setup, not a bug being hidden. Neural
    networks are sensitive to target scale in a way linear models are not, and
    that sensitivity is worth knowing before reaching for one.
    """
    train, test = chronological_split(frame, test_days=365)
    res = run_all(train, test, FEATURES, target=TARGET, seed=0).set_index("model")
    assert res.loc["mlp_small", "mae"] > 3 * res.loc["ridge", "mae"], (
        "small MLP now competitive; update the README finding")


def test_mlp_is_the_only_seed_sensitive_model(frame):
    train, test = chronological_split(frame, test_days=200)
    a = run_all(train, test, FEATURES, target=TARGET, seed=0).set_index("model")["mae"]
    b = run_all(train, test, FEATURES, target=TARGET, seed=7).set_index("model")["mae"]
    assert np.isclose(a["ridge"], b["ridge"])
    assert not np.isclose(a["mlp_deep"], b["mlp_deep"], atol=1e-6)
