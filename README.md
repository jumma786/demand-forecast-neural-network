# Demand Forecasting: Does a Neural Network Actually Help?

A neural network built properly, benchmarked honestly against simpler models on
daily demand forecasting, and **reported losing**.

```bash
pip install -r requirements.txt
python run_analysis.py
python -m pytest tests/ -q
```

---

## The result

Six years of daily demand, trained on 2020–2023, tested on the unseen 365 days
of 2024. Every model sees identical features and an identical chronological split.

| Model | MAE | MAPE | RMSE |
|---|---|---|---|
| **Ridge regression** | **3.81** | **1.18%** | **5.10** |
| Gradient boosting | 4.57 | 1.40% | 6.12 |
| Neural network (128-64-32) | 4.98 | 1.55% | 6.46 |
| Neural network (32) | 5.70 | 1.76% | 7.55 |
| Seasonal naive (last week, same weekday) | 5.85 | 1.81% | 7.37 |

**The linear model wins.** A ridge regression is 24% better than the deep MLP on
MAE and beats gradient boosting too.

The neural networks are not broken. Both clear the seasonal-naive baseline, so
they have genuinely learned the weekly and annual structure. They are simply the
wrong tool for this problem: roughly 1,460 training rows, fifteen engineered
features, and a target that is close to linear in its lags once seasonality is
encoded. That is territory where a penalised linear model is hard to beat, and
where a network mostly finds new ways to overfit.

## The part that matters more than the ranking

| Model | seed 0 | 1 | 2 | 3 | 4 | mean | std |
|---|---|---|---|---|---|---|---|
| ridge | 3.806 | 3.806 | 3.806 | 3.806 | 3.806 | 3.806 | **0.000** |
| gradient_boosting | 4.567 | 4.567 | 4.567 | 4.567 | 4.567 | 4.567 | **0.000** |
| mlp_deep | 4.803 | 4.414 | 4.866 | 4.594 | 4.507 | 4.637 | **0.193** |
| mlp_small | 5.535 | 5.531 | 5.389 | 5.047 | 5.110 | 5.322 | **0.231** |
| seasonal_naive | 5.847 | 5.847 | 5.847 | 5.847 | 5.847 | 5.847 | **0.000** |

**The neural networks are the only models whose score moves when nothing else
changes.** The deep MLP swings 0.45 MAE across seeds. That is larger than the gap
between gradient boosting and the deep MLP, which means a single-run comparison
could rank them either way and the analyst would never know.

Anyone reporting one MLP run as a result is reporting a sample of size one.

## One more thing the logs admit

The deep MLP emits a `ConvergenceWarning`: it reaches the 3,000-iteration cap
without the optimiser settling. Raising the cap or loosening early stopping would
silence the warning, but it would not change the conclusion, because the model is
already ahead of the seasonal baseline and still well behind a ridge regression.
It is recorded here rather than suppressed, since a warning quietly filtered out
of a notebook is how a known limitation becomes an unknown one.

## A documented weakness

On a reduced fixture of roughly 700 training rows, **both MLPs perform worse than
simply predicting the training mean**, while ridge and gradient boosting still
comfortably beat it. The networks are data-hungry, and this dataset is small.

That behaviour is pinned down by a test (`test_mlp_is_data_hungry_a_documented_limitation`)
which fails if a future change makes the MLP competitive on small samples, so the
claim above cannot quietly go stale.

## Guarding against leakage

The failure mode that makes a forecast look brilliant and be worthless:

- **Chronological split only.** No shuffling, no random sampling. Train ends
  before test begins, asserted in a test.
- **Rolling features are shifted before the window is applied**, so a day's own
  value can never enter its own rolling mean. Asserted against a hand-computed
  value.
- **Same-day temperature is dropped** and only `temp_lag_1` is used, because
  today's weather is not known when today's forecast is made.
- Lag alignment is asserted against the raw series rather than assumed.

## Testing

Twelve tests (all passing) covering series reproducibility, chronological and disjoint splits,
the rolling-window leak guard, lag alignment, absence of same-day weather, model
fitting and finite predictions, the sanity floor for deterministic models, the
documented MLP weakness, and MLP seed sensitivity.

## Data and honesty

**The series is synthetic.** It is generated in `src/data.py` from a fully
declared process: level, linear trend, weekly and annual seasonality, a
non-linear temperature response above 19°C, and an AR(1) noise term. Nothing here
is real consumption data from any company.

This matters for how the numbers should be read. A MAPE of 1.18% is a property of
a synthetic series with a known generating process, **not** a claim about
real-world forecast accuracy, where messy demand, meter faults and behavioural
change all widen the error considerably.

What transfers is the method, not the score: the leakage guards, the honest
baseline, the seed-stability check, and the willingness to publish that the
sophisticated model lost.

## Layout

```
src/data.py       synthetic series, feature engineering, chronological split
src/models.py     ridge, two MLPs, gradient boosting, seasonal naive, metrics
run_analysis.py   runs the benchmark and the seed-stability sweep
tests/            12 tests
output/           results.csv, seed_stability.csv, summary.json
```
