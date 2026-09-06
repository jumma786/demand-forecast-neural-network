# Does a Neural Network Actually Help? — On Real GB Demand Data

A neural network built properly, benchmarked honestly against simpler models on
real electricity demand, and **reported losing**.

```bash
python -m src.download      # fetch the NESO CSVs (not committed)
python run_analysis.py
python -m pytest tests/ -q
```

---

## The data

**NESO (National Energy System Operator) Historic Demand Data**, half-hourly
settlement periods 2019–2024, NESO Open Data Licence.
<https://www.neso.energy/data-portal/historic-demand-data>

105,222 half-hourly readings aggregated to **2,192 complete days** of mean
National Demand. Days without a full complement of settlement periods are dropped
rather than averaged, since a partial day would look like a demand dip that never
happened.

After lag features (including a 365-day lag) the usable set is 1,827 days:
**1,462 for training, and the whole of 2024 held out as the test year.**

## The result

| Model | MAE (MW) | MAPE | RMSE |
|---|---|---|---|
| **Ridge regression** | **985.7** | **3.83%** | **1,263.7** |
| Gradient boosting | 1,016.3 | 3.94% | 1,275.9 |
| Neural network (128-64-32) | 1,040.1 | 4.01% | 1,315.4 |
| Seasonal naive (last week, same weekday) | 1,689.4 | 6.33% | 2,206.6 |
| Neural network (32) | **8,258.2** | **32.47%** | 9,807.5 |

**A ridge regression wins.** It beats gradient boosting and the deep network,
and all three comfortably beat the seasonal-naive baseline.

The deep network is close, 5.5% worse on MAE, and it clearly learned the
structure. It is simply not the right tool for 1,462 rows of a strongly
autoregressive series once seasonality is encoded as features. That is territory
a penalised linear model is very hard to beat in.

## The small network's failure is instructive, not embarrassing

The 32-unit network lands at **8,258 MAE against ridge's 986** — eight times
worse, and a 32% MAPE.

This is a scaling effect, and worth understanding rather than patching away. The
target is GB demand in megawatts, of order 27,000. Features are standardised; the
target is not. A small network cannot span that output range within its iteration
budget, while ridge and gradient boosting are entirely indifferent to it.

**Neural networks are sensitive to target scale in a way linear models are not.**
That is a real cost of choosing one, and it belongs in the comparison rather than
being quietly fixed before the table is printed. The behaviour is pinned by a
test that fails if it ever stops being true.

## The part that matters more than the ranking

MAE across five random seeds, everything else identical:

| Model | 0 | 1 | 2 | 3 | 4 | mean | std |
|---|---|---|---|---|---|---|---|
| ridge | 985.7 | 985.7 | 985.7 | 985.7 | 985.7 | 985.7 | **0.00** |
| gradient_boosting | 1,016.3 | 1,016.3 | 1,016.3 | 1,016.3 | 1,016.3 | 1,016.3 | **0.00** |
| mlp_deep | 1,018.1 | 1,025.2 | 1,022.3 | 1,060.4 | 1,060.8 | 1,037.4 | **21.37** |
| seasonal_naive | 1,689.4 | 1,689.4 | 1,689.4 | 1,689.4 | 1,689.4 | 1,689.4 | **0.00** |
| mlp_small | 8,725.0 | 8,425.8 | 9,023.7 | 8,712.7 | 8,960.5 | 8,769.5 | **236.88** |

**The neural networks are the only models whose score moves when nothing else
changes.** The deep MLP swings 43 MW across seeds, comparable to its entire
54 MW deficit against ridge. On a single run it could plausibly appear to beat
gradient boosting or to lose by twice as much, and the analyst would not know
which they were looking at.

Anyone reporting one MLP run as a result is reporting a sample of size one.

## Guarding against leakage

The failure mode that makes a forecast look brilliant and be worthless:

- **Chronological split only.** Train ends before test begins, asserted in a test.
- **Rolling features are shifted before the window applies**, so a day's own value
  cannot enter its own rolling mean. Checked against a hand-computed value.
- **Lag alignment is asserted against the raw series**, not assumed.
- Partial days are excluded, so no artificial dips enter the lags.

## Testing

14 tests covering the real-data span and completeness, physical plausibility,
winter-exceeds-summer, chronological and disjoint splits, the rolling-window leak
guard, lag alignment, finite predictions from every model, ridge beating the
naive baseline, **the documented small-MLP scaling failure**, and MLP seed
sensitivity.

## Honesty

The data is real and openly licensed, and the test year is genuinely held out.

The model set is deliberately plain: no weather covariate, because the NESO demand
file carries no temperature and **temperature is the largest known omitted driver
of electricity demand**. A production forecast would need it. The 2020 COVID
period is left in the training data untouched.

So a 3.83% MAPE here should be read as what these features support on this series,
not as a claim about achievable forecast accuracy in an operational setting.

What transfers is the method: an honest baseline, leakage guards that are tested
rather than asserted, a seed-stability check, and the willingness to publish that
the more sophisticated model lost.

## Layout

```
src/data.py       load NESO half-hourly, aggregate to daily, build features
src/models.py     ridge, two MLPs, gradient boosting, seasonal naive, metrics
src/download.py   fetch the source CSVs
run_analysis.py   benchmark plus the five-seed stability sweep
tests/            14 tests
output/           results.csv, seed_stability.csv, summary.json
```
