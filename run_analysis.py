"""Benchmark a neural network against simpler baselines. Report what happens."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import make_series, add_features, chronological_split, FEATURES
from src.models import run_all

OUT = Path("output"); OUT.mkdir(exist_ok=True)

df = add_features(make_series())
train, test = chronological_split(df, test_days=365)
print(f"train {train.index.min().date()} to {train.index.max().date()}  n={len(train)}")
print(f"test  {test.index.min().date()} to {test.index.max().date()}  n={len(test)}\n")

results = run_all(train, test, FEATURES)
print(results.to_string(index=False))
results.to_csv(OUT / "results.csv", index=False)

# stability across seeds: is any ranking difference real or just noise?
runs = []
for s in (0, 1, 2, 3, 4):
    r = run_all(train, test, FEATURES, seed=s).set_index("model")["mae"]
    runs.append(r.rename(s))
stab = pd.concat(runs, axis=1)
stab["mean"] = stab.mean(axis=1); stab["std"] = stab.iloc[:, :5].std(axis=1)
stab = stab.sort_values("mean")
print("\n=== MAE ACROSS 5 SEEDS ===")
print(stab.round(3).to_string())
stab.to_csv(OUT / "seed_stability.csv")

best = results.iloc[0]
mlp_best = results[results.model.str.startswith("mlp")].sort_values("mae").iloc[0]
summary = {
    "winner": best["model"], "winner_mae": best["mae"],
    "best_mlp": mlp_best["model"], "best_mlp_mae": mlp_best["mae"],
    "mlp_gap_vs_winner_mae": float(mlp_best["mae"] - best["mae"]),
    "mlp_beats_seasonal_naive": bool(
        mlp_best["mae"] < results.set_index("model").loc["seasonal_naive", "mae"]),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print("\n", json.dumps(summary, indent=2))
