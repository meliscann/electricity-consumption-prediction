"""Step 5: naive baselines, evaluated on the validation period only.

A forecast for hour T made `horizon` hours earlier may only use values at or before T - horizon.
Every baseline below uses lags that are >= horizon, which is checked by an assertion.

Usage:
    python -m src.baselines --data-dir data --out-dir outputs/metrics
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import metrics, split_masks

TARGET = "Global_active_power"
HORIZONS = (1, 24)


def lag_mean(y, lags, horizon):
    """Mean of y at the given lags. NaN unless every lag is available (strict)."""
    assert min(lags) >= horizon, f"lag {min(lags)} is not known at horizon {horizon}"
    return pd.concat([y.shift(k) for k in lags], axis=1).mean(axis=1, skipna=False)


def make_baselines(y, horizon):
    """Dict name -> prediction series (index = target hour T) for a given horizon."""
    out = {}
    if horizon <= 1:  # t-1 is only known when forecasting one hour ahead
        out["t-1"] = lag_mean(y, [1], horizon)
    out["t-24"] = lag_mean(y, [24], horizon)
    out["t-168"] = lag_mean(y, [168], horizon)
    out["mean_same_hour_last_7_days"] = lag_mean(y, [24 * k for k in range(1, 8)], horizon)
    out["mean_same_weekday_hour_last_4_weeks"] = lag_mean(y, [168 * k for k in range(1, 5)], horizon)
    return out


def eval_mask(y, mask, horizon):
    """Hours inside `mask` where the target and every baseline are defined (the common scoring set)."""
    preds = pd.DataFrame(make_baselines(y, horizon))
    return mask & y.notna().to_numpy() & preds.notna().all(axis=1).to_numpy()


def evaluate(y, mask, horizon):
    """Score all baselines on the hours where the target and every baseline are defined."""
    preds = pd.DataFrame(make_baselines(y, horizon))
    ok = eval_mask(y, mask, horizon)
    rows = []
    for name in preds.columns:
        rows.append({"horizon": horizon, "baseline": name, "n_hours": int(ok.sum()),
                     **metrics(y[ok], preds.loc[ok, name])})
    return pd.DataFrame(rows), int(ok.sum()), int((mask & y.notna().to_numpy()).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    h = pd.read_parquet(Path(a.data_dir) / "interim" / "hourly.parquet")
    y = h[TARGET]
    _, val, _ = split_masks(y.index)  # the test mask is deliberately never used here

    tables, coverage, best = [], {}, {}
    for horizon in HORIZONS:
        t, n_eval, n_valid = evaluate(y, val, horizon)
        tables.append(t)
        coverage[horizon] = {"evaluated_hours": n_eval, "valid_target_hours_in_val": n_valid,
                             "share": n_eval / n_valid}
        b = t.loc[t["mae"].idxmin()]
        best[horizon] = {"baseline": b["baseline"], "val_mae": float(b["mae"]),
                         "val_rmse": float(b["rmse"]), "val_r2": float(b["r2"])}
    res = pd.concat(tables, ignore_index=True)
    res.to_csv(out / "step5_baselines_val.csv", index=False)
    (out / "step5_best_baseline.json").write_text(json.dumps({"best": best, "coverage": coverage}, indent=2))
    print(res.round(3).to_string(index=False))
    print(json.dumps({"best": best, "coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()
