"""Step 9: flag unexpected hours with thresholds taken from validation residuals of the 1-hour model.

Residual = actual - prediction. Thresholds are the 0.5% and 99.5% quantiles of the validation residuals
(model fitted on the training period only). They are written to disk before the test period is scored.
By construction 1% of validation hours fall outside; a larger share on test indicates distribution shift.

Usage:
    python -m src.anomaly --data-dir data --out-dir outputs
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eda import BLUE, GRAY, INK
from src.evaluate import split_masks
from src.features import build_features
from src.final_eval import fit_fixed

Q_LO, Q_HI = 0.005, 0.995


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - w), float(c + w)


def episodes(idx):
    """Number of runs of consecutive flagged hours."""
    if len(idx) == 0:
        return 0
    return int((pd.Series(idx).diff() != pd.Timedelta("1h")).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    hourly = pd.read_parquet(Path(a.data_dir) / "interim" / "hourly.parquet")
    sel = json.loads((out / "metrics" / "step7_selected.json").read_text())["h1"]["models"]["lightgbm"]
    X, y = build_features(hourly, 1)
    train, val, test = split_masks(y.index)
    valid = y.notna().to_numpy()

    model = fit_fixed(sel["params"], sel["best_iteration"], X[train & valid], y[train & valid])  # training period only
    res_val = y[val & valid] - model.predict(X[val & valid])
    lo, hi = (float(q) for q in res_val.quantile([Q_LO, Q_HI]))
    thr = {"lower": lo, "upper": hi, "quantiles": [Q_LO, Q_HI], "n_val_hours": int(len(res_val)),
           "val_share_outside": float(((res_val < lo) | (res_val > hi)).mean())}
    (out / "metrics" / "step9_thresholds.json").write_text(json.dumps(thr, indent=2))  # frozen before touching test

    res_test = y[test & valid] - model.predict(X[test & valid])
    flag = (res_test < lo) | (res_test > hi)
    n, k = len(res_test), int(flag.sum())
    ci = wilson(k, n)
    f = res_test[flag]
    rep = {
        "thresholds_kw": {"lower": lo, "upper": hi},
        "validation": {"n": int(len(res_val)), "share_outside": thr["val_share_outside"],
                       "residual_mean": float(res_val.mean()), "residual_std": float(res_val.std()),
                       "mae": float(res_val.abs().mean())},
        "test": {"n": n, "flagged": k, "share_flagged": k / n, "share_ci95_wilson_independent_hours": ci,
                 "above_upper": int((res_test > hi).sum()), "below_lower": int((res_test < lo).sum()),
                 "residual_mean": float(res_test.mean()), "residual_std": float(res_test.std()),
                 "mae": float(res_test.abs().mean()), "episodes_of_consecutive_flagged_hours": episodes(f.index),
                 "flagged_by_month": {int(m): int(c) for m, c in f.groupby(f.index.month).size().items()},
                 "flagged_by_hour_top5": {int(m): int(c) for m, c in f.groupby(f.index.hour).size().sort_values(ascending=False).head(5).items()},
                 "share_flagged_by_month": {int(m): round(float(v), 4) for m, v in flag.groupby(flag.index.month).mean().items()}},
        "residual_quantiles_val_vs_test": {str(q): [float(res_val.quantile(q)), float(res_test.quantile(q))]
                                           for q in (0.005, 0.025, 0.25, 0.5, 0.75, 0.975, 0.995)},
    }
    # sensitivity: the step 8 model (train + validation) scored on test, same thresholds
    p8 = pd.read_parquet(out / "models" / "step8_eval_preds_h1.parquet")
    r8 = p8["actual"] - p8["pred"]
    rep["sensitivity_final_model_on_test"] = {"n": int(len(r8)), "share_flagged": float(((r8 < lo) | (r8 > hi)).mean()),
                                              "residual_std": float(r8.std())}
    pd.DataFrame({"residual": f, "actual": y[f.index], "direction": np.where(f > 0, "above", "below")}).to_csv(
        out / "metrics" / "step9_flagged_test_hours.csv")
    (out / "metrics" / "step9_anomaly.json").write_text(json.dumps(rep, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 1.8]})
    bins = np.linspace(-2.5, 2.5, 80)
    ax[0].hist(res_val.clip(-2.5, 2.5), bins=bins, density=True, alpha=.6, color=GRAY, label="validation")
    ax[0].hist(res_test.clip(-2.5, 2.5), bins=bins, density=True, alpha=.6, color=BLUE, label="test")
    for t in (lo, hi):
        ax[0].axvline(t, color=INK, lw=1, ls="--")
    ax[0].set(title="1 h residuals (clipped at ±2.5 kW)", xlabel="actual - predicted (kW)", ylabel="density"); ax[0].legend(frameon=False)
    ax[1].plot(res_test.index, res_test.values, color=GRAY, lw=.5)
    ax[1].scatter(f.index, f.values, color=BLUE, s=7, zorder=3, label=f"flagged ({k})")
    for t in (lo, hi):
        ax[1].axhline(t, color=INK, lw=1, ls="--")
    ax[1].set(title="Test residuals over time", ylabel="kW"); ax[1].legend(frameon=False)
    ax[1].xaxis.set_major_locator(mdates.MonthLocator()); ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout(); fig.savefig(out / "figures" / "step9_anomaly.png", dpi=130); plt.close(fig)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
