"""Build the hourly series from the minute-level original data.

Usage:
    python -m src.hourly --data-dir data --out-dir outputs/metrics
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_quality import VALUE_COLS, load_original

TARGET = "Global_active_power"
MIN_VALID_MINUTES = 45


def build_hourly(minute_df, min_valid=MIN_VALID_MINUTES):
    """Hourly mean target plus helper statistics, on a gapless hourly index.

    Hours with fewer than `min_valid` valid minutes become NaN in every column.
    Nothing is filled: gaps stay NaN.
    """
    g = minute_df.resample("h")
    hourly = g[VALUE_COLS].mean()
    hourly["gap_max"] = g[TARGET].max()
    hourly["gap_min"] = g[TARGET].min()
    hourly["gap_std"] = g[TARGET].std()
    hourly["n_valid"] = g[TARGET].count()
    hourly.loc[hourly["n_valid"] < min_valid, hourly.columns.drop("n_valid")] = np.nan
    return hourly.asfreq("h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder that contains raw/ and interim/")
    ap.add_argument("--out-dir", required=True, help="where to write the report file")
    a = ap.parse_args()
    data, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    minute = load_original(data / "raw" / "household_power_consumption.txt")
    hourly = build_hourly(minute)
    hourly.to_parquet(data / "interim" / "hourly.parquet")

    nv = hourly["n_valid"]
    valid = hourly[TARGET].notna()
    report = {
        "hours_total": int(len(hourly)),
        "index_start": str(hourly.index[0]), "index_end": str(hourly.index[-1]),
        "index_gapless": bool((hourly.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()),
        "hours_valid": int(valid.sum()),
        "hours_nan": int((~valid).sum()),
        "share_valid": float(valid.mean()),
        "nan_hours_with_zero_valid_minutes": int((nv == 0).sum()),
        "nan_hours_with_1_to_44_valid_minutes": int(((nv > 0) & (nv < MIN_VALID_MINUTES)).sum()),
        "valid_hours_with_45_to_59_minutes": int(((nv >= MIN_VALID_MINUTES) & (nv < 60)).sum()),
        "valid_hours_by_year": {str(y): int(n) for y, n in valid.groupby(hourly.index.year).sum().items()},
        "target_hourly_mean_describe": {k: round(float(v), 4) for k, v in hourly[TARGET].describe().items()},
    }

    # why the hourly MEAN and not the snapshot at minute :00 (what the old project used)
    snap = minute[TARGET][minute.index.minute == 0].reindex(hourly.index)
    both = pd.concat([snap.rename("snapshot"), hourly[TARGET].rename("mean")], axis=1).dropna()
    report["snapshot_vs_hourly_mean"] = {
        "hours_compared": int(len(both)),
        "corr": float(both.corr().iloc[0, 1]),
        "mae_kw": float((both["snapshot"] - both["mean"]).abs().mean()),
        "share_abs_diff_gt_0.5kw": float(((both["snapshot"] - both["mean"]).abs() > 0.5).mean()),
    }

    (out / "step2_hourly.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("\nhourly.head():")
    print(hourly.head(3).round(3).to_string())
    print(hourly.drop(columns=[]).describe().T[["count", "mean", "std", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
