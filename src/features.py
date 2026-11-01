"""Step 6: features for forecasting hour T from information available at T - horizon.

Convention: the value stamped H is the mean over [H, H+1h) and is known once hour H is over.
A forecast for hour T made `horizon` hours ahead may use the hourly rows with index <= T - horizon.
Every non-calendar feature is therefore computed from data shifted by at least `horizon`.
Calendar features describe T itself and are known in advance.

Usage:
    python -m src.features --data-dir data --out-dir outputs/metrics
"""
import argparse
import json
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

TARGET = "Global_active_power"
LAGS = [1, 2, 3, 24, 48, 168]
WINDOWS = [6, 24, 168]
SUBHOUR_COLS = ["gap_max", "gap_min", "gap_std", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
GROUP_ORDER = ["calendar", "lag", "roll_profile", "holiday", "subhour"]


def _calendar(idx):
    return pd.DataFrame({
        "hour": idx.hour, "dayofweek": idx.day_of_week, "month": idx.month,
        "dayofyear": idx.day_of_year, "is_weekend": (idx.day_of_week >= 5).astype(int),
    }, index=idx)


def _holiday(idx):
    fr = holidays.France(years=range(idx.year.min() - 1, idx.year.max() + 2))
    hol = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in fr.keys()))
    day = idx.normalize()
    return pd.DataFrame({
        "is_holiday": day.isin(hol).astype(int),
        "is_day_before_holiday": (day + pd.Timedelta(days=1)).isin(hol).astype(int),
        "is_day_after_holiday": (day - pd.Timedelta(days=1)).isin(hol).astype(int),
    }, index=idx)


def _lags(y, h):
    return pd.DataFrame({f"lag_{k}": y.shift(k) for k in LAGS if k >= h})


def _mean_of_shifts(y, shifts, min_count):
    m = pd.concat([y.shift(s) for s in shifts], axis=1)
    return m.mean(axis=1).where(m.notna().sum(axis=1) >= min_count)


def _roll_profile(y, h):
    f = {}
    for w in WINDOWS:
        r = y.rolling(w, min_periods=w // 2)  # at least half of the window must be valid
        f[f"roll_mean_{w}"] = r.mean().shift(h)
        f[f"roll_std_{w}"] = r.std().shift(h)
        f[f"roll_max_{w}"] = r.max().shift(h)
    f["profile_same_hour_7d"] = _mean_of_shifts(y, [24 * k for k in range(1, 8)], 4)
    f["profile_same_weekday_hour_4w"] = _mean_of_shifts(y, [168 * k for k in range(1, 5)], 2)
    df = pd.DataFrame(f)
    # the profiles use lags >= 24 and are known at any horizon <= 24
    assert h <= 24, "profiles assume horizon <= 24"
    return df


def _subhour(hourly, h):
    return pd.DataFrame({f"last_{c}": hourly[c].shift(h) for c in SUBHOUR_COLS})


def build_features(hourly, horizon, groups=GROUP_ORDER):
    """Return (X, y). X has one row per hour T of the (gapless) hourly index, NaNs are kept."""
    assert (hourly.index.to_series().diff().dropna() == pd.Timedelta("1h")).all(), "index must be gapless hourly"
    y = hourly[TARGET]
    parts = []
    if "calendar" in groups:
        parts.append(_calendar(hourly.index))
    if "lag" in groups:
        parts.append(_lags(y, horizon))
    if "roll_profile" in groups:
        parts.append(_roll_profile(y, horizon))
    if "holiday" in groups:
        parts.append(_holiday(hourly.index))
    if "subhour" in groups:
        parts.append(_subhour(hourly, horizon))
    return pd.concat(parts, axis=1), y


def feature_groups(columns):
    """Map each feature column to its group (used by the ablation in Step 8)."""
    def grp(c):
        if c in ("hour", "dayofweek", "month", "dayofyear", "is_weekend"):
            return "calendar"
        if c.startswith("lag_"):
            return "lag"
        if c.startswith(("roll_", "profile_")):
            return "roll_profile"
        if c.startswith("is_"):
            return "holiday"
        return "subhour"
    return {c: grp(c) for c in columns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    data, out = Path(a.data_dir), Path(a.out_dir)
    hourly = pd.read_parquet(data / "interim" / "hourly.parquet")
    train = hourly.index < "2009-07-01"
    report = {}
    for h in (1, 24):
        X, y = build_features(hourly, h)
        X.assign(**{TARGET: y}).to_parquet(data / "interim" / f"features_h{h}.parquet")
        g = feature_groups(X.columns)
        ok = train & y.notna().to_numpy()
        corr = X[ok].apply(lambda c: c.corr(y[ok])).abs().sort_values(ascending=False)
        report[f"h{h}"] = {
            "n_rows": int(len(X)), "n_features": int(X.shape[1]),
            "features_per_group": pd.Series(g).value_counts().to_dict(),
            "nan_share_per_feature_train": {c: round(float(v), 4) for c, v in X[train].isna().mean().items()},
            "top_abs_corr_with_target_train": {c: round(float(v), 3) for c, v in corr.head(8).items()},
        }
    (out / "step6_features.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
