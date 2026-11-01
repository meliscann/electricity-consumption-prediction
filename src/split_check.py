"""Step 4 control: does a random split inflate the score of a model that can memorize?

Same calendar-only features, same model, only the split differs.

Usage:
    python -m src.split_check --data-dir data --out-dir outputs/metrics
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from src.legacy_repro import TARGET, SAME_INSTANT, load, split

PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=0, verbose=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    df = load(Path(a.data_dir) / "interim" / "extracted" / "ModelDatasets" / "hourly_electricity_data.csv")
    df = df.drop(columns=SAME_INSTANT)  # keeps year, month, day, hour, weekday

    res = {}
    for kind in ("random", "chrono"):
        X_tr, y_tr, X_va, y_va, X_te, y_te = split(df, kind)
        m = lgb.LGBMRegressor(**PARAMS).fit(X_tr, y_tr)
        res[kind] = {}
        for name, X, y in (("val", X_va, y_va), ("test", X_te, y_te)):
            p = m.predict(X)
            base = np.full(len(y), y_tr.mean())
            res[kind][name] = {"mae": float(mean_absolute_error(y, p)), "r2": float(r2_score(y, p)),
                               "train_mean_mae": float(mean_absolute_error(y, base)),
                               "mae_gain_vs_train_mean_pct": float((1 - mean_absolute_error(y, p) / mean_absolute_error(y, base)) * 100)}
    (out / "step4_split_check_lgbm.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
