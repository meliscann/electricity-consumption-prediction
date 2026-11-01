"""Step 4: re-run the 2024 setup, measure it on a test set, then change one thing at a time.

Uses the old project's hourly file (minute :00 snapshots, filled gaps included) so that
only the three prescribed changes differ between rows: scaling, chronological split,
removal of same-instant measurements.

Usage:
    python -m src.legacy_repro --data-dir data --out-dir outputs/metrics --seeds 3
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import pandas as pd
from keras import layers, regularizers
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET = "Global_active_power"
SAME_INSTANT = ["Global_reactive_power", "Voltage", "Global_intensity",
                "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
SUBS = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
VAL_START, TEST_START = "2009-07-01", "2010-01-01"

# name -> (split, scale, columns to drop)
CONFIGS = {
    "A_old_setup":              ("random", False, []),
    "B_+scaling":               ("random", True, []),
    "C_+chronological":         ("chrono", True, []),
    "D1_-Global_intensity":     ("chrono", True, ["Global_intensity"]),
    "D2_-intensity,-submeters": ("chrono", True, ["Global_intensity"] + SUBS),
    "D3_-all_same_instant":     ("chrono", True, SAME_INSTANT),
    # control: same features as D3 but random split, to isolate the split effect
    "E_D3_but_random_split":    ("random", True, SAME_INSTANT),
}


def load(path):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    dt = df["datetime"]
    df["year"], df["month"], df["day"] = dt.dt.year, dt.dt.month, dt.dt.day
    df["hour"], df["weekday"] = dt.dt.hour, dt.dt.weekday
    return df.dropna()


def split(df, kind):
    X, y = df.drop(columns=[TARGET, "datetime"]), df[TARGET]
    if kind == "random":  # same calls and order as the 2024 code
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_te, X_va, y_te, y_va = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)
    else:
        d = df["datetime"]
        tr, va, te = d < VAL_START, (d >= VAL_START) & (d < TEST_START), d >= TEST_START
        X_tr, y_tr, X_va, y_va, X_te, y_te = X[tr], y[tr], X[va], y[va], X[te], y[te]
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def build_model(n_in):
    return keras.Sequential([
        layers.Input(shape=(n_in,)),
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
        layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
        layers.Dense(1, activation="linear"),
    ])


def scores(y, p):
    return {"mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": float(r2_score(y, p))}


def run_one(df, cfg, seed, epochs):
    kind, scale, drop = cfg
    X_tr, y_tr, X_va, y_va, X_te, y_te = split(df.drop(columns=drop), kind)
    if scale:  # fit on the training part only
        sc = StandardScaler().fit(X_tr)
        X_tr, X_va, X_te = sc.transform(X_tr), sc.transform(X_va), sc.transform(X_te)
    keras.utils.set_random_seed(seed)
    m = build_model(X_tr.shape[1])
    m.compile(optimizer="adam", loss="mse", metrics=["mae"])
    h = m.fit(np.asarray(X_tr), y_tr, epochs=epochs, batch_size=32,
              validation_data=(np.asarray(X_va), y_va), verbose=0)
    va_mae = np.array(h.history["val_mae"])
    return {
        "n_train": len(y_tr), "n_val": len(y_va), "n_test": len(y_te), "n_features": X_tr.shape[1],
        "val": scores(y_va, m.predict(np.asarray(X_va), verbose=0).ravel()),
        "test": scores(y_te, m.predict(np.asarray(X_te), verbose=0).ravel()),
        # how much the validation MAE jumps from one epoch to the next
        "val_mae_epoch_jump": float(np.abs(np.diff(va_mae)).mean()),
        "val_mae_history": va_mae.tolist(),
    }, (y_tr, y_va, y_te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--only", default="", help="comma separated config names to run (default: all)")
    ap.add_argument("--tag", default="", help="suffix for output file names")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load(Path(a.data_dir) / "interim" / "extracted" / "ModelDatasets" / "hourly_electricity_data.csv")

    runs, rows = {}, []
    names = [n for n in CONFIGS if not a.only or n in a.only.split(",")]
    for name in names:
        cfg = CONFIGS[name]
        runs[name] = []
        for seed in range(a.seeds):
            r, _ = run_one(df, cfg, seed, a.epochs)
            r["seed"] = seed
            runs[name].append(r)
            print(f"{name} seed={seed} val_mae={r['val']['mae']:.3f} test_mae={r['test']['mae']:.3f} "
                  f"test_r2={r['test']['r2']:.3f}", flush=True)
        for split_name in ("val", "test"):
            row = {"config": name, "split": split_name}
            for k in ("mae", "rmse", "r2"):
                v = np.array([r[split_name][k] for r in runs[name]])
                row[k], row[k + "_sd"] = v.mean(), v.std()
            row["n_features"] = runs[name][0]["n_features"]
            row["epoch_jump"] = float(np.mean([r["val_mae_epoch_jump"] for r in runs[name]]))
            rows.append(row)

    # reference: always predict the training mean (chronological split)
    _, y_tr, y_va, y_te = None, *split(df, "chrono")[1::2]
    ref = {s: scores(y, np.full(len(y), y_tr.mean())) for s, y in (("val", y_va), ("test", y_te))}

    pd.DataFrame(rows).to_csv(out / f"step4_summary{a.tag}.csv", index=False)
    (out / f"step4_runs{a.tag}.json").write_text(json.dumps({"runs": runs, "train_mean_reference": ref,
                                                     "seeds": a.seeds, "epochs": a.epochs}, indent=1))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("train-mean reference:", json.dumps(ref))


if __name__ == "__main__":
    main()
