"""Step 7: Ridge, LightGBM, HistGradientBoosting, Keras ANN and an averaged ensemble.

Everything is fitted on the training period and selected on the validation period.
The test period is never touched here.

Usage:
    python -m src.models --data-dir data --out-dir outputs
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.baselines import eval_mask
from src.evaluate import improvement_pct, metrics, split_masks
from src.features import build_features, feature_groups

SEED = 0
CAT = ["hour", "dayofweek", "month"]


def ridge_pipeline(columns, alpha):
    cat = [c for c in CAT if c in columns]
    num = [c for c in columns if c not in cat]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ])
    return make_pipeline(pre, Ridge(alpha=alpha))


def sample_lgbm_params(rng):
    return {
        "objective": str(rng.choice(["regression", "regression_l1", "huber"])),
        "num_leaves": int(rng.choice([15, 31, 63, 127])),
        "learning_rate": float(rng.choice([0.02, 0.03, 0.05, 0.08])),
        "min_child_samples": int(rng.choice([10, 20, 50, 100])),
        "colsample_bytree": float(rng.choice([0.5, 0.7, 0.9, 1.0])),
        "reg_lambda": float(rng.choice([0, 1, 5, 20])),
    }


def fit_lgbm(params, X_tr, y_tr, X_va, y_va):
    m = lgb.LGBMRegressor(n_estimators=3000, metric="l1", random_state=SEED, verbose=-1, **params)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def sample_hgb_params(rng):
    return {
        "loss": str(rng.choice(["squared_error", "absolute_error"])),
        "learning_rate": float(rng.choice([0.03, 0.05, 0.1])),
        "max_leaf_nodes": int(rng.choice([15, 31, 63])),
        "l2_regularization": float(rng.choice([0, 1, 10])),
        "max_iter": int(rng.choice([200, 400, 800])),
        "min_samples_leaf": int(rng.choice([20, 50, 100])),
    }


def fit_ann(X_tr, y_tr, X_va, y_va, seed):
    import keras
    from keras import layers
    pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(X_tr)  # train only
    keras.utils.set_random_seed(seed)
    m = keras.Sequential([layers.Input(shape=(X_tr.shape[1],)), layers.Dense(64, activation="relu"),
                          layers.Dense(32, activation="relu"), layers.Dense(1)])
    m.compile(optimizer="adam", loss="mae")
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    h = m.fit(pre.transform(X_tr), y_tr, validation_data=(pre.transform(X_va), y_va),
              epochs=100, batch_size=64, callbacks=[es], verbose=0)
    return pre, m, len(h.history["loss"]) - es.patience


def run_horizon(hourly, h, out, base):
    X, y = build_features(hourly, h)
    train, val, _ = split_masks(y.index)  # test mask is deliberately unused
    tr = train & y.notna().to_numpy()
    va_all = val & y.notna().to_numpy()
    ev = eval_mask(y, val, h)  # same hours as the baselines
    X_tr, y_tr, X_va, y_va, X_ev, y_ev = X[tr], y[tr], X[va_all], y[va_all], X[ev], y[ev]
    cols = list(X.columns)
    groups = feature_groups(cols)
    results, preds, selected = [], {}, {}

    def record(name, p, extra=None):
        m = metrics(y_ev, p)
        results.append({"horizon": h, "model": name, **m, "improvement_vs_best_baseline_pct": improvement_pct(m["mae"], base)})
        preds[name] = pd.Series(np.asarray(p), index=y_ev.index)
        if extra:
            selected[name] = extra

    # 1) Ridge: calendar+lag as planned, and all features as a fairer linear reference
    for tag, cs in (("ridge_calendar_lag", [c for c in cols if groups[c] in ("calendar", "lag")]),
                    ("ridge_all_features", cols)):
        best = None
        for alpha in (0.1, 1, 10, 100, 1000):
            pipe = ridge_pipeline(cs, alpha).fit(X_tr[cs], y_tr)
            mae = metrics(y_ev, pipe.predict(X_ev[cs]))["mae"]
            if best is None or mae < best[0]:
                best = (mae, alpha, pipe)
        record(tag, best[2].predict(X_ev[cs]), {"alpha": best[1], "n_features": len(cs)})

    # 2) LightGBM: 25 random trials, early stopping on the validation period
    rng = np.random.default_rng(SEED)
    trials = []
    for i in range(25):
        p = sample_lgbm_params(rng)
        t0 = time.time()
        m = fit_lgbm(p, X_tr, y_tr, X_va, y_va)
        sc = metrics(y_ev, m.predict(X_ev))
        trials.append({"trial": i, **p, "best_iteration": int(m.best_iteration_), **sc, "seconds": round(time.time() - t0, 1)})
    tdf = pd.DataFrame(trials).sort_values("mae")
    tdf.to_csv(out / "metrics" / f"step7_lgbm_search_h{h}.csv", index=False)
    b = tdf.iloc[0]
    bp = {k: (b[k].item() if hasattr(b[k], "item") else b[k]) for k in sample_lgbm_params(rng)}
    m = fit_lgbm(bp, X_tr, y_tr, X_va, y_va)
    record("lightgbm", m.predict(X_ev), {"params": bp, "best_iteration": int(m.best_iteration_)})
    imp = pd.Series(m.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
    selected["lightgbm"]["top_gain_features"] = (imp / imp.sum()).head(8).round(3).to_dict()

    # 3) HistGradientBoosting: 10 random trials (no internal random validation split)
    rng = np.random.default_rng(SEED + 1)
    best = None
    hgb_trials = []
    for i in range(10):
        p = sample_hgb_params(rng)
        mdl = HistGradientBoostingRegressor(early_stopping=False, random_state=SEED, **p).fit(X_tr, y_tr)
        mae = metrics(y_ev, mdl.predict(X_ev))["mae"]
        hgb_trials.append({"trial": i, **p, "mae": mae})
        if best is None or mae < best[0]:
            best = (mae, p, mdl)
    pd.DataFrame(hgb_trials).sort_values("mae").to_csv(out / "metrics" / f"step7_hgb_search_h{h}.csv", index=False)
    record("hist_gradient_boosting", best[2].predict(X_ev), {"params": best[1]})

    # 4) Keras ANN: one untuned configuration, 3 seeds averaged
    ann_p, epochs = [], []
    for s in range(3):
        pre, net, ep = fit_ann(X_tr, y_tr, X_va, y_va, s)
        ann_p.append(net.predict(pre.transform(X_ev), verbose=0).ravel())
        epochs.append(int(ep))
    record("ann_3_seed_mean", np.mean(ann_p, axis=0),
           {"best_epochs_per_seed": epochs, "single_seed_mae": [metrics(y_ev, p)["mae"] for p in ann_p]})

    # 5) Ensemble: equal-weight mean of the best 2, 3 or 4 of {lightgbm, hgb, ann, best ridge}
    pool = ["lightgbm", "hist_gradient_boosting", "ann_3_seed_mean",
            min(("ridge_calendar_lag", "ridge_all_features"), key=lambda n: [r for r in results if r["model"] == n][0]["mae"])]
    pool.sort(key=lambda n: [r for r in results if r["model"] == n][0]["mae"])
    for k in (2, 3, 4):
        record(f"ensemble_top{k}", np.mean([preds[n] for n in pool[:k]], axis=0), {"members": pool[:k]})

    res = pd.DataFrame(results)
    preds_df = pd.DataFrame(preds).assign(actual=y_ev)
    preds_df.to_parquet(out / "models" / f"step7_val_preds_h{h}.parquet")
    return res, selected, int(ev.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)
    hourly = pd.read_parquet(Path(a.data_dir) / "interim" / "hourly.parquet")
    best = json.loads((out / "metrics" / "step5_best_baseline.json").read_text())["best"]
    base_mae = {int(k): v["val_mae"] for k, v in best.items()}  # best baseline per horizon, from step 5
    all_res, all_sel = [], {}
    for h in (1, 24):
        t0 = time.time()
        res, sel, n = run_horizon(hourly, h, out, base_mae[h])
        all_res.append(res)
        all_sel[f"h{h}"] = {"eval_hours": n, "baseline_val_mae": base_mae[h], "models": sel}
        print(f"\n=== horizon {h}h ({n} validation hours, {time.time() - t0:.0f}s) ===", flush=True)
        print(res.round(4).to_string(index=False), flush=True)
    pd.concat(all_res).to_csv(out / "metrics" / "step7_models_val.csv", index=False)
    (out / "metrics" / "step7_selected.json").write_text(json.dumps(all_sel, indent=2, default=str))


if __name__ == "__main__":
    main()
