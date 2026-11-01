"""Step 8: final evaluation of the selected LightGBM, rolling-origin backtest, importance, error by hour, ablation.

The model, its parameters and the number of trees come from step 7 (validation only).
The test period is scored once. With --dry-run the validation period plays the role of the test period
(models are fitted on the training period only), which lets us check the code without touching the test set.

Usage:
    python -m src.final_eval --data-dir data --out-dir outputs [--dry-run]
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.baselines import eval_mask, make_baselines
from src.eda import BLUE, GRAY, INK
from src.evaluate import improvement_pct, metrics, split_masks
from src.features import GROUP_ORDER, build_features
from src.models import SEED

GROUP_LABELS = {"calendar": "calendar", "lag": "+ lags", "roll_profile": "+ rolling/profiles",
                "holiday": "+ holidays", "subhour": "+ sub-hour/meters"}


def fit_fixed(params, n_trees, X, y):
    return lgb.LGBMRegressor(n_estimators=n_trees, random_state=SEED, verbose=-1, **params).fit(X, y)


def final_test(X, y, h, params, n_trees, base_name, fit_mask, target_mask):
    ev = eval_mask(y, target_mask, h)
    model = fit_fixed(params, n_trees, X[fit_mask], y[fit_mask])
    pred = pd.Series(model.predict(X[ev]), index=y.index[ev])
    bases = make_baselines(y, h)
    out = {"n_eval_hours": int(ev.sum()), "n_valid_target_hours": int((target_mask & y.notna().to_numpy()).sum()),
           "model": metrics(y[ev], pred), "baselines": {n: metrics(y[ev], b[ev]) for n, b in bases.items()}}
    out["reference_baseline"] = base_name
    out["improvement_vs_reference_baseline_pct"] = improvement_pct(out["model"]["mae"], out["baselines"][base_name]["mae"])
    all_valid = target_mask & y.notna().to_numpy()  # also score every valid hour (GBM tolerates NaN features)
    out["model_on_all_valid_hours"] = metrics(y[all_valid], model.predict(X[all_valid]))
    return out, model, pred, bases[base_name][ev]


def by_hour(y, pred, base_pred):
    d = pd.DataFrame({"y": y.loc[pred.index], "model": pred, "baseline": base_pred})
    g = d.groupby(d.index.hour)
    return pd.DataFrame({"model_mae": g.apply(lambda x: (x.y - x.model).abs().mean()),
                         "baseline_mae": g.apply(lambda x: (x.y - x.baseline).abs().mean())})


def backtest(X, y, h, params, n_trees, base_name, end_excl, folds=5, days=60):
    rows = []
    for i in range(folds):
        start = end_excl - pd.Timedelta(days=days * (folds - i))
        stop = start + pd.Timedelta(days=days)
        window = (y.index >= start) & (y.index < stop)
        ev = eval_mask(y, window, h)
        fit = (y.index < start) & y.notna().to_numpy()
        m = fit_fixed(params, n_trees, X[fit], y[fit])
        mae_m = metrics(y[ev], m.predict(X[ev]))["mae"]
        mae_b = metrics(y[ev], make_baselines(y, h)[base_name][ev])["mae"]
        rows.append({"fold": i + 1, "start": str(start.date()), "end": str((stop - pd.Timedelta(hours=1)).date()),
                     "n_train": int(fit.sum()), "n_eval_hours": int(ev.sum()), "model_mae": mae_m,
                     "baseline_mae": mae_b, "improvement_pct": improvement_pct(mae_m, mae_b)})
    return pd.DataFrame(rows)


def ablation(hourly, h, params, base_val_mae):
    """Cumulative feature groups, fitted on the training period, early stopping and scoring on validation."""
    rows = []
    for k in range(len(GROUP_ORDER)):
        groups = GROUP_ORDER[: k + 1]
        X, y = build_features(hourly, h, groups)
        train, val, _ = split_masks(y.index)
        tr, va = train & y.notna().to_numpy(), val & y.notna().to_numpy()
        ev = eval_mask(y, val, h)
        m = lgb.LGBMRegressor(n_estimators=3000, metric="l1", random_state=SEED, verbose=-1, **params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        sc = metrics(y[ev], m.predict(X[ev]))
        rows.append({"horizon": h, "groups": GROUP_LABELS[GROUP_ORDER[k]], "n_features": X.shape[1],
                     "trees": int(m.best_iteration_), **sc, "improvement_vs_baseline_pct": improvement_pct(sc["mae"], base_val_mae)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dry-run", action="store_true", help="use validation as the evaluation period (test untouched)")
    a = ap.parse_args()
    out = Path(a.out_dir)
    tag = "_dry" if a.dry_run else ""
    hourly = pd.read_parquet(Path(a.data_dir) / "interim" / "hourly.parquet")
    sel = json.loads((out / "metrics" / "step7_selected.json").read_text())
    b5 = json.loads((out / "metrics" / "step5_best_baseline.json").read_text())["best"]

    results, tables = {}, {"hour": [], "importance": [], "backtest": [], "ablation": []}
    for h in (1, 24):
        X, y = build_features(hourly, h)
        train, val, test = split_masks(y.index)
        valid = y.notna().to_numpy()
        params, n_trees = sel[f"h{h}"]["models"]["lightgbm"]["params"], sel[f"h{h}"]["models"]["lightgbm"]["best_iteration"]
        base_name, base_val_mae = b5[str(h)]["baseline"], b5[str(h)]["val_mae"]
        if a.dry_run:
            fit_mask, target_mask, end_excl = train & valid, val, pd.Timestamp("2010-01-01")
        else:
            fit_mask, target_mask, end_excl = (train | val) & valid, test, y.index[-1] + pd.Timedelta(hours=1)

        res, model, pred, base_pred = final_test(X, y, h, params, n_trees, base_name, fit_mask, target_mask)
        res["trees"], res["params"], res["n_fit_rows"] = n_trees, params, int(fit_mask.sum())
        hh = by_hour(y, pred, base_pred).assign(horizon=h)
        imp = pd.Series(model.booster_.feature_importance("gain"), index=X.columns)
        imp = (imp / imp.sum()).sort_values(ascending=False).rename("gain_share").to_frame().assign(horizon=h)
        bt = backtest(X, y, h, params, n_trees, base_name, end_excl)
        res["backtest_folds_won"] = int((bt["model_mae"] < bt["baseline_mae"]).sum())
        ab = ablation(hourly, h, params, base_val_mae)
        results[f"h{h}"] = res
        tables["hour"].append(hh.reset_index(names="hour")); tables["importance"].append(imp.reset_index(names="feature"))
        tables["backtest"].append(bt.assign(horizon=h)); tables["ablation"].append(ab)
        pred.rename("pred").to_frame().assign(actual=y.loc[pred.index], baseline=base_pred).to_parquet(out / "models" / f"step8_eval_preds_h{h}{tag}.parquet")
        model.booster_.save_model(str(out / "models" / f"lgbm_h{h}{tag}.txt"))

    for k, v in tables.items():
        pd.concat(v, ignore_index=True).to_csv(out / "metrics" / f"step8_{k}{tag}.csv", index=False)
    (out / "metrics" / f"step8_results{tag}.json").write_text(json.dumps(results, indent=2, default=str))

    # figures
    hh, imp, bt = (pd.concat(tables[k], ignore_index=True) for k in ("hour", "importance", "backtest"))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for axi, h in zip(ax, (1, 24)):
        d = hh[hh.horizon == h]
        axi.plot(d.hour, d.baseline_mae, color=GRAY, lw=1.6, label="baseline")
        axi.plot(d.hour, d.model_mae, color=BLUE, lw=1.8, label="LightGBM")
        axi.set(title=f"MAE by hour of day, {h} h ahead", xlabel="hour of day", xticks=range(0, 24, 3)); axi.legend(frameon=False)
    ax[0].set_ylabel("MAE (kW)")
    fig.tight_layout(); fig.savefig(out / "figures" / f"step8_error_by_hour{tag}.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for axi, h in zip(ax, (1, 24)):
        d = imp[imp.horizon == h].head(10).iloc[::-1]
        axi.barh(d.feature, d.gain_share, color=BLUE); axi.set(title=f"Importance (gain), {h} h ahead", xlabel="share of total gain")
    fig.tight_layout(); fig.savefig(out / "figures" / f"step8_importance{tag}.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for axi, h in zip(ax, (1, 24)):
        d = bt[bt.horizon == h]; x = np.arange(len(d))
        axi.bar(x - .2, d.baseline_mae, .4, color=GRAY, label="baseline"); axi.bar(x + .2, d.model_mae, .4, color=BLUE, label="LightGBM")
        axi.set(title=f"Rolling-origin backtest, {h} h ahead", xticks=x, xticklabels=[f"{s[5:]}" for s in d.start], xlabel="fold start (month-day)"); axi.legend(frameon=False)
    ax[0].set_ylabel("MAE (kW)")
    fig.tight_layout(); fig.savefig(out / "figures" / f"step8_backtest{tag}.png", dpi=130); plt.close(fig)

    for h in (1, 24):
        r = results[f"h{h}"]
        print(f"\n=== {h} h ahead{' (DRY RUN: validation as eval period)' if a.dry_run else ''} ===")
        print("eval hours", r["n_eval_hours"], "of", r["n_valid_target_hours"], "valid | model", {k: round(v, 4) for k, v in r["model"].items()},
              "| ref baseline", r["reference_baseline"], {k: round(v, 4) for k, v in r["baselines"][r["reference_baseline"]].items()},
              "| improvement %", round(r["improvement_vs_reference_baseline_pct"], 2), "| backtest folds won", r["backtest_folds_won"], "/5")
    print(pd.concat(tables["backtest"]).round(4).to_string(index=False))
    print(pd.concat(tables["ablation"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
