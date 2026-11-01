"""Step 3: exploratory analysis of the hourly series. Writes figures and a JSON summary.

Usage:
    python -m src.eda --data-dir data --out-dir outputs
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf

from src.data_quality import run_lengths

TARGET = "Global_active_power"
BASE_COLS = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
             "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
BLUE, GRAY, INK = "#2f6f9f", "#8a8f98", "#22262b"
SEQ = LinearSegmentedColormap.from_list("seq", ["#eef3f8", "#2f6f9f", "#12304a"])
DIV = LinearSegmentedColormap.from_list("div", ["#b5532f", "#ececec", "#2f6f9f"])
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GRAY, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK,
    "ytick.color": INK, "axes.grid": True, "grid.color": "#e3e5e8", "grid.linewidth": 0.7,
    "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
})


def longest_valid_segment(s):
    starts, lens = run_lengths(s.notna().to_numpy())
    i = int(np.argmax(lens))
    return s.iloc[starts[i]:starts[i] + lens[i]]


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    data, out = Path(a.data_dir), Path(a.out_dir)
    figs, mets = out / "figures", out / "metrics"
    figs.mkdir(parents=True, exist_ok=True)
    mets.mkdir(parents=True, exist_ok=True)

    h = pd.read_parquet(data / "interim" / "hourly.parquet")
    y = h[TARGET]
    yv = y.dropna()
    R = {}

    # 1) distribution
    R["distribution"] = {"skew": float(yv.skew()), "skew_of_log": float(np.log(yv).skew()),
                         "mean": float(yv.mean()), "median": float(yv.median()),
                         "p95": float(yv.quantile(.95)), "p99": float(yv.quantile(.99)), "max": float(yv.max())}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].hist(yv, bins=60, color=BLUE)
    ax[0].axvline(yv.mean(), color=INK, lw=1, ls="--"); ax[0].axvline(yv.median(), color=GRAY, lw=1, ls="--")
    ax[0].set(title="Hourly mean power, linear scale", xlabel="kW", ylabel="hours")
    ax[0].text(yv.mean() + .1, ax[0].get_ylim()[1] * .9, f"mean {yv.mean():.2f}", fontsize=9)
    ax[0].text(yv.median() + .1, ax[0].get_ylim()[1] * .8, f"median {yv.median():.2f}", fontsize=9, color=GRAY)
    ax[1].hist(yv, bins=np.logspace(np.log10(yv.min()), np.log10(yv.max()), 60), color=BLUE)
    ax[1].set(xscale="log", title="Same data, log x-axis", xlabel="kW (log scale)", ylabel="hours")
    ax[1].set_xticks([0.2, 0.5, 1, 2, 5])
    ax[1].set_xticklabels(["0.2", "0.5", "1", "2", "5"])
    ax[1].minorticks_off()
    save(fig, figs / "step3_1_distribution.png")

    # 2) hour x weekday heatmap
    d = pd.DataFrame({"y": y, "hour": y.index.hour, "dow": y.index.dayofweek})
    pv = d.pivot_table(index="hour", columns="dow", values="y", aggfunc="mean")
    pv.columns = DAYS
    R["hour_dow"] = {"max_cell": [int(pv.stack().idxmax()[0]), pv.stack().idxmax()[1], float(pv.stack().max())],
                     "min_cell": [int(pv.stack().idxmin()[0]), pv.stack().idxmin()[1], float(pv.stack().min())],
                     "mean_by_hour": pv.mean(axis=1).round(3).to_dict(), "mean_by_dow": pv.mean().round(3).to_dict()}
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    im = ax.imshow(pv.values, aspect="auto", cmap=SEQ, origin="lower")
    ax.set(xticks=range(7), xticklabels=DAYS, yticks=range(0, 24, 3), title="Mean power by hour and weekday (kW)",
           xlabel="weekday", ylabel="hour of day")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="kW")
    save(fig, figs / "step3_2_hour_weekday_heatmap.png")

    # 3) monthly and yearly
    m = y.resample("MS").mean()
    mcount = y.resample("MS").count()
    m = m[mcount >= 500]  # drop the two stub months (2006-12, 2010-11) and any nearly empty month
    cal = y.groupby(y.index.month).mean()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
    ax[0].plot(m.index, m.values, color=BLUE, lw=1.6, marker="o", ms=3.5)
    ax[0].set(title="Monthly mean power", ylabel="kW", ylim=(0, None))
    ax[1].bar(range(1, 13), cal.values, color=BLUE)
    ax[1].set(title="Mean by calendar month (all years)", xlabel="month", xticks=range(1, 13), ylabel="kW")
    save(fig, figs / "step3_3_monthly.png")
    R["monthly"] = {"calendar_month_mean": cal.round(3).to_dict(),
                    "lowest_month": int(cal.idxmin()), "highest_month": int(cal.idxmax()),
                    "months_shown": int(len(m))}

    jn = y[y.index.month <= 11]  # compare only Jan-Nov (2010 ends in November)
    jn = jn[jn.index.year >= 2007]
    ymean = jn.groupby(jn.index.year).mean()
    daily = jn.resample("D").mean().dropna()
    dby = {int(k): g.values for k, g in daily.groupby(daily.index.year)}
    R["yearly_jan_nov"] = {"mean": ymean.round(3).to_dict(),
                           "daily_mean_median": {k: float(np.median(v)) for k, v in dby.items()},
                           "daily_mean_std": {k: float(np.std(v)) for k, v in dby.items()},
                           "change_2010_vs_2007_pct": float((ymean[2010] / ymean[2007] - 1) * 100)}
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].bar([str(i) for i in ymean.index], ymean.values, color=BLUE)
    ax[0].set(title="Mean power, Jan-Nov of each year", ylabel="kW")
    ax[1].boxplot(list(dby.values()), tick_labels=[str(k) for k in dby], showfliers=True,
                  boxprops=dict(color=BLUE), medianprops=dict(color=INK), whiskerprops=dict(color=GRAY),
                  capprops=dict(color=GRAY), flierprops=dict(markeredgecolor=GRAY, markersize=3))
    ax[1].set(title="Daily mean power, Jan-Nov", ylabel="kW")
    save(fig, figs / "step3_4_yearly.png")

    # low-consumption stretches (house probably empty): >= 5 consecutive days with daily mean < 0.4 kW
    dm = y.resample("D").mean()
    starts, lens = run_lengths((dm < 0.4).to_numpy())
    R["low_stretches_daily_mean_lt_0.4kW_ge_5days"] = [
        {"start": str(dm.index[s].date()), "days": int(l)} for s, l in zip(starts, lens) if l >= 5]
    R["share_days_daily_mean_lt_0.4"] = float((dm.dropna() < 0.4).mean())

    # 4) ACF / PACF / STL on the longest gap-free stretch (diagnostics only, nothing is filled)
    seg = longest_valid_segment(y)
    R["segment"] = {"start": str(seg.index[0]), "end": str(seg.index[-1]), "hours": int(len(seg)),
                    "days": round(len(seg) / 24, 1)}
    nl = 200
    ac, pc = acf(seg, nlags=nl, fft=True), pacf(seg, nlags=nl)
    R["acf"] = {str(k): round(float(ac[k]), 3) for k in [1, 2, 3, 12, 23, 24, 25, 48, 72, 96, 120, 144, 167, 168, 169]}
    R["pacf"] = {str(k): round(float(pc[k]), 3) for k in [1, 2, 3, 12, 23, 24, 25, 48, 168]}
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True)
    for axi, v, name in [(ax[0], ac, "ACF"), (ax[1], pc, "PACF")]:
        axi.vlines(range(nl + 1), 0, v, color=BLUE, lw=1.2)
        axi.axhline(0, color=GRAY, lw=.8)
        axi.set(title=f"{name} of hourly mean power (lags up to {nl} h)", ylabel=name)
        for k in (24, 48, 168):
            axi.axvline(k, color=GRAY, lw=.8, ls=":")
    ax[1].set_xlabel("lag (hours)")
    ax[1].set_xticks([0, 24, 48, 72, 96, 120, 144, 168, 192])
    save(fig, figs / "step3_5_acf_pacf.png")

    stl = STL(seg, period=24, robust=True).fit()
    v = lambda s: float(np.var(s))
    R["stl"] = {"period": 24, "var_share_trend": v(stl.trend) / v(seg), "var_share_seasonal": v(stl.seasonal) / v(seg),
                "var_share_resid": v(stl.resid) / v(seg),
                "seasonal_strength": float(max(0, 1 - v(stl.resid) / v(stl.seasonal + stl.resid))),
                "trend_strength": float(max(0, 1 - v(stl.resid) / v(stl.trend + stl.resid)))}
    z = seg.index[-24 * 14:]
    fig, ax = plt.subplots(4, 1, figsize=(10, 7.5))
    ax[0].plot(seg.index, seg.values, color=BLUE, lw=.5); ax[0].set(title=f"Observed ({seg.index[0].date()} to {seg.index[-1].date()})", ylabel="kW")
    ax[1].plot(stl.trend.index, stl.trend.values, color=BLUE, lw=1.2); ax[1].set(title="Trend", ylabel="kW")
    ax[2].plot(z, stl.seasonal.loc[z].values, color=BLUE, lw=1.2); ax[2].set(title="Daily seasonal component (last 14 days)", ylabel="kW")
    ax[3].plot(stl.resid.index, stl.resid.values, color=GRAY, lw=.5); ax[3].set(title="Residual", ylabel="kW")
    save(fig, figs / "step3_6_stl.png")

    # 5) correlation matrix and the leakage numbers
    hv = h[BASE_COLS].dropna()
    C = hv.corr()
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(C.values, cmap=DIV, vmin=-1, vmax=1)
    short = ["Active\npower", "Reactive\npower", "Voltage", "Intensity", "Sub 1", "Sub 2", "Sub 3"]
    ax.set(xticks=range(7), yticks=range(7), xticklabels=short, yticklabels=short, title="Correlation, hourly means")
    ax.grid(False)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{C.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(C.values[i, j]) > .6 else INK)
    fig.colorbar(im, ax=ax)
    save(fig, figs / "step3_7_correlation.png")

    sub_sum = hv[["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]].sum(axis=1)
    apparent = hv["Voltage"] * hv["Global_intensity"] / 1000
    ok = y.notna() & y.shift(1).notna() & y.shift(24).notna()
    R["leakage"] = {
        "corr_target_vs_intensity_hourly": float(C.loc[TARGET, "Global_intensity"]),
        "corr_target_vs_voltage_x_intensity": float(np.corrcoef(hv[TARGET], apparent)[0, 1]),
        "r2_target_from_intensity_alone": float(C.loc[TARGET, "Global_intensity"] ** 2),
        "corr_target_vs_sub_meter_sum": float(np.corrcoef(hv[TARGET], sub_sum)[0, 1]),
        "corr_target_vs_reactive": float(C.loc[TARGET, "Global_reactive_power"]),
        "legit_corr_t_vs_t_minus_1": float(np.corrcoef(y[ok], y.shift(1)[ok])[0, 1]),
        "legit_corr_t_vs_t_minus_24": float(np.corrcoef(y[ok], y.shift(24)[ok])[0, 1]),
        "legit_corr_t_vs_t_minus_168": float(np.corrcoef(y.dropna().align(y.shift(168).dropna(), join="inner")[0],
                                                          y.dropna().align(y.shift(168).dropna(), join="inner")[1])[0, 1]),
    }
    (mets / "step3_eda.json").write_text(json.dumps(R, indent=2, default=str))
    print(json.dumps(R, indent=2, default=str))


if __name__ == "__main__":
    main()
