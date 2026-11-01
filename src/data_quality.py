"""Data quality helpers: find missing / filled blocks in the minute-level data.

Usage:
    python -m src.data_quality --data-dir data --out-dir outputs/metrics
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VALUE_COLS = [
    "Global_active_power", "Global_reactive_power", "Voltage",
    "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
]


def load_original(path):
    """Original UCI file: ';' separated, missing values are empty fields."""
    df = pd.read_csv(path, sep=";", na_values=["?", ""], low_memory=False)
    df.index = pd.to_datetime(df.pop("Date") + " " + df.pop("Time"), format="%d/%m/%Y %H:%M:%S")
    df.index.name = "datetime"
    return df[VALUE_COLS]


def load_filled(path):
    """Old project's minute file, where missing minutes were filled with numbers."""
    df = pd.read_csv(path, low_memory=False)
    df.index = pd.to_datetime(df.pop("Date") + " " + df.pop("Time"), format="%d/%m/%Y %H:%M:%S")
    df.index.name = "datetime"
    return df[VALUE_COLS]


def run_lengths(mask):
    """Return (start_pos, length) of each run of True in a boolean array."""
    m = np.asarray(mask, dtype=bool)
    padded = np.concatenate([[False], m, [False]])
    d = np.diff(padded.astype(np.int8))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return starts, ends - starts


def nan_blocks(df):
    """Blocks of consecutive rows where every value column is NaN."""
    starts, lens = run_lengths(df[VALUE_COLS].isna().all(axis=1).to_numpy())
    return pd.DataFrame({
        "start": df.index[starts], "end": df.index[starts + lens - 1], "minutes": lens,
    })


def identical_row_blocks(df, min_len=10, keep_first=True):
    """Blocks of >= min_len consecutive rows that are identical in all columns.

    keep_first=True keeps the first row of a run (it could be the last real
    measurement of a forward-filled gap). In the old project's file the gaps
    were filled with a constant, so the first row of the run is a fill too
    and keep_first=False is the right choice there (see the Step 1 report).
    Returns the runs (first row included) and a boolean mask of rows to NaN.
    """
    vals = df[VALUE_COLS].to_numpy()
    same_as_prev = np.zeros(len(df), dtype=bool)
    same_as_prev[1:] = (vals[1:] == vals[:-1]).all(axis=1)
    # a run of k repeated rows = k-1 consecutive "same_as_prev" flags (+ the first row)
    starts, lens = run_lengths(same_as_prev)
    keep = lens + 1 >= min_len
    starts, lens = starts[keep], lens[keep]
    mask = np.zeros(len(df), dtype=bool)
    for s, l in zip(starts, lens):
        # rows s..s+l-1 repeat row s-1; row s-1 is the first row of the run
        mask[(s if keep_first else s - 1):s + l] = True
    blocks = pd.DataFrame({
        "start": df.index[starts - 1], "end": df.index[starts + lens - 1], "minutes": lens + 1,
    })
    return blocks, mask


def mask_filled_blocks(df, min_len=10, keep_first=True):
    """Return a copy of df with repeated rows of long identical blocks set to NaN."""
    _, mask = identical_row_blocks(df, min_len, keep_first)
    out = df.copy()
    out.loc[mask, VALUE_COLS] = np.nan
    return out


def summarize(blocks, total_rows):
    return {
        "blocks": int(len(blocks)),
        "minutes": int(blocks["minutes"].sum()),
        "share_of_rows": float(blocks["minutes"].sum() / total_rows),
        "longest_minutes": int(blocks["minutes"].max()),
        "longest_start": str(blocks.loc[blocks["minutes"].idxmax(), "start"]),
        "longest_end": str(blocks.loc[blocks["minutes"].idxmax(), "end"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder that contains raw/ and interim/")
    ap.add_argument("--out-dir", required=True, help="where to write the report files")
    ap.add_argument("--min-len", type=int, default=10)
    a = ap.parse_args()
    data, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    orig = load_original(data / "raw" / "household_power_consumption.txt")
    filled = load_filled(data / "interim" / "extracted" / "ModelDatasets" / "electricity_data.csv")
    report = {"rows": int(len(orig))}

    # 1) real NaNs in the original
    nb = nan_blocks(orig)
    report["original_nan"] = summarize(nb, len(orig))
    report["original_nan"]["blocks_ge_10"] = int((nb["minutes"] >= a.min_len).sum())
    report["original_nan"]["minutes_in_blocks_ge_10"] = int(nb.loc[nb["minutes"] >= a.min_len, "minutes"].sum())
    nb.to_csv(out / "step1_original_nan_blocks.csv", index=False)

    # 2) do the two files agree on the minutes that were really measured?
    same_index = orig.index.equals(filled.index)
    real = ~orig.isna().all(axis=1).to_numpy()
    diff = (orig.to_numpy()[real] != filled.to_numpy()[real]).any(axis=1).sum() if same_index else None
    report["same_index"] = bool(same_index)
    report["measured_rows_that_differ_between_files"] = None if diff is None else int(diff)

    # 3) identical-row block method on the filled file
    blocks, mask = identical_row_blocks(filled, a.min_len)
    report["filled_identical_blocks"] = summarize(blocks, len(filled))
    blocks.to_csv(out / "step1_filled_identical_blocks.csv", index=False)

    # 4) overlap of the method with the true NaN minutes (both keep_first variants)
    true_nan = orig.isna().all(axis=1).to_numpy()
    report["overlap"] = {}
    for keep_first in (True, False):
        _, m = identical_row_blocks(filled, a.min_len, keep_first)
        report["overlap"][f"keep_first={keep_first}"] = {
            "flagged_minutes": int(m.sum()),
            "flagged_and_truly_nan": int((m & true_nan).sum()),
            "flagged_but_measured": int((m & ~true_nan).sum()),
            "truly_nan_but_not_flagged": int((~m & true_nan).sum()),
        }

    # 5) what constant was used to fill the gaps in the old file?
    fill_vals = filled[true_nan].drop_duplicates()
    report["fill_distinct_rows"] = int(len(fill_vals))
    report["fill_equals_original_column_mean"] = bool(
        np.allclose(fill_vals.iloc[0].to_numpy(), orig.mean().to_numpy(), rtol=1e-9)
    ) if len(fill_vals) == 1 else None
    report["fill_value_global_active_power"] = float(fill_vals.iloc[0]["Global_active_power"])

    # 6) identical blocks that are real measurements in the original
    fp = [(str(r.start), int(r.minutes)) for r in blocks.itertuples()
          if not true_nan[filled.index.get_indexer([r.start])[0] + 1]]
    report["identical_blocks_that_are_real_data"] = fp

    (out / "step1_data_quality.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
