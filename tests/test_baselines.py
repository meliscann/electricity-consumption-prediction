"""Baselines must not use any value after the forecast origin (T - horizon)."""
import numpy as np
import pandas as pd
import pytest

from src.baselines import HORIZONS, make_baselines


@pytest.mark.parametrize("horizon", HORIZONS)
def test_baselines_ignore_values_after_origin(horizon):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2008-01-01", periods=24 * 60, freq="h")
    y = pd.Series(rng.random(len(idx)), index=idx)
    base = pd.DataFrame(make_baselines(y, horizon))
    for pos in rng.choice(np.arange(24 * 40, len(idx)), size=30, replace=False):
        corrupted = y.copy()
        corrupted.iloc[pos - horizon + 1:] = 999.0  # destroy everything after the origin
        altered = pd.DataFrame(make_baselines(corrupted, horizon))
        pd.testing.assert_series_equal(base.iloc[pos], altered.iloc[pos], check_names=False)
