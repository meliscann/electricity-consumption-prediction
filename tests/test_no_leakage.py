"""Leakage test: features of hour T must not change if every measurement after T - horizon is destroyed.

Run after adding any new feature group. Uses a synthetic hourly frame, so it needs no data files.
"""
import numpy as np
import pandas as pd
import pytest

from src.features import GROUP_ORDER, SUBHOUR_COLS, TARGET, build_features

HORIZONS = (1, 24)
CUMULATIVE_GROUPS = [GROUP_ORDER[: i + 1] for i in range(len(GROUP_ORDER))]


@pytest.fixture(scope="module")
def hourly():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2008-01-01", periods=24 * 400, freq="h")
    df = pd.DataFrame({c: rng.random(len(idx)) for c in [TARGET] + SUBHOUR_COLS}, index=idx)
    df = df.mask(rng.random(df.shape) < 0.02)  # sprinkle NaNs like the real data
    return df.asfreq("h")


def row_changes(builder, hourly, pos, horizon):
    """True if the feature row at position `pos` differs after corrupting data after the origin."""
    base, _ = builder(hourly, horizon)
    corrupted = hourly.copy()
    corrupted.iloc[pos - horizon + 1:] = 999.0  # everything strictly after the origin T - horizon
    after, _ = builder(corrupted, horizon)
    a, b = base.iloc[pos], after.iloc[pos]
    return not ((a == b) | (a.isna() & b.isna())).all()


@pytest.mark.parametrize("horizon", HORIZONS)
@pytest.mark.parametrize("groups", CUMULATIVE_GROUPS, ids=lambda g: "+".join(g))
def test_features_ignore_data_after_origin(hourly, horizon, groups):
    rng = np.random.default_rng(1)
    builder = lambda d, h: build_features(d, h, groups)
    for pos in rng.choice(np.arange(24 * 40, len(hourly)), size=25, replace=False):
        assert not row_changes(builder, hourly, int(pos), horizon), f"leak at row {pos}, groups {groups}"


def test_the_test_detects_a_deliberate_leak(hourly):
    """Control: an unshifted rolling mean must be caught, otherwise the test above proves nothing."""
    def leaky(d, h):
        X, y = build_features(d, h)
        X["leaky_roll_mean_6"] = d[TARGET].rolling(6, min_periods=3).mean()  # missing .shift(h)
        return X, y
    assert row_changes(leaky, hourly, 24 * 100, 24)


def test_24h_horizon_has_no_short_lags(hourly):
    X, _ = build_features(hourly, 24)
    assert not {"lag_1", "lag_2", "lag_3"} & set(X.columns)
