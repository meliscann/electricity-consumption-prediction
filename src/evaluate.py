"""Shared evaluation helpers: chronological split bounds and metrics."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Chronological split (CLAUDE.md): train 2006-12 -> 2009-06, validation 2009-07 -> 2009-12, test 2010-01 ->
VAL_START = "2009-07-01"
TEST_START = "2010-01-01"


def split_masks(index):
    """Boolean masks (train, val, test) over a DatetimeIndex."""
    train = index < VAL_START
    val = (index >= VAL_START) & (index < TEST_START)
    test = index >= TEST_START
    return train, val, test


def metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def improvement_pct(model_mae, baseline_mae):
    """Percent MAE improvement over a baseline (positive = model is better)."""
    return float((1 - model_mae / baseline_mae) * 100)
