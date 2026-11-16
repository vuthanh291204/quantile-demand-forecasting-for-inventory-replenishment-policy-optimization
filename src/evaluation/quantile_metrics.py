"""Quantile diagnostics per series; rates are fractions, not percentages.

Inputs follow metrics.py: one row per series and one column per day.
Inverted intervals count as uncovered; their widths are undefined.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import _masked_mean, _prepare, _wrap


def _finite_pair(y_true, y_pred, mask):
    y, f, keep, index = _prepare(y_true, y_pred, mask)
    if not np.isfinite(y).all() or not np.isfinite(f).all():
        raise ValueError("Values must be finite")
    return y, f, keep, index


def pinball_per_series(y_true, y_pred, alpha, mask=None):
    """Mean pinball loss at one level, in units of the target variable."""
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between zero and one")
    y, f, keep, index = _finite_pair(y_true, y_pred, mask)
    error = y - f
    loss = np.maximum(alpha * error, (alpha - 1) * error)
    return _wrap(_masked_mean(loss, keep), index)


def quantile_coverage_per_series(y_true, y_pred, mask=None, strict=False):
    """Fraction y <= q (default) or y < q (strict); ties are explicit."""
    y, f, keep, index = _finite_pair(y_true, y_pred, mask)
    covered = y < f if strict else y <= f
    return _wrap(_masked_mean(covered, keep), index)


def interval_diagnostics_per_series(y_true, lower, upper, mask=None):
    """Inclusive interval coverage, width on ordered bounds, and counts.

    Four disjoint rates sum to one: covered, below, above, invalid.
    Invalid means lower > upper; never silently swap bounds or drop misses.
    Width averages only ordered intervals; n_ordered_interval gives its count.
    All rates/widths are NaN if their respective denominators are zero.
    """
    y, lo, keep, index = _finite_pair(y_true, lower, mask)
    _, hi, _, _ = _finite_pair(y_true, upper, mask)
    ordered = lo <= hi
    return pd.DataFrame({
        "n_evaluated": keep.sum(axis=1),
        "n_ordered_interval": (keep & ordered).sum(axis=1),
        "interval_coverage": _masked_mean(ordered & (lo <= y) & (y <= hi), keep),
        "interval_below_rate": _masked_mean(ordered & (y < lo), keep),
        "interval_above_rate": _masked_mean(ordered & (y > hi), keep),
        "interval_invalid_rate": _masked_mean(~ordered, keep),
        "interval_mean_width_ordered": _masked_mean(hi - lo, keep & ordered),
    }, index=index)


def crossing_per_series(quantile_predictions, mask=None):
    """Fraction with any adjacent crossing; inputs must follow increasing alpha."""
    if len(quantile_predictions) < 2:
        raise ValueError("Need at least two quantile predictions")
    reference = quantile_predictions[0]
    arrays = []
    for predicted in quantile_predictions:
        _, f, keep, index = _finite_pair(reference, predicted, mask)
        arrays.append(f)
    crossed = np.any(np.diff(np.stack(arrays), axis=0) < 0, axis=0)
    return _wrap(_masked_mean(crossed, keep), index)
