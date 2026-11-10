"""Forecast accuracy metrics, computed per series.

Conventions
- y_true, y_pred: tables with one row per series and one column per day
  (pandas DataFrame or numpy array, same shape).
- mask (optional): True = count this day, False = ignore it (e.g. Christmas).
  Either one value per day (1-D) or one per series and day (2-D).
- Every metric returns ONE number per series. Average across series afterwards
  with `summarize`, so one big series does not drown out the small ones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _prepare(y_true, y_pred, mask):
    """Check inputs and return (y, f, keep, index) as numpy arrays."""
    if isinstance(y_true, pd.DataFrame) and isinstance(y_pred, pd.DataFrame):
        if not y_true.index.equals(y_pred.index) or not y_true.columns.equals(y_pred.columns):
            raise ValueError("y_true and y_pred must have the same rows and columns")
    index = y_true.index if isinstance(y_true, pd.DataFrame) else None

    y = np.asarray(y_true, dtype=float)
    f = np.asarray(y_pred, dtype=float)
    if y.ndim != 2 or y.shape != f.shape:
        raise ValueError(f"y_true and y_pred must be 2-D with the same shape, got {y.shape} and {f.shape}")
    if np.isnan(y).any() or np.isnan(f).any():
        raise ValueError("NaN found in y_true or y_pred")

    if mask is None:
        keep = np.ones(y.shape, dtype=bool)
    else:
        keep = np.asarray(mask, dtype=bool)
        if keep.ndim == 1:
            if keep.shape[0] != y.shape[1]:
                raise ValueError("1-D mask must have one value per day")
            keep = np.broadcast_to(keep, y.shape)
        if keep.shape != y.shape:
            raise ValueError("mask must have one value per day or the same shape as y_true")
    return y, f, keep, index


def _masked_mean(values: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Mean over kept days of each row; NaN if a row has no kept day."""
    n = keep.sum(axis=1)
    total = np.where(keep, values, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, total / n, np.nan)


def _wrap(values: np.ndarray, index):
    return pd.Series(values, index=index) if index is not None else values


def mae_per_series(y_true, y_pred, mask=None):
    """Mean absolute error (in units sold per day), one value per series."""
    y, f, keep, index = _prepare(y_true, y_pred, mask)
    return _wrap(_masked_mean(np.abs(y - f), keep), index)


def rmse_per_series(y_true, y_pred, mask=None):
    """Root mean squared error (in units), one value per series."""
    y, f, keep, index = _prepare(y_true, y_pred, mask)
    return _wrap(np.sqrt(_masked_mean((y - f) ** 2, keep)), index)


def smape_per_series(y_true, y_pred, mask=None, both_zero: str = "zero"):
    """Symmetric MAPE in percent (0 to 200), one value per series.

    One day: 2 * |y - f| / (|y| + |f|).
    When y = 0 and f = 0 the formula is 0/0, so:
      both_zero="zero": that day scores 0 (perfect forecast)   [default]
      both_zero="skip": that day is left out of the average
    A forecast above 0 on a day with 0 sales always scores the maximum (200%).
    """
    if both_zero not in ("zero", "skip"):
        raise ValueError('both_zero must be "zero" or "skip"')
    y, f, keep, index = _prepare(y_true, y_pred, mask)
    denom = np.abs(y) + np.abs(f)
    zero_denom = denom == 0
    term = np.where(zero_denom, 0.0, 2 * np.abs(y - f) / np.where(zero_denom, 1.0, denom))
    if both_zero == "skip":
        keep = keep & ~zero_denom
    return _wrap(100 * _masked_mean(term, keep), index)


def rmsse_per_series(y_true, y_pred, y_history, mask=None):
    """Root mean squared scaled error, one value per series.

    RMSSE = sqrt( mean squared error on the evaluation days
                  / mean squared one-step change on the history ).
    The history part starts at the first day with sales, so days before a
    product was launched do not count. About 1.0 means "as wrong as guessing
    that tomorrow equals today on the past data"; lower is better.

    y_history: table with one row per series (same order as y_true) holding the
    days BEFORE the evaluation window. NaN if the history has no sales or never changes.
    """
    y, f, keep, index = _prepare(y_true, y_pred, mask)
    if isinstance(y_history, pd.DataFrame) and index is not None:
        if not y_history.index.equals(index):
            raise ValueError("y_history must have the same rows as y_true")
    h = np.asarray(y_history, dtype=float)
    if h.ndim != 2 or h.shape[0] != y.shape[0]:
        raise ValueError("y_history must be 2-D with one row per series")
    if np.isnan(h).any():
        raise ValueError("NaN found in y_history")

    mse = _masked_mean((y - f) ** 2, keep)

    diffs = np.diff(h, axis=1) ** 2              # diffs[:, t] = (h[t+1] - h[t])^2
    has_sale = (h > 0).any(axis=1)
    first = np.argmax(h > 0, axis=1)             # first day with sales
    cols = np.arange(diffs.shape[1])
    use = (cols[None, :] >= first[:, None]) & has_sale[:, None]
    scale = _masked_mean(diffs, use)

    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(scale > 0, np.sqrt(mse / scale), np.nan)
    return _wrap(out, index)


def _stats(values: pd.Series) -> dict:
    return {"mean": values.mean(), "median": values.median(), "n_series": int(values.notna().sum())}


def summarize(per_series: pd.Series, groups: pd.Series | None = None) -> pd.DataFrame:
    """Mean, median and number of series of a per-series metric.

    per_series: Series indexed by series id.
    groups: optional Series indexed by series id (e.g. store_id or volume_group);
            adds one row per group. Series with NaN are skipped.
    """
    rows = {"all": _stats(per_series)}
    if groups is not None:
        aligned = groups.reindex(per_series.index)
        for name, part in per_series.groupby(aligned):
            rows[f"group {name}"] = _stats(part)
    return pd.DataFrame(rows).T