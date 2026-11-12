"""Simple forecasting baselines.

Every function takes
- history: array (n_series, n_days) with past sales, up to and including the forecast origin
- horizon: number of future days to forecast
and returns an array (n_series, horizon).
Only `history` is used, so nothing from the future can leak in.
"""
from __future__ import annotations

import numpy as np


def _check(history, horizon: int) -> np.ndarray:
    history = np.asarray(history, dtype=float)
    if history.ndim != 2 or history.shape[1] == 0:
        raise ValueError("history must be 2-D with at least one day")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    return history


def zero_forecast(history, horizon: int) -> np.ndarray:
    """Reference only: always forecast 0."""
    history = _check(history, horizon)
    return np.zeros((history.shape[0], horizon))


def naive_forecast(history, horizon: int) -> np.ndarray:
    """Every future day = the last observed day."""
    history = _check(history, horizon)
    return np.repeat(history[:, -1:], horizon, axis=1)


def seasonal_naive_forecast(history, horizon: int, period: int = 7) -> np.ndarray:
    """Every future day = the same weekday in the last `period` days (repeated)."""
    history = _check(history, horizon)
    if history.shape[1] < period:
        raise ValueError(f"need at least {period} days of history, got {history.shape[1]}")
    last = history[:, -period:]
    repeats = -(-horizon // period)  # ceiling division
    return np.tile(last, (1, repeats))[:, :horizon]


def _keep_flags(closed, n_days: int) -> np.ndarray:
    """True for days that count. `closed` marks store-closed days (e.g. Christmas)."""
    if closed is None:
        return np.ones(n_days, dtype=bool)
    closed = np.asarray(closed, dtype=bool)
    if closed.shape != (n_days,):
        raise ValueError(f"closed must have one value per history day ({n_days}), got {closed.shape}")
    return ~closed


def moving_average_forecast(history, horizon: int, window: int, closed=None) -> np.ndarray:
    """Every future day = mean of the last `window` days (closed days left out)."""
    history = _check(history, horizon)
    if window < 1 or history.shape[1] < window:
        raise ValueError(f"need at least {window} days of history, got {history.shape[1]}")
    keep = _keep_flags(closed, history.shape[1])[-window:]
    total = (history[:, -window:] * keep).sum(axis=1)
    level = total / max(int(keep.sum()), 1)
    return np.repeat(level[:, None], horizon, axis=1)


def seasonal_moving_average_forecast(
    history, horizon: int, weeks: int, period: int = 7, closed=None
) -> np.ndarray:
    """Each future day = mean of the same weekday over the last `weeks` weeks.

    weeks=1 gives exactly the seasonal naive forecast. Closed days are left out.
    """
    history = _check(history, horizon)
    n_days = period * weeks
    if weeks < 1 or history.shape[1] < n_days:
        raise ValueError(f"need at least {n_days} days of history, got {history.shape[1]}")
    keep = _keep_flags(closed, history.shape[1])[-n_days:].reshape(weeks, period)
    recent = history[:, -n_days:].reshape(history.shape[0], weeks, period)
    total = (recent * keep).sum(axis=1)  # (n_series, period)
    per_weekday = total / np.maximum(keep.sum(axis=0), 1)
    repeats = -(-horizon // period)
    return np.tile(per_weekday, (1, repeats))[:, :horizon]


def with_closed_days(fn, closed_by_day: np.ndarray, **kwargs):
    """Turn fn(history, horizon, closed=..., **kwargs) into forecast_fn(history, horizon).

    closed_by_day[i] is True when day number i + 1 is a closed day. The history
    given to the forecast always starts at day 1, so its closed flags are the
    first `n_days` entries.
    """

    def wrapped(history, horizon):
        return fn(history, horizon, closed=closed_by_day[: history.shape[1]], **kwargs)

    return wrapped