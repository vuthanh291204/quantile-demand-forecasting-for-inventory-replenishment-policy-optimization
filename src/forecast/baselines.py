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