"""Statistical forecasting models from the statsforecast library.

Same interface as the simple baselines: forecast_fn(history, horizon) where
history is an array (n_series, n_days) with past sales and the result is an
array (n_series, horizon). Only `history` is used, so nothing from the future
can leak in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecast.baselines import _check


def _forecast_with_statsforecast(
    history, horizon: int, model, fallback=None, stats: dict | None = None
) -> np.ndarray:
    """Fit `model` (a statsforecast model object) on each series and forecast.

    - Each series is used from its first day with sales (days before launch are dropped).
    - A series with no sales in the history is forecast as 0.
    - Negative forecasts are set to 0 (sales cannot be negative). If `stats` is a
      dict, it counts how many forecasts were made and how many were negative.
    """
    from statsforecast import StatsForecast

    history = _check(history, horizon)
    n_series, n_days = history.shape
    out = np.zeros((n_series, horizon))

    sold = history > 0
    series_idx = np.flatnonzero(sold.any(axis=1))
    if len(series_idx) == 0:
        return out
    first = sold.argmax(axis=1)  # first day with sales

    keep = np.arange(n_days)[None, :] >= first[series_idx, None]
    rows, days = np.nonzero(keep)
    df = pd.DataFrame(
        {"unique_id": series_idx[rows], "ds": days + 1, "y": history[series_idx[rows], days]}
    )

    sf = StatsForecast(models=[model], freq=1, n_jobs=1, fallback_model=fallback)
    fc = sf.forecast(h=horizon, df=df).sort_values(["unique_id", "ds"])
    ids = fc["unique_id"].to_numpy().reshape(len(series_idx), horizon)[:, 0]
    if not np.array_equal(ids, series_idx):
        raise RuntimeError("statsforecast returned the series in an unexpected order")
    values = fc[model.alias].to_numpy().reshape(len(series_idx), horizon)
    if np.isnan(values).any():
        raise ValueError(f"{model.alias} returned NaN forecasts")

    if stats is not None:
        stats["forecasts"] = stats.get("forecasts", 0) + values.size
        stats["negative"] = stats.get("negative", 0) + int((values < 0).sum())
    out[series_idx] = np.maximum(values, 0.0)
    return out


def ets_forecast(
    history, horizon: int, season_length: int = 7, model: str = "ZZZ", stats: dict | None = None
) -> np.ndarray:
    """ETS (exponential smoothing). model="ZZZ": the library picks error, trend and season."""
    from statsforecast.models import AutoETS, HistoricAverage

    return _forecast_with_statsforecast(
        history,
        horizon,
        AutoETS(season_length=season_length, model=model),
        fallback=HistoricAverage(),  # used by the library if ETS fails on a series
        stats=stats,
    )


def croston_sba_forecast(history, horizon: int, stats: dict | None = None) -> np.ndarray:
    """Croston method (SBA variant) for intermittent demand.

    It estimates the average size of a sale and the average number of days between
    sales, and forecasts size / gap as one flat level for every future day.
    """
    from statsforecast.models import CrostonSBA

    return _forecast_with_statsforecast(history, horizon, CrostonSBA(), stats=stats)