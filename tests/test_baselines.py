import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import backtest
from src.forecast.baselines import naive_forecast, seasonal_naive_forecast, zero_forecast


def test_naive_repeats_last_value():
    hist = np.array([[1, 2, 3], [0, 0, 5]])
    assert naive_forecast(hist, 4).tolist() == [[3, 3, 3, 3], [5, 5, 5, 5]]


def test_seasonal_naive_repeats_last_period():
    hist = np.array([[1, 2, 3, 4, 5, 6]])
    assert seasonal_naive_forecast(hist, 5, period=3).tolist() == [[4, 5, 6, 4, 5]]


def test_seasonal_naive_needs_enough_history():
    with pytest.raises(ValueError):
        seasonal_naive_forecast(np.array([[1, 2, 3]]), 4)  # default period is 7


def test_zero_forecast_shape():
    out = zero_forecast(np.ones((3, 10)), 28)
    assert out.shape == (3, 28) and not out.any()


def _weekly_data(n_days: int = 84) -> pd.DataFrame:
    """Two series with a pure weekly pattern: day k has value (k-1) % 7 (times 1 or 2)."""
    pattern = np.arange(n_days) % 7
    values = np.vstack([pattern, 2 * pattern])
    cols = [f"d_{i}" for i in range(1, n_days + 1)]
    return pd.DataFrame(values, index=pd.Index(["a", "b"], name="id"), columns=cols)


def test_seasonal_naive_is_perfect_on_a_pure_weekly_pattern():
    closed = pd.Series(False, index=range(1, 85))
    res = backtest(_weekly_data(), seasonal_naive_forecast, [(57, 70), (71, 84)], closed)
    for metric in ["mae", "rmse", "smape", "rmsse"]:
        assert res[metric].abs().max() == pytest.approx(0)


def test_closed_days_are_ignored():
    all_days = _weekly_data()
    all_days.loc["a", "d_75"] = 100  # a strange day...
    closed = pd.Series(False, index=range(1, 85))
    closed.loc[75] = True  # ...but the store was closed
    res = backtest(all_days, seasonal_naive_forecast, [(71, 84)], closed)
    assert res.loc["a", "mae"] == pytest.approx(0)


def test_forecast_function_only_sees_the_past():
    closed = pd.Series(False, index=range(1, 85))
    seen = []

    def spy(history, horizon):
        seen.append((history.shape[1], horizon))
        return np.zeros((history.shape[0], horizon))

    backtest(_weekly_data(), spy, [(57, 70), (71, 84)], closed)
    assert seen == [(56, 14), (70, 14)]