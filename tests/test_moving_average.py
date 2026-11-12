import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import backtest
from src.forecast.baselines import (
    moving_average_forecast,
    seasonal_moving_average_forecast,
    seasonal_naive_forecast,
    with_closed_days,
)


def test_moving_average_by_hand():
    hist = np.array([[1, 2, 3, 4]])
    assert moving_average_forecast(hist, 3, window=2).tolist() == [[3.5, 3.5, 3.5]]


def test_moving_average_leaves_out_closed_days():
    hist = np.array([[10, 0, 20]])
    closed = np.array([False, True, False])
    assert moving_average_forecast(hist, 2, window=3, closed=closed).tolist() == [[15.0, 15.0]]


def test_moving_average_input_checks():
    with pytest.raises(ValueError):
        moving_average_forecast(np.ones((1, 5)), 2, window=10)  # not enough history
    with pytest.raises(ValueError):
        moving_average_forecast(np.ones((1, 5)), 2, window=3, closed=np.array([False] * 4))


def test_seasonal_average_by_hand():
    hist = np.array([[1, 2, 3, 5, 6, 7]])  # two "weeks" of 3 days
    out = seasonal_moving_average_forecast(hist, 4, weeks=2, period=3)
    assert out.tolist() == [[3, 4, 5, 3]]  # (1+5)/2, (2+6)/2, (3+7)/2, then repeat


def test_seasonal_average_leaves_out_closed_days():
    hist = np.array([[1, 2, 3, 100, 6, 7]])
    closed = np.array([False, False, False, True, False, False])
    out = seasonal_moving_average_forecast(hist, 3, weeks=2, period=3, closed=closed)
    assert out.tolist() == [[1, 4, 5]]  # first slot uses only the open day


def test_seasonal_average_with_one_week_is_seasonal_naive():
    rng = np.random.default_rng(0)
    hist = rng.integers(0, 10, size=(3, 30))
    np.testing.assert_allclose(
        seasonal_moving_average_forecast(hist, 10, weeks=1),
        seasonal_naive_forecast(hist, 10),
    )


def test_seasonal_average_needs_enough_history():
    with pytest.raises(ValueError):
        seasonal_moving_average_forecast(np.ones((1, 10)), 7, weeks=2)  # needs 14 days


def test_with_closed_days_passes_the_matching_slice():
    seen = []

    def spy(history, horizon, closed, extra):
        seen.append((history.shape[1], closed.tolist(), extra))
        return np.zeros((history.shape[0], horizon))

    closed_by_day = np.array([False, True, False, False, True, False, False, False])
    with_closed_days(spy, closed_by_day, extra="x")(np.ones((1, 5)), 3)
    assert seen == [(5, [False, True, False, False, True], "x")]


def test_closed_day_in_history_does_not_bias_the_average():
    cols = [f"d_{i}" for i in range(1, 85)]
    all_days = pd.DataFrame(5, index=pd.Index(["a"], name="id"), columns=cols)
    all_days.loc["a", "d_70"] = 0  # store closed that day
    closed = pd.Series(False, index=range(1, 85))
    closed.loc[70] = True
    fn = with_closed_days(moving_average_forecast, closed.to_numpy(), window=14)
    res = backtest(all_days, fn, [(71, 84)], closed)
    assert res.loc["a", "mae"] == pytest.approx(0)