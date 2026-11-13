import numpy as np
import pytest

from src.forecast.statistical import croston_sba_forecast, ets_forecast


def _sparse_history(n_series=3, n_days=140, seed=0):
    rng = np.random.default_rng(seed)
    lam = rng.uniform(0.5, 3, size=(n_series, 1))
    return rng.poisson(lam, size=(n_series, n_days)).astype(float)


@pytest.mark.parametrize("fn", [ets_forecast, croston_sba_forecast])
def test_shape_finite_and_non_negative(fn):
    out = fn(_sparse_history(), 14)
    assert out.shape == (3, 14)
    assert np.isfinite(out).all()
    assert (out >= 0).all()


@pytest.mark.parametrize("fn", [ets_forecast, croston_sba_forecast])
def test_series_without_sales_is_forecast_as_zero(fn):
    hist = _sparse_history()
    hist[1] = 0
    out = fn(hist, 14)
    assert (out[1] == 0).all()
    assert out[0].sum() > 0 and out[2].sum() > 0


@pytest.mark.parametrize("fn", [ets_forecast, croston_sba_forecast])
def test_all_series_without_sales(fn):
    assert not fn(np.zeros((2, 50)), 7).any()


@pytest.mark.parametrize("fn", [ets_forecast, croston_sba_forecast])
def test_ignores_days_before_first_sale(fn):
    base = _sparse_history(n_series=1, n_days=120)
    base[0, 0] = 1.0  # make sure the first day has a sale
    padded = np.hstack([np.zeros((1, 60)), base])
    np.testing.assert_allclose(fn(base, 14), fn(padded, 14), rtol=1e-6)


@pytest.mark.parametrize("fn", [ets_forecast, croston_sba_forecast])
def test_input_checks(fn):
    with pytest.raises(ValueError):
        fn(np.ones((2, 30)), 0)
    with pytest.raises(ValueError):
        fn(np.ones(30), 7)


def test_stats_counts_forecasts():
    stats = {}
    ets_forecast(_sparse_history(), 14, stats=stats)
    assert stats["forecasts"] == 3 * 14
    assert stats["negative"] >= 0
    assert stats["negative_by_row"].shape == (3,)
    assert stats["negative_by_row"].sum() == stats["negative"]


def test_croston_is_flat_and_close_to_the_true_average():
    hist = np.zeros((1, 200))
    hist[0, ::4] = 2.0  # sells 2 units every 4th day -> average 0.5 per day
    out = croston_sba_forecast(hist, 28)
    assert np.allclose(out, out[0, 0])  # same value on every future day
    assert 0.4 < out[0, 0] < 0.55