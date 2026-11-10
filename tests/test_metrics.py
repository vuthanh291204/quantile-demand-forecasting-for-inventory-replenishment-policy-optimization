import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    mae_per_series,
    rmse_per_series,
    rmsse_per_series,
    smape_per_series,
    summarize,
)

Y = np.array([[0, 2, 4, 0]])
F = np.array([[0, 1, 6, 1]])


def test_mae_and_rmse_by_hand():
    assert mae_per_series(Y, F)[0] == pytest.approx(1.0)
    assert rmse_per_series(Y, F)[0] == pytest.approx(np.sqrt(1.5))


def test_smape_both_zero_modes():
    # terms: 0 (both zero), 2/3, 0.4, 2
    assert smape_per_series(Y, F, both_zero="zero")[0] == pytest.approx(100 * 3.0666667 / 4)
    assert smape_per_series(Y, F, both_zero="skip")[0] == pytest.approx(100 * 3.0666667 / 3)


def test_smape_rejects_unknown_mode():
    with pytest.raises(ValueError):
        smape_per_series(Y, F, both_zero="other")


def test_all_zero_forecast_on_sparse_series():
    y = np.array([[0, 3, 0, 1]])
    f = np.zeros_like(y)
    assert mae_per_series(y, f)[0] == pytest.approx(1.0)
    assert smape_per_series(y, f)[0] == pytest.approx(100.0)  # 2 sale days of 4, each scores 200%


def test_mask_ignores_days():
    mask = np.array([True, True, False, True])  # drop day 3
    assert mae_per_series(Y, F, mask=mask)[0] == pytest.approx(2 / 3)


def test_mask_that_removes_everything_gives_nan():
    mask = np.array([False, False, False, False])
    assert np.isnan(mae_per_series(Y, F, mask=mask)[0])


def test_dataframe_input_keeps_series_ids_and_series_are_independent():
    y = pd.DataFrame([[1, 1], [10, 10]], index=["a", "b"], columns=["d_1", "d_2"])
    f = pd.DataFrame([[1, 1], [0, 0]], index=["a", "b"], columns=["d_1", "d_2"])
    mae = mae_per_series(y, f)
    assert mae.loc["a"] == 0 and mae.loc["b"] == 10
    assert smape_per_series(y, f).loc["b"] == pytest.approx(200.0)
    assert rmse_per_series(y, f).loc["b"] == pytest.approx(10.0)


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError):
        mae_per_series(np.array([[1.0, np.nan]]), np.array([[1.0, 1.0]]))
    with pytest.raises(ValueError):
        mae_per_series(np.array([[1, 2]]), np.array([[1, 2, 3]]))
    with pytest.raises(ValueError):
        mae_per_series(np.array([[1, 2]]), np.array([[1, 2]]), mask=np.array([True]))


def test_summarize_overall_and_by_group():
    values = pd.Series([1.0, 2.0, 3.0, 4.0], index=["a", "b", "c", "d"])
    groups = pd.Series([0, 0, 1, 1], index=["a", "b", "c", "d"])
    out = summarize(values, groups)
    assert out.loc["all", "mean"] == pytest.approx(2.5)
    assert out.loc["group 0", "mean"] == pytest.approx(1.5)
    assert out.loc["group 1", "median"] == pytest.approx(3.5)
    assert out.loc["all", "n_series"] == 4


def test_rmsse_by_hand():
    hist = np.array([[0, 0, 2, 4, 4, 6]])
    y = np.array([[1, 3]])
    f = np.array([[2, 1]])
    assert rmsse_per_series(y, f, hist)[0] == pytest.approx(np.sqrt(2.5 / (8 / 3)))


def test_rmsse_ignores_days_before_first_sale():
    y = np.array([[1, 3]])
    f = np.array([[2, 1]])
    short = np.array([[0, 0, 2, 4, 4, 6]])
    long = np.array([[0, 0, 0, 0, 0, 0, 2, 4, 4, 6]])
    assert rmsse_per_series(y, f, long)[0] == pytest.approx(rmsse_per_series(y, f, short)[0])


def test_rmsse_is_nan_when_history_has_no_scale():
    y = np.array([[1, 3]])
    f = np.array([[2, 1]])
    assert np.isnan(rmsse_per_series(y, f, np.array([[3, 3, 3]]))[0])  # never changes
    assert np.isnan(rmsse_per_series(y, f, np.array([[0, 0, 0]]))[0])  # never sold


def test_rmsse_perfect_forecast_is_zero_and_respects_mask():
    hist = np.array([[0, 0, 2, 4, 4, 6]])
    y = np.array([[1, 3, 5]])
    assert rmsse_per_series(y, y, hist)[0] == 0
    f = np.array([[1, 3, 50]])
    mask = np.array([True, True, False])  # the bad day is ignored
    assert rmsse_per_series(y, f, hist, mask=mask)[0] == 0


def test_rmsse_rejects_wrong_history_shape():
    with pytest.raises(ValueError):
        rmsse_per_series(np.array([[1, 3]]), np.array([[2, 1]]), np.array([[1, 2], [3, 4]]))