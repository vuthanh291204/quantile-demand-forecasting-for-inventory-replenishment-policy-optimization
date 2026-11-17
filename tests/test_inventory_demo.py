"""Check the M5 adapter and plot using invented sales and calendar dates."""

import json

import pandas as pd
import pytest
from PIL import Image

from src.data.split import SplitConfig
from src.simulation.run_demo import simulate_demo, write_demo_outputs


@pytest.fixture
def invented_sales():
    split = SplitConfig(1, 4, 6, 8, 2)
    long = pd.DataFrame({
        "id": "a",
        "d": [f"d_{day}" for day in range(1, 9)],
        "date": pd.date_range("2015-12-20", periods=8),
        "sales": [2, 2, 2, 2, 1, 9, 8, 8],
    })
    return long, split


def test_history_only_target_and_closed_day_receipt_match_hand_calculation(invented_sales):
    long, split = invented_sales
    original = long.copy(deep=True)
    result, info = simulate_demo(
        long.sample(frac=1, random_state=29), split,
        series_id="a", lead_time_days=1, history_days=2,
    )
    assert info["history_mean"] == 2
    assert info["target_stock"] == info["initial_stock"] == 4
    assert info["closed_days"] == 1
    assert result.daily["day"].tolist() == [5, 6]
    assert result.daily["date"].tolist() == list(pd.date_range("2015-12-24", periods=2))
    assert result.daily["observed_sales"].tolist() == [1, 9]
    assert result.daily["demand"].tolist() == [1, 0]
    assert result.daily["received"].tolist() == [0, 1]
    assert result.daily["order_placed"].tolist() == [1, 0]
    assert result.daily["closing_stock"].tolist() == [3, 4]
    assert result.pending_orders == {}
    pd.testing.assert_frame_equal(long, original)


def test_future_sales_do_not_change_fixed_target_or_initial_state(invented_sales):
    long, split = invented_sales
    _, before = simulate_demo(long, split, series_id="a", lead_time_days=1, history_days=2)
    changed = long.copy()
    changed.loc[4:, "sales"] = 99999
    _, after = simulate_demo(changed, split, series_id="a", lead_time_days=1, history_days=2)
    assert after["history_mean"] == before["history_mean"]
    assert after["target_stock"] == before["target_stock"]
    assert after["initial_stock"] == before["initial_stock"]


@pytest.mark.parametrize("problem", ["missing_day", "duplicate_day", "date_gap"])
def test_missing_duplicate_or_nonconsecutive_days_are_rejected(invented_sales, problem):
    long, split = invented_sales
    if problem == "missing_day":
        long = long.drop(index=4)
    elif problem == "duplicate_day":
        long = pd.concat([long, long.iloc[[4]]], ignore_index=True)
    else:
        long.loc[4, "date"] = pd.Timestamp("2015-12-25")
    with pytest.raises(ValueError, match="one row per|consecutive"):
        simulate_demo(long, split, series_id="a", lead_time_days=1, history_days=2)


def test_plot_and_saved_ledger_replay_the_simulation_without_overwriting(invented_sales, tmp_path):
    long, split = invented_sales
    result, info = simulate_demo(long, split, series_id="a", lead_time_days=1, history_days=2)
    first = write_demo_outputs(result, info, tmp_path)
    second = write_demo_outputs(result, info, tmp_path)
    assert first != second
    for output_dir in (first, second):
        assert {path.name for path in output_dir.iterdir()} == {
            "inventory.png", "daily.csv", "run_info.json",
        }
        with Image.open(output_dir / "inventory.png") as figure:
            assert figure.width >= 1500 and figure.height >= 1000
        saved = pd.read_csv(output_dir / "daily.csv", parse_dates=["date"])
        pd.testing.assert_frame_equal(saved, result.daily, check_dtype=False)
        assert json.loads((output_dir / "run_info.json").read_text(encoding="utf-8")) == info
