import pandas as pd
import pytest

from src.data.split import (
    SplitConfig,
    closed_days,
    day_columns,
    load_config,
    make_windows,
    split_wide,
)


def test_real_config_matches_chosen_split():
    cfg = load_config()
    assert cfg.ranges == {
        "train": (1, 1605),
        "valid": (1606, 1773),
        "test": (1774, 1941),
    }
    assert cfg.horizon == 28


def test_ranges_are_contiguous():
    cfg = SplitConfig(first_day=1, train_end=10, valid_end=14, last_day=18, horizon=4)
    r = cfg.ranges
    assert r["train"][1] + 1 == r["valid"][0]
    assert r["valid"][1] + 1 == r["test"][0]


def test_bad_config_is_rejected():
    with pytest.raises(ValueError):  # validation = 50 days, not a multiple of 28
        SplitConfig(first_day=1, train_end=100, valid_end=150, last_day=200, horizon=28)
    with pytest.raises(ValueError):  # cut-off days in wrong order
        SplitConfig(first_day=1, train_end=100, valid_end=90, last_day=200, horizon=10)


def test_make_windows():
    assert make_windows(1, 8, 4) == [(1, 4), (5, 8)]
    with pytest.raises(ValueError):
        make_windows(1, 10, 4)


def test_split_wide_has_no_overlap_and_keeps_values():
    cfg = SplitConfig(first_day=1, train_end=6, valid_end=8, last_day=10, horizon=2)
    wide = pd.DataFrame({"id": ["a", "b"], "store_id": ["CA_3", "TX_2"]})
    for day in range(1, 11):
        wide[f"d_{day}"] = [day, day * 10]
    parts = split_wide(wide, cfg)
    assert list(parts["train"].columns) == day_columns(1, 6)
    assert list(parts["valid"].columns) == day_columns(7, 8)
    assert list(parts["test"].columns) == day_columns(9, 10)
    assert parts["test"].loc["b", "d_10"] == 100
    all_cols = [c for p in parts.values() for c in p.columns]
    assert len(all_cols) == len(set(all_cols)) == 10


def test_closed_days_flags_only_christmas():
    cfg = SplitConfig(first_day=1, train_end=6, valid_end=8, last_day=10, horizon=2)
    dates = pd.Series(pd.to_datetime(["2015-12-24", "2015-12-25", "2015-12-26"]))
    assert closed_days(dates, cfg).tolist() == [False, True, False]