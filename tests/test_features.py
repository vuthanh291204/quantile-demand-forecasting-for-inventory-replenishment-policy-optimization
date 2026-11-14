import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import src.evaluation.check_leakage as leakage
from src.data.features import (
    FeatureConfig,
    build_features,
    build_training_table,
    load_feature_config,
    prepare_forecast_frame,
)


@pytest.fixture
def daily_data():
    """Invented daily sales: a sells day units, b sells ten times that."""
    days = np.arange(1, 91)
    parts = []
    for name, state, scale in (("a", "CA", 1), ("b", "TX", 10)):
        parts.append(pd.DataFrame({
            "id": name, "d": [f"d_{day}" for day in days],
            "date": pd.date_range("2015-11-01", periods=len(days)),
            "sales": days * scale, "sell_price": 2.0 + days / 100,
            "item_id": f"item_{name}", "dept_id": "FOODS_1",
            "store_id": f"{state}_1", "state_id": state,
            "event_name_1": None, "event_type_1": None,
            "event_name_2": None, "event_type_2": None,
            "snap_CA": 1, "snap_TX": 0, "snap_WI": 0,
            # These full-period EDA fields must never enter the feature table.
            "volume_group": 2, "mean_live": 123.0, "live": True,
        }))
    return pd.concat(parts, ignore_index=True)


def test_feature_config_matches_initial_choices():
    cfg = load_feature_config()
    assert cfg.lags == (7, 14, 28)
    assert cfg.rolling_windows == (7, 28)
    assert cfg.history_days == 28
    assert len(cfg.feature_columns) == 21
    assert "sales" not in cfg.feature_columns


@pytest.mark.parametrize("values", [[], [0], [-1], [7, 7], [1.5]])
def test_invalid_feature_windows_are_rejected(values):
    with pytest.raises(ValueError):
        FeatureConfig(rolling_windows=values)


def test_hand_calculated_lags_rolls_and_series_boundaries(daily_data):
    cfg = FeatureConfig()
    # Sorting inside the builder must restore daily order within each series.
    frame = build_features(daily_data.sample(frac=1, random_state=29), cfg)
    rows = frame[frame["day"] == 40].set_index("id")
    assert rows.loc["a", "sales_lag_7"] == 33
    assert rows.loc["a", "sales_lag_14"] == 26
    assert rows.loc["a", "sales_lag_28"] == 12
    assert rows.loc["a", "sales_roll_mean_7"] == pytest.approx(36)
    assert rows.loc["a", "sales_roll_mean_28"] == pytest.approx(25.5)
    assert rows.loc["b", "sales_roll_mean_7"] == pytest.approx(360)
    assert rows.loc["a", "sell_price_lag_1"] == pytest.approx(2.39)
    assert rows.loc["a", "snap"] == 1
    assert rows.loc["b", "snap"] == 0
    assert rows.loc["a", "day_of_week"] == 3  # 2015-12-10, Thursday
    assert rows.loc["a", "month"] == 12
    assert rows["features_ready"].all()
    assert not {"volume_group", "mean_live", "live", "first_date"} & set(frame.columns)
    assert not frame[frame["day"] <= 28]["features_ready"].any()


def test_current_and_later_sales_do_not_change_current_features(daily_data):
    cfg = FeatureConfig()
    before = build_features(daily_data, cfg)
    changed = daily_data.copy()
    future = changed["date"] >= pd.Timestamp("2015-12-10")  # d_40
    changed.loc[future, "sales"] = 999999
    changed.loc[future, "sell_price"] = 999
    after = build_features(changed, cfg)
    cols = ["id", "day", *cfg.feature_columns, "features_ready"]
    assert_frame_equal(before.loc[before["day"] <= 40, cols], after.loc[after["day"] <= 40, cols])


def test_training_cutoff_excludes_validation_and_test(daily_data):
    cfg = FeatureConfig()
    before = build_training_table(daily_data, end_day=60, cfg=cfg)
    changed = daily_data.copy()
    changed.loc[changed["date"] > pd.Timestamp("2015-12-30"), "sales"] = -999
    after = build_training_table(changed, end_day=60, cfg=cfg)
    assert_frame_equal(before, after)
    assert before["day"].max() == 60
    assert before["day"].min() == 29


def test_prelaunch_days_are_not_treated_as_zero_demand(daily_data):
    cfg = FeatureConfig()
    daily_data.loc[(daily_data["id"] == "a") & daily_data["date"].lt("2015-11-05"), "sales"] = 0
    train = build_training_table(daily_data, 60, cfg)
    assert train.loc[train["id"] == "a", "day"].min() == 33
    assert train.loc[train["id"] == "b", "day"].min() == 29


def test_closed_day_is_excluded_from_mean_but_preserved_in_raw_lag(daily_data):
    cfg = FeatureConfig(lags=(1, 7, 28))
    daily_data.loc[daily_data["date"] == "2015-12-25", "sales"] = 0
    frame = build_features(daily_data, cfg)
    row = frame[(frame["id"] == "a") & (frame["date"] == "2015-12-26")].iloc[0]
    assert row["sales_lag_1"] == 0
    assert row["sales_roll_mean_7"] == pytest.approx(51.5)  # mean of d_49..d_54
    assert row["features_ready"]


def test_future_events_are_preserved_as_known_calendar(daily_data):
    cfg = FeatureConfig()
    daily_data.loc[daily_data["d"] == "d_61", "event_name_2"] = "ExampleHoliday"
    daily_data.loc[daily_data["d"] == "d_61", "event_type_2"] = "Cultural"
    frame = build_features(prepare_forecast_frame(daily_data, 60, 28), cfg)
    rows = frame[frame["day"] == 61]
    assert rows["event_name_2"].eq("ExampleHoliday").all()
    assert rows["has_event"].eq(1).all()


def test_forecast_window_erases_future_truth_and_freezes_origin_price(daily_data):
    cfg = FeatureConfig()
    safe = prepare_forecast_frame(daily_data, 60, 28)
    future = safe[safe["date"] > pd.Timestamp("2015-12-30")]
    assert future["sales"].isna().all()
    assert future["sell_price"].eq(2.6).all()
    changed = daily_data.copy()
    future_mask = changed["date"] > pd.Timestamp("2015-12-30")
    changed.loc[future_mask, "sales"] = -999
    changed.loc[future_mask, "sell_price"] = 999
    assert_frame_equal(safe, prepare_forecast_frame(changed, 60, 28))
    features = build_features(safe, cfg)
    ready = features[(features["day"] > 60) & features["features_ready"]]
    assert len(ready) == 2
    assert ready["day"].eq(61).all()
    actual = build_features(daily_data, cfg)
    cols = ["id", *cfg.feature_columns]
    assert_frame_equal(
        features.loc[features["day"] == 61, cols].reset_index(drop=True),
        actual.loc[actual["day"] == 61, cols].reset_index(drop=True),
    )


def test_later_forecast_steps_use_predictions_instead_of_observed_sales(daily_data):
    cfg = FeatureConfig()
    safe = prepare_forecast_frame(daily_data, 60, 28)
    for day in range(61, 68):
        safe.loc[safe["d"] == f"d_{day}", "sales"] = 1000 + day
    features = build_features(safe, cfg)
    next_day = features[features["day"] == 68]
    assert next_day["features_ready"].all()
    assert next_day["sales_lag_7"].eq(1061).all()
    assert next_day["sales_roll_mean_7"].eq(1064).all()
    assert next_day["sell_price_lag_1"].to_numpy() == pytest.approx([2.6, 2.6])
    assert not features.loc[features["day"] == 69, "features_ready"].any()


@pytest.mark.parametrize("kind", ["gap", "duplicate", "date"])
def test_bad_daily_alignment_is_rejected(daily_data, kind):
    if kind == "gap":
        bad = daily_data.drop(index=5)
    elif kind == "duplicate":
        bad = pd.concat([daily_data, daily_data.iloc[[5]]])
    else:
        bad = daily_data.copy()
        bad.loc[5, "date"] = pd.Timestamp("2015-11-07")
    with pytest.raises(ValueError):
        build_features(bad, FeatureConfig())


def test_incomplete_forecast_window_is_rejected(daily_data):
    with pytest.raises(ValueError):
        prepare_forecast_frame(daily_data, origin_day=70, horizon=28)


def test_training_leakage_audit_checks_cutoff_and_explicit_feature_list(daily_data):
    result = leakage.audit_training(daily_data, end_day=60, cfg=FeatureConfig())
    assert result["n_rows"] == 64
    assert result["n_series"] == 2
    assert result["held_out_mutation_invariant"]
    assert result["forbidden_feature_columns_absent"]


@pytest.mark.parametrize("origin", [30, 58, 60])
def test_leakage_audit_checks_all_28_steps_without_changing_source(daily_data, origin):
    original = daily_data.copy(deep=True)
    result = leakage.audit_window(daily_data, origin, horizon=28, cfg=FeatureConfig())
    assert result["synthetic_rows_checked"] == 56  # two series, 28 days each
    assert result["all_step_lags_and_means_match_past_only_oracle"]
    assert_frame_equal(daily_data, original)


def test_audit_rejects_retained_future_sales(daily_data, monkeypatch):
    def bad_prepare(long, origin, horizon):
        frame = prepare_forecast_frame(long, origin, horizon)
        future = frame["d"].str.slice(2).astype(int) > origin
        frame.loc[future, "sales"] = 123.0
        return frame

    monkeypatch.setattr(leakage, "prepare_forecast_frame", bad_prepare)
    with pytest.raises(AssertionError, match="Future observed sales were retained"):
        leakage.audit_window(daily_data, 60, 28, FeatureConfig())


def test_audit_rejects_current_label_in_a_rolling_feature(daily_data, monkeypatch):
    def bad_features(long, cfg):
        frame = build_features(long, cfg)
        frame["sales_roll_mean_7"] += frame["sales"].fillna(0)
        return frame

    monkeypatch.setattr(leakage, "build_features", bad_features)
    with pytest.raises(AssertionError):
        leakage.audit_window(daily_data, 60, 28, FeatureConfig())


def test_audit_rejects_unfrozen_future_price(daily_data, monkeypatch):
    def bad_prepare(long, origin, horizon):
        frame = prepare_forecast_frame(long, origin, horizon)
        future = frame["d"].str.slice(2).astype(int) > origin
        frame.loc[future, "sell_price"] = 999.0
        return frame

    monkeypatch.setattr(leakage, "prepare_forecast_frame", bad_prepare)
    with pytest.raises(AssertionError):
        leakage.audit_window(daily_data, 60, 28, FeatureConfig())


def test_audit_rejects_full_period_training_statistic(daily_data, monkeypatch):
    def bad_training(long, end_day, cfg):
        frame = build_training_table(long, end_day, cfg)
        frame["sales_roll_mean_7"] = long["sales"].mean()
        return frame

    monkeypatch.setattr(leakage, "build_training_table", bad_training)
    with pytest.raises(AssertionError):
        leakage.audit_training(daily_data, 60, FeatureConfig())
