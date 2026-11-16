import json
from dataclasses import replace
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.data.direct_features import (
    build_direct_forecast_table,
    build_direct_training_table,
    direct_feature_columns,
)
from src.data.features import FeatureConfig
from src.data.split import SplitConfig
from src.forecast.quantile_model import (
    load_quantile_config,
    model_inputs,
    predict_quantiles,
    quantile_name,
    run_first_window,
)
from src.tracking import TrackingConfig, configure_tracking


@pytest.fixture
def daily_data():
    """Invented data, not M5: a sells day units; b sells ten times that."""
    days = np.arange(1, 121)
    parts = []
    for name, state, scale in (("a", "CA", 1), ("b", "TX", 10)):
        parts.append(pd.DataFrame({
            "id": name, "d": [f"d_{d}" for d in days],
            "date": pd.date_range("2015-11-01", periods=len(days)),
            "sales": days * scale, "sell_price": 2 + days / 100,
            "item_id": f"item_{name}", "dept_id": "FOODS_1",
            "store_id": f"{state}_1", "state_id": state,
            "event_name_1": np.where(days == 65, "new_future_event", None),
            "event_type_1": None, "event_name_2": None, "event_type_2": None,
            "snap_CA": 1, "snap_TX": 0, "snap_WI": 0,
            "mean_live": 999999, "volume_group": 2,
        }))
    return pd.concat(parts, ignore_index=True)


def test_hand_calculated_origin_history_and_known_target_calendar(daily_data):
    table = build_direct_forecast_table(daily_data, 50, 28, FeatureConfig())
    assert len(table) == 2 * 28
    a = table.loc[table["id"].eq("a")]
    assert a["origin_sales"].eq(50).all()
    assert a["origin_sales_lag_7"].eq(43).all()
    assert a["origin_sales_lag_28"].eq(22).all()
    assert a["origin_sales_roll_mean_7"].eq(47).all()  # d_44..d_50
    assert a["origin_sales_roll_mean_28"].eq(36.5).all()  # d_23..d_50
    assert a["origin_sell_price"].eq(2.5).all()
    b = table.loc[table["id"].eq("b")]
    np.testing.assert_allclose(b["origin_sales_roll_mean_7"], 470)
    assert a["snap"].eq(1).all() and b["snap"].eq(0).all()
    assert a["day"].tolist() == list(range(51, 79))
    assert a["day_of_week"].tolist() == a["date"].dt.dayofweek.tolist()
    event = a.loc[a["day"].eq(65)].iloc[0]
    assert event["event_name_1"] == "new_future_event" and event["has_event"] == 1
    assert not {"sales", "mean_live", "volume_group"} & set(table.columns)


def test_origin_rolling_mean_excludes_assumed_closed_day(daily_data):
    table = build_direct_forecast_table(daily_data, 60, 7, FeatureConfig())
    row = table.iloc[0]
    # d_55 is 25 December. The last seven calendar days are d_54..d_60.
    assert row["origin_sales_roll_mean_7"] == pytest.approx(np.mean([54, 56, 57, 58, 59, 60]))
    assert row["origin_sales"] == 60


def test_future_truth_and_prices_cannot_change_forecast_features(daily_data):
    before = daily_data.copy(deep=True)
    changed = daily_data.copy()
    future = changed["d"].str.slice(2).astype(int).gt(50)
    changed.loc[future, "sales"] = -999999
    changed.loc[future, "sell_price"] = np.inf
    expected = build_direct_forecast_table(daily_data, 50, 28, FeatureConfig())
    actual = build_direct_forecast_table(changed, 50, 28, FeatureConfig())
    assert_frame_equal(expected, actual)
    assert_frame_equal(daily_data, before)


def test_training_blocks_are_complete_and_cutoff_is_applied_first(daily_data):
    cfg = FeatureConfig()
    expected = build_direct_training_table(daily_data, 60, 7, 7, cfg)
    assert expected["origin_day"].unique().tolist() == [32, 39, 46, 53]
    assert len(expected) == 2 * 4 * 7
    assert expected["day"].max() == 60
    assert expected.groupby(["id", "origin_day"])["horizon"].nunique().eq(7).all()
    changed = daily_data.copy()
    future = changed["d"].str.slice(2).astype(int).gt(60)
    changed.loc[future, "sales"] = -1
    changed.loc[future, "sell_price"] = np.inf
    assert_frame_equal(expected, build_direct_training_table(changed, 60, 7, 7, cfg))
    a = expected.loc[expected["id"].eq("a")]
    np.testing.assert_array_equal(a["sales"], a["day"])


def test_training_targets_do_not_become_features_of_earlier_origins(daily_data):
    expected = build_direct_training_table(daily_data, 60, 7, 7, FeatureConfig())
    changed = daily_data.copy()
    changed.loc[changed["d"].eq("d_60"), "sales"] = 999999
    actual = build_direct_training_table(changed, 60, 7, 7, FeatureConfig())
    assert_frame_equal(expected.drop(columns="sales"), actual.drop(columns="sales"))
    assert actual.loc[actual["day"].eq(60), "sales"].eq(999999).all()


def test_prelaunch_history_cannot_make_a_series_eligible(daily_data):
    changed = daily_data.copy()
    days = changed["d"].str.slice(2).astype(int)
    changed.loc[changed["id"].eq("b") & days.lt(35), "sales"] = 0
    table = build_direct_training_table(changed, 60, 7, 7, FeatureConfig())
    assert set(table["id"]) == {"a"}
    with pytest.raises(ValueError, match="Every forecast series"):
        build_direct_forecast_table(changed, 50, 7, FeatureConfig())


def test_missing_future_calendar_is_rejected(daily_data):
    changed = daily_data.loc[~(daily_data["id"].eq("a") & daily_data["d"].eq("d_78"))]
    with pytest.raises(ValueError, match="consecutive|every forecast day"):
        build_direct_forecast_table(changed, 50, 28, FeatureConfig())


@pytest.mark.parametrize("quantiles", [
    (), (0.0,), (1.0,), (np.nan,), (0.95, 0.05), (0.5, 0.5), (0.5, 0.5000000000001),
])
def test_invalid_quantiles_are_rejected(quantiles):
    with pytest.raises(ValueError, match="Quantiles"):
        replace(load_quantile_config(), quantiles=quantiles)


def test_four_models_persist_and_reload_without_scoring_validation(daily_data, tmp_path):
    cfg = load_quantile_config()
    cfg = replace(cfg, model_params={
        **cfg.model_params, "n_estimators": 8, "min_child_samples": 2, "n_jobs": 1,
    })
    features = FeatureConfig()
    split = SplitConfig(1, 60, 88, 116, 28)
    tracking = TrackingConfig("synthetic-quantile-test", tmp_path / "mlruns")
    previous_uri = mlflow.get_tracking_uri()
    try:
        run_id, output = run_first_window(
            daily_data, split, cfg, features, tracking, tmp_path / "forecasts"
        )
        forecasts = pd.read_csv(output)
        assert forecasts.shape == (56, 13)
        assert forecasts["day"].min() == 61 and forecasts["day"].max() == 88
        client, _ = configure_tracking(tracking)
        run = client.get_run(run_id)
        assert run.info.status == "FINISHED"
        assert run.data.tags["validation_scored"] == "false"
        assert run.data.tags["test_evaluated"] == "false"
        assert run.data.metrics["n_features"] == 23
        assert not any("pinball" in name or "coverage" in name for name in run.data.metrics)
        levels_path = client.download_artifacts(run_id, "models/category_levels.json")
        levels = json.loads(Path(levels_path).read_text(encoding="utf-8"))
        future = build_direct_forecast_table(daily_data, 60, 28, features)
        X, _ = model_inputs(future, features, levels)
        assert X.loc[future["day"].eq(65), "event_name_1"].isna().all()
        models = {}
        for alpha in cfg.quantiles:
            name = quantile_name(alpha)
            model_path = client.download_artifacts(run_id, f"models/{name}.txt")
            models[name] = lgb.Booster(model_str=Path(model_path).read_text(encoding="utf-8"))
            assert float(models[name].params["alpha"]) == alpha
        restored = predict_quantiles(models, future, features, levels)
        columns = [f"{quantile_name(q)}_raw" for q in cfg.quantiles]
        np.testing.assert_allclose(forecasts[columns], restored[columns])
        assert set(direct_feature_columns(features)) == set(X.columns)
    finally:
        mlflow.set_tracking_uri(previous_uri)


def test_predictions_preserve_crossing_and_negative_raw_values(daily_data):
    class Constant:
        def __init__(self, value):
            self.value = value

        def predict(self, X, **kwargs):
            return np.full(len(X), self.value)

    table = build_direct_forecast_table(daily_data, 50, 7, FeatureConfig())
    _, levels = model_inputs(table, FeatureConfig())
    out = predict_quantiles({"q05": Constant(2), "q95": Constant(-1)},
                            table, FeatureConfig(), levels)
    assert out["q05"].eq(2).all() and out["q95"].eq(0).all()
    assert out["q95_raw"].eq(-1).all()  # No sorting that could hide crossing.


@pytest.mark.parametrize("problem", ["nan", "shape", "duplicate"])
def test_invalid_predictions_and_duplicate_keys_are_rejected(daily_data, problem):
    class Invalid:
        def predict(self, X, **kwargs):
            if problem == "shape":
                return np.zeros((len(X), 1))
            return np.full(len(X), np.nan if problem == "nan" else 1)

    table = build_direct_forecast_table(daily_data, 50, 7, FeatureConfig())
    _, levels = model_inputs(table, FeatureConfig())
    if problem == "duplicate":
        table = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Invalid predictions|keys must be unique"):
        predict_quantiles({"q50": Invalid()}, table, FeatureConfig(), levels)
