from dataclasses import replace

import mlflow
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import src.evaluation.backtest_quantiles as backtest
from src.data.features import FeatureConfig
from src.data.split import SplitConfig
from src.evaluation.score_quantiles import summarize_quantile_scores
from src.forecast.quantile_model import load_quantile_config, run_first_window
from src.tracking import TrackingConfig, configure_tracking


@pytest.fixture
def window_scores():
    """Invented scores whose median depends on averaging windows first."""
    rows = []
    for variant in ("clipped", "raw"):
        for name, values in (("a", [0.0, 100.0]), ("b", [0.0, 0.0])):
            for window, value in enumerate(values, start=1):
                rows.append({
                    "variant": variant, "id": name, "window": window,
                    "pinball_q90": value, "interval90_mean_width_ordered": 2.0 * window,
                    "n_evaluated": 4, "n_ordered_interval": 4,
                })
    return pd.DataFrame(rows)


def test_aggregation_averages_windows_before_median_and_sums_counts(window_scores):
    metrics = ["pinball_q90", "interval90_mean_width_ordered"]
    result = backtest.aggregate_quantile_windows(window_scores, ["a", "b"], 2, metrics)
    primary = result.query("variant == 'clipped'").set_index("id")
    assert primary.loc["a", "pinball_q90"] == 50
    assert primary.loc["b", "pinball_q90"] == 0
    assert primary["n_evaluated"].eq(8).all()
    assert primary["n_windows"].eq(2).all()
    summary = summarize_quantile_scores(result, metrics)
    row = summary.query("variant == 'clipped' and metric == 'pinball_q90'").iloc[0]
    assert row["median"] == 25


@pytest.mark.parametrize("problem", ["missing", "duplicate"])
def test_incomplete_or_duplicated_window_scores_cannot_be_aggregated(window_scores, problem):
    if problem == "missing":
        scores = window_scores.iloc[1:]
    else:
        scores = pd.concat([window_scores, window_scores.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="exactly one score"):
        backtest.aggregate_quantile_windows(
            scores, ["a", "b"], 2, ["pinball_q90", "interval90_mean_width_ordered"]
        )


def test_undefined_width_is_reported_with_its_defined_window_count(window_scores):
    missing = window_scores["id"].eq("a") & window_scores["window"].eq(2)
    window_scores.loc[missing, "interval90_mean_width_ordered"] = np.nan
    window_scores.loc[missing, "n_ordered_interval"] = 0
    metrics = ["pinball_q90", "interval90_mean_width_ordered"]
    result = backtest.aggregate_quantile_windows(window_scores, ["a", "b"], 2, metrics)
    assert result.loc[result["id"].eq("a"), "interval90_mean_width_ordered"].isna().all()
    assert result.loc[result["id"].eq("a"), "width_windows_defined"].eq(1).all()
    assert result.loc[result["id"].eq("b"), "width_windows_defined"].eq(2).all()
    summary = summarize_quantile_scores(result, metrics)
    width = summary.query("metric == 'interval90_mean_width_ordered'")
    assert width["n_series_defined"].eq(1).all()


def test_six_window_run_uses_archived_settings_and_expanding_cutoffs(tmp_path, monkeypatch):
    """Small invented panel exercises the training, scoring and artifact flow."""
    days = np.arange(1, 109)
    dates = pd.Series(pd.date_range("2015-11-01", periods=len(days)), index=days)
    parts = []
    for name, state, scale in (("a", "CA", 1), ("b", "TX", 10)):
        parts.append(pd.DataFrame({
            "id": name, "d": [f"d_{day}" for day in days], "date": dates.to_numpy(),
            "sales": days * scale, "sell_price": 2 + days / 100,
            "item_id": f"item_{name}", "dept_id": "FOODS_1",
            "store_id": f"{state}_1", "state_id": state,
            "event_name_1": None, "event_type_1": None,
            "event_name_2": None, "event_type_2": None,
            "snap_CA": 1, "snap_TX": 0, "snap_WI": 0,
        }))
    long = pd.concat(parts, ignore_index=True)
    positive_long = long.copy(deep=True)
    # Day 90 is the assumed closure, inside the second test window.
    split = SplitConfig(1, 60, 84, 108, 4, closed_month=1, closed_day=29)
    labels = long["d"].str.slice(2).astype(int)
    wide = long.loc[labels.between(61, 84)].pivot(index="id", columns="d", values="sales")
    # Held-out test data contains invalid values to catch accidental use by the runner.
    long.loc[labels.gt(84), "sales"] = -999999
    long.loc[labels.gt(84), "sell_price"] = np.inf
    original_long = long.copy(deep=True)
    cfg = load_quantile_config()
    cfg = replace(cfg, origin_stride=4, model_params={
        **cfg.model_params, "n_estimators": 2, "min_child_samples": 2, "n_jobs": 1,
    })
    tracking = TrackingConfig("invented-backtest", tmp_path / "mlruns")
    previous_uri = mlflow.get_tracking_uri()
    try:
        source_id, _ = run_first_window(
            long, split, cfg, FeatureConfig(closed_month=1, closed_day=29),
            tracking, tmp_path / "first-window"
        )
        # Give the invented source run a snapshot of its own two series.
        subset_path = tmp_path / "subset_ids.csv"
        pd.DataFrame({"id": wide.index}).to_csv(subset_path, index=False)
        with mlflow.start_run(run_id=source_id):
            mlflow.log_artifact(str(subset_path), artifact_path="snapshot")
        cutoffs = []
        build_training = backtest.build_direct_training_table

        def audit_training(frame, origin, horizon, stride, features):
            table = build_training(frame, origin, horizon, stride, features)
            assert table["day"].max() <= origin
            assert frame["d"].str.slice(2).astype(int).max() == split.valid_end
            cutoffs.append(origin)
            return table

        monkeypatch.setattr(backtest, "build_direct_training_table", audit_training)
        client, experiment_id = configure_tracking(tracking)
        run_id, folder = backtest.run_validation_backtest(
            client, experiment_id, source_id, split, long, wide, dates, tmp_path / "backtest"
        )
        assert cutoffs == [60, 64, 68, 72, 76, 80]
        assert_frame_equal(long, original_long)
        saved = client.get_run(run_id)
        assert saved.info.status == "FINISHED"
        assert saved.data.params["n_estimators"] == "2"  # Uses the archived two-tree configuration.
        assert saved.data.params["origin_stride"] == "4"
        assert saved.data.tags["source_training_run_id"] == source_id
        assert saved.data.tags["test_evaluated"] == "false"
        assert saved.data.metrics["n_windows"] == 6
        assert saved.data.metrics["forecast_rows"] == 48
        assert saved.data.metrics["evaluated_observations"] == 48
        forecasts = pd.read_csv(folder / "forecasts_valid.csv")
        assert forecasts.shape == (48, 14)
        assert not forecasts.duplicated(["id", "day"]).any()
        assert forecasts["day"].min() == 61 and forecasts["day"].max() == 84
        per_series = pd.read_csv(folder / "per_series_valid.csv")
        assert len(per_series) == 4 and per_series["n_windows"].eq(6).all()
        assert per_series["n_evaluated"].eq(24).all()
        info = pd.read_csv(folder / "window_info_valid.csv")
        assert info["last_training_target"].tolist() == cutoffs
        for window in range(1, 7):
            models = client.list_artifacts(run_id, f"models/window_{window}")
            assert sum(item.path.endswith(".txt") for item in models) == 4
        assert client.get_run(source_id).info.status == "FINISHED"

        test_wide = positive_long.loc[labels.between(85, 108)].pivot(
            index="id", columns="d", values="sales"
        )
        test_cutoffs = []

        def audit_test_training(frame, origin, horizon, stride, features):
            table = build_training(frame, origin, horizon, stride, features)
            assert table["day"].max() <= origin
            assert frame["d"].str.slice(2).astype(int).max() == split.last_day
            test_cutoffs.append(origin)
            return table

        monkeypatch.setattr(backtest, "build_direct_training_table", audit_test_training)
        test_id, test_folder = backtest.run_backtest(
            client, experiment_id, run_id, split, positive_long, test_wide, dates,
            tmp_path / "test-backtest", evaluation_split="test",
        )
        assert test_cutoffs == [84, 88, 92, 96, 100, 104]
        test_run = client.get_run(test_id)
        assert test_run.info.status == "FINISHED"
        assert test_run.data.tags["source_validation_run_id"] == run_id
        assert test_run.data.tags["test_evaluated"] == "true"
        assert test_run.data.params["n_estimators"] == "2"
        assert test_run.data.metrics["forecast_rows"] == 48
        assert test_run.data.metrics["closed_days_excluded"] == 1
        assert test_run.data.metrics["evaluated_observations"] == 46
        test_forecasts = pd.read_csv(test_folder / "forecasts_test.csv")
        assert test_forecasts["day"].min() == 85 and test_forecasts["day"].max() == 108
        test_series = pd.read_csv(test_folder / "per_series_test.csv")
        assert test_series["n_evaluated"].eq(23).all()
        with pytest.raises(ValueError, match="completed validation backtest"):
            backtest.run_backtest(
                client, experiment_id, source_id, split, positive_long, test_wide, dates,
                tmp_path / "rejected", evaluation_split="test",
            )
    finally:
        mlflow.set_tracking_uri(previous_uri)
