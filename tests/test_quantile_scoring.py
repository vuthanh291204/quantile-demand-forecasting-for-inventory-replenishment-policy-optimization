from dataclasses import asdict, replace
from types import SimpleNamespace

import mlflow
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.metrics import mean_pinball_loss

from src.data.split import SplitConfig
from src.evaluation.quantile_metrics import (
    crossing_per_series,
    interval_diagnostics_per_series,
    pinball_per_series,
    quantile_coverage_per_series,
)
from src.evaluation.score_quantiles import (
    QUANTILES,
    load_validation_source,
    score_quantile_window,
    score_saved_validation_run,
)
from src.forecast.quantile_model import quantile_name
from src.tracking import TrackingConfig, configure_tracking


def test_pinball_matches_hand_calculation_and_sklearn():
    y = pd.DataFrame([[10, 2], [0, 2]], index=["a", "b"])
    q = pd.DataFrame([[6, 6], [-1, 1]], index=y.index)
    loss = pinball_per_series(y, q, 0.9)
    assert loss.loc["a"] == pytest.approx((3.6 + 0.4) / 2)
    for name in y.index:
        assert loss.loc[name] == pytest.approx(mean_pinball_loss(y.loc[name], q.loc[name], alpha=0.9))
    np.testing.assert_allclose(pinball_per_series(y, q, 0.5), np.abs(y - q).mean(axis=1) / 2)


def test_quantile_coverage_exposes_ties_for_discrete_sales():
    y = np.array([[0, 0, 0, 1]])
    q = np.zeros_like(y)
    assert quantile_coverage_per_series(y, q)[0] == 0.75
    assert quantile_coverage_per_series(y, q, strict=True)[0] == 0


def test_inclusive_interval_coverage_and_inversions_have_explicit_denominators():
    y = np.array([[0, 2, 6, 5]])
    lo = np.array([[0, 1, 7, 4]])
    hi = np.array([[0, 3, 4, 4]])
    row = interval_diagnostics_per_series(y, lo, hi).iloc[0]
    assert row["n_evaluated"] == 4 and row["n_ordered_interval"] == 3
    assert row["interval_coverage"] == 0.5  # Includes the zero-width interval [0, 0].
    assert row["interval_above_rate"] == 0.25
    assert row["interval_below_rate"] == 0
    assert row["interval_invalid_rate"] == 0.25
    assert row["interval_mean_width_ordered"] == pytest.approx(2 / 3)
    assert row[["interval_coverage", "interval_below_rate", "interval_above_rate",
                "interval_invalid_rate"]].sum() == pytest.approx(1)
    invalid = interval_diagnostics_per_series([[1, 1]], [[2, 3]], [[1, 0]]).iloc[0]
    assert invalid["n_ordered_interval"] == 0
    assert np.isnan(invalid["interval_mean_width_ordered"])
    assert invalid["interval_invalid_rate"] == 1 and invalid["interval_coverage"] == 0


def test_masks_counts_and_all_excluded_days():
    y = np.array([[10, 2], [0, 2]])
    q = np.array([[6, 6], [0, 1]])
    keep = np.array([[False, False], [True, False]])
    loss = pinball_per_series(y, q, 0.9, keep)
    assert np.isnan(loss[0]) and loss[1] == 0
    intervals = interval_diagnostics_per_series(y, q, q, keep)
    assert intervals["n_evaluated"].tolist() == [0, 1]
    assert np.isnan(intervals.loc[0, "interval_coverage"])
    assert intervals.loc[1, "interval_coverage"] == 1
    # A one-dimensional mask excludes the second day for both series.
    np.testing.assert_allclose(pinball_per_series(y, q, 0.9, [True, False]), [3.6, 0])


def test_crossing_detects_any_adjacent_inversion_but_not_ties():
    values = [np.array([[0, 1, 2]]), np.array([[0, 3, 2]]), np.array([[1, 2, 3]])]
    assert crossing_per_series(values)[0] == pytest.approx(1 / 3)
    assert crossing_per_series(values, [True, False, True])[0] == 0


@pytest.mark.parametrize("alpha", [0, 1, -0.1, np.nan, np.inf])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        pinball_per_series([[1]], [[1]], alpha)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_are_rejected_even_on_masked_days(value):
    with pytest.raises(ValueError, match="NaN|finite"):
        pinball_per_series([[1, 1]], [[1, value]], 0.9, [True, False])


def test_metrics_reject_mismatched_series_or_bound_labels():
    y = pd.DataFrame([[1, 2]], index=["a"], columns=["d_1", "d_2"])
    wrong = pd.DataFrame([[1, 2]], index=["b"], columns=y.columns)
    with pytest.raises(ValueError, match="same rows and columns"):
        interval_diagnostics_per_series(y, y, wrong)
    with pytest.raises(ValueError, match="at least two"):
        crossing_per_series([y])


@pytest.fixture
def invented_window():
    """Invented forecasts with known crossings; not outputs from an M5 model."""
    dates = pd.Series(pd.date_range("2015-11-01", periods=68), index=range(1, 69))
    wide = pd.DataFrame([[0, 2, 6, 5], [0, 20, 60, 50]],
                        index=pd.Index(["a", "b"], name="id"),
                        columns=[f"d_{d}" for d in range(61, 65)])
    parts = []
    for name, scale in (("a", 1), ("b", 10)):
        frame = pd.DataFrame({
            "id": name, "origin_day": 60, "day": range(61, 65),
            "date": dates.loc[61:64].to_numpy(), "horizon": range(1, 5),
        })
        for label, values in {"q05": [-1, 1, 7, 4], "q50": [1, 2, 6, 5],
                              "q90": [2, 3, 5, 6], "q95": [3, 4, 4, 7]}.items():
            frame[f"{label}_raw"] = np.array(values, dtype=float) * scale
            frame[label] = np.maximum(frame[f"{label}_raw"], 0)
        parts.append(frame)
    return wide, pd.concat(parts, ignore_index=True), dates, pd.Series(False, index=dates.index)


def test_window_scoring_aligns_keys_and_preserves_raw_clipped_and_crossings(invented_window):
    wide, forecasts, dates, closed = invented_window
    series, horizons, summary = score_quantile_window(wide, forecasts, 60, 4, dates, closed)
    assert len(series) == 4 and len(horizons) == 8
    primary = series.loc[series["variant"].eq("clipped")].set_index("id")
    assert primary["crossing_rate"].eq(0.25).all()
    assert primary["crossing_q05_gt_q50"].eq(0.25).all()
    assert primary["crossing_q50_gt_q90"].eq(0.25).all()
    assert primary["crossing_q90_gt_q95"].eq(0.25).all()
    assert primary["interval90_invalid_rate"].eq(0.25).all()
    assert primary["n_ordered_interval"].eq(3).all()
    row = summary.query("variant == 'clipped' and metric == 'pinball_q90'").iloc[0]
    assert row["mean"] == pytest.approx(primary["pinball_q90"].mean())
    raw = series.loc[series["variant"].eq("raw")].set_index("id")
    assert (raw["pinball_q05"] > primary["pinball_q05"]).all()
    shuffled = forecasts.sample(frac=1, random_state=29).reset_index(drop=True)
    actual = score_quantile_window(wide, shuffled, 60, 4, dates, closed)
    for expected, observed in zip((series, horizons, summary), actual, strict=True):
        assert_frame_equal(expected, observed)


def test_closed_horizon_has_zero_count_and_undefined_rates(invented_window):
    wide, forecasts, dates, closed = invented_window
    closed.loc[64] = True
    series, horizons, _ = score_quantile_window(wide, forecasts, 60, 4, dates, closed)
    assert series["n_evaluated"].eq(3).all()
    masked = horizons.loc[horizons["horizon"].eq(4)]
    assert masked["n_evaluated"].eq(0).all()
    assert masked["pinball_q90"].isna().all() and masked["interval90_coverage"].isna().all()


def test_calendar_and_csv_can_have_different_datetime_resolutions(invented_window):
    wide, forecasts, dates, closed = invented_window
    forecasts["date"] = forecasts["date"].astype("datetime64[ns]")
    expected = score_quantile_window(wide, forecasts, 60, 4, dates, closed)
    actual = score_quantile_window(wide, forecasts, 60, 4, dates.astype("datetime64[us]"), closed)
    for before, after in zip(expected, actual, strict=True):
        assert_frame_equal(before, after)


@pytest.mark.parametrize("problem", ["duplicate", "missing", "origin", "horizon", "date",
                                     "nan", "clipping", "reordered"])
def test_invalid_forecast_panel_is_rejected(invented_window, problem):
    wide, frame, dates, closed = invented_window
    if problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif problem == "missing":
        frame = frame.iloc[1:].copy()
    elif problem == "origin":
        frame.loc[0, "origin_day"] = 61
    elif problem == "horizon":
        frame.loc[0, "horizon"] = 2
    elif problem == "date":
        frame.loc[0, "date"] = pd.NaT
    elif problem == "nan":
        frame.loc[0, "q95_raw"] = np.inf
    elif problem == "clipping":
        frame.loc[0, "q05"] = -1
    else:
        frame.loc[2, ["q05", "q50", "q90", "q95"]] = [4, 5, 6, 7]
    with pytest.raises(ValueError):
        score_quantile_window(wide, frame, 60, 4, dates, closed)


def test_scoring_persists_separate_run_and_rejects_changed_split(invented_window, tmp_path):
    wide, forecasts, dates, _ = invented_window
    split = SplitConfig(1, 60, 64, 68, 4)
    tracking = TrackingConfig("invented-scoring-test", tmp_path / "mlruns")
    previous_uri = mlflow.get_tracking_uri()
    try:
        client, experiment_id = configure_tracking(tracking)
        with mlflow.start_run(experiment_id=experiment_id, tags={
            "stage": "quantile_first_window_training", "forecast_strategy": "direct_pooled_horizon",
        }) as source:
            mlflow.log_params({"forecast_origin_day": 60, "horizon_days": 4})
            mlflow.log_dict({"split": asdict(split), "quantiles": {"quantiles": QUANTILES}},
                            "effective_config.json")
            subset_path = tmp_path / "subset_ids.csv"
            pd.DataFrame({"id": wide.index}).to_csv(subset_path, index=False)
            mlflow.log_artifact(str(subset_path), artifact_path="snapshot")
            forecast_path = tmp_path / "forecasts_valid_first_window.csv"
            forecasts.to_csv(forecast_path, index=False)
            mlflow.log_artifact(str(forecast_path), artifact_path="forecasts")
            source_id = source.info.run_id
        with pytest.raises(ValueError, match="Split/calendar"):
            load_validation_source(client, experiment_id, source_id, replace(split, closed_day=24))
        score_id, output_dir = score_saved_validation_run(
            client, experiment_id, source_id, split, wide, dates, tmp_path / "reports"
        )
        saved = client.get_run(score_id)
        assert saved.info.status == "FINISHED" and score_id != source_id
        assert saved.data.tags["source_forecast_run_id"] == source_id
        assert saved.data.tags["model_trained"] == "false"
        assert saved.data.tags["test_evaluated"] == "false"
        assert saved.data.metrics["evaluated_observations"] == 8
        assert saved.data.metrics["clipped_crossing_rate_mean"] == 0.25
        assert client.get_run(source_id).data.metrics == {}
        assert {p.name for p in output_dir.iterdir()} == {
            "per_series_window.csv", "per_horizon_window.csv", "summary_window.csv",
        }
        summary = pd.read_csv(output_dir / "summary_window.csv")
        for alpha in QUANTILES:
            metric = f"pinball_{quantile_name(alpha)}"
            assert metric in set(summary["metric"])
    finally:
        mlflow.set_tracking_uri(previous_uri)


def test_source_from_test_window_is_rejected_before_reading_artifacts():
    class Client:
        def get_run(self, run_id):
            return SimpleNamespace(
                info=SimpleNamespace(status="FINISHED", experiment_id="1"),
                data=SimpleNamespace(
                    tags={"stage": "quantile_first_window_training",
                          "forecast_strategy": "direct_pooled_horizon"},
                    params={"forecast_origin_day": "64", "horizon_days": "4"},
                ),
            )

        def download_artifacts(self, *args):
            pytest.fail("Test-window artifacts must not be accessed")

    with pytest.raises(ValueError, match="first validation window"):
        load_validation_source(Client(), "1", "invented-test-run", SplitConfig(1, 60, 64, 68, 4))
