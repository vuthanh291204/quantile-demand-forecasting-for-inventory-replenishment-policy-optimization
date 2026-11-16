"""Score saved q05/q50/q90/q95 forecasts on the first validation window.

Run: python -m src.evaluation.score_quantiles --run-id <training-run-id>
No model is fitted; no test labels or model-selection decisions are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from itertools import pairwise
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.data.direct_features import FORECAST_KEYS
from src.data.split import (
    REPO_ROOT,
    WIDE_PATH,
    closed_days,
    day_columns,
    load_config,
    load_day_dates,
)
from src.evaluation.quantile_metrics import (
    crossing_per_series,
    interval_diagnostics_per_series,
    pinball_per_series,
    quantile_coverage_per_series,
)
from src.forecast.quantile_model import load_quantile_config, quantile_name
from src.tracking import configure_tracking, load_tracking_config

QUANTILES = (0.05, 0.50, 0.90, 0.95)
VARIANTS = ("clipped", "raw")
COUNT_COLUMNS = ("n_evaluated", "n_ordered_interval")


def _diagnostics(actual, predicted, keep):
    scores = interval_diagnostics_per_series(actual, predicted["q05"], predicted["q95"], keep)
    scores = scores.rename(columns={
        c: c.replace("interval_", "interval90_", 1) for c in scores if c.startswith("interval_")
    })
    for alpha in QUANTILES:
        name = quantile_name(alpha)
        scores[f"pinball_{name}"] = pinball_per_series(actual, predicted[name], alpha, keep)
        scores[f"coverage_le_{name}"] = quantile_coverage_per_series(actual, predicted[name], keep)
        scores[f"coverage_lt_{name}"] = quantile_coverage_per_series(
            actual, predicted[name], keep, strict=True
        )
    names = [quantile_name(q) for q in QUANTILES]
    scores["crossing_rate"] = crossing_per_series([predicted[n] for n in names], keep)
    for low, high in pairwise(names):
        # strict=True measures high < low, exactly the adjacent crossing event.
        scores[f"crossing_{low}_gt_{high}"] = quantile_coverage_per_series(
            predicted[high], predicted[low], keep, strict=True
        )
    return scores


def validate_forecasts(wide, forecasts, origin, horizon, dates):
    names = [quantile_name(q) for q in QUANTILES]
    columns = [*FORECAST_KEYS, *names, *[f"{n}_raw" for n in names]]
    missing = set(columns) - set(forecasts)
    if missing:
        raise ValueError(f"Missing forecast columns: {sorted(missing)}")
    if wide.index.has_duplicates or wide.empty:
        raise ValueError("Need unique, non-empty actual series")
    expected = pd.MultiIndex.from_product(
        [wide.index, range(origin + 1, origin + horizon + 1)], names=["id", "day"]
    )
    keys = pd.MultiIndex.from_frame(forecasts[["id", "day"]])
    if forecasts.duplicated(["id", "day"]).any() or not keys.sort_values().equals(
        expected.sort_values()
    ):
        raise ValueError("Forecasts must contain every series/day exactly once")
    if not forecasts["origin_day"].eq(origin).all() or not forecasts["horizon"].eq(
        forecasts["day"] - origin
    ).all():
        raise ValueError("Origin/horizon metadata does not match forecast days")
    saved_dates = pd.to_datetime(forecasts["date"], errors="raise")
    expected_dates = pd.to_datetime(forecasts["day"].map(dates))
    # CSV parsing may yield ns while the parquet calendar uses us. Compare values,
    # not Series.equals(), which also requires identical datetime resolutions.
    if saved_dates.isna().any() or expected_dates.isna().any() or not saved_dates.eq(expected_dates).all():
        raise ValueError("Forecast dates do not match the evaluation calendar")
    values = forecasts[[*names, *[f"{n}_raw" for n in names]]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Forecast values must be finite")
    for name in names:
        if not np.allclose(forecasts[name], np.maximum(forecasts[f"{name}_raw"], 0),
                           rtol=1e-12, atol=0):
            raise ValueError("Clipped forecasts must equal maximum(raw, 0); no sorting allowed")
    actual = wide[day_columns(origin + 1, origin + horizon)]
    if not np.isfinite(actual.to_numpy(dtype=float)).all() or actual.lt(0).any().any():
        raise ValueError("Actual sales must be finite and non-negative")


def score_quantile_window(wide, forecasts, origin, horizon, dates, closed):
    """Return per-series, per-horizon and mean/median diagnostics for both variants."""
    validate_forecasts(wide, forecasts, origin, horizon, dates)
    actual = wide[day_columns(origin + 1, origin + horizon)]
    keep = ~closed.reindex(range(origin + 1, origin + horizon + 1)).to_numpy(dtype=bool)
    if closed.reindex(range(origin + 1, origin + horizon + 1)).isna().any():
        raise ValueError("Need a closure flag for every forecast day")
    per_series = []
    per_horizon = []
    for variant in VARIANTS:
        predicted = {}
        for alpha in QUANTILES:
            name = quantile_name(alpha)
            column = name if variant == "clipped" else f"{name}_raw"
            matrix = forecasts.pivot(index="id", columns="day", values=column)
            matrix = matrix.rename(columns=lambda d: f"d_{d}")
            predicted[name] = matrix.reindex(index=actual.index, columns=actual.columns)
        scores = _diagnostics(actual, predicted, keep)
        per_series.append(scores.rename_axis("id").reset_index().assign(variant=variant))
        # Transposing makes each horizon a row and the series its observations.
        horizon_keep = np.broadcast_to(keep[:, None], actual.T.shape)
        steps = _diagnostics(actual.T, {n: p.T for n, p in predicted.items()}, horizon_keep)
        steps.index = pd.Index(range(1, horizon + 1), name="horizon")
        per_horizon.append(steps.reset_index().assign(variant=variant))
    per_series = pd.concat(per_series, ignore_index=True)
    per_horizon = pd.concat(per_horizon, ignore_index=True)
    metrics = [c for c in per_series if c not in ("id", "variant", *COUNT_COLUMNS)]
    return per_series, per_horizon, summarize_quantile_scores(per_series, metrics)


def summarize_quantile_scores(per_series, metrics):
    """Give each series equal weight and report the defined-series count."""
    summary_rows = []
    for variant in VARIANTS:
        part = per_series.loc[per_series["variant"].eq(variant)]
        for metric in metrics:
            values = part[metric]
            summary_rows.append({
                "variant": variant, "metric": metric,
                "mean": values.mean(), "median": values.median(),
                "n_series_defined": int(values.notna().sum()),
            })
    return pd.DataFrame(summary_rows)


def print_quantile_summary(summary):
    """Print clipped diagnostics with percentage units for rates."""
    for row in summary.loc[summary["variant"].eq("clipped")].itertuples():
        if row.metric.startswith("pinball_") or row.metric in (
            "coverage_le_q90", "coverage_le_q95", "interval90_coverage",
            "interval90_mean_width_ordered", "interval90_invalid_rate", "crossing_rate",
        ):
            is_rate = row.metric.startswith("coverage_") or "rate" in row.metric or (
                row.metric == "interval90_coverage"
            )
            scale, unit = (100, "%") if is_rate else (1, "")
            print(f"{row.metric}: mean={row.mean * scale:.6f}{unit}; "
                  f"median={row.median * scale:.6f}{unit}; defined_series={row.n_series_defined}")


def load_validation_source(client, experiment_id, run_id, split):
    source = client.get_run(run_id)
    if source.info.status != "FINISHED" or source.info.experiment_id != experiment_id:
        raise ValueError("Source must be a finished run in the quantile experiment")
    if source.data.tags.get("stage") != "quantile_first_window_training" or (
        source.data.tags.get("forecast_strategy") != "direct_pooled_horizon"
    ):
        raise ValueError("Need a first-window direct quantile training run")
    params = source.data.params
    if int(params["forecast_origin_day"]) != split.train_end or int(
        params["horizon_days"]
    ) != split.horizon:
        raise ValueError("This command only scores the first validation window")
    paths = {
        name: Path(client.download_artifacts(run_id, name)) for name in (
            "effective_config.json", "snapshot/subset_ids.csv",
            "forecasts/forecasts_valid_first_window.csv",
        )
    }
    archived = json.loads(paths["effective_config.json"].read_text(encoding="utf-8"))
    if archived["split"] != asdict(split):
        raise ValueError("Split/calendar settings changed since the forecast run")
    if tuple(archived["quantiles"]["quantiles"]) != QUANTILES:
        raise ValueError("This scorer requires q05, q50, q90 and q95")
    ids = pd.read_csv(paths["snapshot/subset_ids.csv"])["id"]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("Archived series IDs must be unique and non-missing")
    return pd.read_csv(paths["forecasts/forecasts_valid_first_window.csv"]), ids, paths


def score_saved_validation_run(client, experiment_id, source_id, split, wide, dates, output_root):
    forecasts, ids, source_paths = load_validation_source(client, experiment_id, source_id, split)
    if set(ids) != set(wide.index):
        raise ValueError("Actual series differ from the archived forecast subset")
    closed = closed_days(dates, split)
    per_series, per_horizon, summary = score_quantile_window(
        wide, forecasts, split.train_end, split.horizon, dates, closed
    )
    window_days = range(split.train_end + 1, split.train_end + split.horizon + 1)
    excluded_days = int(closed.reindex(window_days).sum())
    clipped = per_series.loc[per_series["variant"].eq("clipped")]
    with mlflow.start_run(
        experiment_id=experiment_id, run_name="quantiles-score-first-window",
        tags={
            "stage": "quantile_first_window_scoring", "source_forecast_run_id": source_id,
            "model_trained": "false", "validation_scored": "true", "test_evaluated": "false",
        },
    ) as run:
        output_dir = Path(output_root) / source_id / "scoring" / run.info.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in (
            ("per_series_window.csv", per_series),
            ("per_horizon_window.csv", per_horizon), ("summary_window.csv", summary),
        ):
            path = output_dir / name
            frame.to_csv(path, index=False)
            mlflow.log_artifact(str(path), artifact_path="scores")
        mlflow.log_params({
            "forecast_origin_day": split.train_end, "horizon_days": split.horizon,
            "quantiles": str(QUANTILES), "primary_variant": "clipped",
            "interval": "inclusive_q05_q95", "invalid_interval_policy": "uncovered_width_undefined",
            "aggregation": "per_series_then_equal_weight_mean_and_median",
            "postprocessing_added": "none",
        })
        logged = {
            f"{r.variant}_{r.metric}_{stat}": float(getattr(r, stat))
            for r in summary.itertuples() for stat in ("mean", "median")
            if np.isfinite(getattr(r, stat))
        }
        logged.update({
            "n_series": len(wide), "forecast_rows": len(forecasts),
            "closed_days_excluded": excluded_days,
            "evaluated_observations": int(clipped["n_evaluated"].sum()),
            "undefined_series_metric_values": int(summary["n_series_defined"].rsub(len(wide)).sum()),
        })
        mlflow.log_metrics(logged)
        mlflow.log_dict({
            "source_forecast_run_id": source_id,
            "source_sha256": {n: hashlib.sha256(p.read_bytes()).hexdigest()
                              for n, p in source_paths.items()},
            "split": asdict(split), "n_series": len(wide),
            "evaluation_labels_sha256": hashlib.sha256(
                json.dumps(day_columns(split.train_end + 1, split.train_end + split.horizon)).encode()
                + pd.util.hash_pandas_object(
                    wide[day_columns(split.train_end + 1, split.train_end + split.horizon)].sort_index(),
                    index=True,
                ).to_numpy().tobytes()
            ).hexdigest(),
        }, "scoring_provenance.json")
        for relative in ("src/evaluation/quantile_metrics.py", "src/evaluation/score_quantiles.py",
                         "src/evaluation/metrics.py", "src/data/split.py"):
            mlflow.log_artifact(str(REPO_ROOT / relative), artifact_path="code")
        scoring_id = run.info.run_id
    print("Source forecast run:", source_id)
    print(f"Window: d_{split.train_end + 1}..d_{split.train_end + split.horizon}")
    print(f"Series: {len(wide)}; forecast rows: {len(forecasts)}")
    print(f"Closed days excluded: {excluded_days}; evaluated observations: {int(clipped['n_evaluated'].sum())}")
    print("Primary variant: clipped; rates below are percentages; pinball/width are in units")
    print_quantile_summary(summary)
    print("Raw diagnostics: saved alongside clipped diagnostics in CSV and MLflow")
    print("Scoring run ID:", scoring_id)
    print("Status:", client.get_run(scoring_id).info.status)
    print("Saved scores:", output_dir)
    print("Model trained: false; test evaluated: false")
    return scoring_id, output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Completed first-window quantile training run")
    args = parser.parse_args()
    split = load_config()
    tracking = replace(load_tracking_config(), experiment_name=load_quantile_config().experiment_name)
    client, experiment_id = configure_tracking(tracking)
    columns = ["id", *day_columns(split.train_end + 1, split.train_end + split.horizon)]
    wide = pd.read_parquet(WIDE_PATH, columns=columns).set_index("id")
    score_saved_validation_run(
        client, experiment_id, args.run_id, split, wide, load_day_dates(),
        REPO_ROOT / "reports" / "quantiles",
    )


if __name__ == "__main__":
    main()
