"""Refit quantile models at rolling origins using archived settings.

Run: python -m src.evaluation.backtest_quantiles --source-run-id <training-run-id>
Test: python -m src.evaluation.backtest_quantiles --split test --source-run-id <validation-run-id>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.data.direct_features import (
    build_direct_forecast_table,
    build_direct_training_table,
    direct_feature_columns,
)
from src.data.features import FeatureConfig
from src.data.split import (
    LONG_PATH,
    REPO_ROOT,
    WIDE_PATH,
    closed_days,
    day_columns,
    load_config,
    load_day_dates,
    make_windows,
)
from src.evaluation.score_quantiles import (
    COUNT_COLUMNS,
    VARIANTS,
    load_validation_source,
    print_quantile_summary,
    score_quantile_window,
    summarize_quantile_scores,
    validate_forecasts,
)
from src.forecast.quantile_model import (
    QuantileConfig,
    fit_quantiles,
    load_quantile_config,
    predict_quantiles,
)
from src.tracking import configure_tracking, load_tracking_config


def aggregate_quantile_windows(scores, ids, n_windows, metrics):
    """Average windows within each series, retaining undefined window counts."""
    expected = pd.MultiIndex.from_product(
        [VARIANTS, range(1, n_windows + 1), ids], names=["variant", "window", "id"]
    )
    actual = pd.MultiIndex.from_frame(scores[["variant", "window", "id"]])
    if actual.has_duplicates or not actual.sort_values().equals(expected.sort_values()):
        raise ValueError("Need exactly one score per variant/series/window")
    values = scores[metrics].to_numpy(dtype=float)
    width = "interval90_mean_width_ordered"
    finite_metrics = [m for m in metrics if m != width]
    if np.isinf(values).any() or not np.isfinite(scores[finite_metrics].to_numpy()).all():
        raise ValueError("Window metrics must be finite except undefined interval widths")
    grouped = scores.groupby(["variant", "id"], sort=True)
    defined = grouped[metrics].count()
    result = grouped[metrics].mean().where(defined.eq(n_windows))
    result[list(COUNT_COLUMNS)] = grouped[list(COUNT_COLUMNS)].sum()
    result["n_windows"] = n_windows
    result["width_windows_defined"] = defined[width]
    return result.reset_index()


def run_validation_backtest(client, experiment_id, source_id, split, long, wide, dates, output_root):
    return run_backtest(client, experiment_id, source_id, split, long, wide, dates, output_root)


def run_backtest(client, experiment_id, source_id, split, long, wide, dates, output_root,
                 evaluation_split="valid"):
    if evaluation_split not in ("valid", "test"):
        raise ValueError("evaluation_split must be valid or test")
    if evaluation_split == "valid":
        source_forecasts, ids, source_paths = load_validation_source(
            client, experiment_id, source_id, split
        )
        validate_forecasts(wide, source_forecasts, split.train_end, split.horizon, dates)
    else:
        source = client.get_run(source_id)
        if source.info.status != "FINISHED" or source.info.experiment_id != experiment_id:
            raise ValueError("Need a finished validation run in the quantile experiment")
        if source.data.tags.get("stage") != "quantile_validation_backtest":
            raise ValueError("Test settings must come from a completed validation backtest")
        source_paths = {name: Path(client.download_artifacts(source_id, name)) for name in (
            "effective_config.json", "configs/subset_ids.csv",
        )}
        ids = pd.read_csv(source_paths["configs/subset_ids.csv"])["id"]
        if ids.isna().any() or ids.duplicated().any():
            raise ValueError("Archived series IDs must be unique and non-missing")
    if set(ids) != set(wide.index) or set(ids) != set(long["id"]):
        raise ValueError("Processed series must match the archived subset")
    archived = json.loads(source_paths["effective_config.json"].read_text(encoding="utf-8"))
    if archived["split"] != asdict(split):
        raise ValueError("Split/calendar settings changed since the source run")
    settings = archived["quantiles"]
    cfg = QuantileConfig(
        tuple(settings["quantiles"]), settings["origin_stride"],
        settings["model_params"], settings["experiment_name"],
    )
    feature_cfg = FeatureConfig(**archived["features"])
    if (feature_cfg.closed_month, feature_cfg.closed_day) != (split.closed_month, split.closed_day):
        raise ValueError("Feature closure settings must match the split calendar")
    windows = make_windows(*split.ranges[evaluation_split], split.horizon)
    end_day = split.valid_end if evaluation_split == "valid" else split.last_day
    days = long["d"].str.slice(2).astype(int)
    long = long.loc[days <= end_day].copy()
    closed = closed_days(dates, split)
    window_scores = []
    horizon_scores = []
    forecast_parts = []
    window_info = []

    source_tag = "source_training_run_id" if evaluation_split == "valid" else "source_validation_run_id"
    stage = "quantile_validation_backtest" if evaluation_split == "valid" else "quantile_test_backtest"
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=f"quantiles-expanding-{evaluation_split}",
        tags={
            "stage": stage, source_tag: source_id,
            "forecast_strategy": "direct_pooled_horizon", "refit_policy": "expanding_window",
            "model_trained": "true", "validation_scored": str(evaluation_split == "valid").lower(),
            "test_evaluated": str(evaluation_split == "test").lower(),
        },
    ) as run:
        output_dir = Path(output_root) / run.info.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        mlflow.log_params({
            **cfg.model_params, "quantiles": str(cfg.quantiles),
            "origin_stride": cfg.origin_stride, "evaluation_split": evaluation_split,
            "horizon_days": split.horizon, "n_windows": len(windows),
            "primary_variant": "clipped", "price_policy": "observed_at_origin",
            "aggregation": "mean_windows_within_series_then_mean_and_median_over_series",
            "undefined_window_policy": "series_metric_requires_all_windows_defined",
        })
        mlflow.log_dict({
            "quantiles": asdict(cfg), "split": asdict(split), "features": asdict(feature_cfg),
        }, "effective_config.json")
        mlflow.log_dict({
            source_tag: source_id,
            "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in source_paths.items()},
        }, "source_provenance.json")
        mlflow.log_dict({
            "python": platform.python_version(),
            "libraries": {name: version(name) for name in ("lightgbm", "pandas", "numpy", "mlflow")},
        }, "environment.json")
        subset_key = "snapshot/subset_ids.csv" if evaluation_split == "valid" else "configs/subset_ids.csv"
        mlflow.log_artifact(str(source_paths[subset_key]), artifact_path="configs")
        for relative in (
            "src/evaluation/backtest_quantiles.py", "src/evaluation/score_quantiles.py",
            "src/evaluation/quantile_metrics.py", "src/evaluation/metrics.py",
            "src/forecast/quantile_model.py", "src/data/direct_features.py",
            "src/data/features.py", "src/data/split.py", "src/tracking.py",
        ):
            mlflow.log_artifact(str(REPO_ROOT / relative), artifact_path="code")

        for number, (first, last) in enumerate(windows, start=1):
            origin = first - 1
            print(f"Window {number}/{len(windows)}: fit through d_{origin}; forecast d_{first}..d_{last}",
                  flush=True)
            train = build_direct_training_table(
                long, origin, split.horizon, cfg.origin_stride, feature_cfg
            )
            future = build_direct_forecast_table(long, origin, split.horizon, feature_cfg)

            def save_model(name, model, window_number=number):
                mlflow.log_text(model.booster_.model_to_string(), f"models/window_{window_number}/{name}.txt")
                iterations = model.booster_.current_iteration()
                mlflow.log_metric(f"{name}_iterations", iterations, step=window_number)
                print(f"Window {window_number} fitted {name}: iterations={iterations}", flush=True)

            models, levels = fit_quantiles(train, cfg, feature_cfg, on_fitted=save_model)
            mlflow.log_dict(levels, f"models/window_{number}/category_levels.json")
            mlflow.log_dict({"feature_columns": direct_feature_columns(feature_cfg)},
                            f"models/window_{number}/feature_schema.json")
            forecasts = predict_quantiles(models, future, feature_cfg, levels)
            scores, steps, window_summary = score_quantile_window(
                wide, forecasts, origin, split.horizon, dates, closed
            )
            window_scores.append(scores.assign(window=number, origin_day=origin))
            horizon_scores.append(steps.assign(window=number, origin_day=origin))
            forecast_parts.append(forecasts.assign(window=number))
            excluded = int(closed.loc[first:last].sum())
            window_info.append({
                "window": number, "origin_day": origin, "first_day": first, "last_day": last,
                "train_rows": len(train), "train_origins": train["origin_day"].nunique(),
                "last_training_target": int(train["day"].max()), "closed_days_excluded": excluded,
            })
            mlflow.log_metrics({
                f"window_{row.variant}_{row.metric}_mean": float(row.mean)
                for row in window_summary.itertuples() if np.isfinite(row.mean)
            }, step=number)
            print(f"Window {number} completed: forecast rows={len(forecasts)}; train rows={len(train)}",
                  flush=True)

        per_window = pd.concat(window_scores, ignore_index=True)
        metrics = window_summary["metric"].drop_duplicates().tolist()
        per_series = aggregate_quantile_windows(per_window, wide.index, len(windows), metrics)
        summary = summarize_quantile_scores(per_series, metrics)
        all_forecasts = pd.concat(forecast_parts, ignore_index=True)
        outputs = {
            f"forecasts_{evaluation_split}.csv": all_forecasts,
            f"per_window_{evaluation_split}.csv": per_window,
            f"per_series_{evaluation_split}.csv": per_series,
            f"per_horizon_{evaluation_split}.csv": pd.concat(horizon_scores, ignore_index=True),
            f"summary_{evaluation_split}.csv": summary,
            f"window_info_{evaluation_split}.csv": pd.DataFrame(window_info),
        }
        for filename, frame in outputs.items():
            path = output_dir / filename
            frame.to_csv(path, index=False)
            mlflow.log_artifact(str(path), artifact_path="results")
        evaluated = int(per_series.loc[per_series["variant"].eq("clipped"), "n_evaluated"].sum())
        mlflow.log_metrics({
            **{f"{evaluation_split}_{row.variant}_{row.metric}_{stat}": float(getattr(row, stat))
               for row in summary.itertuples() for stat in ("mean", "median")
               if np.isfinite(getattr(row, stat))},
            "n_windows": len(windows), "n_series": len(wide), "forecast_rows": len(all_forecasts),
            "evaluated_observations": evaluated,
            "closed_days_excluded": sum(info["closed_days_excluded"] for info in window_info),
            "undefined_series_metric_values": int(summary["n_series_defined"].rsub(len(wide)).sum()),
        })
        run_id = run.info.run_id

    print("Source training run:" if evaluation_split == "valid" else "Source validation run:", source_id)
    label = "Validation" if evaluation_split == "valid" else "Test"
    print(f"{label} windows: {len(windows)}; series: {len(wide)}")
    print("Forecast shape:", all_forecasts.shape)
    print("Forecast days:", all_forecasts["day"].min(), all_forecasts["day"].max())
    print("Evaluated observations:", evaluated)
    print("Closed days excluded:", sum(info["closed_days_excluded"] for info in window_info))
    print("Aggregation: mean windows within each series, then mean/median over series")
    print("Primary variant: clipped; rates below are percentages; pinball/width are in units")
    print_quantile_summary(summary)
    print("Run ID:", run_id)
    print("Status:", client.get_run(run_id).info.status)
    print("Saved results:", output_dir)
    print("Test evaluated:", str(evaluation_split == "test").lower())
    return run_id, output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", required=True, help="Training run for valid, validation run for test")
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    args = parser.parse_args()
    split = load_config()
    tracking = replace(load_tracking_config(), experiment_name=load_quantile_config().experiment_name)
    client, experiment_id = configure_tracking(tracking)
    columns = ["id", *day_columns(*split.ranges[args.split])]
    run_backtest(
        client, experiment_id, args.source_run_id, split, pd.read_parquet(LONG_PATH),
        pd.read_parquet(WIDE_PATH, columns=columns).set_index("id"), load_day_dates(),
        REPO_ROOT / "reports" / "quantiles", evaluation_split=args.split,
    )


if __name__ == "__main__":
    main()
