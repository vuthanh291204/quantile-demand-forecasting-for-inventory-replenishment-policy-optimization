import argparse
import json
import tomllib

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from src.data.features import load_feature_config
from src.data.split import (
    LONG_PATH,
    REPO_ROOT,
    WIDE_PATH,
    closed_days,
    load_config,
    load_day_dates,
    make_windows,
)
from src.evaluation.run_final_baselines import overall_table
from src.evaluation.score_lightgbm import METRICS, score_window
from src.forecast.lightgbm_model import (
    prepare_training_data,
    recursive_forecast,
)
from src.tracking import configure_tracking, load_tracking_config


def read_toml(path):
    with open(path, "rb") as file:
        return tomllib.load(file)


def aggregate_windows(scores, ids, n_windows):
    expected = pd.MultiIndex.from_product(
        [range(1, n_windows + 1), ids],
        names=["window", "id"],
    )
    actual = pd.MultiIndex.from_frame(
        scores[["window", "id"]]
    )

    if (
        scores.duplicated(["window", "id"]).any()
        or not actual.sort_values().equals(
            expected.sort_values()
        )
    ):
        raise ValueError(
            "Need exactly one score per series/window"
        )

    if not np.isfinite(scores[METRICS].to_numpy()).all():
        raise ValueError("Window scores must be finite")

    return scores.groupby("id")[METRICS].mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-run-id", required=True)
    args = parser.parse_args()

    split = load_config()
    feature_cfg = load_feature_config()
    windows = make_windows(
        *split.ranges["valid"],
        split.horizon,
    )

    client, experiment_id = configure_tracking(
        load_tracking_config()
    )
    source = client.get_run(args.selection_run_id)

    if (
        source.info.status != "FINISHED"
        or source.info.experiment_id != experiment_id
        or "best_iteration" not in source.data.metrics
    ):
        raise ValueError("Need a completed selection run")

    archived = {
        name: client.download_artifacts(
            args.selection_run_id,
            f"configs/{name}",
        )
        for name in (
            "lightgbm.toml",
            "features.toml",
            "split.toml",
        )
    }

    for name in ("features.toml", "split.toml"):
        if read_toml(archived[name]) != read_toml(
            REPO_ROOT / "configs" / name
        ):
            raise ValueError(
                f"{name} changed since model selection"
            )

    params = read_toml(archived["lightgbm.toml"])["model"]
    params["n_estimators"] = int(
        source.data.metrics["best_iteration"]
    )

    wide = pd.read_parquet(WIDE_PATH).set_index("id")
    long = pd.read_parquet(LONG_PATH)
    closed = closed_days(load_day_dates(), split)

    baseline_dir = REPO_ROOT / "reports" / "baselines"
    baseline_path = baseline_dir / "per_series_valid.csv"
    info_path = baseline_dir / "run_info.json"

    info = json.loads(info_path.read_text(encoding="utf-8"))

    if (
        info["horizon_days"] != split.horizon
        or info["n_series"] != len(wide)
        or info["splits"]["valid"]["days"]
        != list(split.ranges["valid"])
        or info["splits"]["valid"]["windows"] != len(windows)
    ):
        raise ValueError("Baseline evaluation settings differ")

    baselines = pd.read_csv(baseline_path)

    if (
        not baselines["split"].eq("valid").all()
        or not np.isfinite(
            baselines[METRICS].to_numpy()
        ).all()
    ):
        raise ValueError("Invalid baseline results")

    results = {}

    for name, part in baselines.groupby("method"):
        if (
            part["id"].duplicated().any()
            or set(part["id"]) != set(wide.index)
        ):
            raise ValueError(
                f"Baseline series differ: {name}"
            )

        results[name] = part.set_index("id")[METRICS]

    window_parts = []
    forecast_parts = []
    method = f"lightgbm_{params['n_estimators']}_refit"

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="lightgbm-vs-baselines-valid",
        tags={
            "stage": "validation_baseline_comparison",
            "selection_run_id": args.selection_run_id,
            "refit_policy": "expanding_window",
            "model_trained": "true",
            "test_evaluated": "false",
        },
    ) as run:
        mlflow.log_params(params)
        mlflow.log_params({
            "evaluation_split": "valid",
            "horizon_days": split.horizon,
            "n_windows": len(windows),
        })

        for number, (first, last) in enumerate(
            windows, start=1
        ):
            origin = first - 1
            print(
                f"Window {number}/{len(windows)}: "
                f"fit through d_{origin}",
                flush=True,
            )

            X, y, levels = prepare_training_data(
                end_day=origin
            )
            model = lgb.LGBMRegressor(**params)
            model.fit(X, y, categorical_feature="auto")

            forecasts = recursive_forecast(
                model=model,
                long=long,
                origin_day=origin,
                horizon=last - first + 1,
                feature_cfg=feature_cfg,
                category_levels=levels,
                verbose=True,
            )
            scores = score_window(
                wide, forecasts, origin, split.horizon, closed
            )

            part = scores.rename_axis("id").reset_index()
            part["window"] = number
            window_parts.append(part)
            forecast_parts.append(forecasts)

            mlflow.log_metric(
                "window_rmsse_mean",
                float(scores["rmsse"].mean()),
                step=number,
            )
            mlflow.log_text(
                model.booster_.model_to_string(),
                f"models/window_{number}/lightgbm.txt",
            )
            mlflow.log_dict(
                levels,
                f"models/window_{number}/category_levels.json",
            )

        window_scores = pd.concat(
            window_parts, ignore_index=True
        )
        forecasts_all = pd.concat(
            forecast_parts, ignore_index=True
        )

        per_series = aggregate_windows(
            window_scores,
            wide.index,
            len(windows),
        )
        results[method] = per_series

        comparison = overall_table(results).sort_values(
            "rmsse mean"
        )

        mlflow.log_metrics({
            f"lightgbm_{column.replace(' ', '_')}": float(value)
            for column, value in comparison.loc[method].items()
        })

        output_dir = (
            REPO_ROOT / "reports" / "lightgbm" / run.info.run_id
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = [
            ("per_window_valid.csv", window_scores),
            (
                "per_series_valid.csv",
                per_series.rename_axis("id").reset_index(),
            ),
            (
                "comparison_valid.csv",
                comparison.rename_axis("method").reset_index(),
            ),
        ]

        for filename, frame in outputs:
            path = output_dir / filename
            frame.to_csv(path, index=False)
            mlflow.log_artifact(
                str(path), artifact_path="results"
            )

        forecast_path = (
            REPO_ROOT
            / "data"
            / "processed"
            / "lightgbm_valid_all_windows.csv"
        )
        forecasts_all.to_csv(forecast_path, index=False)
        mlflow.log_artifact(
            str(forecast_path),
            artifact_path="forecasts",
        )

        for path in archived.values():
            mlflow.log_artifact(
                str(path), artifact_path="configs"
            )

        for path in (baseline_path, info_path):
            mlflow.log_artifact(
                str(path), artifact_path="baselines"
            )

        for relative in (
            "src/evaluation/compare_lightgbm.py",
            "src/forecast/lightgbm_model.py",
            "src/evaluation/score_lightgbm.py",
        ):
            mlflow.log_artifact(
                str(REPO_ROOT / relative),
                artifact_path="code",
            )

        run_id = run.info.run_id

    print("Windows:", len(windows))
    print("Forecast rows:", len(forecasts_all))
    print("Methods:", len(comparison))
    print(comparison.round(6).to_string())
    print("Run ID:", run_id)
    print("Status:", client.get_run(run_id).info.status)


if __name__ == "__main__":
    main()
