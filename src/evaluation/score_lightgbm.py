import argparse

import mlflow
import numpy as np
import pandas as pd

from src.data.split import (
    REPO_ROOT,
    WIDE_PATH,
    closed_days,
    day_columns,
    load_config,
    load_day_dates,
)
from src.evaluation.metrics import (
    mae_per_series,
    rmse_per_series,
    rmsse_per_series,
    smape_per_series,
)
from src.tracking import configure_tracking, load_tracking_config

METRICS = ["mae", "rmse", "smape", "rmsse"]


def score_window(wide, forecasts, origin_day, horizon, closed):
    first = origin_day + 1
    last = origin_day + horizon

    expected_keys = pd.MultiIndex.from_product(
        [wide.index, range(first, last + 1)],
        names=["id", "day"],
    )
    actual_keys = pd.MultiIndex.from_frame(
        forecasts[["id", "day"]]
    )

    if (
        forecasts.duplicated(["id", "day"]).any()
        or not actual_keys.sort_values().equals(
            expected_keys.sort_values()
        )
    ):
        raise ValueError(
            "Forecasts must contain every series/day exactly once"
        )

    values = forecasts["y_pred"].to_numpy(dtype=float)

    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(
            "Predictions must be finite and non-negative"
        )

    actual = wide[day_columns(first, last)]
    history = wide[day_columns(1, origin_day)]

    predicted = forecasts.pivot(
        index="id",
        columns="day",
        values="y_pred",
    )
    predicted = predicted.rename(
        columns=lambda day: f"d_{day}"
    )
    predicted = predicted.reindex(
        index=actual.index,
        columns=actual.columns,
    )

    keep = ~closed.loc[first:last].to_numpy()

    return pd.DataFrame({
        "mae": mae_per_series(
            actual, predicted, mask=keep
        ),
        "rmse": rmse_per_series(
            actual, predicted, mask=keep
        ),
        "smape": smape_per_series(
            actual, predicted, mask=keep
        ),
        "rmsse": rmsse_per_series(
            actual, predicted, history, mask=keep
        ),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    split = load_config()
    client, experiment_id = configure_tracking(
        load_tracking_config()
    )

    source_run = client.get_run(args.run_id)

    if (
        source_run.info.status != "FINISHED"
        or source_run.info.experiment_id != experiment_id
    ):
        raise ValueError(
            "Source run must be finished and in this experiment"
        )

    params = source_run.data.params

    if params.get("forecast_strategy") != "recursive":
        raise ValueError(
            "Source run must contain recursive forecasts"
        )

    origin = int(params["forecast_origin_day"])
    horizon = int(params["forecast_horizon_days"])

    if not (
        origin >= split.train_end
        and origin + horizon <= split.valid_end
        and horizon == split.horizon
    ):
        raise ValueError(
            "This command only scores validation windows"
        )

    forecast_path = client.download_artifacts(
        args.run_id,
        "forecasts/lightgbm_valid_first_window.csv",
    )
    forecasts = pd.read_csv(forecast_path)

    wide = pd.read_parquet(WIDE_PATH).set_index("id")
    closed = closed_days(load_day_dates(), split)

    scores = score_window(
        wide, forecasts, origin, horizon, closed
    )

    n_defined = int(scores["rmsse"].notna().sum())

    if n_defined == 0:
        raise ValueError(
            "RMSSE is undefined for every series"
        )

    summary = {
        f"valid_window_{metric}_{stat}": float(
            getattr(scores[metric], stat)()
        )
        for metric in METRICS
        for stat in ("mean", "median")
    }
    summary["valid_window_rmsse_defined_series"] = n_defined

    output_dir = (
        REPO_ROOT / "reports" / "lightgbm" / args.run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "per_series_window.csv"
    scores.rename_axis("id").reset_index().to_csv(
        output_path,
        index=False,
    )

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="lightgbm-score-first-window",
        tags={
            "stage": "validation_window_scoring",
            "source_forecast_run_id": args.run_id,
            "model_trained": "false",
            "test_evaluated": "false",
        },
    ) as run:
        mlflow.log_params({
            "forecast_origin_day": origin,
            "forecast_horizon_days": horizon,
            "forecast_strategy": "recursive",
        })
        mlflow.log_metrics(summary)
        mlflow.log_artifact(
            str(output_path),
            artifact_path="scores",
        )
        mlflow.log_artifact(
            __file__,
            artifact_path="code",
        )
        mlflow.log_artifact(
            str(REPO_ROOT / "configs" / "split.toml"),
            artifact_path="configs",
        )
        scoring_run_id = run.info.run_id

    print("Source forecast run:", args.run_id)
    print(f"Window: d_{origin + 1}..d_{origin + horizon}")
    print("Series:", len(scores))
    print("RMSSE defined series:", n_defined)

    for metric in METRICS:
        print(
            f"{metric}: "
            f"mean={scores[metric].mean():.6f}; "
            f"median={scores[metric].median():.6f}"
        )

    print("Scoring run ID:", scoring_run_id)
    print(
        "Status:",
        client.get_run(scoring_run_id).info.status,
    )
    print("Saved scores:", output_path)


if __name__ == "__main__":
    main()