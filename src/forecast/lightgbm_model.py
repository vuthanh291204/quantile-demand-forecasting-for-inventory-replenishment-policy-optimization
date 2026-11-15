import tomllib

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from src.data.features import (
    CATEGORICAL_FEATURES,
    build_features,
    build_training_table,
    load_feature_config,
    prepare_forecast_frame,
)
from src.data.split import LONG_PATH, REPO_ROOT, load_config
from src.tracking import configure_tracking, load_tracking_config


def prepare_training_data(end_day=None):
    split = load_config()
    feature_cfg = load_feature_config()

    long = pd.read_parquet(LONG_PATH)
    cutoff = split.train_end if end_day is None else end_day

    train = build_training_table(
        long,
        end_day=cutoff,
        cfg=feature_cfg,
    )

    X = train[feature_cfg.feature_columns].copy()
    y = train["sales"].astype("float32")

    category_levels = {}

    for column in CATEGORICAL_FEATURES:
        levels = sorted(X[column].dropna().unique().tolist())
        category_levels[column] = levels

        X[column] = pd.Categorical(
            X[column],
            categories=levels,
        )

    return X, y, category_levels


def load_model_params():
    config_path = REPO_ROOT / "configs" / "lightgbm.toml"

    with open(config_path, "rb") as file:
        config = tomllib.load(file)

    return config["model"]


def recursive_forecast(
    model,
    long,
    origin_day,
    horizon,
    feature_cfg,
    category_levels,
    verbose=False,
):
    frame = prepare_forecast_frame(
        long,
        origin_day=origin_day,
        horizon=horizon,
    )

    series_ids = pd.Index(
        sorted(frame["id"].unique()),
        name="id",
    )
    results = []

    forecast_days = range(
        origin_day + 1,
        origin_day + horizon + 1,
    )

    for step, day in enumerate(forecast_days, start=1):
        features = build_features(frame, feature_cfg)

        daily = (
            features.loc[features["day"].eq(day)]
            .set_index("id")
            .sort_index()
        )

        if (
            not daily.index.equals(series_ids)
            or not daily["features_ready"].all()
        ):
            raise ValueError(
                f"Missing or incomplete features for d_{day}"
            )

        X_day = daily[feature_cfg.feature_columns].copy()

        for column, levels in category_levels.items():
            X_day[column] = pd.Categorical(
                X_day[column],
                categories=levels,
            )

        raw_predictions = np.asarray(
            model.predict(X_day, validate_features=True),
            dtype=float,
        )

        if (
            raw_predictions.shape != (len(series_ids),)
            or not np.isfinite(raw_predictions).all()
        ):
            raise ValueError(f"Invalid predictions for d_{day}")

        predictions = pd.Series(
            np.maximum(raw_predictions, 0.0),
            index=daily.index,
        )

        target_rows = frame["d"].eq(f"d_{day}")
        frame.loc[target_rows, "sales"] = (
            frame.loc[target_rows, "id"].map(predictions)
        )

        daily_result = daily[["day", "date"]].copy()
        daily_result["y_pred"] = predictions
        results.append(daily_result.reset_index())

        if verbose and (step == 1 or step % 7 == 0):
            print(
                f"Forecast {step}/{horizon}: d_{day}",
                flush=True,
            )

    return (
        pd.concat(results, ignore_index=True)
        .sort_values(["id", "day"])
        .reset_index(drop=True)
    )


def main():
    X, y, category_levels = prepare_training_data()
    params = load_model_params()

    model = lgb.LGBMRegressor(**params)

    client, experiment_id = configure_tracking(
        load_tracking_config()
    )

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="lightgbm-first-window-check",
        tags={
            "stage": "recursive_forecast_check",
            "model_trained": "true",
            "early_stopping_used": "false",
            "test_evaluated": "false",
        },
    ) as run:
        mlflow.log_params(params)

        for filename in (
            "lightgbm.toml",
            "features.toml",
            "split.toml",
        ):
            mlflow.log_artifact(
                str(REPO_ROOT / "configs" / filename),
                artifact_path="configs",
            )

        mlflow.log_artifact(__file__, artifact_path="code")

        print("Training started...", flush=True)

        model.fit(
            X,
            y,
            categorical_feature=list(CATEGORICAL_FEATURES),
        )

        mlflow.log_metrics({
            "train_rows": len(X),
            "n_features": X.shape[1],
            "training_iterations": model.n_estimators_,
        })

        mlflow.log_text(
            model.booster_.model_to_string(),
            "model/lightgbm.txt",
        )

        mlflow.log_dict(
            category_levels,
            "model/category_levels.json",
        )

        mlflow.log_dict(
            {"feature_columns": X.columns.tolist()},
            "model/feature_schema.json",
        )

        split = load_config()
        feature_cfg = load_feature_config()
        long = pd.read_parquet(LONG_PATH)

        forecasts = recursive_forecast(
            model=model,
            long=long,
            origin_day=split.train_end,
            horizon=split.horizon,
            feature_cfg=feature_cfg,
            category_levels=category_levels,
            verbose=True,
        )

        output_path = (
            REPO_ROOT
            / "data"
            / "processed"
            / "lightgbm_valid_first_window.csv"
        )
        forecasts.to_csv(output_path, index=False)

        mlflow.log_params({
            "forecast_origin_day": split.train_end,
            "forecast_horizon_days": split.horizon,
            "forecast_strategy": "recursive",
        })
        mlflow.log_metric("forecast_rows", len(forecasts))
        mlflow.log_artifact(
            str(output_path),
            artifact_path="forecasts",
        )

        print("Forecast shape:", forecasts.shape)
        print("Forecast series:", forecasts["id"].nunique())
        print(
            "Forecast days:",
            forecasts["day"].min(),
            forecasts["day"].max(),
        )

        run_id = run.info.run_id

    saved_run = client.get_run(run_id)

    print("Run ID:", run_id)
    print("Status:", saved_run.info.status)
    print("Feature count:", model.n_features_in_)
    print("Fitted iterations:", model.n_estimators_)


if __name__ == "__main__":
    main()