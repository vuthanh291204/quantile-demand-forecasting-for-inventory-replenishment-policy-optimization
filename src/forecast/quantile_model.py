"""Train four direct quantile models and forecast the first validation window.

Run: python -m src.forecast.quantile_model
This command does not score validation labels, select hyperparameters or use test.
"""
from __future__ import annotations

import argparse
import platform
import tomllib
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from src.data.direct_features import (
    FORECAST_KEYS,
    build_direct_forecast_table,
    build_direct_training_table,
    direct_feature_columns,
)
from src.data.features import CATEGORICAL_FEATURES, load_feature_config
from src.data.split import LONG_PATH, REPO_ROOT, load_config
from src.tracking import configure_tracking, load_tracking_config

DEFAULT_CONFIG = REPO_ROOT / "configs" / "quantiles.toml"


@dataclass(frozen=True)
class QuantileConfig:
    quantiles: tuple[float, ...]
    origin_stride: int
    model_params: dict
    experiment_name: str

    def __post_init__(self):
        if not self.quantiles or any(
            not np.isfinite(q) or not 0 < q < 1 for q in self.quantiles
        ):
            raise ValueError("Quantiles must be finite and strictly between zero and one")
        if list(self.quantiles) != sorted(set(self.quantiles)):
            raise ValueError("Quantiles must be unique and increasing")
        if len({quantile_name(q) for q in self.quantiles}) != len(self.quantiles):
            raise ValueError("Quantiles must have distinct output names")
        if type(self.origin_stride) is not int or self.origin_stride < 1:
            raise ValueError("origin_stride must be a positive integer")
        if self.model_params.get("objective") != "quantile" or "alpha" in self.model_params:
            raise ValueError("Use objective=quantile and supply alpha through quantiles only")
        if not self.experiment_name.strip():
            raise ValueError("Need a non-empty experiment name")


def load_quantile_config(path=DEFAULT_CONFIG):
    with open(path, "rb") as file:
        raw = tomllib.load(file)
    return QuantileConfig(
        quantiles=tuple(raw["forecast"]["quantiles"]),
        origin_stride=raw["forecast"]["origin_stride"],
        model_params=raw["model"],
        experiment_name=raw["mlflow"]["experiment_name"],
    )


def quantile_name(alpha):
    return f"q{alpha * 100:02g}".replace(".", "p")


def model_inputs(table, feature_cfg, category_levels=None):
    X = table[direct_feature_columns(feature_cfg)].copy()
    if category_levels is None:
        category_levels = {
            col: sorted(X[col].dropna().unique().tolist()) for col in CATEGORICAL_FEATURES
        }
    if set(category_levels) != set(CATEGORICAL_FEATURES):
        raise ValueError("Category levels must describe every categorical feature")
    for col, levels in category_levels.items():
        X[col] = pd.Categorical(X[col], categories=levels)
    return X, category_levels


def fit_quantiles(train, cfg, feature_cfg, on_fitted=None):
    X, levels = model_inputs(train, feature_cfg)
    y = train["sales"].astype("float32")
    if len(X) == 0 or not np.isfinite(y).all():
        raise ValueError("Need non-empty training data with finite labels")
    models = {}
    for alpha in cfg.quantiles:
        name = quantile_name(alpha)
        print(f"Training {name}: alpha={alpha}; rows={len(X)}", flush=True)
        model = lgb.LGBMRegressor(**cfg.model_params, alpha=alpha)
        model.fit(X, y, categorical_feature=list(CATEGORICAL_FEATURES))
        models[name] = model
        if on_fitted is not None:
            on_fitted(name, model)
    return models, levels


def predict_quantiles(models, table, feature_cfg, category_levels):
    X, _ = model_inputs(table, feature_cfg, category_levels)
    if table.duplicated(["id", "origin_day", "horizon"]).any():
        raise ValueError("Forecast keys must be unique")
    out = table[FORECAST_KEYS].copy()
    for name, model in models.items():
        raw = np.asarray(model.predict(X, validate_features=True), dtype=float)
        if raw.shape != (len(table),) or not np.isfinite(raw).all():
            raise ValueError(f"Invalid predictions from {name}")
        out[f"{name}_raw"] = raw
        out[name] = np.maximum(raw, 0.0)
    # Keep crossing visible. No sorting or rounding of the quantile outputs.
    return out


def run_first_window(long, split, cfg, feature_cfg, tracking_cfg, output_root):
    origin = split.train_end
    train = build_direct_training_table(
        long, origin, split.horizon, cfg.origin_stride, feature_cfg
    )
    future = build_direct_forecast_table(long, origin, split.horizon, feature_cfg)
    print(f"Train targets end: d_{train['day'].max()}", flush=True)
    print(f"Train rows: {len(train)}; origins: {train['origin_day'].nunique()}", flush=True)
    print(f"Feature count: {len(direct_feature_columns(feature_cfg))}", flush=True)
    client, experiment_id = configure_tracking(tracking_cfg)
    with mlflow.start_run(
        experiment_id=experiment_id, run_name="quantiles-direct-first-window",
        tags={
            "stage": "quantile_first_window_training",
            "forecast_strategy": "direct_pooled_horizon",
            "model_trained": "true", "early_stopping_used": "false",
            "validation_scored": "false", "test_evaluated": "false",
        },
    ) as run:
        mlflow.log_params({
            **cfg.model_params, "quantiles": str(cfg.quantiles),
            "origin_stride": cfg.origin_stride, "forecast_origin_day": origin,
            "horizon_days": split.horizon,
            "price_policy": "observed_at_origin",
            "postprocessing": "clip_negative_to_zero_no_sorting_no_rounding",
        })
        mlflow.log_dict({
            "quantiles": asdict(cfg), "split": asdict(split),
            "features": asdict(feature_cfg),
        }, "effective_config.json")
        mlflow.log_dict({
            "python": platform.python_version(),
            "libraries": {n: version(n) for n in ("lightgbm", "pandas", "numpy", "mlflow")},
        }, "environment.json")
        for relative in (
            "configs/quantiles.toml", "configs/features.toml", "configs/split.toml",
            "configs/tracking.toml", "configs/subset_ids.csv",
            "src/data/direct_features.py", "src/data/features.py", "src/data/split.py",
            "src/forecast/quantile_model.py", "src/tracking.py",
        ):
            mlflow.log_artifact(str(REPO_ROOT / relative), artifact_path="snapshot")

        def save_model(name, model):
            mlflow.log_text(model.booster_.model_to_string(), f"models/{name}.txt")
            mlflow.log_metric(f"{name}_iterations", model.booster_.current_iteration())
            print(f"Fitted {name}: iterations={model.booster_.current_iteration()}", flush=True)

        models, levels = fit_quantiles(train, cfg, feature_cfg, on_fitted=save_model)
        mlflow.log_dict(levels, "models/category_levels.json")
        mlflow.log_dict({"feature_columns": direct_feature_columns(feature_cfg)},
                        "models/feature_schema.json")
        forecasts = predict_quantiles(models, future, feature_cfg, levels)
        output = Path(output_root) / run.info.run_id / "forecasts_valid_first_window.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        forecasts.to_csv(output, index=False)
        mlflow.log_artifact(str(output), artifact_path="forecasts")
        mlflow.log_metrics({
            "train_rows": len(train), "train_origins": train["origin_day"].nunique(),
            "n_features": len(direct_feature_columns(feature_cfg)),
            "forecast_rows": len(forecasts), "forecast_series": forecasts["id"].nunique(),
        })
        run_id = run.info.run_id
    print("Experiment:", tracking_cfg.experiment_name)
    print("Run ID:", run_id)
    print("Status:", client.get_run(run_id).info.status)
    print("Forecast shape:", forecasts.shape)
    print("Forecast series:", forecasts["id"].nunique())
    print("Forecast days:", forecasts["day"].min(), forecasts["day"].max())
    print("Quantiles:", ", ".join(models))
    print("Saved forecasts:", output)
    print("Validation scored: false; test evaluated: false")
    return run_id, output


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    cfg = load_quantile_config()
    tracking = replace(load_tracking_config(), experiment_name=cfg.experiment_name)
    run_first_window(
        pd.read_parquet(LONG_PATH), load_config(), cfg, load_feature_config(),
        tracking, REPO_ROOT / "reports" / "quantiles",
    )


if __name__ == "__main__":
    main()
