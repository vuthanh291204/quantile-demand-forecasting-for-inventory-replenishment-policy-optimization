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
)
from src.evaluation.score_lightgbm import score_window
from src.forecast.lightgbm_model import (
    load_model_params,
    prepare_training_data,
    recursive_forecast,
)
from src.tracking import configure_tracking, load_tracking_config


def fit_by_checkpoints(X, y, params, settings, evaluate):
    common_params = params.copy()
    max_rounds = common_params.pop("n_estimators")
    check_every = settings["check_every"]
    patience = settings["patience_checks"]

    if min(max_rounds, check_every, patience) < 1:
        raise ValueError("Selection settings must be positive")

    previous = None
    best_model = None
    best_score = float("inf")
    rounds = 0
    bad_checks = 0
    history = []
    stop_reason = "max_rounds"

    while rounds < max_rounds:
        additional = min(check_every, max_rounds - rounds)

        print(
            f"Training toward {rounds + additional} trees...",
            flush=True,
        )

        candidate = lgb.LGBMRegressor(
            n_estimators=additional,
            **common_params,
        )
        candidate.fit(
            X,
            y,
            categorical_feature="auto",
            init_model=previous,
        )

        current = candidate.booster_.current_iteration()

        if current <= rounds:
            stop_reason = "no_new_trees"
            break

        rounds = current
        previous = candidate.booster_
        score = float(evaluate(candidate))

        if not np.isfinite(score) or score < 0:
            raise ValueError(
                "RMSSE must be finite and non-negative"
            )

        if score < best_score:
            best_score = score
            best_model = candidate
            bad_checks = 0
        else:
            bad_checks += 1

        history.append({
            "iteration": rounds,
            "rmsse": score,
            "best_iteration": (
                best_model.booster_.current_iteration()
            ),
            "checks_without_improvement": bad_checks,
        })

        if bad_checks >= patience:
            stop_reason = "patience"
            break

    if best_model is None:
        raise RuntimeError("No usable checkpoint was produced")

    return best_model, pd.DataFrame(history), stop_reason


def main():
    split = load_config()
    feature_cfg = load_feature_config()
    params = load_model_params()

    with open(
        REPO_ROOT / "configs" / "lightgbm.toml",
        "rb",
    ) as file:
        settings = tomllib.load(file)["selection"]

    X, y, category_levels = prepare_training_data()
    long = pd.read_parquet(LONG_PATH)
    wide = pd.read_parquet(WIDE_PATH).set_index("id")
    closed = closed_days(load_day_dates(), split)

    client, experiment_id = configure_tracking(
        load_tracking_config()
    )
    evaluations = {}

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="lightgbm-round-selection",
        tags={
            "stage": "validation_round_selection",
            "early_stopping_method": "checkpoint_patience",
            "model_trained": "true",
            "test_evaluated": "false",
        },
    ) as run:
        mlflow.log_params(params)
        mlflow.log_params({
            "check_every": settings["check_every"],
            "patience_checks": settings["patience_checks"],
            "forecast_origin_day": split.train_end,
            "forecast_horizon_days": split.horizon,
            "forecast_strategy": "recursive",
        })

        for filename in (
            "lightgbm.toml",
            "features.toml",
            "split.toml",
        ):
            mlflow.log_artifact(
                str(REPO_ROOT / "configs" / filename),
                artifact_path="configs",
            )

        for relative in (
            "src/forecast/select_lightgbm.py",
            "src/forecast/lightgbm_model.py",
            "src/evaluation/score_lightgbm.py",
        ):
            mlflow.log_artifact(
                str(REPO_ROOT / relative),
                artifact_path="code",
            )

        def evaluate(candidate):
            iteration = (
                candidate.booster_.current_iteration()
            )
            forecasts = recursive_forecast(
                model=candidate,
                long=long,
                origin_day=split.train_end,
                horizon=split.horizon,
                feature_cfg=feature_cfg,
                category_levels=category_levels,
                verbose=True,
            )
            scores = score_window(
                wide,
                forecasts,
                split.train_end,
                split.horizon,
                closed,
            )

            if not np.isfinite(scores["rmsse"]).all():
                raise ValueError(
                    "RMSSE must be defined for every series"
                )

            value = float(scores["rmsse"].mean())
            evaluations[iteration] = (forecasts, scores)

            mlflow.log_metric(
                "selection_window_rmsse",
                value,
                step=iteration,
            )
            print(
                f"Checkpoint {iteration}: RMSSE={value:.6f}",
                flush=True,
            )
            return value

        best_model, history, stop_reason = (
            fit_by_checkpoints(
                X, y, params, settings, evaluate
            )
        )

        best_iteration = (
            best_model.booster_.current_iteration()
        )
        best_forecasts, best_scores = evaluations[
            best_iteration
        ]

        mlflow.log_metrics({
            "best_iteration": best_iteration,
            "stopped_iteration": int(
                history["iteration"].iloc[-1]
            ),
            "best_selection_rmsse": float(
                best_scores["rmsse"].mean()
            ),
            "selection_checks": len(history),
        })
        mlflow.set_tag("stop_reason", stop_reason)

        mlflow.log_text(
            best_model.booster_.model_to_string(),
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

        output_dir = (
            REPO_ROOT / "reports" / "lightgbm" / run.info.run_id
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = [
            ("selection_history.csv", history, "selection"),
            (
                "lightgbm_valid_first_window.csv",
                best_forecasts,
                "forecasts",
            ),
            (
                "per_series_window.csv",
                best_scores.rename_axis("id").reset_index(),
                "scores",
            ),
        ]

        for filename, frame, artifact_dir in outputs:
            path = output_dir / filename
            frame.to_csv(path, index=False)
            mlflow.log_artifact(
                str(path),
                artifact_path=artifact_dir,
            )

        run_id = run.info.run_id

    print("Run ID:", run_id)
    print("Status:", client.get_run(run_id).info.status)
    print("Selected iteration:", best_iteration)
    print("Stop reason:", stop_reason)
    print(
        "Selected-window RMSSE:",
        float(best_scores["rmsse"].mean()),
    )


if __name__ == "__main__":
    main()