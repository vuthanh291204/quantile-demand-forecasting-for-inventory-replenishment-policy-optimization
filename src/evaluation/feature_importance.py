"""Summarize training gain/split importance from saved validation models.

Run: python -m src.evaluation.feature_importance --run-id <comparison-run>
No fitting, prediction or test evaluation is performed.
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import matplotlib
import mlflow
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.split import REPO_ROOT
from src.tracking import configure_tracking, load_tracking_config


def load_saved_booster(path):
    # MLflow log_text on Windows may write CRLF, changing tree_sizes offsets.
    # Universal-newline reading restores LF before native string parsing.
    return lgb.Booster(model_str=Path(path).read_text(encoding="utf-8"))


def summarize_importance(raw):
    if raw.empty or raw.duplicated(["window", "feature"]).any():
        raise ValueError("Need unique window/feature pairs")
    if not np.isfinite(raw[["gain", "split"]].to_numpy()).all():
        raise ValueError("Importance must be finite")
    if (raw[["gain", "split"]] < 0).any().any():
        raise ValueError("Importance must be non-negative")
    expected = pd.MultiIndex.from_product(
        [raw["window"].unique(), raw["feature"].unique()],
        names=["window", "feature"],
    )
    actual = pd.MultiIndex.from_frame(raw[["window", "feature"]])
    if not actual.sort_values().equals(expected.sort_values()):
        raise ValueError("Each window must contain the same features")
    totals = raw.groupby("window")["gain"].transform("sum")
    if totals.le(0).any():
        raise ValueError("Each model needs positive total gain")
    normalized = raw.assign(gain_pct=100 * raw["gain"] / totals)
    summary = normalized.groupby("feature").agg(
        gain_pct_mean=("gain_pct", "mean"),
        gain_pct_std=("gain_pct", "std"),
        gain_pct_min=("gain_pct", "min"),
        gain_pct_max=("gain_pct", "max"),
        split_mean=("split", "mean"),
    ).sort_values("gain_pct_mean", ascending=False)
    return normalized, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    client, experiment_id = configure_tracking(load_tracking_config())
    source = client.get_run(args.run_id)
    if (source.info.status != "FINISHED"
            or source.info.experiment_id != experiment_id
            or source.data.tags.get("stage") != "validation_baseline_comparison"):
        raise ValueError("Need a finished validation comparison run")
    n_windows = int(source.data.params["n_windows"])
    parts, expected_features = [], None
    for window in range(1, n_windows + 1):
        path = client.download_artifacts(
            args.run_id, f"models/window_{window}/lightgbm.txt"
        )
        model = load_saved_booster(path)
        names = model.feature_name()
        if expected_features is not None and names != expected_features:
            raise ValueError("Model feature schemas differ")
        expected_features = names
        parts.append(pd.DataFrame({
            "window": window, "feature": names,
            "gain": model.feature_importance("gain", iteration=-1),
            "split": model.feature_importance("split", iteration=-1),
        }))
    raw, summary = summarize_importance(pd.concat(parts, ignore_index=True))
    output = REPO_ROOT / "reports" / "lightgbm" / args.run_id / "importance"
    output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output / "per_window.csv", index=False)
    summary.rename_axis("feature").reset_index().to_csv(output / "summary.csv", index=False)
    top = summary.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index, top["gain_pct_mean"], color="#287D8E")
    ax.set_xlabel("Mean normalized training gain (%) across saved models")
    ax.set_title(f"LightGBM feature importance — {n_windows} validation-window models")
    ax.text(0, -0.16, "Descriptive training importance; not causal or validation performance gain.",
            transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "gain_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    with mlflow.start_run(experiment_id=experiment_id, run_name="lightgbm-feature-importance",
                          tags={"stage": "feature_importance", "source_comparison_run_id": args.run_id,
                                "model_trained": "false", "test_evaluated": "false"}) as run:
        mlflow.log_params({"n_windows": n_windows, "n_features": len(summary),
                           "aggregation": "normalize gain per model then average equally"})
        for path in output.iterdir():
            mlflow.log_artifact(str(path), artifact_path="importance")
        mlflow.log_artifact(__file__, artifact_path="code")
        importance_run_id = run.info.run_id
    print(summary.head(10).round(4).to_string())
    print("Importance run ID:", importance_run_id)
    print("Saved to:", output)


if __name__ == "__main__":
    main()
