"""Local MLflow tracking and a preflight run using the saved feature audit.

Run: python -m src.tracking
View from the repo root:
    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --host 127.0.0.1 --port 5000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import tomllib
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.data.features import load_feature_config
from src.data.split import REPO_ROOT, load_config, make_windows

DEFAULT_CONFIG = REPO_ROOT / "configs" / "tracking.toml"
AUDIT_PATH = REPO_ROOT / "reports" / "leakage_checks.json"


@dataclass(frozen=True)
class TrackingConfig:
    experiment_name: str
    storage_dir: Path

    @property
    def tracking_uri(self) -> str:
        return f"sqlite:///{(self.storage_dir.resolve() / 'mlflow.db').as_posix()}"

    @property
    def artifact_uri(self) -> str:
        return (self.storage_dir.resolve() / "artifacts").as_uri()


def load_tracking_config(path: Path = DEFAULT_CONFIG) -> TrackingConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)["mlflow"]
    directory = (REPO_ROOT / raw["storage_dir"]).resolve()
    if not directory.is_relative_to(REPO_ROOT) or directory == REPO_ROOT:
        raise ValueError("MLflow storage_dir must be a subdirectory of the repo")
    if not raw["experiment_name"].strip():
        raise ValueError("MLflow experiment_name must not be empty")
    return TrackingConfig(raw["experiment_name"], directory)


def configure_tracking(cfg: TrackingConfig) -> tuple[MlflowClient, str]:
    """Configure the local backend; return a client and a reusable experiment id."""
    cfg.storage_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(cfg.tracking_uri)
    client = MlflowClient(tracking_uri=cfg.tracking_uri)
    experiment = client.get_experiment_by_name(cfg.experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            cfg.experiment_name, artifact_location=cfg.artifact_uri
        )
    else:
        if experiment.lifecycle_stage != "active":
            raise ValueError("The configured MLflow experiment has been deleted")
        if experiment.artifact_location != cfg.artifact_uri:
            raise ValueError("Existing experiment uses a different artifact location")
        experiment_id = experiment.experiment_id
    return client, experiment_id


def _read_matching_audit(path: Path) -> dict:
    """Reject an incomplete audit or one with different split/feature settings."""
    report = json.loads(path.read_text(encoding="utf-8"))
    split, features = load_config(), load_feature_config()
    expected_features = json.loads(json.dumps(asdict(features)))
    train = report.get("training", {})
    windows = report.get("validation_windows", [])
    expected_windows = make_windows(*split.ranges["valid"], split.horizon)
    valid = (
        report.get("status") == "passed"
        and report.get("feature_config") == expected_features
        and train.get("cutoff_day") == split.train_end
        and train.get("n_features") == len(features.feature_columns)
        and train.get("n_rows", 0) > 0
        and train.get("n_series", 0) > 0
        and train.get("held_out_mutation_invariant") is True
        and train.get("forbidden_feature_columns_absent") is True
        and len(windows) == len(expected_windows)
    )
    checks = (
        "future_sales_erased", "future_price_frozen", "held_out_mutation_invariant",
        "only_first_future_day_ready_initially", "all_step_lags_and_means_match_past_only_oracle",
    )
    if valid:
        for window, (first, last) in zip(windows, expected_windows):
            valid = valid and (
                (window.get("origin_day"), window.get("first_day"), window.get("last_day"))
                == (first - 1, first, last)
                and window.get("n_series") == train["n_series"]
                and window.get("synthetic_rows_checked") == train["n_series"] * split.horizon
                and all(window.get(key) is True for key in checks)
            )
    if not valid:
        raise ValueError("Audit is incomplete or settings changed. Run python -m src.evaluation.check_leakage first.")
    return report


def _git_tags() -> dict[str, str]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()

    try:
        return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": str(bool(git("status", "--porcelain"))).lower()}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "git_dirty": "unknown"}


def record_audit_run(
    cfg: TrackingConfig, report_path: Path = AUDIT_PATH, run_name: str = "tracking-preflight"
):
    """Record measured audit counts, configs and code; no model accuracy is logged.

    This checks report structure/settings; it does not rerun the leakage audit
    or prove the report is current after a change to source data or code.
    """
    report = _read_matching_audit(report_path)
    split, features = load_config(), load_feature_config()
    metrics = {
        "audit_train_rows": report["training"]["n_rows"],
        "audit_n_series": report["training"]["n_series"],
        "audit_n_features": report["training"]["n_features"],
        "audit_validation_windows": len(report["validation_windows"]),
        "audit_synthetic_rows_checked": sum(w["synthetic_rows_checked"] for w in report["validation_windows"]),
    }
    client, experiment_id = configure_tracking(cfg)
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, tags={
        "stage": "tracking_preflight", "model_trained": "false", "test_evaluated": "false",
        **_git_tags(),
    }) as active:
        mlflow.log_params({
            "train_end_day": split.train_end, "valid_end_day": split.valid_end,
            "horizon_days": split.horizon, "lags": json.dumps(features.lags),
            "rolling_windows": json.dumps(features.rolling_windows),
            "price_policy": "last_observed_at_origin", "calendar_known_ahead": True,
        })
        mlflow.log_metrics(metrics)
        mlflow.log_dict({
            "experiment_name": cfg.experiment_name,
            "tracking_uri": cfg.tracking_uri,
            "artifact_uri": cfg.artifact_uri,
        }, "tracking_storage.json")
        mlflow.log_artifact(str(report_path), artifact_path="audit")
        snapshots = {
            "configs": ["configs/tracking.toml", "configs/features.toml", "configs/split.toml", "configs/subset_ids.csv"],
            "code": ["src/tracking.py", "src/data/features.py", "src/evaluation/check_leakage.py"],
        }
        hashes = {}
        for folder, paths in snapshots.items():
            for relative in paths:
                path = REPO_ROOT / relative
                mlflow.log_artifact(str(path), artifact_path=folder)
                hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes["audit_report"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
        mlflow.log_dict(hashes, "snapshot_sha256.json")
        mlflow.log_dict({
            "python": platform.python_version(),
            "libraries": {lib: version(lib) for lib in ("mlflow", "lightgbm", "pandas", "numpy")},
        }, "environment.json")
        run_id = active.info.run_id
    saved = client.get_run(run_id)
    if saved.info.status != "FINISHED" or saved.data.metrics != metrics:
        raise RuntimeError("MLflow run did not persist all audit metrics successfully")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="tracking-preflight")
    args = parser.parse_args()
    cfg = load_tracking_config()
    saved = record_audit_run(cfg, run_name=args.run_name)
    print(f"Tracking URI: {cfg.tracking_uri}")
    print(f"Experiment: {cfg.experiment_name}")
    print(f"Run ID: {saved.info.run_id}")
    print(f"Status: {saved.info.status}")
    for key, value in sorted(saved.data.metrics.items()):
        print(f"{key}: {value:g}")
    print("No model trained; no forecast accuracy metrics logged.")
    print("View from repo root: mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --host 127.0.0.1 --port 5000")


if __name__ == "__main__":
    main()
