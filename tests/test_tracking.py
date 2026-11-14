import json
from pathlib import Path

import mlflow
import pytest

from src.tracking import (
    AUDIT_PATH,
    TrackingConfig,
    configure_tracking,
    load_tracking_config,
    record_audit_run,
)


@pytest.fixture
def store(tmp_path):
    previous_uri = mlflow.get_tracking_uri()
    yield TrackingConfig("test-demand-forecasting", tmp_path / "mlruns")
    mlflow.set_tracking_uri(previous_uri)


def test_real_tracking_settings_are_local_and_inside_ignored_directory():
    cfg = load_tracking_config()
    assert cfg.storage_dir.name == "mlruns"
    assert cfg.tracking_uri.startswith("sqlite:///")
    assert cfg.artifact_uri.startswith("file:///")


def test_audit_run_persists_metrics_params_artifacts_and_separate_runs(store, tmp_path):
    first = record_audit_run(store, run_name="first-check")
    second = record_audit_run(store, run_name="second-check")
    client, experiment_id = configure_tracking(store)
    saved = client.get_run(first.info.run_id)
    assert saved.info.status == "FINISHED"
    assert first.info.run_id != second.info.run_id
    assert first.info.experiment_id == second.info.experiment_id == experiment_id
    report = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    assert saved.data.metrics["audit_train_rows"] == report["training"]["n_rows"]
    assert saved.data.metrics["audit_n_series"] == report["training"]["n_series"]
    assert saved.data.metrics["audit_validation_windows"] == len(report["validation_windows"])
    assert saved.data.params["price_policy"] == "last_observed_at_origin"
    assert saved.data.tags["model_trained"] == "false"
    assert not {"mae", "rmse", "rmsse", "smape"} & saved.data.metrics.keys()
    assert {item.path for item in client.list_artifacts(first.info.run_id, "configs")} == {
        "configs/tracking.toml", "configs/features.toml", "configs/split.toml", "configs/subset_ids.csv",
    }
    downloaded = client.download_artifacts(
        first.info.run_id, "audit/leakage_checks.json", dst_path=str(tmp_path)
    )
    assert json.loads(Path(downloaded).read_text(encoding="utf-8")) == report


@pytest.mark.parametrize("problem", ["failed", "features", "window", "check"])
def test_stale_or_incomplete_audit_is_rejected_before_creating_a_store(store, tmp_path, problem):
    report = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if problem == "failed":
        report["status"] = "failed"
    elif problem == "features":
        report["feature_config"]["lags"] = [1]
    elif problem == "window":
        report["validation_windows"].pop()
    else:
        report["validation_windows"][0]["future_sales_erased"] = False
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="Run python -m src.evaluation.check_leakage first"):
        record_audit_run(store, report_path=path)
    assert not store.storage_dir.exists()
