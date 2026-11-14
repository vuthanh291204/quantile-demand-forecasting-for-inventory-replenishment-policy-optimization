"""Audit feature timing on validation without fitting a model or scoring test.

Run: python -m src.evaluation.check_leakage

Synthetic forecast values are used only to check lag/rolling calculations.
They are not model predictions, accuracy results or observations from M5.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.data.features import (
    FeatureConfig,
    build_features,
    build_training_table,
    load_feature_config,
    prepare_forecast_frame,
)
from src.data.split import LONG_PATH, REPO_ROOT, load_config, make_windows

FORBIDDEN_FEATURES = {
    "sales", "sell_price", "id", "d", "day", "date", "features_ready",
    "live", "first_date", "volume_group", "mean_live", "mean_daily",
    "days_live", "zero_share_live", "rel", "rel_year", "yr_mean",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _days(long: pd.DataFrame) -> pd.Series:
    return long["d"].str.slice(2).astype(int)


def _poison_after(long: pd.DataFrame, origin: int) -> pd.DataFrame:
    """Change withheld outcomes/prices while leaving assumed-known inputs alone."""
    changed = long.copy()
    future = _days(changed) > origin
    changed.loc[future, "sales"] = 999999
    changed.loc[future, "sell_price"] = 999999.0
    return changed


def audit_training(long: pd.DataFrame, end_day: int, cfg: FeatureConfig) -> dict:
    """Validation sales/prices must not change training rows, labels or inputs."""
    _require(not FORBIDDEN_FEATURES.intersection(cfg.feature_columns),
             "A label, metadata column or full-period statistic is in feature_columns")
    _require((_days(long) > end_day).any(), "Need held-out rows to test cutoff invariance")
    train = build_training_table(long, end_day, cfg)
    changed = build_training_table(_poison_after(long, end_day), end_day, cfg)
    assert_frame_equal(train, changed)
    _require(not train.empty, "Training table is empty")
    _require(train["day"].max() <= end_day, "A training label is after the fitting cutoff")
    _require(not train[cfg.feature_columns].isna().any().any(), "Missing training features")
    return {
        "cutoff_day": end_day,
        "n_rows": len(train),
        "n_series": train["id"].nunique(),
        "n_features": len(cfg.feature_columns),
        "held_out_mutation_invariant": True,
        "forbidden_feature_columns_absent": True,
    }


def audit_window(
    long: pd.DataFrame, origin_day: int, horizon: int, cfg: FeatureConfig
) -> dict:
    """Check every series/day in one window against an independent array oracle.

    Fill all future labels with a predetermined synthetic sequence. Since the
    oracle only slices values BEFORE each target day, this also tests that a
    synthetic value at or after that day cannot enter its lag or rolling mean.
    This is a feature audit, not the model's recursive prediction loop.
    """
    frame = prepare_forecast_frame(long, origin_day, horizon)
    changed = prepare_forecast_frame(_poison_after(long, origin_day), origin_day, horizon)
    assert_frame_equal(frame, changed)
    days = _days(frame)
    future = days > origin_day
    _require(frame.loc[future, "sales"].isna().all(), "Future observed sales were retained")
    past = frame.loc[~future]
    series = pd.Index(sorted(past["id"].unique()), name="id")
    last_prices = past.groupby("id")["sell_price"].last()
    expected_prices = frame.loc[future, "id"].map(last_prices)
    np.testing.assert_allclose(frame.loc[future, "sell_price"], expected_prices, rtol=0, atol=0)

    features = build_features(frame, cfg)
    ready = features.loc[(features["day"] > origin_day) & features["features_ready"]]
    _require(len(ready) == len(series) and ready["day"].eq(origin_day + 1).all(),
             "Before recursion, exactly the first day must be ready for every series")

    # Vary by series and day so an off-by-one or series-mixing bug is visible.
    synthetic = (
        1000.0 + np.arange(len(series))[:, None]
        + np.arange(1, horizon + 1)[None, :] / 10
    )
    keys = pd.MultiIndex.from_product(
        [series, range(origin_day + 1, origin_day + horizon + 1)], names=["id", "day"]
    )
    synthetic_by_key = pd.Series(synthetic.ravel(), index=keys)
    future_keys = pd.MultiIndex.from_arrays([frame.loc[future, "id"], days.loc[future]])
    filled = frame.copy()
    filled.loc[future, "sales"] = synthetic_by_key.reindex(future_keys).to_numpy()
    checked = build_features(filled, cfg)
    checked = checked.loc[checked["day"] > origin_day].set_index(["id", "day"]).reindex(keys)
    _require(checked["features_ready"].all(), "Synthetic history did not make all steps ready")

    lookback = cfg.history_days
    first_history = origin_day - lookback + 1
    history = past.assign(day=_days(past)).pivot(index="id", columns="day", values="sales")
    history = history.reindex(index=series, columns=range(first_history, origin_day + 1))
    _require(not history.isna().any().any(), "Need a complete history for the array oracle")
    values = np.concatenate([history.to_numpy(dtype=float), synthetic], axis=1)
    targets = lookback + np.arange(horizon)
    for lag in cfg.lags:
        expected = values[:, targets - lag]
        actual = checked[f"sales_lag_{lag}"].to_numpy().reshape(len(series), horizon)
        np.testing.assert_allclose(actual, expected.astype("float32"), rtol=0, atol=0)

    dates = frame.assign(day=days)[["day", "date"]].drop_duplicates("day").set_index("day")["date"]
    dates = pd.to_datetime(dates.reindex(range(first_history, origin_day + horizon + 1)))
    open_days = ~((dates.dt.month == cfg.closed_month) & (dates.dt.day == cfg.closed_day)).to_numpy()
    for window in cfg.rolling_windows:
        expected = np.column_stack([
            values[:, target - window:target][:, open_days[target - window:target]].mean(axis=1)
            for target in targets
        ])
        actual = checked[f"sales_roll_mean_{window}"].to_numpy().reshape(len(series), horizon)
        np.testing.assert_allclose(actual, expected.astype("float32"), rtol=0, atol=0)

    expected = np.repeat(last_prices.reindex(series).to_numpy()[:, None], horizon, axis=1)
    actual = checked["sell_price_lag_1"].to_numpy().reshape(len(series), horizon)
    np.testing.assert_allclose(actual, expected.astype("float32"), rtol=0, atol=0)
    return {
        "origin_day": origin_day,
        "first_day": origin_day + 1,
        "last_day": origin_day + horizon,
        "n_series": len(series),
        "synthetic_rows_checked": len(keys),
        "future_sales_erased": True,
        "future_price_frozen": True,
        "held_out_mutation_invariant": True,
        "only_first_future_day_ready_initially": True,
        "all_step_lags_and_means_match_past_only_oracle": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "reports" / "leakage_checks.json")
    args = parser.parse_args()
    split = load_config()
    cfg = load_feature_config()
    long = pd.read_parquet(LONG_PATH)
    # Audit validation only; no test predictions, scores or parameter selection.
    long = long.loc[_days(long) <= split.valid_end].copy()
    train = audit_training(long, split.train_end, cfg)
    print(f"PASS train cutoff invariance: {train['n_rows']} rows, {train['n_series']} series", flush=True)
    print("PASS feature allowlist: no label, metadata or full-period EDA statistics", flush=True)
    windows = []
    for first, last in make_windows(*split.ranges["valid"], split.horizon):
        result = audit_window(long, first - 1, last - first + 1, cfg)
        windows.append(result)
        print(
            f"PASS valid d_{first}..d_{last}: future mutation invariant; "
            f"{result['synthetic_rows_checked']} synthetic rows checked",
            flush=True,
        )
    report = {
        "status": "passed",
        "scope": "feature timing audit on train and validation; no model fitted or test scored",
        "feature_config": asdict(cfg),
        "training": train,
        "validation_windows": windows,
        "assumptions": [
            "Calendar events and SNAP schedules are known at the forecast origin.",
            "Future price inputs are frozen at the last observed origin price.",
        ],
        "limitations": [
            "The fixed subset was selected using full-period statistics, including validation/test.",
            "Synthetic values check feature calculations, not model accuracy or its prediction loop.",
            "Train-only categorical encoding and model fitting must be checked when implemented.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS all {len(windows)} validation windows; no model fitted or test scored")
    print(f"Saved audit to {args.output}")


if __name__ == "__main__":
    main()
