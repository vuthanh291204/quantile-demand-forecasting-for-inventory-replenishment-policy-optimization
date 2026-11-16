"""Origin-only demand history and known target calendars for direct forecasting.

Each training example is (series, origin, horizon) -> sales[origin + horizon].
The same origin history is used at every horizon; no prediction is fed back.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.features import (
    CALENDAR_FEATURES,
    CATEGORICAL_FEATURES,
    FeatureConfig,
    build_features,
    prepare_forecast_frame,
)

STATIC_FEATURES = ("item_id", "dept_id", "store_id", "state_id")
EVENT_FEATURES = tuple(c for c in CATEGORICAL_FEATURES if c not in STATIC_FEATURES)
TARGET_FEATURES = (*EVENT_FEATURES, *CALENDAR_FEATURES)
FORECAST_KEYS = ["id", "origin_day", "day", "date", "horizon"]


def direct_feature_columns(cfg: FeatureConfig) -> list[str]:
    return [
        "origin_sales",
        *[f"origin_sales_lag_{lag}" for lag in cfg.lags],
        *[f"origin_sales_roll_mean_{w}" for w in cfg.rolling_windows],
        "origin_sell_price",
        *STATIC_FEATURES,
        *TARGET_FEATURES,
        "horizon",
    ]


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _through_day(long, end_day):
    _positive_integer(end_day, "end_day")
    if not long["d"].str.fullmatch(r"d_[1-9][0-9]*").fillna(False).all():
        raise ValueError("Invalid day labels")
    days = long["d"].str.slice(2).astype(int)
    return long.loc[days <= end_day]


def _origin_snapshots(source, cfg):
    # Reuse the existing daily alignment, non-negative sales and category checks.
    base = build_features(source, cfg).merge(
        source[["id", "d", "sell_price"]],
        on=["id", "d"], validate="one_to_one",
    )
    ids = base["id"]
    started = base["sales"].gt(0).groupby(ids, sort=False).cummax()
    open_sales = base["sales"].where(started & base["is_closed"].eq(0))
    snapshots = base[["id", "day", *STATIC_FEATURES]].rename(
        columns={"day": "origin_day"}
    ).copy()
    snapshots["origin_sales"] = base["sales"]
    snapshots["origin_sell_price"] = base["sell_price"]
    for lag in cfg.lags:
        snapshots[f"origin_sales_lag_{lag}"] = base[f"sales_lag_{lag}"]
    for window in cfg.rolling_windows:
        # Includes the observed origin day; unlike week 3's target-day rolling mean.
        snapshots[f"origin_sales_roll_mean_{window}"] = open_sales.groupby(
            ids, sort=False
        ).transform(lambda s, w=window: s.rolling(w, min_periods=1).mean()).astype("float32")
    ready = base["features_ready"] & base["sales"].notna()
    return snapshots.loc[ready].copy(), base


def _expand(snapshots, calendar, horizon, cfg, with_labels):
    if snapshots.empty:
        raise ValueError("No eligible origins with sufficient observed history")
    numeric = [c for c in snapshots if c.startswith("origin_")]
    if not np.isfinite(snapshots[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Origin history and price must be finite")
    if snapshots["origin_sell_price"].le(0).any():
        raise ValueError("Origin price must be positive")
    table = snapshots.merge(
        pd.DataFrame({"horizon": np.arange(1, horizon + 1)}), how="cross"
    )
    table["day"] = table["origin_day"] + table["horizon"]
    target_columns = ["id", "day", "date", *TARGET_FEATURES]
    if with_labels:
        target_columns.append("sales")
    table = table.merge(
        calendar[target_columns], on=["id", "day"], how="left",
        validate="many_to_one", indicator=True,
    )
    if not table["_merge"].eq("both").all():
        raise ValueError("Need every target day for every eligible origin")
    if with_labels and not np.isfinite(table["sales"].to_numpy(dtype=float)).all():
        raise ValueError("Training targets must be observed and finite")
    columns = list(dict.fromkeys([*FORECAST_KEYS, *direct_feature_columns(cfg)]))
    if with_labels:
        columns.append("sales")
    return table[columns].sort_values(["id", "origin_day", "horizon"]).reset_index(drop=True)


def build_direct_training_table(long, end_day, horizon, origin_stride, cfg):
    """Use complete target blocks ending by end_day; never read later labels.

    Origins: end_day-horizon, then backwards by origin_stride. Eligibility is
    checked per series, using only past history since its first positive sale.
    """
    _positive_integer(horizon, "horizon")
    _positive_integer(origin_stride, "origin_stride")
    past = _through_day(long, end_day)
    snapshots, calendar = _origin_snapshots(past, cfg)
    origins = range(end_day - horizon, cfg.history_days, -origin_stride)
    snapshots = snapshots.loc[snapshots["origin_day"].isin(origins)]
    return _expand(snapshots, calendar, horizon, cfg, with_labels=True)


def build_direct_forecast_table(long, origin_day, horizon, cfg):
    """Future sales/prices are erased before validation or feature construction."""
    _positive_integer(origin_day, "origin_day")
    _positive_integer(horizon, "horizon")
    redacted = prepare_forecast_frame(long, origin_day, horizon)
    snapshots, calendar = _origin_snapshots(redacted, cfg)
    snapshots = snapshots.loc[snapshots["origin_day"].eq(origin_day)]
    if set(snapshots["id"]) != set(redacted["id"]):
        raise ValueError("Every forecast series needs sufficient origin history")
    return _expand(snapshots, calendar, horizon, cfg, with_labels=False)
