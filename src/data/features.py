"""Past-only features for one-day training and recursive 28-day forecasting.

Run: python -m src.data.features

Do not build validation features from its observed sales. Start each forecast
window with prepare_forecast_frame(), then fill sales with model predictions
one day at a time. Only the first future day is ready before that loop.
"""
from __future__ import annotations

import argparse
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.split import LONG_PATH, REPO_ROOT, load_config

DEFAULT_CONFIG = REPO_ROOT / "configs" / "features.toml"
CATEGORICAL_FEATURES = (
    "item_id", "dept_id", "store_id", "state_id",
    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
)
CALENDAR_FEATURES = (
    "day_of_week", "month", "day_of_month", "year", "snap", "has_event", "is_closed",
)
SOURCE_COLUMNS = (
    "id", "d", "date", "sales", "sell_price", *CATEGORICAL_FEATURES,
    "snap_CA", "snap_TX", "snap_WI",
)


@dataclass(frozen=True)
class FeatureConfig:
    lags: tuple[int, ...] = (7, 14, 28)
    rolling_windows: tuple[int, ...] = (7, 28)
    closed_month: int = 12
    closed_day: int = 25

    def __post_init__(self) -> None:
        for name in ("lags", "rolling_windows"):
            values = tuple(getattr(self, name))
            if not values or any(type(v) is not int or v < 1 for v in values):
                raise ValueError(f"{name} must contain positive integers")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
            object.__setattr__(self, name, values)

    @property
    def history_days(self) -> int:
        return max(*self.lags, *self.rolling_windows)

    @property
    def feature_columns(self) -> list[str]:
        return (
            [f"sales_lag_{lag}" for lag in self.lags]
            + [f"sales_roll_mean_{window}" for window in self.rolling_windows]
            + list(CATEGORICAL_FEATURES)
            + list(CALENDAR_FEATURES)
            + ["sell_price_lag_1"]
        )


def load_feature_config(path: Path = DEFAULT_CONFIG) -> FeatureConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    split = load_config()
    return FeatureConfig(
        **raw["features"], closed_month=split.closed_month, closed_day=split.closed_day
    )


def _day_numbers(long: pd.DataFrame) -> pd.Series:
    if not long["d"].str.fullmatch(r"d_[1-9][0-9]*").all():
        raise ValueError("d must contain M5 day labels such as d_1")
    return long["d"].str.slice(2).astype(int)


def _prepare(long: pd.DataFrame) -> pd.DataFrame:
    missing = set(SOURCE_COLUMNS) - set(long.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    # An allowlist keeps full-period EDA statistics out of the model table.
    frame = long.loc[:, list(SOURCE_COLUMNS)].copy()
    if frame.empty or frame["id"].isna().any():
        raise ValueError("Need non-empty data with non-missing series ids")
    frame["day"] = _day_numbers(frame)
    frame["date"] = pd.to_datetime(frame["date"])
    if frame["date"].isna().any():
        raise ValueError("Dates must not be missing")
    frame = frame.sort_values(["id", "day"]).reset_index(drop=True)
    gaps = frame.groupby("id", sort=False)["day"].diff().dropna()
    if not gaps.eq(1).all():
        raise ValueError("Each series must have unique, consecutive daily rows")
    offsets = frame["date"] - pd.to_timedelta(frame["day"], unit="D")
    if offsets.nunique() != 1:
        raise ValueError("Day labels and calendar dates must agree across all series")
    observed = frame["sales"].dropna()
    if not np.isfinite(observed).all() or observed.lt(0).any():
        raise ValueError("Observed sales must be finite and non-negative")
    if not frame["state_id"].isin(["CA", "TX", "WI"]).all():
        raise ValueError("Unknown state_id for SNAP selection")
    return frame


def build_features(long: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Return labels, features and a readiness flag; never use current sales in X.

    A ready row needs a complete lookback since the first observed positive
    sale. Rolling windows cover calendar days but exclude assumed closed days
    from their mean. Raw lags retain the observed zeros on those days.
    Categorical vocabularies must be fitted on training data when modelling.
    """
    frame = _prepare(long)
    ids = frame["id"]
    dates = frame["date"].dt
    frame["day_of_week"] = dates.dayofweek.astype("int8")  # Monday=0, Sunday=6
    frame["month"] = dates.month.astype("int8")
    frame["day_of_month"] = dates.day.astype("int8")
    frame["year"] = dates.year.astype("int16")
    frame["is_closed"] = (
        (dates.month == cfg.closed_month) & (dates.day == cfg.closed_day)
    ).astype("int8")
    frame["has_event"] = (
        frame["event_name_1"].notna() | frame["event_name_2"].notna()
    ).astype("int8")
    frame["snap"] = np.select(
        [frame["state_id"].eq(state) for state in ("CA", "TX", "WI")],
        [frame[f"snap_{state}"] for state in ("CA", "TX", "WI")],
    ).astype("int8")
    for col in CATEGORICAL_FEATURES:
        frame[col] = frame[col].fillna("none")

    started = frame["sales"].gt(0).groupby(ids, sort=False).cummax()
    live_sales = frame["sales"].where(started)
    for lag in cfg.lags:
        frame[f"sales_lag_{lag}"] = live_sales.groupby(ids, sort=False).shift(lag).astype("float32")
    open_sales = live_sales.where(frame["is_closed"].eq(0))
    for window in cfg.rolling_windows:
        frame[f"sales_roll_mean_{window}"] = open_sales.groupby(ids, sort=False).transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
        ).astype("float32")
    frame["sell_price_lag_1"] = (
        frame.groupby("id", sort=False)["sell_price"].shift(1).astype("float32")
    )
    known = started & frame["sales"].notna()
    count = known.groupby(ids, sort=False).transform(
        lambda s: s.shift(1, fill_value=False).rolling(cfg.history_days).sum()
    )
    frame["features_ready"] = (
        count.eq(cfg.history_days) & frame[cfg.feature_columns].notna().all(axis=1)
    )
    return frame[["id", "d", "day", "date", "sales", *cfg.feature_columns, "features_ready"]]


def build_training_table(long: pd.DataFrame, end_day: int, cfg: FeatureConfig) -> pd.DataFrame:
    """Restrict to the fitting cutoff BEFORE constructing features."""
    past = long.loc[_day_numbers(long) <= end_day]
    frame = build_features(past, cfg)
    return frame.loc[frame["features_ready"] & frame["sales"].notna()].reset_index(drop=True)


def prepare_forecast_frame(long: pd.DataFrame, origin_day: int, horizon: int) -> pd.DataFrame:
    """Redact future sales and freeze prices at the last observed origin price.

    Calendar events and state SNAP flags are assumed known at the origin.
    After predicting day t, put that prediction into sales for day t before
    building day t+1 features. Never put its observed sales into this frame.
    """
    if origin_day < 1 or horizon < 1:
        raise ValueError("origin_day and horizon must be positive")
    days = _day_numbers(long)
    frame = long.loc[days <= origin_day + horizon, list(SOURCE_COLUMNS)].copy()
    future = _day_numbers(frame) > origin_day
    # Erase held-out values before validation or feature calculation.
    frame.loc[future, "sales"] = np.nan
    frame.loc[future, "sell_price"] = np.nan
    frame = _prepare(frame)
    future = frame["day"] > origin_day
    past = frame.loc[~future]
    series = pd.Index(past["id"].unique())
    expected = pd.MultiIndex.from_product(
        [series, range(origin_day, origin_day + horizon + 1)], names=["id", "day"]
    )
    actual = pd.MultiIndex.from_frame(frame.loc[frame["day"] >= origin_day, ["id", "day"]])
    if len(series) == 0 or not actual.sort_values().equals(expected.sort_values()):
        raise ValueError("Need the origin day and every forecast day for each historical series")
    last_prices = past.groupby("id", sort=False)["sell_price"].last()
    frame.loc[future, "sell_price"] = frame.loc[future, "id"].map(last_prices)
    return frame.loc[:, list(SOURCE_COLUMNS)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional training parquet path")
    args = parser.parse_args()
    split = load_config()
    cfg = load_feature_config()
    long = pd.read_parquet(LONG_PATH)
    train = build_training_table(long, split.train_end, cfg)
    forecast = prepare_forecast_frame(long, split.train_end, split.horizon)
    first_day = build_features(forecast, cfg)
    future = first_day[first_day["day"] > split.train_end]
    ready = future[future["features_ready"]]
    print(f"Train cutoff: d_{split.train_end}; rows={len(train)}; series={train['id'].nunique()}")
    print(f"Features ({len(cfg.feature_columns)}): {', '.join(cfg.feature_columns)}")
    print(f"Missing training feature values: {int(train[cfg.feature_columns].isna().sum().sum())}")
    print(f"Future observed sales retained: {int(future['sales'].notna().sum())}")
    print(f"Ready future rows before recursion: {len(ready)} (first forecast day only)")
    print(f"Ready future days: {sorted(ready['day'].unique().tolist())}")
    print("Future prices: last observed price at origin. Calendar/SNAP: assumed known ahead.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        train.to_parquet(args.output, index=False)
        print(f"Saved training features to {args.output}")


if __name__ == "__main__":
    main()
