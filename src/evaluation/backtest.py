"""Rolling-origin backtest: forecast window by window, only using the past.

Run:  python -m src.evaluation.backtest --split valid
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.data.split import (
    LONG_PATH,
    WIDE_PATH,
    closed_days,
    day_columns,
    load_config,
    load_day_dates,
    make_windows,
)
from src.evaluation.metrics import (
    mae_per_series,
    rmse_per_series,
    rmsse_per_series,
    smape_per_series,
    summarize,
)
from src.forecast.baselines import naive_forecast, seasonal_naive_forecast, zero_forecast

METRICS = ["mae", "rmse", "smape", "rmsse"]


def backtest(
    all_days: pd.DataFrame,
    forecast_fn,
    windows: list[tuple[int, int]],
    closed: pd.Series,
) -> pd.DataFrame:
    """One row per series: each metric averaged over the windows.

    all_days: rows = series id, columns d_1 ... d_N (all days, from day 1).
    forecast_fn(history_array, horizon) -> forecast array (n_series, horizon).
    windows: list of (first_day, last_day) numbers, inclusive.
    closed: bool Series indexed by day number; True = store closed (ignored in metrics).
    """
    per_window = []
    for first, last in windows:
        history = all_days[day_columns(1, first - 1)]  # only days BEFORE the window
        actual = all_days[day_columns(first, last)]
        horizon = last - first + 1
        forecast = pd.DataFrame(
            forecast_fn(history.to_numpy(dtype=float), horizon),
            index=actual.index,
            columns=actual.columns,
        )
        keep = ~closed.loc[first:last].to_numpy()
        per_window.append(
            pd.DataFrame(
                {
                    "mae": mae_per_series(actual, forecast, mask=keep),
                    "rmse": rmse_per_series(actual, forecast, mask=keep),
                    "smape": smape_per_series(actual, forecast, mask=keep),
                    "rmsse": rmsse_per_series(actual, forecast, history, mask=keep),
                }
            )
        )
    return pd.concat(per_window).groupby(level=0).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    args = parser.parse_args()

    cfg = load_config()
    all_days = pd.read_parquet(WIDE_PATH).set_index("id")[day_columns(cfg.first_day, cfg.last_day)]
    dates = load_day_dates()
    closed = closed_days(dates, cfg)
    groups = (
        pd.read_parquet(LONG_PATH, columns=["id", "volume_group"])
        .drop_duplicates("id")
        .set_index("id")["volume_group"]
    )

    first, last = cfg.ranges[args.split]
    windows = make_windows(first, last, cfg.horizon)
    print(f"split={args.split}  d_{first}..d_{last}  windows={len(windows)}  series={len(all_days)}")
    print(f"closed days excluded from metrics: {int(closed.loc[first:last].sum())}")

    methods = {
        "zero (reference)": zero_forecast,
        "naive": naive_forecast,
        "seasonal_naive_7": seasonal_naive_forecast,
    }
    results = {name: backtest(all_days, fn, windows, closed) for name, fn in methods.items()}

    print("\n== Overall: mean and median over series ==")
    rows = {}
    for name, res in results.items():
        rows[name] = {}
        for m in METRICS:
            rows[name][f"{m} mean"] = res[m].mean()
            rows[name][f"{m} median"] = res[m].median()
    print(pd.DataFrame(rows).T.round(3).to_string())

    for name, res in results.items():
        print(f"\n== {name}: by volume group (0 = low, 2 = high; relative within each store) ==")
        table = pd.concat(
            {m: summarize(res[m], groups)[["mean", "median"]] for m in METRICS}, axis=1
        )
        print(table.round(3).to_string())


if __name__ == "__main__":
    main()