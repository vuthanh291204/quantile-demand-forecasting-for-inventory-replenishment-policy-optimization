"""Evaluate Croston and ETS on the VALIDATION split and compare them with the earlier baselines.

Run:  python -m src.evaluation.run_statistical             (all 297 series, takes several minutes)
      python -m src.evaluation.run_statistical --every 10  (every 10th series, quick check)

The test split is not used here.
"""
from __future__ import annotations

import argparse
import time
import tomllib

import pandas as pd

from src.data.split import (
    LONG_PATH,
    REPO_ROOT,
    WIDE_PATH,
    closed_days,
    day_columns,
    load_config,
    load_day_dates,
    make_windows,
)
from src.evaluation.backtest import backtest
from src.evaluation.metrics import summarize
from src.evaluation.run_baselines import METRICS, overall_table
from src.forecast.baselines import (
    moving_average_forecast,
    naive_forecast,
    seasonal_moving_average_forecast,
    seasonal_naive_forecast,
    with_closed_days,
    zero_forecast,
)
from src.forecast.statistical import croston_sba_forecast, ets_forecast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--every", type=int, default=1, help="use every N-th series (quick check)")
    args = parser.parse_args()

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    with open(REPO_ROOT / "configs" / "baselines.toml", "rb") as f:
        chosen = tomllib.load(f)
    ma_days = chosen["moving_average"]["window_days"]
    sa_weeks = chosen["seasonal_average"]["weeks"]

    cfg = load_config()
    all_days = pd.read_parquet(WIDE_PATH).set_index("id")[day_columns(cfg.first_day, cfg.last_day)]
    all_days = all_days.iloc[:: args.every]
    closed = closed_days(load_day_dates(), cfg)
    closed_by_day = closed.sort_index().to_numpy()
    groups = (
        pd.read_parquet(LONG_PATH, columns=["id", "volume_group"])
        .drop_duplicates("id")
        .set_index("id")["volume_group"]
    )

    first, last = cfg.ranges["valid"]
    windows = make_windows(first, last, cfg.horizon)
    print(f"split=valid  d_{first}..d_{last}  windows={len(windows)}  series={len(all_days)}")
    print(f"closed days excluded from metrics: {int(closed.loc[first:last].sum())}")

    ets_stats: dict = {}
    ma_name, sa_name = f"moving_avg_{ma_days}d", f"seasonal_avg_{sa_weeks}w"
    croston_name, ets_name = "croston_sba", "ets_ZZZ_season7"
    methods = {
        "zero (reference)": zero_forecast,
        "naive": naive_forecast,
        "seasonal_naive_7": seasonal_naive_forecast,
        ma_name: with_closed_days(moving_average_forecast, closed_by_day, window=ma_days),
        sa_name: with_closed_days(seasonal_moving_average_forecast, closed_by_day, weeks=sa_weeks),
        croston_name: croston_sba_forecast,
        ets_name: lambda history, horizon: ets_forecast(history, horizon, stats=ets_stats),
    }

    results = {}
    for name, fn in methods.items():
        if name == ets_name:
            print("\nRunning ETS (slow: it is fitted again on every series and every window)...", flush=True)
        start = time.time()
        results[name] = backtest(all_days, fn, windows, closed)
        print(f"  {name}: {time.time() - start:.1f} s", flush=True)

    print("\n== Overall on validation: mean / median over series ==")
    print(overall_table(results).round(3).to_string())

    for name in (croston_name, ets_name):
        print(f"\n== {name}: by volume group (0 = low, 2 = high) ==")
        table = pd.concat(
            {m: summarize(results[name][m], groups)[["mean", "median"]] for m in METRICS}, axis=1
        )
        print(table.round(3).to_string())

    n, neg = ets_stats.get("forecasts", 0), ets_stats.get("negative", 0)
    print(f"\nETS forecasts that were negative and set to 0: {neg} of {n} ({100 * neg / max(n, 1):.2f}%)")
    for name in (croston_name, ets_name):
        better = (results[name]["rmsse"] < results[ma_name]["rmsse"]).mean()
        print(f"Share of series where {name} has lower RMSSE than {ma_name}: {100 * better:.1f}%")


if __name__ == "__main__":
    main()