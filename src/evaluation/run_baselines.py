"""Compare baselines on the VALIDATION split and pick the window lengths.

Run:  python -m src.evaluation.run_baselines

Selection rule (fixed before looking at results): inside each family, the
setting with the lowest MEAN RMSSE over series. The test split is not used here.
"""
from __future__ import annotations

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
from src.evaluation.backtest import backtest
from src.evaluation.metrics import summarize
from src.forecast.baselines import (
    moving_average_forecast,
    naive_forecast,
    seasonal_moving_average_forecast,
    seasonal_naive_forecast,
    with_closed_days,
    zero_forecast,
)

METRICS = ["mae", "rmse", "smape", "rmsse"]
MA_WINDOWS = [7, 14, 28, 56, 112]  # days
SMA_WEEKS = [2, 4, 8, 16]  # weeks


def overall_table(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {}
    for name, res in results.items():
        rows[name] = {
            f"{m} {stat}": getattr(res[m], stat)()
            for m in ["rmsse", "mae", "rmse", "smape"]
            for stat in ("mean", "median")
        }
    return pd.DataFrame(rows).T


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    cfg = load_config()
    all_days = pd.read_parquet(WIDE_PATH).set_index("id")[day_columns(cfg.first_day, cfg.last_day)]
    closed = closed_days(load_day_dates(), cfg)
    closed_by_day = closed.sort_index().to_numpy()  # position i <-> day i + 1
    groups = (
        pd.read_parquet(LONG_PATH, columns=["id", "volume_group"])
        .drop_duplicates("id")
        .set_index("id")["volume_group"]
    )

    first, last = cfg.ranges["valid"]
    windows = make_windows(first, last, cfg.horizon)
    print(f"split=valid  d_{first}..d_{last}  windows={len(windows)}  series={len(all_days)}")
    print(f"closed days excluded from metrics: {int(closed.loc[first:last].sum())}")

    methods = {
        "zero (reference)": zero_forecast,
        "naive": naive_forecast,
        "seasonal_naive_7": seasonal_naive_forecast,
    }
    for k in MA_WINDOWS:
        methods[f"moving_avg_{k}d"] = with_closed_days(moving_average_forecast, closed_by_day, window=k)
    for w in SMA_WEEKS:
        methods[f"seasonal_avg_{w}w"] = with_closed_days(
            seasonal_moving_average_forecast, closed_by_day, weeks=w
        )

    results = {name: backtest(all_days, fn, windows, closed) for name, fn in methods.items()}

    print("\n== Overall on validation: mean / median over series ==")
    print(overall_table(results).round(3).to_string())

    def best(prefix: str) -> str:
        names = [n for n in results if n.startswith(prefix)]
        return min(names, key=lambda n: results[n]["rmsse"].mean())

    chosen = [best("moving_avg_"), best("seasonal_avg_")]
    print(f"\nSelected by lowest mean RMSSE: {chosen[0]}, {chosen[1]}")

    for name in chosen:
        print(f"\n== {name}: by volume group (0 = low, 2 = high) ==")
        table = pd.concat(
            {m: summarize(results[name][m], groups)[["mean", "median"]] for m in METRICS}, axis=1
        )
        print(table.round(3).to_string())


if __name__ == "__main__":
    main()