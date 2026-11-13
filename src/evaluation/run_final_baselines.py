"""Run every baseline on validation and test, and save the results.

Run:  python -m src.evaluation.run_final_baselines             (full run, several minutes)
      python -m src.evaluation.run_final_baselines --dry-run   (every 10th series, validation only,
                                                                writes nothing)

The settings (moving-average days, seasonal-average weeks) come from configs/baselines.toml and were
chosen on VALIDATION. The TEST split is evaluated once: if the test result file already exists the
script stops, unless --overwrite is given. Do not change settings because of test results.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import tomllib
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np
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
from src.forecast.baselines import (
    moving_average_forecast,
    naive_forecast,
    seasonal_moving_average_forecast,
    seasonal_naive_forecast,
    with_closed_days,
    zero_forecast,
)
from src.forecast.statistical import croston_sba_forecast, ets_forecast

METRICS = ["rmsse", "mae", "rmse", "smape"]
OUT_DIR = REPO_ROOT / "reports" / "baselines"
ZERO, NAIVE, SNAIVE = "zero (reference)", "naive", "seasonal_naive_7"
CROSTON, ETS = "croston_sba", "ets_ZZZ_season7"


def make_methods(closed_by_day, ma_days: int, sa_weeks: int, ets_stats: dict) -> dict:
    return {
        ZERO: zero_forecast,
        NAIVE: naive_forecast,
        SNAIVE: seasonal_naive_forecast,
        f"moving_avg_{ma_days}d": with_closed_days(moving_average_forecast, closed_by_day, window=ma_days),
        f"seasonal_avg_{sa_weeks}w": with_closed_days(
            seasonal_moving_average_forecast, closed_by_day, weeks=sa_weeks
        ),
        CROSTON: croston_sba_forecast,
        ETS: lambda history, horizon: ets_forecast(history, horizon, stats=ets_stats),
    }


def overall_table(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {
        name: {f"{m} {stat}": getattr(res[m], stat)() for m in METRICS for stat in ("mean", "median")}
        for name, res in results.items()
    }
    return pd.DataFrame(rows).T


def per_series_frame(split, results, meta, groups, mean_sales, negative_by_series) -> pd.DataFrame:
    frames = []
    for name, res in results.items():
        df = res[METRICS].rename_axis("id").reset_index()
        df.insert(0, "method", name)
        df.insert(0, "split", split)
        df["store_id"] = df["id"].map(meta["store_id"])
        df["volume_group"] = df["id"].map(groups)
        df["mean_daily_sales"] = df["id"].map(mean_sales)
        df["n_negative_forecasts"] = df["id"].map(negative_by_series) if name == ETS else np.nan
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summary_table(split, results, groups) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        g = groups.reindex(res.index)
        parts = [("all", res)] + [(f"group {k}", res[g == k]) for k in sorted(g.dropna().unique())]
        for label, part in parts:
            row = {"split": split, "method": name, "group": label, "n_series": len(part)}
            for m in METRICS:
                row[f"{m}_mean"] = part[m].mean()
                row[f"{m}_median"] = part[m].median()
            rows.append(row)
    return pd.DataFrame(rows)


def top_series_report(results, mean_sales, names) -> None:
    top = mean_sales.idxmax()
    print(f"\nSeries with the highest mean daily sales: {top} ({mean_sales[top]:.1f} units/day)")
    for name in names:
        mae = results[name]["mae"]
        print(
            f"  {name}: its MAE = {mae[top]:.3f}; MAE mean over series: "
            f"all = {mae.mean():.3f}, without this series = {mae.drop(top).mean():.3f}"
        )


def negative_report(negative_by_series: pd.Series, groups, n_forecast_days: int) -> None:
    any_neg = negative_by_series > 0
    print(f"\nETS: series with at least one negative forecast (set to 0): {int(any_neg.sum())} of {len(any_neg)}")
    table = pd.DataFrame(
        {
            "group": groups.reindex(negative_by_series.index),
            "any_negative": any_neg,
            "share": negative_by_series / n_forecast_days,
        }
    ).groupby("group").agg(
        n_series=("any_negative", "size"),
        series_with_negative=("any_negative", "sum"),
        mean_share_negative=("share", "mean"),
    )
    print(table.round(3).to_string())
    print("Series with the most negative forecasts:")
    print(negative_by_series.sort_values(ascending=False).head(5).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="every 10th series, validation only, no files")
    parser.add_argument("--overwrite", action="store_true", help="allow overwriting existing test results")
    args = parser.parse_args()

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    test_file = OUT_DIR / "per_series_test.csv"
    if not args.dry_run and test_file.exists() and not args.overwrite:
        print(f"{test_file} already exists. The test split is meant to be evaluated once.")
        print("Run with --overwrite only if you really need to redo it (and do not tune on test results).")
        sys.exit(1)

    with open(REPO_ROOT / "configs" / "baselines.toml", "rb") as f:
        chosen = tomllib.load(f)
    ma_days = chosen["moving_average"]["window_days"]
    sa_weeks = chosen["seasonal_average"]["weeks"]

    cfg = load_config()
    wide = pd.read_parquet(WIDE_PATH).set_index("id")
    meta = wide[["store_id"]]
    all_days = wide[day_columns(cfg.first_day, cfg.last_day)]
    if args.dry_run:
        all_days = all_days.iloc[::10]
    closed = closed_days(load_day_dates(), cfg)
    closed_by_day = closed.sort_index().to_numpy()
    groups = (
        pd.read_parquet(LONG_PATH, columns=["id", "volume_group"])
        .drop_duplicates("id")
        .set_index("id")["volume_group"]
    )

    splits = ["valid"] if args.dry_run else ["valid", "test"]
    overall, per_series_parts, summary_parts, run_info_splits = {}, [], [], {}
    for split in splits:
        first, last = cfg.ranges[split]
        windows = make_windows(first, last, cfg.horizon)
        excluded = int(closed.loc[first:last].sum())
        print(f"\n===== split={split}  d_{first}..d_{last}  windows={len(windows)}  series={len(all_days)} =====")
        print(f"closed days excluded from metrics: {excluded}")

        ets_stats: dict = {}
        methods = make_methods(closed_by_day, ma_days, sa_weeks, ets_stats)
        results = {}
        for name, fn in methods.items():
            start = time.time()
            results[name] = backtest(all_days, fn, windows, closed)
            print(f"  {name}: {time.time() - start:.1f} s", flush=True)

        mean_sales = all_days[day_columns(first, last)].mean(axis=1)
        negative = pd.Series(ets_stats["negative_by_row"], index=all_days.index)

        overall[split] = overall_table(results)
        print(f"\n== {split}: mean / median over series ==")
        print(overall[split].round(3).to_string())
        top_series_report(results, mean_sales, [ZERO, f"moving_avg_{ma_days}d", ETS])
        negative_report(negative, groups, len(windows) * cfg.horizon)

        per_series_parts.append(per_series_frame(split, results, meta, groups, mean_sales, negative))
        summary_parts.append(summary_table(split, results, groups))
        run_info_splits[split] = {
            "days": [first, last],
            "windows": len(windows),
            "closed_days_excluded": excluded,
        }

    if len(splits) == 2:
        cmp = pd.DataFrame(
            {
                "rmsse valid": overall["valid"]["rmsse mean"],
                "rmsse test": overall["test"]["rmsse mean"],
                "mae valid": overall["valid"]["mae mean"],
                "mae test": overall["test"]["mae mean"],
            }
        )
        print("\n== Validation vs test (mean over series) ==")
        print(cmp.round(3).to_string())

    if args.dry_run:
        print("\nDry run: nothing was saved.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, frame in zip(splits, per_series_parts):
        frame.to_csv(OUT_DIR / f"per_series_{split}.csv", index=False)
    pd.concat(summary_parts, ignore_index=True).to_csv(OUT_DIR / "summary.csv", index=False)
    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "libraries": {lib: version(lib) for lib in ("pandas", "numpy", "statsforecast")},
        "horizon_days": cfg.horizon,
        "n_series": len(all_days),
        "splits": run_info_splits,
        "settings": {
            "moving_average_days": ma_days,
            "seasonal_average_weeks": sa_weeks,
            "ets": "AutoETS, model ZZZ, season_length 7",
            "croston": "CrostonSBA",
            "selection_rule": chosen["selection"]["rule"],
        },
        "notes": [
            "Settings were chosen on validation only; test was evaluated once.",
            "25/12 is excluded from metrics and from the moving averages; ETS and Croston "
            "still see the 25/12 zeros in their history.",
            "ETS negative forecasts are set to 0.",
        ],
    }
    with open(OUT_DIR / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"\nSaved to {OUT_DIR}: per_series_valid.csv, per_series_test.csv, summary.csv, run_info.json")


if __name__ == "__main__":
    main()