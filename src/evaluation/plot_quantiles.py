"""Plot saved validation quantiles for train-volume examples and a diagnostic case.

Run: python -m src.evaluation.plot_quantiles --run-id <validation-backtest-run-id>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
import mlflow
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.data.split import (
    REPO_ROOT,
    WIDE_PATH,
    SplitConfig,
    closed_days,
    day_columns,
    load_day_dates,
    make_windows,
)
from src.evaluation.score_quantiles import validate_forecasts
from src.forecast.quantile_model import load_quantile_config
from src.tracking import configure_tracking, load_tracking_config

ROLE_LABELS = {
    "train_low": "Mức bán thấp theo train",
    "train_middle": "Mức bán giữa theo train",
    "train_high": "Mức bán cao theo train",
    "validation_low_coverage": "Chuỗi độ phủ thấp, chọn sau khi đánh giá validation",
}


def training_volume_stats(wide, split, dates):
    """Use open days since the first positive sale, restricted to the training period."""
    history = wide[day_columns(split.first_day, split.train_end)].to_numpy(dtype=float)
    if not np.isfinite(history).all() or (history < 0).any():
        raise ValueError("Training sales must be finite and non-negative")
    live = np.maximum.accumulate(history > 0, axis=1)
    open_days = ~closed_days(dates, split).loc[split.first_day:split.train_end].to_numpy()
    keep = live & open_days
    counts = keep.sum(axis=1)
    if (counts == 0).any():
        raise ValueError("Every candidate series needs observed open training days")
    return pd.DataFrame({
        "id": wide.index,
        "train_mean": np.where(keep, history, 0).sum(axis=1) / counts,
        "train_zero_share": ((history == 0) & keep).sum(axis=1) / counts,
        "train_days": counts,
    })


def select_plot_series(train_stats, per_series):
    scores = per_series.loc[per_series["variant"].eq("clipped")]
    columns = ["id", "interval90_coverage", "interval90_mean_width_ordered", "crossing_rate"]
    candidates = train_stats.merge(scores[columns], on="id", validate="one_to_one")
    if len(candidates) != len(train_stats) or candidates.empty:
        raise ValueError("Need matching training statistics and validation scores")
    ordered = candidates.sort_values(["train_mean", "id"]).reset_index(drop=True)
    selected = []
    selected_ids = set()
    for role, fraction in (("train_low", 0.1), ("train_middle", 0.5), ("train_high", 0.9)):
        row = ordered.iloc[round(fraction * (len(ordered) - 1))].to_dict()
        if row["id"] not in selected_ids:
            selected.append({**row, "role": role, "selection_basis": "training_period_only"})
            selected_ids.add(row["id"])
    remaining = candidates.loc[~candidates["id"].isin(selected_ids)].sort_values(
        ["interval90_coverage", "id"]
    )
    if not remaining.empty:
        row = remaining.iloc[0].to_dict()
        selected.append({**row, "role": "validation_low_coverage",
                         "selection_basis": "posthoc_validation_diagnostic"})
    return pd.DataFrame(selected)


def build_plot_panel(wide, forecasts, selected, split, dates):
    ids = selected["id"].tolist()
    actual = wide.loc[ids, day_columns(*split.ranges["valid"])].rename_axis("id").reset_index()
    actual = actual.melt(id_vars="id", var_name="d", value_name="sales")
    actual["day"] = actual["d"].str.slice(2).astype(int)
    panel = forecasts.loc[forecasts["id"].isin(ids)].merge(
        actual[["id", "day", "sales"]], on=["id", "day"], validate="one_to_one"
    )
    if len(panel) != len(ids) * (split.valid_end - split.train_end):
        raise ValueError("Selected series need every validation day")
    panel["date"] = pd.to_datetime(panel["date"])
    panel["evaluated"] = ~panel["day"].map(closed_days(dates, split))
    return panel.sort_values(["id", "day"]).reset_index(drop=True)


def _draw_series(ax, panel, detail=False):
    x = mdates.date2num(panel["date"].to_numpy())
    ax.plot(x, panel["sales"], color="#222222", linewidth=0.85, marker=".",
            markersize=3, label="Số bán thực tế", zorder=4)
    for index, (_, window) in enumerate(panel.groupby("window", sort=True)):
        dates = mdates.date2num(window["date"].to_numpy())
        lo = window["q05"].to_numpy(dtype=float)
        hi = window["q95"].to_numpy(dtype=float)
        ax.fill_between(dates, lo, hi, where=lo <= hi, color="#287D8E", alpha=0.19,
                        label="[q05, q95], mức danh nghĩa 90%" if index == 0 else None)
        ax.plot(dates, hi, color="#287D8E", linewidth=0.7, alpha=0.85)
        ax.plot(dates, lo, color="#287D8E", linewidth=0.7, alpha=0.85)
        ax.plot(dates, window["q50"], color="#2463B4", linewidth=1.35,
                label="Trung vị q50" if index == 0 else None)
        ax.plot(dates, window["q90"], color="#8850A2", linewidth=0.9, linestyle=":",
                label="Phân vị q90" if index == 0 else None)
        if index > 0:
            ax.axvline(dates[0] - 0.5, color="#888888", linestyle="--", linewidth=0.75)
    outside = panel["evaluated"] & (
        panel["sales"].lt(panel["q05"]) | panel["sales"].gt(panel["q95"])
        | panel["q05"].gt(panel["q95"])
    )
    ax.scatter(x[outside], panel.loc[outside, "sales"], color="#BD443C", marker="^",
               s=22, label="Ngày ngoài khoảng", zorder=5)
    ax.set_ylabel("Số lượng bán (đơn vị/ngày)")
    ax.set_xlabel("Ngày")
    ax.set_ylim(bottom=0)
    ax.set_xlim(x.min() - 0.6, x.max() + 0.6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.xaxis_date()
    if detail:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)


def write_figures(panel, selected, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        for row in selected.itertuples():
            series = panel.loc[panel["id"].eq(row.id)]
            first = series.loc[series["window"].eq(series["window"].min())]
            fig, axes = plt.subplots(2, 1, figsize=(14, 7.2), layout="constrained")
            fig.suptitle(f"{row.id}\n{ROLE_LABELS[row.role]}", fontsize=12)
            _draw_series(axes[0], series)
            axes[0].set_title(
                f"Toàn bộ validation · Độ phủ TB cửa sổ {row.interval90_coverage:.2%} · "
                f"Độ rộng TB {row.interval90_mean_width_ordered:.2f} · "
                f"Crossing {row.crossing_rate:.2%}", fontsize=10
            )
            _draw_series(axes[1], first, detail=True)
            axes[1].set_title("Phóng to cửa sổ validation đầu tiên", fontsize=10)
            axes[0].legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.9)
            fig.supxlabel(
                "Nguồn M5 · Dự báo trực tiếp, cập nhật mỗi 28 ngày · Vạch đứt dọc đánh dấu cửa sổ mới",
                fontsize=9,
            )
            path = output_dir / f"{row.role}.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            files.append(path)
    selected.to_csv(output_dir / "selected_series.csv", index=False)
    panel.to_csv(output_dir / "plot_data.csv", index=False)
    return files


def run_plots(client, experiment_id, source_id, output_root):
    source = client.get_run(source_id)
    if source.info.status != "FINISHED" or source.info.experiment_id != experiment_id:
        raise ValueError("Need a finished run in the quantile experiment")
    if source.data.tags.get("stage") != "quantile_validation_backtest":
        raise ValueError("Need a validation quantile backtest run")
    paths = {name: Path(client.download_artifacts(source_id, name)) for name in (
        "effective_config.json", "configs/subset_ids.csv", "results/forecasts_valid.csv",
        "results/per_series_valid.csv",
    )}
    archived = json.loads(paths["effective_config.json"].read_text(encoding="utf-8"))
    split = SplitConfig(**archived["split"])
    forecasts = pd.read_csv(paths["results/forecasts_valid.csv"])
    scores = pd.read_csv(paths["results/per_series_valid.csv"])
    wide = pd.read_parquet(WIDE_PATH, columns=["id", *day_columns(split.first_day, split.valid_end)])
    wide = wide.set_index("id")
    ids = pd.read_csv(paths["configs/subset_ids.csv"])["id"]
    if ids.duplicated().any() or set(ids) != set(wide.index):
        raise ValueError("Actual series must match the archived subset")
    dates = load_day_dates()
    windows = make_windows(*split.ranges["valid"], split.horizon)
    if set(forecasts["window"]) != set(range(1, len(windows) + 1)):
        raise ValueError("Need every saved validation window")
    for number, (first, _) in enumerate(windows, start=1):
        validate_forecasts(wide, forecasts.loc[forecasts["window"].eq(number)],
                           first - 1, split.horizon, dates)
    selected = select_plot_series(training_volume_stats(wide, split, dates), scores)
    panel = build_plot_panel(wide, forecasts, selected, split, dates)
    with mlflow.start_run(
        experiment_id=experiment_id, run_name="quantiles-validation-figures",
        tags={"stage": "quantile_validation_plots", "source_backtest_run_id": source_id,
              "model_trained": "false", "test_evaluated": "false"},
    ) as run:
        output_dir = Path(output_root) / source_id / "plots" / run.info.run_id
        files = write_figures(panel, selected, output_dir)
        for path in [*files, output_dir / "selected_series.csv", output_dir / "plot_data.csv"]:
            mlflow.log_artifact(str(path), artifact_path="plots")
        mlflow.log_artifact(__file__, artifact_path="code")
        mlflow.log_dict({
            "source_backtest_run_id": source_id,
            "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in paths.items()},
            "selection": "train_mean_rank_10_50_90_percent_and_lowest_remaining_validation_coverage",
            "detail_window": 1, "interval": "saved_clipped_q05_q95",
        }, "plot_provenance.json")
        mlflow.log_metrics({"n_selected_series": len(selected), "n_figures": len(files),
                            "n_plot_rows": len(panel)})
        run_id = run.info.run_id
    for row in selected.itertuples():
        print(f"Selected {row.id}: {row.role}; coverage={row.interval90_coverage:.6%}")
    print("Source backtest run:", source_id)
    print("Selected series:", len(selected))
    print("Figures:", len(files))
    print("Plot run ID:", run_id)
    print("Status:", client.get_run(run_id).info.status)
    print("Saved figures:", output_dir)
    return run_id, output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    tracking = replace(load_tracking_config(), experiment_name=load_quantile_config().experiment_name)
    client, experiment_id = configure_tracking(tracking)
    run_plots(client, experiment_id, args.run_id, REPO_ROOT / "reports" / "quantiles")


if __name__ == "__main__":
    main()
