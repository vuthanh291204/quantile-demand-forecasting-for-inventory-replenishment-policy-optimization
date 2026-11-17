"""Simulate and plot one M5 series on the first validation window."""

from __future__ import annotations

import argparse
import json
import math
import tomllib
from pathlib import Path
from uuid import uuid4

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.data.split import LONG_PATH, REPO_ROOT, SplitConfig, closed_days, load_config
from src.simulation.inventory import SimulationResult, _integer, simulate_inventory

DEFAULT_CONFIG = REPO_ROOT / "configs" / "inventory_demo.toml"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "simulation" / "week5_demo"


def simulate_demo(
    long: pd.DataFrame,
    split: SplitConfig,
    *,
    series_id: str,
    lead_time_days: int,
    history_days: int,
) -> tuple[SimulationResult, dict]:
    """Set a fixed order-up-to level using only days before validation.

    Illustrative choices are a target of ceil(recent mean * (L + 1)),
    initial stock equal to that target and no initial outstanding orders.
    Review and demand stop on assumed closed days, while receipts and
    calendar lead time continue. Return the ledger and replay settings.
    """
    lead_time = _integer(lead_time_days, "lead_time_days", minimum=1)
    history_length = _integer(history_days, "history_days", minimum=1)
    first_day = split.train_end + 1
    last_day = split.train_end + split.horizon
    history_start = first_day - history_length
    if history_start < split.first_day:
        raise ValueError("history_days exceeds the available training period")

    series = long.loc[long["id"].eq(series_id), ["d", "date", "sales"]].copy()
    series["day"] = series["d"].str.slice(2).astype(int)
    series = series.loc[series["day"].between(history_start, last_day)].sort_values("day")
    if series["day"].tolist() != list(range(history_start, last_day + 1)):
        raise ValueError("Selected series needs exactly one row per history and validation day")
    series["date"] = pd.to_datetime(series["date"])
    date_steps = np.diff(series["date"].to_numpy(dtype="datetime64[ns]"))
    if not (date_steps == np.timedelta64(1, "D")).all():
        raise ValueError("Selected dates must be consecutive calendar days")
    sales = series["sales"].to_numpy(dtype=float)
    if not np.isfinite(sales).all() or (sales < 0).any() or (sales != np.floor(sales)).any():
        raise ValueError("Sales must be finite nonnegative whole-unit quantities")
    series["sales"] = series["sales"].astype(int)
    series = series.set_index("day")
    closed = closed_days(series["date"], split)
    history = series.loc[history_start:split.train_end]
    open_history = history.loc[~closed.loc[history.index], "sales"]
    if open_history.empty:
        raise ValueError("Need at least one open history day")
    # Avoid treating pre-launch days as zero demand.
    prior_sales = long.loc[long["id"].eq(series_id), ["d", "sales"]]
    prior_days = prior_sales["d"].str.slice(2).astype(int)
    live_before_history = (prior_days.le(history_start) & prior_sales["sales"].gt(0)).any()
    if not live_before_history:
        raise ValueError("Selected series must have started selling by the history start")

    history_mean = float(open_history.mean())
    target_stock = math.ceil(history_mean * (lead_time + 1))

    def replenish(state):
        if closed.loc[state.day]:
            return 0
        return max(target_stock - state.inventory_position, 0)

    window = series.loc[first_day:last_day]
    demand = window["sales"].where(~closed.loc[window.index], 0)
    result = simulate_inventory(
        demand, initial_stock=target_stock, lead_time_days=lead_time,
        start_day=first_day, policy=replenish,
    )
    daily = result.daily.assign(
        id=series_id,
        date=window["date"].to_numpy(),
        observed_sales=window["sales"].to_numpy(),
        is_closed=closed.loc[window.index].to_numpy(),
    )
    info = {
        "series_id": series_id,
        "first_day": first_day,
        "last_day": last_day,
        "start_date": window["date"].iloc[0].strftime("%Y-%m-%d"),
        "end_date": window["date"].iloc[-1].strftime("%Y-%m-%d"),
        "history_start": history_start,
        "history_end": split.train_end,
        "history_days": history_length,
        "open_history_days": len(open_history),
        "history_mean": history_mean,
        "lead_time_days": lead_time,
        "policy": "order_up_to_fixed_target",
        "target_rule": "ceil(open_history_mean * (lead_time_days + 1))",
        "target_stock": target_stock,
        "initial_stock": target_stock,
        "initial_orders": {},
        "shortage_mode": "lost_sales",
        "closed_day_rule": "zero demand and no orders; receipts and calendar time continue",
        "closed_days": int(daily["is_closed"].sum()),
        "total_demand": int(daily["demand"].sum()),
        "total_fulfilled": int(daily["fulfilled"].sum()),
        "total_lost_sales": int(daily["lost_sales"].sum()),
        "ending_stock": int(daily["closing_stock"].iloc[-1]),
        "total_ordered": int(daily["order_placed"].sum()),
        "pending_orders": result.pending_orders,
        "assumptions": "M5 sales represent demand; lead time, target rule and initial state are illustrative",
    }
    return SimulationResult(daily, result.pending_orders), info


def write_demo_outputs(result: SimulationResult, info: dict, output_root: Path) -> Path:
    """Write a separate directory per run; write the replay record last."""
    output_dir = Path(output_root) / uuid4().hex
    output_dir.mkdir(parents=True)
    daily = result.daily
    x = mdates.date2num(daily["date"].to_numpy())
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, layout="constrained")
        fig.suptitle(
            f"Mô phỏng M5 · {info['series_id']}\n"
            f"{info['start_date']} – {info['end_date']} · L = {info['lead_time_days']} ngày · "
            f"Mức đặt tới = {info['target_stock']} · Tồn đầu = {info['initial_stock']}\n"
            "L, quy tắc mức đặt và tồn đầu là giả định minh họa", fontsize=12,
        )
        axes[0].step(x, daily["closing_stock"], where="mid", color="#2364AA",
                     linewidth=1.8, label="Tồn cuối ngày")
        axes[0].step(x, daily["on_order"], where="mid", color="#6B7280",
                     linestyle="--", linewidth=1.4, label="Hàng đang về, sau đặt hàng")
        axes[0].set_ylabel("Đơn vị hàng")
        axes[1].bar(x, daily["fulfilled"], width=0.7, color="#3F8C70", label="Đáp ứng")
        axes[1].bar(x, daily["lost_sales"], bottom=daily["fulfilled"], width=0.7,
                    color="#D97934", label="Bán mất")
        axes[1].set_ylabel("Nhu cầu (đơn vị/ngày)")
        axes[2].bar(x - 0.18, daily["received"], width=0.35,
                    color="#2364AA", label="Nhận đầu ngày")
        axes[2].bar(x + 0.18, daily["order_placed"], width=0.35,
                    color="#8A63AD", label="Đặt cuối ngày")
        axes[2].set_ylabel("Đơn vị/ngày")
        axes[2].set_xlabel("Ngày")
        for ax in axes:
            ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False)
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.grid(axis="y", alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
        axes[2].xaxis_date()
        axes[2].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        fig.savefig(output_dir / "inventory.png", dpi=160)
        plt.close(fig)
    daily.to_csv(output_dir / "daily.csv", index=False)
    (output_dir / "run_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with args.config.open("rb") as stream:
        settings = tomllib.load(stream)["demo"]
    long = pd.read_parquet(LONG_PATH, filters=[("id", "=", settings["series_id"])],
                           columns=["id", "d", "date", "sales"])
    result, info = simulate_demo(long, load_config(), **settings)
    info["source_data"] = str(LONG_PATH.relative_to(REPO_ROOT))
    info["demo_config"] = settings
    output_dir = write_demo_outputs(result, info, args.output_dir)
    print(f"Series: {info['series_id']}")
    print(f"Validation window: d_{info['first_day']}..d_{info['last_day']}")
    print(f"History mean: {info['history_mean']:.6f} units/open day")
    print(f"Illustrative assumptions: L={info['lead_time_days']}, "
          f"target={info['target_stock']}, initial stock={info['initial_stock']}")
    print(f"Demand={info['total_demand']}, fulfilled={info['total_fulfilled']}, "
          f"lost sales={info['total_lost_sales']}, ending stock={info['ending_stock']}")
    print(f"Pending orders (arrival day -> quantity): {result.pending_orders}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
