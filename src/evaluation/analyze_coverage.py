"""Review empirical coverage using saved validation and test results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.split import REPO_ROOT, day_columns, load_config, load_day_dates
from src.evaluation.plot_quantiles import training_volume_stats


def coverage_review(summary, per_series, per_window, forecasts):
    primary = per_series.loc[per_series["variant"].eq("clipped")].copy()
    windows = per_window.loc[per_window["variant"].eq("clipped")].copy()
    metrics = summary.loc[summary["variant"].eq("clipped")].set_index("metric")
    counts = windows["interval90_coverage"] * windows["n_evaluated"]
    if not np.allclose(counts, np.rint(counts), rtol=0, atol=1e-8):
        raise ValueError("Saved coverage/count pairs must describe integer event counts")
    windows["n_covered"] = np.rint(counts).astype(int)
    coverage = primary["interval90_coverage"]
    overall = {
        "n_series": len(primary), "n_forecasts": len(forecasts),
        "n_evaluated": int(windows["n_evaluated"].sum()),
        "n_covered": int(windows["n_covered"].sum()),
        "coverage_mean": float(metrics.loc["interval90_coverage", "mean"]),
        "coverage_median": float(metrics.loc["interval90_coverage", "median"]),
        "coverage_pooled": float(windows["n_covered"].sum() / windows["n_evaluated"].sum()),
        "coverage_gap_pp": float(100 * (metrics.loc["interval90_coverage", "mean"] - 0.9)),
        "n_series_below_90": int(coverage.lt(0.9).sum()),
        "coverage_min": float(coverage.min()), "coverage_max": float(coverage.max()),
        "width_mean": float(metrics.loc["interval90_mean_width_ordered", "mean"]),
        "crossing_mean": float(metrics.loc["crossing_rate", "mean"]),
        "q05_zero_forecasts": int(forecasts["q05"].eq(0).sum()),
        "n_series_q05_le_below_05": int(primary["coverage_le_q05"].lt(0.05).sum()),
    }
    by_window = windows.groupby("window", sort=True).agg(
        coverage_mean=("interval90_coverage", "mean"),
        width_mean=("interval90_mean_width_ordered", "mean"),
        crossing_mean=("crossing_rate", "mean"),
        n_evaluated=("n_evaluated", "sum"), n_covered=("n_covered", "sum"),
    ).reset_index()
    quantiles = pd.DataFrame([
        {"quantile": alpha,
         "cdf_lt": metrics.loc[f"coverage_lt_q{name}", "mean"],
         "cdf_le": metrics.loc[f"coverage_le_q{name}", "mean"]}
        for alpha, name in ((0.05, "05"), (0.50, "50"), (0.90, "90"), (0.95, "95"))
    ])
    return overall, by_window, quantiles, primary


def write_coverage_figure(series, windows, output):
    colors = {"valid": "#2463B4", "test": "#C16C2D"}
    labels = {"valid": "Validation", "test": "Test"}
    lower = min(0.9, np.floor(series["interval90_coverage"].min() * 20) / 20)
    bins = np.linspace(lower, 1, 26) * 100
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
        for split in ("valid", "test"):
            data = series.loc[series["split"].eq(split)]
            axes[0].hist(data["interval90_coverage"] * 100, bins=bins, histtype="step",
                         linewidth=1.7, color=colors[split], label=labels[split])
            data = windows.loc[windows["split"].eq(split)]
            axes[1].plot(data["window"], data["coverage_mean"] * 100, marker="o",
                         color=colors[split], label=labels[split])
        axes[0].axvline(90, color="#555555", linestyle="--", label="Mức danh nghĩa 90%")
        axes[1].axhline(90, color="#555555", linestyle="--", label="Mức danh nghĩa 90%")
        axes[0].set(title="Độ phủ từng chuỗi sau khi gộp 6 cửa sổ",
                    xlabel="Độ phủ [q05, q95] (%)", ylabel="Số chuỗi")
        axes[1].set(title="Độ phủ trung bình qua 297 chuỗi ở từng cửa sổ",
                    xlabel="Cửa sổ trong mỗi tập", ylabel="Độ phủ [q05, q95] (%)")
        axes[1].set_xticks(range(1, 7))
        for axis in axes:
            axis.legend(fontsize=8)
            axis.grid(axis="y", alpha=0.2)
            axis.spines[["top", "right"]].set_visible(False)
        fig.supxlabel("Nguồn M5 · Dự báo clipped đã lưu · Validation và test thuộc hai giai đoạn khác nhau",
                      fontsize=9)
        fig.savefig(output, dpi=160)
        plt.close(fig)


def run_review(valid_id, test_id, root=REPO_ROOT):
    root = Path(root)
    overall_rows, window_parts, quantile_parts, series_parts, horizon_parts = [], [], [], [], []
    hashes = {}
    for split, run_id in (("valid", valid_id), ("test", test_id)):
        base = root / "reports" / "quantiles" / run_id
        frames = {}
        for name in ("summary", "per_series", "per_window", "per_horizon", "forecasts"):
            path = base / f"{name}_{split}.csv"
            frames[name] = pd.read_csv(path)
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        overall, windows, quantiles, series = coverage_review(
            frames["summary"], frames["per_series"], frames["per_window"], frames["forecasts"]
        )
        overall_rows.append({"split": split, "run_id": run_id, **overall})
        window_parts.append(windows.assign(split=split))
        quantile_parts.append(quantiles.assign(split=split))
        series_parts.append(series.assign(split=split))
        horizons = frames["per_horizon"].query("variant == 'clipped'")
        horizons = horizons.groupby("horizon", sort=True).agg(
            coverage_mean=("interval90_coverage", "mean"),
            width_mean=("interval90_mean_width_ordered", "mean"),
            n_windows=("interval90_coverage", "count"),
            n_evaluated=("n_evaluated", "sum"),
        ).reset_index()
        horizon_parts.append(horizons.assign(split=split))
    overall = pd.DataFrame(overall_rows)
    windows = pd.concat(window_parts, ignore_index=True)
    quantiles = pd.concat(quantile_parts, ignore_index=True)
    series = pd.concat(series_parts, ignore_index=True)
    split_config = load_config(root / "configs" / "split.toml")
    wide_path = root / "data" / "processed" / "sales_subset_wide.parquet"
    long_path = root / "data" / "processed" / "subset_long.parquet"
    wide = pd.read_parquet(wide_path, columns=[
        "id", *day_columns(split_config.first_day, split_config.train_end)
    ]).set_index("id")
    train_stats = training_volume_stats(wide, split_config, load_day_dates(long_path))
    if set(train_stats["id"]) != set(series["id"]):
        raise ValueError("Training volume groups must match the scored series")
    train_stats = train_stats.sort_values(["train_mean", "id"]).reset_index(drop=True)
    train_stats["volume_group"] = pd.qcut(
        np.arange(len(train_stats)), 3, labels=["low", "middle", "high"]
    )
    series = series.merge(train_stats, on="id", validate="many_to_one")
    groups = series.groupby(["split", "volume_group"], observed=True).agg(
        n_series=("id", "size"), coverage_mean=("interval90_coverage", "mean"),
        width_mean=("interval90_mean_width_ordered", "mean"),
        train_mean_min=("train_mean", "min"), train_mean_max=("train_mean", "max"),
    ).reset_index()
    for path in (root / "configs" / "split.toml", wide_path, long_path):
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    output = root / "reports" / "quantiles" / "coverage" / test_id
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("overall.csv", overall), ("per_window.csv", windows),
                        ("per_quantile.csv", quantiles), ("per_series.csv", series),
                        ("per_volume_group.csv", groups),
                        ("per_horizon.csv", pd.concat(horizon_parts, ignore_index=True)),
                        ("training_volume_groups.csv", train_stats)):
        frame.to_csv(output / name, index=False)
    write_coverage_figure(series, windows, output / "coverage_diagnostics.png")
    (output / "provenance.json").write_text(json.dumps({
        "validation_run_id": valid_id, "test_run_id": test_id, "source_sha256": hashes,
        "nominal_coverage": 0.9,
        "aggregation": "saved_per_series_mean_of_window_scores_then_equal_weight_mean_over_series",
        "below_90_rule": "observed_series_coverage < 0.9",
        "volume_groups": "three equal-size groups sorted by train_mean then id; descriptive choice",
        "horizon_aggregation": "equal weight over windows with defined coverage; save their counts",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in overall.itertuples():
        print(f"{row.split}: mean coverage={row.coverage_mean:.6%}; "
              f"gap={row.coverage_gap_pp:.6f} percentage points; "
              f"series below 90%={row.n_series_below_90}/{row.n_series}; "
              f"zero q05={row.q05_zero_forecasts}/{row.n_forecasts}")
    print("Saved coverage review:", output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-run-id", required=True)
    parser.add_argument("--test-run-id", required=True)
    args = parser.parse_args()
    run_review(args.validation_run_id, args.test_run_id)


if __name__ == "__main__":
    main()
