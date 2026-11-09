"""Time-based train / validation / test split for the M5 subset.

Rules:
- Every series uses the same cut-off days (never split per series).
- Day numbers refer to the M5 column names d_1 ... d_1941.
- The cut-off days live in configs/split.toml, not in the code.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "split.toml"
LONG_PATH = REPO_ROOT / "data" / "processed" / "subset_long.parquet"
WIDE_PATH = REPO_ROOT / "data" / "processed" / "sales_subset_wide.parquet"


@dataclass(frozen=True)
class SplitConfig:
    first_day: int
    train_end: int
    valid_end: int
    last_day: int
    horizon: int
    closed_month: int = 12
    closed_day: int = 25

    def __post_init__(self) -> None:
        if not (self.first_day <= self.train_end < self.valid_end < self.last_day):
            raise ValueError("Need first_day <= train_end < valid_end < last_day")
        valid_len = self.valid_end - self.train_end
        test_len = self.last_day - self.valid_end
        for name, length in (("validation", valid_len), ("test", test_len)):
            if length % self.horizon != 0:
                raise ValueError(
                    f"{name} has {length} days, not a multiple of horizon {self.horizon}"
                )

    @property
    def ranges(self) -> dict[str, tuple[int, int]]:
        """Inclusive (first_day, last_day) of each split."""
        return {
            "train": (self.first_day, self.train_end),
            "valid": (self.train_end + 1, self.valid_end),
            "test": (self.valid_end + 1, self.last_day),
        }


def load_config(path: Path = DEFAULT_CONFIG) -> SplitConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return SplitConfig(**raw["split"], **raw["calendar"])


def day_columns(start: int, end: int) -> list[str]:
    """['d_<start>', ..., 'd_<end>'] (both ends included)."""
    return [f"d_{i}" for i in range(start, end + 1)]


def make_windows(start: int, end: int, horizon: int) -> list[tuple[int, int]]:
    """Cut start..end into consecutive windows of `horizon` days."""
    length = end - start + 1
    if length % horizon != 0:
        raise ValueError(f"{length} days is not a multiple of horizon {horizon}")
    return [(s, s + horizon - 1) for s in range(start, end + 1, horizon)]


def split_wide(wide: pd.DataFrame, cfg: SplitConfig) -> dict[str, pd.DataFrame]:
    """Cut the wide table (one row per series, columns d_1 ...) into 3 splits.

    Each result has one row per series (index = id) and only day columns.
    """
    wide = wide.set_index("id")
    return {
        name: wide[day_columns(first, last)]
        for name, (first, last) in cfg.ranges.items()
    }


def load_day_dates(long_path: Path = LONG_PATH) -> pd.Series:
    """Map day number (1, 2, ...) to its calendar date."""
    cal = pd.read_parquet(long_path, columns=["d", "date"]).drop_duplicates()
    cal["day"] = cal["d"].str.slice(2).astype(int)
    return cal.set_index("day")["date"].sort_index()


def closed_days(dates: pd.Series, cfg: SplitConfig) -> pd.Series:
    """True on days when stores are assumed closed (Christmas). [Assumption]"""
    return (dates.dt.month == cfg.closed_month) & (dates.dt.day == cfg.closed_day)


def main() -> None:
    """Print a summary so the split can be checked by eye."""
    cfg = load_config()
    dates = load_day_dates()
    closed = closed_days(dates, cfg)

    print("== Split ==")
    for name, (first, last) in cfg.ranges.items():
        d0, d1 = dates[first], dates[last]
        print(
            f"{name:5s} d_{first}..d_{last}  {d0:%Y-%m-%d} ({d0:%A}) -> "
            f"{d1:%Y-%m-%d} ({d1:%A})  {last - first + 1} days"
        )

    print(f"\n== Windows ({cfg.horizon} days each) ==")
    for name in ("valid", "test"):
        first, last = cfg.ranges[name]
        for i, (s, e) in enumerate(make_windows(first, last, cfg.horizon), 1):
            print(f"{name} window {i}: d_{s}..d_{e}  {dates[s]:%Y-%m-%d} -> {dates[e]:%Y-%m-%d}")

    def split_of(day: int) -> str:
        for name, (a, b) in cfg.ranges.items():
            if a <= day <= b:
                return name
        return "?"

    print("\n== Closed days (25/12) ==")
    wide = pd.read_parquet(WIDE_PATH)
    for day in dates.index[closed.to_numpy()]:
        total = int(wide[f"d_{day}"].sum())
        print(f"d_{day}  {dates[day]:%Y-%m-%d}  split={split_of(day)}  total sales, all series = {total}")

    print("\n== Checks ==")
    print("number of days in calendar:", len(dates), "(expected", cfg.last_day, ")")
    base = pd.Timestamp("2011-01-29")
    expected = base + pd.to_timedelta(dates.index - 1, unit="D")
    print("dates match 2011-01-29 + (day - 1):", bool((dates.to_numpy() == expected.to_numpy()).all()))

    series = pd.read_parquet(LONG_PATH, columns=["id", "first_date"]).drop_duplicates("id")
    valid_start = dates[cfg.ranges["valid"][0]]
    train_last = dates[cfg.train_end]
    n_not_live = int((series["first_date"] > valid_start).sum())
    history = (train_last - series["first_date"]).dt.days + 1
    print(f"series not selling yet at start of validation: {n_not_live} of {len(series)}")
    print(f"days of history inside train (from first sale): min={history.min()}, median={history.median():.0f}")


if __name__ == "__main__":
    main()