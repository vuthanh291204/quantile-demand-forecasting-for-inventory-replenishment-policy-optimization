import pandas as pd
import pytest

from src.evaluation.analyze_coverage import coverage_review


def saved_scores():
    rows = [
        {"metric": "interval90_coverage", "mean": 0.75, "median": 0.75},
        {"metric": "interval90_mean_width_ordered", "mean": 2.0, "median": 2.0},
        {"metric": "crossing_rate", "mean": 0.0, "median": 0.0},
    ]
    for name in ("05", "50", "90", "95"):
        for comparison in ("lt", "le"):
            rows.append({"metric": f"coverage_{comparison}_q{name}",
                         "mean": 0.5, "median": 0.5})
    summary = pd.DataFrame(rows).assign(variant="clipped")
    series = pd.DataFrame({"id": ["a"], "variant": ["clipped"],
                           "interval90_coverage": [0.75], "coverage_le_q05": [0.5]})
    windows = pd.DataFrame({
        "window": [1, 2], "variant": "clipped", "n_evaluated": [2, 4],
        "interval90_coverage": [1.0, 0.5], "interval90_mean_width_ordered": 2.0,
        "crossing_rate": 0.0,
    })
    forecasts = pd.DataFrame({"q05": [0.0] * 7})  # One unscored forecast remains saved.
    return summary, series, windows, forecasts


def test_coverage_keeps_macro_and_pooled_denominators_distinct():
    overall, windows, _, _ = coverage_review(*saved_scores())
    assert overall["coverage_mean"] == 0.75
    assert overall["coverage_pooled"] == pytest.approx(4 / 6)
    assert overall["n_covered"] == 4
    assert overall["n_evaluated"] == 6
    assert overall["q05_zero_forecasts"] == overall["n_forecasts"] == 7
    assert windows["n_covered"].tolist() == [2, 2]


def test_coverage_rejects_fractional_reconstructed_event_counts():
    summary, series, windows, forecasts = saved_scores()
    windows.loc[0, "interval90_coverage"] = 0.6
    with pytest.raises(ValueError, match="integer event counts"):
        coverage_review(summary, series, windows, forecasts)
