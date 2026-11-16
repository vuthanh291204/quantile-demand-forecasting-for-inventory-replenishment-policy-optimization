import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from PIL import Image

from src.data.split import SplitConfig
from src.evaluation.plot_quantiles import (
    build_plot_panel,
    select_plot_series,
    training_volume_stats,
    write_figures,
)


def test_selection_uses_training_sales_and_labels_the_validation_diagnostic():
    split = SplitConfig(1, 60, 64, 68, 4)
    dates = pd.Series(pd.date_range("2015-11-01", periods=68), index=range(1, 69))
    ids = [f"item_{i}" for i in range(10)]
    wide = pd.DataFrame(
        {f"d_{day}": np.arange(1, 11) for day in range(1, 69)}, index=ids
    )
    scores = pd.DataFrame({
        "id": ids, "variant": "clipped", "interval90_coverage": [0.9] * 9 + [0.4],
        "interval90_mean_width_ordered": 3.0, "crossing_rate": 0.0,
    })
    before = training_volume_stats(wide, split, dates)
    wide.loc[:, [f"d_{day}" for day in range(61, 69)]] = 999999
    assert_frame_equal(before, training_volume_stats(wide, split, dates))
    selected = select_plot_series(before, scores)
    assert selected["id"].tolist() == ["item_1", "item_4", "item_8", "item_9"]
    assert selected["selection_basis"].tolist() == [
        "training_period_only", "training_period_only", "training_period_only",
        "posthoc_validation_diagnostic",
    ]


def test_rendering_joins_actuals_by_key_and_keeps_saved_crossing_values(tmp_path):
    split = SplitConfig(1, 60, 64, 68, 4)
    dates = pd.Series(pd.date_range("2015-11-01", periods=68), index=range(1, 69))
    wide = pd.DataFrame([[2, 0, 4, 1]], index=["a"], columns=[f"d_{d}" for d in range(61, 65)])
    forecasts = pd.DataFrame({
        "id": "a", "day": range(61, 65), "date": dates.loc[61:64].to_numpy(),
        "window": 1, "q05": 0.0, "q50": [1.0, 1.0, 2.0, 1.0],
        "q90": [3.0, 2.0, 6.0, 2.0], "q95": [4.0, 3.0, 5.0, 3.0],
    }).sample(frac=1, random_state=29)
    original = forecasts.copy(deep=True)
    selected = pd.DataFrame({
        "id": ["a"], "role": ["train_low"], "interval90_coverage": [1.0],
        "interval90_mean_width_ordered": [3.75], "crossing_rate": [0.25],
    })
    panel = build_plot_panel(wide, forecasts, selected, split, dates)
    assert panel["sales"].tolist() == [2, 0, 4, 1]
    assert panel.loc[panel["day"].eq(63), "q90"].item() == 6
    assert panel.loc[panel["day"].eq(63), "q95"].item() == 5
    files = write_figures(panel, selected, tmp_path)
    with Image.open(files[0]) as figure:
        assert figure.width >= 1500 and figure.height >= 800
    saved = pd.read_csv(tmp_path / "plot_data.csv")
    assert saved.loc[saved["day"].eq(63), "q90"].item() == 6
    assert_frame_equal(forecasts, original)
