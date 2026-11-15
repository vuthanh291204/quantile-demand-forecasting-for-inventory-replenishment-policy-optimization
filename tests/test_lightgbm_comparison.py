import pandas as pd
import pytest

from src.evaluation.compare_lightgbm import aggregate_windows


def example_scores():
    scores = pd.DataFrame({
        "window": [1, 2, 1, 2],
        "id": ["a", "a", "b", "b"],
    })
    for metric in ("mae", "rmse", "smape", "rmsse"):
        scores[metric] = [1.0, 3.0, 2.0, 4.0]
    return scores


def test_aggregation_averages_windows_within_each_series():
    scores = example_scores().sample(
        frac=1, random_state=29
    )
    result = aggregate_windows(scores, ["a", "b"], 2)

    assert result.loc["a", "rmse"] == 2.0
    assert result.loc["b", "rmsse"] == 3.0


@pytest.mark.parametrize("problem", ["missing", "duplicate"])
def test_aggregation_rejects_missing_or_duplicate_windows(problem):
    scores = example_scores()

    if problem == "missing":
        scores = scores.iloc[:-1]
    else:
        scores = pd.concat(
            [scores, scores.iloc[[0]]],
            ignore_index=True,
        )

    with pytest.raises(ValueError):
        aggregate_windows(scores, ["a", "b"], 2)