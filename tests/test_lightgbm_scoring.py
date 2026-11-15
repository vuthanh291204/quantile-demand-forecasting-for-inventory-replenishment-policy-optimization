import pandas as pd
import pytest

from src.evaluation.score_lightgbm import score_window


def make_example():
    wide = pd.DataFrame(
        [
            [1, 2, 1, 2, 1, 2, 3, 4],
            [2, 4, 2, 4, 2, 4, 6, 8],
        ],
        index=pd.Index(["a", "b"], name="id"),
        columns=[f"d_{day}" for day in range(1, 9)],
    )
    forecasts = pd.DataFrame({
        "id": ["a", "a", "b", "b"],
        "day": [7, 8, 7, 8],
        "y_pred": [3, 4, 6, 8],
    })
    closed = pd.Series(False, index=range(1, 9))
    return wide, forecasts, closed


def test_scoring_aligns_by_id_and_day():
    wide, forecasts, closed = make_example()
    shuffled = forecasts.sample(frac=1, random_state=29)

    scores = score_window(wide, shuffled, 6, 2, closed)

    assert scores.eq(0).all().all()


@pytest.mark.parametrize("problem", ["missing", "duplicate"])
def test_scoring_rejects_incomplete_or_duplicate_rows(problem):
    wide, forecasts, closed = make_example()

    if problem == "missing":
        forecasts = forecasts.iloc[:-1]
    else:
        forecasts = pd.concat(
            [forecasts, forecasts.iloc[[0]]],
            ignore_index=True,
        )

    with pytest.raises(ValueError):
        score_window(wide, forecasts, 6, 2, closed)