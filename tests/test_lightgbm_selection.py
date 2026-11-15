import numpy as np
import pandas as pd
import pytest

from src.forecast.select_lightgbm import fit_by_checkpoints


def test_selection_continues_training_and_keeps_best_model():
    X = pd.DataFrame({"x": np.arange(20, dtype=float)})
    y = pd.Series([0.0] * 10 + [1.0] * 10)

    params = {
        "objective": "regression",
        "n_estimators": 10,
        "learning_rate": 0.1,
        "num_leaves": 2,
        "min_child_samples": 1,
        "random_state": 29,
        "n_jobs": 1,
        "verbosity": -1,
    }
    settings = {
        "check_every": 2,
        "patience_checks": 2,
    }

    fake_scores = {
        2: 0.90,
        4: 0.80,
        6: 0.81,
        8: 0.82,
        10: 0.10,
    }
    seen = []

    def evaluate(model):
        iteration = model.booster_.current_iteration()
        seen.append(iteration)
        return fake_scores[iteration]

    best, history, reason = fit_by_checkpoints(
        X, y, params, settings, evaluate
    )

    assert seen == [2, 4, 6, 8]
    assert reason == "patience"
    assert best.booster_.current_iteration() == 4
    assert history["iteration"].tolist() == seen
    assert history["rmsse"].min() == pytest.approx(0.80)