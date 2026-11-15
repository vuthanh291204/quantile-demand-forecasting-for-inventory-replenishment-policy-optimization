import numpy as np
import pandas as pd
import pytest

from src.evaluation.feature_importance import load_saved_booster, summarize_importance


def example():
    return pd.DataFrame({"window": [1, 1, 2, 2], "feature": ["a", "b", "a", "b"],
                         "gain": [9.0, 1.0, 100.0, 900.0], "split": [2, 1, 3, 5]})


def test_normalizes_each_model_before_averaging():
    normalized, summary = summarize_importance(example())
    assert normalized.groupby("window")["gain_pct"].sum().tolist() == [100, 100]
    assert summary.loc["a", "gain_pct_mean"] == pytest.approx(50)
    assert summary.loc["b", "gain_pct_mean"] == pytest.approx(50)


def test_loads_windows_text_artifact_without_changing_predictions(tmp_path):
    X = pd.DataFrame({"x": np.arange(20, dtype=float)})
    model = lgb.LGBMRegressor(n_estimators=2, min_child_samples=1, n_jobs=1, verbosity=-1)
    model.fit(X, np.arange(20, dtype=float))
    path = tmp_path / "model.txt"
    path.write_bytes(model.booster_.model_to_string().replace("\n", "\r\n").encode("utf-8"))
    restored = load_saved_booster(path)
    np.testing.assert_allclose(model.predict(X), restored.predict(X, num_threads=1))
    np.testing.assert_allclose(model.booster_.feature_importance("gain"),
                               restored.feature_importance("gain"))


@pytest.mark.parametrize("problem", ["missing", "duplicate", "zero", "negative", "nan"])
def test_rejects_invalid_saved_importance(problem):
    raw = example()
    if problem == "missing":
        raw = raw.iloc[:-1]
    elif problem == "duplicate":
        raw = pd.concat([raw, raw.iloc[[0]]])
    elif problem == "zero":
        raw.loc[raw["window"] == 1, "gain"] = 0
    elif problem == "negative":
        raw.loc[0, "gain"] = -1
    else:
        raw.loc[0, "gain"] = np.nan
    with pytest.raises(ValueError):
        summarize_importance(raw)
import lightgbm as lgb
