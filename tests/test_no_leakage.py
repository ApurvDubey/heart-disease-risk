"""Guard against re-introducing the original leakage bug."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import make_preprocessor, split  # noqa: E402
from train import build_models  # noqa: E402


def _toy(n=2000, pos=0.09, seed=0):
    rng = np.random.default_rng(seed)
    y = pd.Series((rng.random(n) < pos).astype(int))
    X = pd.DataFrame({
        "BMI": rng.normal(27, 5, n),
        "Smoking": rng.choice(["Yes", "No"], n),
        "Race": rng.choice(["A", "B", "C"], n),
    })
    return X, y


def test_split_keeps_natural_balance():
    X, y = _toy()
    _, Xte, _, yte = split(X, y)
    assert abs(yte.mean() - y.mean()) < 0.01


def test_resampling_only_inside_pipeline():
    X, _ = _toy()
    for name, model in build_models(X).items():
        steps = [s for s, _ in model.steps]
        if "smote" in steps:
            # SMOTE must be a pipeline step (fit on training folds only),
            # which is only possible when the estimator is an imblearn Pipeline.
            assert type(model).__module__.startswith("imblearn")


def test_preprocessor_unfitted_until_pipeline_fit():
    X, _ = _toy()
    pre = make_preprocessor(X)
    assert not hasattr(pre, "transformers_")
