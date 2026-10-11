"""Reproduce the original MSc mistake and show how much it inflates the score.

Original approach (coursework, 2022): scale and SMOTE the WHOLE dataset, then
call train_test_split. Synthetic rows built from training neighbours land in
the test set, and the test set is artificially 50/50.

Compare with the correct approach: split first, resample inside the pipeline
on training data only, evaluate on the untouched natural distribution.

Usage: python src/leakage_demo.py
"""
import json
from pathlib import Path

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import average_precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from common import RANDOM_STATE, load_data, make_preprocessor, split

OUT = Path(__file__).resolve().parents[1] / "results" / "leakage_demo.json"


def leaky(X, y):
    pre = make_preprocessor(X)
    Xt = pre.fit_transform(X)  # scaler sees test rows too
    Xr, yr = SMOTE(random_state=RANDOM_STATE).fit_resample(Xt, y)  # before split
    Xtr, Xte, ytr, yte = train_test_split(
        Xr, yr, test_size=0.20, random_state=0
    )
    model = DecisionTreeClassifier(max_depth=None, random_state=RANDOM_STATE)
    model.fit(Xtr, ytr)
    p = model.predict(Xte)
    return {
        "test_positive_rate": float(yte.mean()),
        "recall": float(recall_score(yte, p)),
        "f1": float(f1_score(yte, p)),
        "pr_auc": float(average_precision_score(yte, model.predict_proba(Xte)[:, 1])),
    }


def correct(X, y):
    Xtr, Xte, ytr, yte = split(X, y)
    pipe = ImbPipeline(
        [
            ("pre", make_preprocessor(X)),
            ("smote", SMOTE(random_state=RANDOM_STATE)),  # fit on train only
            ("clf", DecisionTreeClassifier(max_depth=None, random_state=RANDOM_STATE)),
        ]
    )
    pipe.fit(Xtr, ytr)
    p = pipe.predict(Xte)
    return {
        "test_positive_rate": float(yte.mean()),
        "recall": float(recall_score(yte, p)),
        "f1": float(f1_score(yte, p)),
        "pr_auc": float(average_precision_score(yte, pipe.predict_proba(Xte)[:, 1])),
    }


if __name__ == "__main__":
    X, y = load_data()
    result = {"leaky": leaky(X, y), "correct": correct(X, y)}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
