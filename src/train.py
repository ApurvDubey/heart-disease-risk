"""Train and honestly evaluate heart disease risk models.

Design choices:
- Split first, stratified, on the natural ~8.6% positive rate.
- Imbalance handled INSIDE cross-validation / the pipeline (class weights or
  SMOTE on training folds only), never on the evaluation data.
- Primary metrics are PR-AUC and recall/precision at a chosen threshold, not
  accuracy: predicting "no disease" for everyone scores ~91% accuracy.
- The decision threshold is chosen from out-of-fold predictions on the
  training set, then applied once to the untouched test set.

Usage: python src/train.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from common import RANDOM_STATE, load_data, make_preprocessor, split

RESULTS = Path(__file__).resolve().parents[1] / "results"
TARGET_RECALL = 0.80  # screening use case: catch most true cases


def build_models(X):
    pre = make_preprocessor(X)
    return {
        "baseline_majority": Pipeline(
            [("pre", pre), ("clf", DummyClassifier(strategy="prior"))]
        ),
        "logreg_class_weight": Pipeline(
            [("pre", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]
        ),
        "logreg_smote": ImbPipeline(
            [
                ("pre", pre),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        ),
        "hist_gb_class_weight": Pipeline(
            [
                ("pre", pre),
                ("clf", HistGradientBoostingClassifier(
                    class_weight="balanced", max_depth=4, learning_rate=0.1,
                    random_state=RANDOM_STATE)),
            ]
        ),
    }


def threshold_for_recall(y_true, scores, target_recall):
    """Highest threshold that still achieves the target recall (max precision)."""
    prec, rec, thr = precision_recall_curve(y_true, scores)
    ok = np.where(rec[:-1] >= target_recall)[0]
    return float(thr[ok[-1]]) if len(ok) else float(thr[0])


def main():
    X, y = load_data()
    Xtr, Xte, ytr, yte = split(X, y)
    print(f"train {len(ytr):,} rows ({ytr.mean():.1%} positive) | "
          f"test {len(yte):,} rows ({yte.mean():.1%} positive)")

    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    results, curves = {}, {}
    for name, model in build_models(X).items():
        oof = cross_val_predict(model, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        thr = threshold_for_recall(ytr, oof, TARGET_RECALL)
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        results[name] = {
            "roc_auc": float(roc_auc_score(yte, proba)),
            "pr_auc": float(average_precision_score(yte, proba)),
            "threshold": thr,
            "recall": float(recall_score(yte, pred)),
            "precision": float(precision_score(yte, pred, zero_division=0)),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }
        if not name.startswith("baseline"):  # constant predictor: degenerate curve
            curves[name] = precision_recall_curve(yte, proba)[:2]
        r = results[name]
        print(f"{name:24s} PR-AUC {r['pr_auc']:.3f}  ROC-AUC {r['roc_auc']:.3f}  "
              f"recall {r['recall']:.2f}  precision {r['precision']:.2f}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(
        {"test_positive_rate": float(yte.mean()), "target_recall": TARGET_RECALL,
         "models": results}, indent=2))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, (p, r) in curves.items():
        ax.plot(r, p, label=name)
    ax.axhline(yte.mean(), ls=":", c="grey", label="no-skill (prevalence)")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall on untouched test set")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "pr_curves.png", dpi=150)


if __name__ == "__main__":
    main()
