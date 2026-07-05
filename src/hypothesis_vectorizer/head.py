"""STAGE 3 — the classifier head, chosen honestly.

Head family AND regularization are selected by CV on train only, then refit on
full train. This is the systemic variance fix (+2 pts measured) and the honest
reporting protocol: picking the best of several heads by their test scores
inflated results +2.2 pts. There is exactly one head in the output: pool_cv.
"""

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import cross_val_score

# thread-based n_jobs only (sklearn tree building releases the GIL); process
# workers forked from a CUDA-holding parent have crashed this machine before
_GRID = [
    dict(kind="rf", min_samples_leaf=msl, max_features=mf) for msl in (2, 5, 10) for mf in (0.3, 0.6, 1.0)
] + [dict(kind="hgb", learning_rate=lr, l2_regularization=l2) for lr in (0.06, 0.12) for l2 in (0.01, 0.3)]


def _build(params: dict, seed: int):
    p = dict(params)
    if p.pop("kind") == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=4, **p)
    return HistGradientBoostingClassifier(max_iter=200, random_state=seed, **p)


def cv_selected_head(x: np.ndarray, y: np.ndarray, seed: int, folds: int = 4):
    """Best (model family, regularization) by CV-on-train; refit on full train.

    Returns (fitted head, chosen params, cv accuracy).
    """
    scored = [
        (float(cross_val_score(_build(p, seed), x, y, cv=folds).mean()), i) for i, p in enumerate(_GRID)
    ]
    cv_acc, best_i = max(scored)
    head = clone(_build(_GRID[best_i], seed))
    head.fit(x, y)
    return head, _GRID[best_i], cv_acc


def fit_head(params: dict, x: np.ndarray, y: np.ndarray, seed: int):
    """Fit exactly the head a run already selected (skips re-running the CV grid).

    cv_selected_head is deterministic, so refitting the saved params reproduces
    the reported head at a fraction of the cost — used by post-hoc tools."""
    head = _build(params, seed)
    head.fit(x, y)
    return head


def evaluate(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> dict:
    pred = proba.argmax(axis=1)
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "macro_f1": round(float(f1_score(y_true, pred, average="macro")), 4),
        "logloss": round(float(log_loss(y_true, proba, labels=list(range(n_classes)))), 4),
    }
