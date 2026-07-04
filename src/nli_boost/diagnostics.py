"""Error decomposition: WHERE is accuracy being lost, and is anything reward-hacked?

Each deficiency has a distinct signature and a distinct fix:
- coverage gap   no hypothesis separates a confused pair       -> fix: generation
- redundancy     effective rank << feature count               -> fix: dedup/selection
- fit gaps       train >> CV (variance) / val vs test drift    -> fix: regularization / bigger val
- label noise    confident errors, likely mislabels            -> fix: none; caps expectations
- artifacts      hypothesis scores correlated with text LENGTH -> reward-hacking flag
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from .cache import ScoreCache
from .config import RunConfig
from .costs import CostTracker
from .data import load
from .encoder import EntailmentScorer
from .head import cv_selected_head


def _effective_rank(x: np.ndarray) -> float:
    """Participation ratio of the covariance spectrum: independent signals paid for."""
    s = np.linalg.svd(x - x.mean(axis=0), compute_uv=False) ** 2
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def _pair_coverage(
    x: np.ndarray, y: np.ndarray, m: int, pairs: list[tuple[int, int]], pool: list[str], names: list[str]
) -> list[dict]:
    """Best single-hypothesis separation available for each confused class pair."""
    out = []
    for a, b in pairs:
        mask = (y == a) | (y == b)
        yy = (y[mask] == a).astype(int)
        best_auc, best_h = 0.5, None
        for j in range(m):
            for col in (x[mask, j], x[mask, m + j]):
                if np.std(col) < 1e-9:
                    continue
                auc = roc_auc_score(yy, col)
                auc = max(auc, 1 - auc)
                if auc > best_auc:
                    best_auc, best_h = auc, pool[j]
        out.append(
            {
                "pair": f"{names[a]} vs {names[b]}",
                "best_separator_auc": round(float(best_auc), 3),
                "best_hypothesis": best_h,
                "verdict": (
                    "COVERAGE GAP — no hypothesis separates this pair" if best_auc < 0.75 else "covered"
                ),
            }
        )
    return out


def _length_flags(x: np.ndarray, texts: list[str], pool: list[str]) -> list[dict]:
    """The classic NLI reward-hacking channel: scores that track premise length."""
    lengths = np.array([len(t) for t in texts], dtype=float)
    flags = []
    for j, h in enumerate(pool):
        col = x[:, j]
        if np.std(col) < 1e-9 or np.std(lengths) < 1e-9:
            continue
        r = float(np.corrcoef(col, lengths)[0, 1])
        if abs(r) > 0.5:
            flags.append({"hypothesis": h, "length_corr": round(r, 3)})
    return flags


def diagnose_run(run_dir: Path) -> dict:
    cfg = RunConfig.from_yaml(run_dir / "config.yaml")
    bundle = load(cfg.data, cfg.seed)
    scorer = EntailmentScorer(cfg.encoder, ScoreCache(cfg.cache_dir / "nli_scores.sqlite"), CostTracker())
    pool = json.loads((run_dir / "model.json").read_text())["hypotheses"]
    m = len(pool)
    names = bundle.class_names

    xtr = scorer.features(bundle.train_texts, pool)
    xva = scorer.features(bundle.val_texts, pool)
    xte = scorer.features(bundle.test_texts, pool)
    head, _, cv_acc = cv_selected_head(xtr, bundle.y_train, cfg.seed)

    accs = {
        "train": float(head.score(xtr, bundle.y_train)),
        "cv_train": cv_acc,
        "val": float(head.score(xva, bundle.y_val)),
        "test": float(head.score(xte, bundle.y_test)),
    }

    pred = head.predict(xte)
    proba = head.predict_proba(xte)
    prec, rec, f1s, support = precision_recall_fscore_support(
        bundle.y_test, pred, labels=range(len(names)), zero_division=0
    )
    confusions = Counter((int(t), int(p)) for t, p in zip(bundle.y_test, pred) if t != p)
    confident_errors = [
        {
            "text": bundle.test_texts[i][:140],
            "true": names[bundle.y_test[i]],
            "pred": names[pred[i]],
            "confidence": round(float(proba[i].max()), 3),
        }
        for i in np.flatnonzero(pred != bundle.y_test)
        if proba[i].max() > 0.9
    ]

    diag = {
        "accuracy": {k: round(v, 4) for k, v in accs.items()},
        "fit_gaps": {
            "train_minus_cv": round(accs["train"] - accs["cv_train"], 4),
            "val_minus_test": round(accs["val"] - accs["test"], 4),
        },
        "per_class": [
            {
                "class": names[c],
                "precision": round(float(prec[c]), 3),
                "recall": round(float(rec[c]), 3),
                "f1": round(float(f1s[c]), 3),
                "support": int(support[c]),
            }
            for c in range(len(names))
        ],
        "top_confusions": [
            {"pair": f"{names[t]} -> {names[p]}", "count": c} for (t, p), c in confusions.most_common(6)
        ],
        "confusion_coverage": _pair_coverage(
            xtr, bundle.y_train, m, [pr for pr, _ in confusions.most_common(6)], pool, names
        ),
        "redundancy": {
            "n_hypotheses": m,
            "n_features": int(xtr.shape[1]),
            "effective_rank": round(_effective_rank(xtr), 1),
        },
        "length_artifact_flags": _length_flags(xtr, bundle.train_texts, pool),
        "suspected_label_noise": {
            "confident_errors": len(confident_errors),
            "examples": confident_errors[:8],
        },
    }
    (run_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2))
    return diag
