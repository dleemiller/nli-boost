"""Composite pool-quality reward for instruction optimization (GEPA).

Scores a GENERATED pool (pre-evolution) on a frozen context. Every term encodes
something the experiments taught us, and the whole thing is honest (train-CV
only, test never touched):

- cv_skill   noise-AVERAGED held-out CV accuracy, normalized above majority.
             Averaging over fold-seeds is load-bearing: a single 4-fold CV wobbles
             ~0.003 from HistGBM thread-nondeterminism (measured 2026-07-04), so an
             un-averaged reward lets the optimizer hack jitter. This is the primary term.
- diversity  effective rank of the entailment columns / #hypotheses. Independent
             separating directions are what GBDT converts into accuracy; this is the
             explicit anti-collapse pressure (a label-paraphrase pool scores ~0.13,
             a diverse pool ~0.40 on TREC).
- anti_hack  1 - (fraction of hypotheses that track text length + fraction that are
             near-constant/vacuous). Penalizes the two NLI reward-hacking channels
             from diagnostics.py directly.
- judge      optional LM semantic score (passed in), blind to the numbers above.

Per-dataset composite = weighted sum. Across datasets, aggregate with a GEOMETRIC
mean so a candidate that tanks any single dataset craters (cross-dataset
generalization pressure — the failure mode that shelved the first GEPA attempt).
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from threadpoolctl import threadpool_limits


@dataclass
class RewardConfig:
    cv_seeds: int = 3  # fold-seeds averaged to beat the ~0.003 CV noise floor
    cv_folds: int = 4
    cpu_threads: int = 4  # cap HGB's per-core OpenMP pool so it stays civil on a shared box
    # Live, continuous terms only (anti_hack was always 1.0 -> now a penalty multiplier, not a
    # weighted term). More discriminative terms => the reward separates near-identical pools
    # instead of collapsing to a few values.
    weights: dict = field(
        default_factory=lambda: {
            "cv_skill": 0.4,  # downstream CV accuracy above majority (primary target proxy)
            "min_coverage": 0.2,  # worst-covered class's best single-hypothesis separation
            "mean_coverage": 0.1,  # average per-class separation
            "diversity": 0.15,  # effective rank / #hypotheses (independent directions)
            "judge": 0.15,  # multi-dimensional LM rubric (averaged sub-scores)
        }
    )
    length_corr_thresh: float = 0.5
    const_std_thresh: float = 0.02


def _class_coverage(x_entail: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Per-class best single-hypothesis separation (one-vs-rest AUC, rescaled to [0,1]).
    Continuous and discriminative where CV accuracy saturates; min rewards covering the
    WORST class, mean rewards overall separability. Returns (min, mean)."""
    classes = np.unique(y)
    if len(classes) < 2 or x_entail.shape[1] == 0:
        return 0.0, 0.0
    per_class = []
    for c in classes:
        yc = (y == c).astype(int)
        best = 0.5
        for j in range(x_entail.shape[1]):
            col = x_entail[:, j]
            if np.std(col) < 1e-9:
                continue
            auc = roc_auc_score(yc, col)
            best = max(best, auc, 1.0 - auc)
        per_class.append(max(0.0, (best - 0.5) * 2.0))  # AUC [0.5,1] -> [0,1]
    return float(np.min(per_class)), float(np.mean(per_class))


def effective_rank(x: np.ndarray) -> float:
    """Participation ratio of the covariance spectrum (shared with diagnostics.py)."""
    s = np.linalg.svd(x - x.mean(axis=0), compute_uv=False) ** 2
    if s.sum() == 0:
        return 1.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def _cv_skill(x: np.ndarray, y: np.ndarray, cfg: RewardConfig) -> tuple[float, float, float]:
    """Held-out CV accuracy above the majority baseline, averaged over fold-seeds.
    Normalizing above majority makes datasets of different class balance comparable."""
    base = float(np.bincount(y).max() / len(y))
    accs = []
    with threadpool_limits(limits=cfg.cpu_threads):  # bound OpenMP; no thread swarm on a shared box
        for s in range(cfg.cv_seeds):
            skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=1000 + s)
            clf = HistGradientBoostingClassifier(max_iter=150, random_state=0)
            accs.append(float(cross_val_score(clf, x, y, cv=skf).mean()))
    acc = float(np.mean(accs))
    return max(0.0, (acc - base) / (1.0 - base + 1e-9)), acc, float(np.std(accs))


def _anti_hack(x_entail: np.ndarray, texts: list[str], cfg: RewardConfig) -> tuple[float, int, int]:
    lengths = np.array([len(t) for t in texts], dtype=float)
    m = x_entail.shape[1]
    n_len, n_const = 0, 0
    for j in range(m):
        col = x_entail[:, j]
        if np.std(col) < cfg.const_std_thresh:
            n_const += 1
            continue
        if np.std(lengths) > 1e-9 and abs(np.corrcoef(col, lengths)[0, 1]) > cfg.length_corr_thresh:
            n_len += 1
    penalty = (n_len + n_const) / max(1, m)
    return max(0.0, 1.0 - penalty), n_len, n_const


def pool_reward(
    x: np.ndarray,
    y: np.ndarray,
    pool: list[str],
    texts: list[str],
    cfg: RewardConfig | None = None,
    judge_score: float | None = None,
) -> dict:
    """Composite reward + per-term components + a feedback string for GEPA reflection.

    x is the (n, 2m) [P(entail)|P(contradict)] matrix; entail columns are x[:, :m].
    """
    cfg = cfg or RewardConfig()
    m = len(pool)
    x_entail = x[:, :m]

    cv_skill, cv_acc, cv_noise = _cv_skill(x, y, cfg)
    eff = effective_rank(x_entail)
    diversity = eff / max(1, m)
    min_cov, mean_cov = _class_coverage(x_entail, y)
    anti_hack, n_len, n_const = _anti_hack(x_entail, texts, cfg)  # penalty multiplier, not a term

    components = {
        "cv_skill": cv_skill,
        "min_coverage": min_cov,
        "mean_coverage": mean_cov,
        "diversity": diversity,
    }
    if judge_score is not None:
        components["judge"] = float(judge_score)

    w = {k: cfg.weights[k] for k in components}
    wsum = sum(w.values()) or 1.0
    base = sum(components[k] * w[k] for k in components) / wsum
    score = base * anti_hack  # anti_hack is 1.0 unless real artifacts appear, then it bites

    feedback = (
        f"Pool of {m} hypotheses. Held-out CV accuracy {cv_acc:.4f} "
        f"(skill above majority {cv_skill:.3f}, cross-seed noise +/-{cv_noise:.4f}). "
        f"Per-class separation: worst class {min_cov:.2f}, average {mean_cov:.2f} (0=none, 1=perfect) — "
        + (
            f"the weakest class is poorly covered ({min_cov:.2f}); add hypotheses that isolate the "
            "hardest-to-separate class. "
            if min_cov < 0.5
            else "every class has at least one strong separator. "
        )
        + f"Effective rank {eff:.1f}/{m} (diversity {diversity:.2f}) — "
        + (
            "LOW: the pool is collapsing onto a few directions; write hypotheses from more "
            "independent angles (entities, intent, syntax, topic), not paraphrases. "
            if diversity < 0.35
            else "healthy spread of independent directions. "
        )
        + (
            f"{n_len} hypotheses track text length and {n_const} are near-constant/vacuous — "
            "these are surface artifacts, replace them with content statements. "
            if (n_len + n_const)
            else "no length or vacuity artifacts detected. "
        )
    )
    return {
        "score": round(score, 4),
        "components": {k: round(v, 4) for k, v in components.items()},
        "anti_hack_penalty": round(anti_hack, 4),
        "cv_accuracy": round(cv_acc, 4),
        "cv_noise": round(cv_noise, 4),
        "min_coverage": round(min_cov, 4),
        "mean_coverage": round(mean_cov, 4),
        "effective_rank": round(eff, 2),
        "n_length_artifacts": n_len,
        "n_vacuous": n_const,
        "feedback": feedback,
    }


def geometric_mean(scores: list[float]) -> float:
    """Cross-dataset aggregation: craters if any dataset scores near zero, so a
    candidate must generalize rather than win one dataset and tank another."""
    scores = [max(0.0, s) for s in scores]
    if not scores:
        return 0.0
    return float(np.exp(np.mean([np.log(s + 1e-9) for s in scores])))
