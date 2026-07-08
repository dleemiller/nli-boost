"""STAGE 2 (alternative) — tree-guided, LLM-in-the-loop pool growth.

Where the stability method (evolve.py) prunes a large pool by CV importance, this GROWS a small
pool by feedback: fit an entropy decision tree on the current features, find the leaf it is most
confused about (highest entropy x count), and ask the LLM — shown K labelled examples from exactly
that leaf — for ONE hypothesis whose entailment score would carve those classes apart. The candidate
is chosen by an in-house refine loop (proposer.split_leaf): each attempt is scored gain x novelty,
failures are fed back to the LLM as MEASURED LANGUAGE ("collinear with <named hypothesis>"), best
attempt wins. Add it, refit the tree, repeat. The tree is the gradient signal: it says precisely
where the current features fail, one confused region at a time.

The result is a pool; the RF/HGB head is fit afterward by the runner (honest protocol: pool_cv).
"""

from collections import Counter

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from ..config import PoolConfig
from ..dedup import Deduper, norm_statement
from ..encoder import EntailmentScorer
from .data import Bundle
from .proposer import Proposer


def _entropy(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _best_split_gain(scores: np.ndarray, y: np.ndarray, h0: float, n_thresholds: int = 32) -> float:
    """Information gain (bits) of the best single threshold on `scores` vs the labels `y`.
    Candidate thresholds are quantiles, so cost is bounded regardless of leaf size."""
    n = len(y)
    if n < 2 or np.std(scores) < 1e-9:
        return 0.0
    thresholds = np.unique(np.quantile(scores, np.linspace(0.05, 0.95, n_thresholds)))
    best = 0.0
    for t in thresholds:
        left = scores <= t
        n_l = int(left.sum())
        if n_l == 0 or n_l == n:
            continue
        child = (n_l * _entropy(y[left]) + (n - n_l) * _entropy(y[~left])) / n
        best = max(best, h0 - child)
    return best


def _pick_leaf(leaf_id: np.ndarray, y: np.ndarray, min_samples: int) -> int | None:
    """The impure leaf with the most total confusion (entropy x count) and enough examples."""
    best, best_score = None, 0.0
    for lid in np.unique(leaf_id):
        mask = leaf_id == lid
        n = int(mask.sum())
        if n < min_samples:
            continue
        h = _entropy(y[mask])
        if h <= 0.0:  # pure leaf: nothing to split
            continue
        score = h * n
        if score > best_score:
            best, best_score = int(lid), score
    return best


def _sample_shots(idx: np.ndarray, y: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    """K leaf examples, stratified across the classes present so every confused class is shown."""
    classes = np.unique(y[idx])
    per = max(1, k // len(classes))
    picked: list[int] = []
    for c in classes:
        ci = idx[y[idx] == c].copy()
        rng.shuffle(ci)
        picked.extend(int(i) for i in ci[:per])
    rng.shuffle(picked)
    return picked[:k]


def _max_abs_corr(col: np.ndarray, pool_cols: np.ndarray) -> tuple[float, int]:
    """(max |Pearson|, argmax column index) between a candidate's score vector and every existing
    pool feature column (entail AND contradict), on the leaf texts. 'Novelty' = 1 - max."""
    if np.std(col) < 1e-9 or pool_cols.shape[1] == 0:
        return 0.0, -1
    c = (col - col.mean()) / col.std()
    z = pool_cols - pool_cols.mean(axis=0)
    s = pool_cols.std(axis=0)
    corr = np.zeros(pool_cols.shape[1])
    valid = s > 1e-9
    corr[valid] = np.abs((z[:, valid] / s[valid]).T @ c) / len(col)
    j = int(np.argmax(corr))
    return float(corr[j]), j


def _make_evaluator(
    scorer: EntailmentScorer,
    leaf_texts: list[str],
    leaf_y: np.ndarray,
    pool: list[str],
    pool_leaf_feats: np.ndarray,
):
    """evaluate(hyp) -> {score, gain, novelty, covariant_with}. score = gain x novelty, both [0,1]:

    - gain: fraction of the leaf's entropy removed by the candidate's best threshold split
      (better of entail/contradict).
    - novelty: 1 - max |corr| with the EXISTING pool's feature columns on this leaf; a candidate
      collinear with a feature the tree already has scores ~0 however well it splits.
    - covariant_with: the TEXT of the most-correlated existing hypothesis — fed back to the LLM
      verbatim (the wording is the useful feedback, not the scalar).
    Every evaluation is printed live and kept on `evaluate.attempts`."""
    h0 = _entropy(leaf_y)
    m = len(pool)

    def evaluate(hyp: str) -> dict:
        feats = scorer.features(leaf_texts, [hyp])  # (k, 2) = [entail | contradict]
        gains = [_best_split_gain(feats[:, i], leaf_y, h0) for i in (0, 1)] if h0 > 0 else [0.0, 0.0]
        best = int(np.argmax(gains))
        gain_norm = gains[best] / h0 if h0 > 0 else 0.0
        max_corr, j = _max_abs_corr(feats[:, best], pool_leaf_feats)
        novelty = 1.0 - max_corr
        r = {
            "hypothesis": hyp,
            "score": round(gain_norm * novelty, 4),
            "gain": round(gain_norm, 4),
            "novelty": round(novelty, 4),
            "covariant_with": pool[j % m] if j >= 0 else None,
        }
        evaluate.attempts.append(r)
        print(
            f"      attempt {len(evaluate.attempts)}: score={r['score']:.3f} "
            f"(gain {gain_norm:.3f} x novelty {novelty:.3f}"
            + (f', ~ "{r["covariant_with"][:60]}"' if max_corr > 0.5 else "")
            + f") | {hyp[:90]}",
            flush=True,
        )
        return r

    evaluate.attempts = []
    return evaluate


def _related_hypotheses(x_leaf: np.ndarray, leaf_y: np.ndarray, pool: list[str], top: int = 10) -> list[str]:
    """The existing hypotheses most relevant to this leaf, each with the (insufficient) fraction
    of the leaf's entropy its best split resolves — shown to the LLM so it knows what the model
    already reads here and complements rather than re-derives it."""
    h0 = _entropy(leaf_y)
    if h0 <= 0:
        return []
    m = len(pool)
    scored = []
    for j in range(m):
        g = max(_best_split_gain(x_leaf[:, j], leaf_y, h0), _best_split_gain(x_leaf[:, m + j], leaf_y, h0))
        scored.append((g / h0, pool[j]))
    scored.sort(reverse=True)
    return [f"resolves {g:.0%} of this leaf's confusion: {s}" for g, s in scored[:top]]


def tree_evolve(
    bundle: Bundle,
    pool: list[str],
    scorer: EntailmentScorer,
    proposer: Proposer,
    deduper: Deduper,
    cfg: PoolConfig,
    seed: int,
    baseline_train: np.ndarray | None = None,
) -> tuple[list[str], list[dict]]:
    """Grow `pool` by tree-guided LLM proposals for up to `cfg.tree.rounds` rounds, returning
    (final pool, per-round history). `baseline_train` (n, d), if given, joins the tree's features
    so leaves already resolved by the baseline (TF-IDF / tabular / fixed hypotheses) are never
    targeted — new hypotheses address only the confusion the baseline leaves behind."""
    tcfg = cfg.tree
    rng = np.random.default_rng(seed)
    texts, y, names = bundle.train_texts, bundle.y_train, bundle.class_names
    pool = list(pool)
    seen = {norm_statement(s) for s in pool}
    rejected_redundant: list[str] = []  # told to the LLM via `avoid` so it stops re-deriving them
    history: list[dict] = []
    since_add = 0

    for round_i in range(tcfg.rounds):
        x = scorer.features(texts, pool)  # (n, 2m); layout is irrelevant to the tree
        feat = x if baseline_train is None else np.concatenate([x, baseline_train], axis=1)
        tree = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=tcfg.max_depth,
            min_samples_leaf=tcfg.min_samples_leaf,
            random_state=seed,
        ).fit(feat, y)
        leaf_id = tree.apply(feat)
        target = _pick_leaf(leaf_id, y, tcfg.leaf_min_samples)
        if target is None:
            print("--- tree-evolve stop: no impure leaf with enough samples", flush=True)
            break

        mask = leaf_id == target
        leaf_local = np.flatnonzero(mask)
        leaf_texts = [texts[i] for i in leaf_local]
        leaf_y = y[mask]
        present = Counter(int(c) for c in leaf_y)
        classes_present = [f"{names[c]}: {n}" for c, n in present.most_common()]
        shot_idx = _sample_shots(leaf_local, y, tcfg.leaf_shots, rng)
        examples = [f"[{names[y[i]]}] {texts[i][:400]}" for i in shot_idx]

        print(
            f"--- tree-evolve round {round_i}: targeting leaf {target} "
            f"(n={int(mask.sum())}, H={_entropy(leaf_y):.3f}, {'/'.join(classes_present)})",
            flush=True,
        )
        evaluate_fn = _make_evaluator(scorer, leaf_texts, leaf_y, pool, x[mask])
        hyp, reward = proposer.split_leaf(
            task=bundle.task,
            class_definitions=bundle.class_descriptions,
            confused_examples=examples,
            classes_present=classes_present,
            related_hypotheses=_related_hypotheses(x[mask], leaf_y, pool),
            avoid=pool + rejected_redundant,
            evaluate_fn=evaluate_fn,
            attempts=tcfg.refine_attempts,
            strategy=tcfg.strategy,
        )

        added = None
        if hyp:
            kept, rejects = deduper.filter([hyp], against=pool, seen=seen)
            if kept:
                added = kept[0]
                pool.append(added)
            else:  # survived the reward's leaf-novelty but is redundant on the FULL train:
                rejected_redundant.append(hyp)  # tell the LLM explicitly in later rounds

        history.append(
            {
                "round": round_i,
                "leaf": int(target),
                "leaf_size": int(mask.sum()),
                "leaf_entropy": round(_entropy(leaf_y), 4),
                "classes_present": classes_present,
                "hypothesis": hyp,
                "info_gain": round(float(reward), 4),
                "attempts": evaluate_fn.attempts,  # every LLM attempt: score/gain/novelty/covariant
                "added": added is not None,
                "pool_size": len(pool),
            }
        )
        print(
            f"    round {round_i} result: best score {reward:.3f} over {len(evaluate_fn.attempts)} "
            f"attempts, {'ADDED' if added else 'no-add'} -> pool {len(pool)}",
            flush=True,
        )

        since_add = 0 if added else since_add + 1
        if since_add >= tcfg.patience:
            print(f"--- tree-evolve stop: {tcfg.patience} rounds with no new hypothesis", flush=True)
            break

    return pool, history
