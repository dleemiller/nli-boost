"""STAGE 2 (alternative) — tree-guided, LLM-in-the-loop pool growth.

Where the stability method (evolve.py) prunes a large pool by CV importance, this GROWS a small
pool by feedback: fit an entropy decision tree on the current features, find the leaf it is most
confused about (highest entropy x count), and ask the LLM — shown K labelled examples from exactly
that leaf — for ONE hypothesis whose entailment score would carve those classes apart. The candidate
is chosen by a Refine/BestOfN loop whose reward is the information gain of its best threshold split
on the leaf. Add it, refit the tree, repeat. The tree is the gradient signal: it says precisely
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


def _make_reward(scorer: EntailmentScorer, leaf_texts: list[str], leaf_y: np.ndarray):
    """reward(inputs, pred) -> normalized info gain in [0, 1]: how much of the leaf's entropy the
    candidate hypothesis' best threshold split removes (taking the better of entail/contradict)."""
    h0 = _entropy(leaf_y)

    def reward(*args) -> float:
        pred = args[-1]
        hyp = (getattr(pred, "hypothesis", "") or "").strip()
        if not hyp or h0 <= 0.0:
            return 0.0
        feats = scorer.features(leaf_texts, [hyp])  # (k, 2) = [entail | contradict]
        gain = max(_best_split_gain(feats[:, 0], leaf_y, h0), _best_split_gain(feats[:, 1], leaf_y, h0))
        return gain / h0

    return reward


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

        reward_fn = _make_reward(scorer, leaf_texts, leaf_y)
        hyp, reward = proposer.split_leaf(
            task=bundle.task,
            class_definitions=bundle.class_descriptions,
            confused_examples=examples,
            classes_present=classes_present,
            avoid=pool,
            reward_fn=reward_fn,
            attempts=tcfg.refine_attempts,
            strategy=tcfg.strategy,
        )

        added = None
        if hyp:
            kept, _ = deduper.filter([hyp], against=pool, seen=seen)
            if kept:
                added = kept[0]
                pool.append(added)

        history.append(
            {
                "round": round_i,
                "leaf": int(target),
                "leaf_size": int(mask.sum()),
                "leaf_entropy": round(_entropy(leaf_y), 4),
                "classes_present": classes_present,
                "hypothesis": hyp,
                "info_gain": round(float(reward), 4),
                "added": added is not None,
                "pool_size": len(pool),
            }
        )
        print(
            f"--- tree-evolve round {round_i}: leaf {target} "
            f"(n={int(mask.sum())}, H={_entropy(leaf_y):.3f}, {'/'.join(classes_present)}), "
            f"gain {reward:.3f}, {'added' if added else 'no-add'} -> pool {len(pool)}",
            flush=True,
        )

        since_add = 0 if added else since_add + 1
        if since_add >= tcfg.patience:
            print(f"--- tree-evolve stop: {tcfg.patience} rounds with no new hypothesis", flush=True)
            break

    return pool, history
