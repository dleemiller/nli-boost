#!/usr/bin/env python
"""LLM-free core + headroom diagnostic for the entropy-reduction (tree-splitter) evolution loop.

Before spending any LLM budget on dspy.Refine generation, this validates the plumbing and answers
the NECESSARY-condition question: how much class impurity does the CURRENT hypothesis pool leave
unresolved, and is that residual concentrated in a few populated regions worth targeting?

  * fit a diagnostic tree on entail/contradict features -> leaves = regions the pool can't separate
  * rank leaves by entropy(y_leaf) * support
  * report residual weighted impurity (the target mass for generation)
  * OOF information-gain scorer (oof_ig): the exact reward the dspy.Refine loop will maximize,
    validated here on EXISTING hypotheses so we know the metric behaves before generating new ones.

Headroom is NECESSARY not SUFFICIENT: a big residual means there is room to improve; whether new
hypotheses can actually claim it (encoder-reducible) is what the generation loop tests next.

    uv run python experiments/scripts/diag_leaf_headroom.py --dataset trec --n-train 2000
    uv run python experiments/scripts/diag_leaf_headroom.py --dataset goemotions --n-train 3000
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from hvexp import datasets, hypotheses  # noqa: E402
from hvexp.features import NLIFeaturizer  # noqa: E402

ENCODER = "dleemiller/finecat-nli-l"


def entropy(y: np.ndarray, n_classes: int) -> float:
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=n_classes) / len(y)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def oof_ig(scores: np.ndarray, y: np.ndarray, n_classes: int, seed: int = 0) -> float:
    """Out-of-fold information gain of a single continuous feature `scores` on labels `y`.

    Pick the best threshold on half the rows, measure the gain it delivers on the other half.
    Rewards a split that GENERALIZES within the leaf, not one that overfits its rows. This is the
    reward dspy.Refine will maximize over candidate hypotheses.
    """
    n = len(y)
    if n < 6 or len(np.unique(y)) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    a, b = perm[: n // 2], perm[n // 2:]  # a = pick threshold, b = measure
    order = np.argsort(scores[a])
    sa, ya = scores[a][order], y[a][order]
    best_thr, best_gain_a = None, -1.0
    Ha = entropy(ya, n_classes)
    for i in range(1, len(sa)):
        if sa[i] == sa[i - 1]:
            continue
        thr = (sa[i] + sa[i - 1]) / 2
        left, right = ya[:i], ya[i:]
        g = Ha - (len(left) * entropy(left, n_classes) + len(right) * entropy(right, n_classes)) / len(ya)
        if g > best_gain_a:
            best_gain_a, best_thr = g, thr
    if best_thr is None:
        return 0.0
    sb, yb = scores[b], y[b]
    Hb = entropy(yb, n_classes)
    left, right = yb[sb <= best_thr], yb[sb > best_thr]
    if len(left) == 0 or len(right) == 0:
        return 0.0
    return Hb - (len(left) * entropy(left, n_classes) + len(right) * entropy(right, n_classes)) / len(yb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="trec")
    ap.add_argument("--pool-json", type=pathlib.Path, default=None,
                    help="JSON list of hypothesis strings (e.g. a union pool); default = expert_pool")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-leaf", type=int, default=20, help="min_samples_leaf for the diagnostic tree")
    ap.add_argument("--impure-bits", type=float, default=0.3, help="leaf entropy (bits) above which = impure")
    args = ap.parse_args()

    raw = datasets.load_raw(args.dataset, test_size=500, test_seed=7)
    n_classes = raw.n_classes
    # stratified-ish subsample of train for a fast, cache-friendly diagnostic
    y = raw.y_train
    idx = datasets.subsample_indices(y, args.n_train // n_classes, seed=0)
    texts = [raw.train_texts[i] for i in idx]
    yy = y[idx]

    fz = NLIFeaturizer(encoder=ENCODER, device=args.device, verbose=True)
    if args.pool_json:
        import json
        pool = json.loads(args.pool_json.read_text())
        if pool and isinstance(pool[0], dict):  # tolerate [{text,...}] pools too
            pool = [h["text"] for h in pool]
        print(f"[pool] {args.pool_json.name}: {len(pool)} hyps (external)")
    else:
        pool, _tags = hypotheses.expert_pool(args.dataset)
    X = fz.features(texts, pool, "entail_contradict")  # (n, 2*len(pool)); hits the disk cache
    m = len(pool)

    tree = DecisionTreeClassifier(min_samples_leaf=args.min_leaf, random_state=0).fit(X, yy)
    leaf_id = tree.apply(X)
    leaves = np.unique(leaf_id)

    rows = []
    total = len(yy)
    residual_mass = 0.0
    for lf in leaves:
        mask = leaf_id == lf
        yl = yy[mask]
        H = entropy(yl, n_classes)
        support = int(mask.sum())
        residual_mass += H * support  # weighted impurity contributed by this leaf
        rows.append((H, support, lf, mask))

    impure = [(H, s, lf, mask) for (H, s, lf, mask) in rows if H > args.impure_bits]
    impure.sort(key=lambda r: -r[0] * r[1])
    frac_in_impure = sum(s for _, s, _, _ in impure) / total

    print(f"\n=== leaf-impurity headroom — {args.dataset} (n_train={total}, pool={m} hyps) ===")
    print(f"diagnostic tree: {len(leaves)} leaves (min_samples_leaf={args.min_leaf})")
    print(f"train accuracy of tree on current features: {tree.score(X, yy):.3f}")
    print(f"total residual weighted impurity: {residual_mass:.0f} bit-examples "
          f"({residual_mass/total:.3f} bits/example)")
    print(f"impure leaves (H>{args.impure_bits}): {len(impure)}  holding "
          f"{frac_in_impure*100:.0f}% of train rows  <- the mass generation would target")

    # validate the OOF-IG reward on the top impure leaves using EXISTING hypotheses as candidates
    print(f"\ntop impure leaves — best OOF-IG achievable by an EXISTING pool hypothesis:")
    print(f"{'leaf':>6} {'H(bits)':>8} {'support':>8} {'top-2 classes in leaf':<34} {'best existing IG':>16} {'hyp':<40}")
    ent = fz.features(texts, pool, "entail")  # (n, m) P(entail) per hypothesis, for candidate scoring
    for H, s, lf, mask in impure[:8]:
        yl = yy[mask]
        cnt = np.bincount(yl, minlength=n_classes)
        top2 = cnt.argsort()[::-1][:2]
        pair = f"{raw.class_names[top2[0]]}({cnt[top2[0]]}) vs {raw.class_names[top2[1]]}({cnt[top2[1]]})"
        best_ig, best_h = -1.0, ""
        for j in range(m):
            ig = oof_ig(ent[mask, j], yl, n_classes)
            if ig > best_ig:
                best_ig, best_h = ig, pool[j]
        print(f"{lf:>6} {H:>8.2f} {s:>8} {pair:<34} {best_ig:>16.3f} {best_h[:40]:<40}")

    print("\ninterpretation:")
    print("  * small residual (e.g. SST-2) -> pool already separates; generation has little to do.")
    print("  * large residual + low best-existing-IG in impure leaves -> current features are spent")
    print("    THERE; the leaf needs a NEW hypothesis. That is exactly the generation target set.")
    print("  * large residual but high best-existing-IG -> the tree just didn't use available signal")
    print("    (tree suboptimality, not a missing hypothesis) -> generation would be wasted there.")


if __name__ == "__main__":
    main()
