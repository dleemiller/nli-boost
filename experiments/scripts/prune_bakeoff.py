"""Abundance-then-prune bake-off on cached hypotheses (GPU-free).

Assembles every hypothesis already scored on BOTH the TREC train and test splits under -l (the union
of all prior runs' pools), then compares SUBSET-pruning strategies: does pruning an abundant pool to
k beat generating k directly (curated-32 = 0.960; trec_full-62 = 0.964)? All scores are cached, so
this needs no GPU — only sklearn head fits + selection. Selection uses TRAIN only; one test eval each.

    uv run python experiments/scripts/prune_bakeoff.py --k 32
"""

import argparse

import numpy as np


def _cached_pool(scorer, train_texts, test_texts, model):
    """Every hypothesis fully scored (>=99%) on both splits under `model`, from the cache."""
    import sqlite3

    from hypothesis_vectorizer.encoder import digest, normalize

    conn = sqlite3.connect("cache/nli_scores.sqlite")
    hh2text = dict(conn.execute("SELECT hyp_hash, text FROM hypotheses"))

    def covered(texts):
        th = list({digest(normalize(t, 1200)) for t in texts})
        cov = {}
        for i in range(0, len(th), 400):
            ch = th[i : i + 400]
            qs = ",".join("?" * len(ch))
            for hh, c in conn.execute(
                f"SELECT hyp_hash, COUNT(*) FROM nli_scores WHERE model=? AND text_hash IN ({qs}) GROUP BY hyp_hash",
                (model, *ch),
            ):
                cov[hh] = cov.get(hh, 0) + c
        return cov, len(th)

    ctr, ntr = covered(train_texts)
    cte, nte = covered(test_texts)
    hh = [h for h in ctr if ctr[h] >= ntr * 0.99 and cte.get(h, 0) >= nte * 0.99]
    return [hh2text[h] for h in hh if h in hh2text]


def _eval_pool(pool, xtr, xte, ytr, yte, n_classes, seed):
    # FIXED RF head across all methods (fair comparison + fast); not the full cv_selected grid, so
    # absolute numbers run ~0.5pt under the grid-selected 0.960/0.964 baselines — compare methods
    # to each other and to the abundance-N row, not to the grid baselines in absolute terms.
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    from hypothesis_vectorizer.train.head import evaluate

    rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=4)
    cv = float(cross_val_score(rf, xtr, ytr, cv=4, n_jobs=1).mean())
    rf.fit(xtr, ytr)
    r = evaluate(yte, rf.predict_proba(xte), n_classes)
    return r, cv, "rf300"


def _hyp_importance(x, y, m, seed):
    """Summed RF permutation-free importance per hypothesis (over its 2 columns), on TRAIN."""
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=4).fit(x, y)
    imp = rf.feature_importances_
    return imp[:m] + imp[m : 2 * m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--data", default="trec")
    ap.add_argument("--encoder", default="dleemiller/finecat-nli-l")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from sklearn.cluster import AgglomerativeClustering

    from hypothesis_vectorizer.cache import ScoreCache
    from hypothesis_vectorizer.config import DataConfig, EncoderConfig
    from hypothesis_vectorizer.costs import CostTracker
    from hypothesis_vectorizer.encoder import EntailmentScorer
    from hypothesis_vectorizer.train.data import load

    b = load(DataConfig(name=args.data, train_size=5452, val_size=0, test_size=2000), seed=args.seed)
    scorer = EntailmentScorer(
        EncoderConfig(model=args.encoder, device="cuda"), ScoreCache("cache/nli_scores.sqlite"), CostTracker()
    )
    pool = _cached_pool(scorer, b.train_texts, b.test_texts, args.encoder)
    m = len(pool)
    print(f"abundance pool: {m} hypotheses (cached train+test on {args.encoder})")

    def feats(texts):
        f = scorer.features(texts, pool)  # (n, 2m)
        return f

    xtr, xte = feats(b.train_texts), feats(b.test_texts)
    assert scorer.costs.encoder_gpu_pairs == 0, f"NOT cached — {scorer.costs.encoder_gpu_pairs} gpu pairs"
    ytr, yte, nc = b.y_train, b.y_test, b.n_classes

    def sub(idx):  # feature columns for a hypothesis subset (entail block + contradict block)
        idx = list(idx)
        cols = idx + [m + i for i in idx]
        return xtr[:, cols], xte[:, cols]

    rows = []
    # 1) full abundance pool
    r, cv, kind = _eval_pool(pool, xtr, xte, ytr, yte, nc, args.seed)
    rows.append((f"abundance-{m}", m, r, cv, kind))

    imp = _hyp_importance(xtr, ytr, m, args.seed)

    # 2) prune -> k by RF importance (top-k hypotheses)
    top = np.argsort(imp)[::-1][: args.k]
    xtr2, xte2 = sub(top)
    r, cv, kind = _eval_pool([pool[i] for i in top], xtr2, xte2, ytr, yte, nc, args.seed)
    rows.append((f"rf-importance->{args.k}", args.k, r, cv, kind))

    # 3) prune -> k by agglomerative clustering (on entail-score correlation), keep top-importance medoid
    E = xtr[:, :m]
    corr = np.corrcoef(E.T)
    dist = 1.0 - np.abs(np.nan_to_num(corr))
    labels = AgglomerativeClustering(n_clusters=args.k, metric="precomputed", linkage="average").fit_predict(
        dist
    )
    keep = [max(np.where(labels == c)[0], key=lambda i: imp[i]) for c in range(args.k)]
    xtr3, xte3 = sub(keep)
    r, cv, kind = _eval_pool([pool[i] for i in keep], xtr3, xte3, ytr, yte, nc, args.seed)
    rows.append((f"cluster-medoid->{args.k}", len(keep), r, cv, kind))

    # 4) covariance dedup (behavioral) — variable count
    from hypothesis_vectorizer.dedup import Deduper

    sub_idx = np.random.default_rng(args.seed).choice(len(b.train_texts), 400, replace=False)
    ded = Deduper(scorer, [b.train_texts[i] for i in sub_idx], corr_threshold=0.9, min_std=0.02)
    kept, _ = ded.filter(pool, against=[], seen=set())
    keep_idx = [pool.index(h) for h in kept]
    xtr4, xte4 = sub(keep_idx)
    r, cv, kind = _eval_pool(kept, xtr4, xte4, ytr, yte, nc, args.seed)
    rows.append(("covariance-dedup(0.9)", len(kept), r, cv, kind))

    print(f"\n{'method':26} {'k':>4} {'acc':>7} {'macro_f1':>9} {'logloss':>8} {'cv_tr':>7} head")
    for name, k, r, cv, kind in rows:
        print(
            f"{name:26} {k:>4} {r['accuracy']:>7.4f} {r['macro_f1']:>9.4f} {r['logloss']:>8.4f} {cv:>7.4f} {kind}"
        )
    print("\nbaselines (same test/seed): curated-32 0.960 | trec_full-62 0.964")


if __name__ == "__main__":
    main()
