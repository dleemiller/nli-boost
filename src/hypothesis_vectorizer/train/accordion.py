"""STAGE 2 (alternative) — the 'accordion': expand <-> compact until the pool saturates.

Each round EXPANDS (LLM generates `gen_size` new hypotheses, told to avoid the current kept set and
to target its confusion hot-spots) then COMPACTS the pooled set back down. The compaction is the
crux, and its size is NOT fixed — it is the pool's EFFECTIVE RANK (how many independent behavioral
directions the score matrix spans, from its variance spectrum). That number grows early (each
generation adds new directions) and plateaus once generations only re-derive existing ones — the
plateau is the accordion's natural resting size and stopping signal (NOTES 2026-07-10).

Compaction, in order (all measured choices, NOTES 2026-07-09):
  1. behavioral dedup (covariance) — free removal of near-duplicate score vectors;
  2. count = effective rank (components for `var_threshold` of variance) of the deduped matrix;
  3. selection = one CV-importance-best MEDOID per cluster (coverage beats importance-ranking, which
     over-prunes rare-class detectors). Keeps REAL hypotheses (interpretable; re-summarizing axes
     underperformed at matched budget).

The result is a compact pool of real hypotheses; the RF/HGB head is fit afterward (honest protocol).
GPU is touched only to score each round's NEW hypotheses; everything else is cached/CPU.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

from ..dedup import Deduper
from ..encoder import EntailmentScorer
from .data import Bundle, labeled_examples, stratified_indices
from .evolve import hotspots
from .proposer import Proposer


def effective_rank(x_std: np.ndarray, var_threshold: float = 0.90) -> int:
    """Number of principal components needed to reach `var_threshold` of the variance of the
    standardized feature matrix — the pool's independent-direction count."""
    if x_std.shape[1] <= 1:
        return x_std.shape[1]
    # eigenvalues of the correlation matrix (x already standardized) = PCA explained variance
    ev = np.linalg.svd(x_std, compute_uv=False) ** 2
    ratio = ev / ev.sum()
    return int(np.searchsorted(np.cumsum(ratio), var_threshold) + 1)


def _keep_representatives(pool: list[str], E: np.ndarray, y: np.ndarray, k: int, seed: int) -> list[str]:
    """Cluster the hypotheses by entail-vector correlation into k groups; keep each cluster's
    highest-RF-importance member (coverage-preserving selection of REAL hypotheses)."""
    if k >= len(pool):
        return list(pool)
    imp = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=4).fit(E, y).feature_importances_
    corr = np.corrcoef(E.T)
    dist = 1.0 - np.abs(np.nan_to_num(corr))
    labels = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(dist)
    keep = [max(np.flatnonzero(labels == c), key=lambda i: imp[i]) for c in range(k)]
    return [pool[i] for i in sorted(keep)]


def _confusion_evidence(
    kept: list[str], E: np.ndarray, y: np.ndarray, names: list[str], texts: list[str], seed: int
) -> list[str]:
    """Where the current kept set still fails: CV-predict with an RF, find mutually-confused class
    groups, and hand the LLM a few example errors per hot-spot so the next batch targets the gaps."""
    if not kept:
        return []
    pred = cross_val_predict(
        RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=4), E, y, cv=4
    )
    errors = [(i, int(p)) for i, p in enumerate(pred) if p != y[i]]
    groups = hotspots(errors, y, len(names))
    evidence = []
    for g in groups[:3]:
        gset = set(g)
        ex = [(i, p) for i, p in errors if y[i] in gset and p in gset][:6]
        if not ex:
            continue
        lines = [f"HOT SPOT — {{{', '.join(names[c] for c in g)}}} are mutually confused; carve them apart:"]
        lines += [f"  [true {names[y[i]]}, pred {names[p]}] {texts[i][:180]}" for i, p in ex]
        evidence.append("\n".join(lines))
    return evidence


def accordion(
    bundle: Bundle,
    scorer: EntailmentScorer,
    proposer: Proposer,
    seed: int,
    *,
    gen_size: int = 64,
    rounds: int = 6,
    var_threshold: float = 0.90,
    patience: int = 2,
    min_keep: int = 8,
    sample: int = 1000,
    deduper: Deduper | None = None,
) -> tuple[list[str], list[dict]]:
    """Expand<->compact until the kept-count stops growing. Returns (final kept pool, per-round
    history). Internal decisions (rank/cluster/hotspots) use a stratified train subsample of size
    `sample` for speed; the runner fits the final head on full train."""
    rng = np.random.default_rng(seed)
    texts, y, names = bundle.train_texts, bundle.y_train, bundle.class_names
    sub = stratified_indices(y, min(sample, len(y)), rng)
    sub_texts, sub_y = [texts[i] for i in sub], y[sub]
    examples = labeled_examples(texts, y, names, per_class=3, rng=rng)
    deduper = deduper or Deduper(scorer, sub_texts, corr_threshold=0.95)

    kept: list[str] = []
    history: list[dict] = []
    prev_k, flat = 0, 0

    for round_i in range(rounds):
        if not kept:  # first expand: broad generation
            cand = proposer.generate(bundle.task, bundle.class_descriptions, examples, gen_size, avoid=[])
        else:  # later expands: target the kept set's confusion, avoid paraphrasing survivors
            E_kept = scorer.features(sub_texts, kept)[:, : len(kept)]
            evidence = _confusion_evidence(kept, E_kept, sub_y, names, sub_texts, seed)
            cand = proposer.refill(
                bundle.task, bundle.class_descriptions, examples, kept, [], evidence, n=gen_size
            )

        # pool kept + new, behavioral-dedup (fresh `seen` so kept survive), then compact
        pooled, _ = deduper.filter(kept + cand, against=[], seen=set())
        E = scorer.features(sub_texts, pooled)[:, : len(pooled)]  # new hyps: GPU here; rest cached
        k = min(len(pooled), max(min_keep, effective_rank(StandardScaler().fit_transform(E), var_threshold)))
        kept = _keep_representatives(pooled, E, sub_y, k, seed)

        history.append(
            {
                "round": round_i,
                "generated": len(cand),
                "deduped": len(pooled),
                "eff_rank": k,
                "kept": len(kept),
            }
        )
        print(
            f"--- accordion round {round_i}: +{len(cand)} gen -> {len(pooled)} deduped "
            f"-> eff_rank {k} -> kept {len(kept)}",
            flush=True,
        )

        flat = flat + 1 if k <= prev_k else 0
        prev_k = k
        if flat >= patience:
            print(f"--- accordion stop: kept-count plateaued for {patience} rounds", flush=True)
            break

    return kept, history
