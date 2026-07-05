"""STAGE 2 — evolve the pool. The measured design (METHOD.md):

- rank hypotheses by CV-fold permutation importance + cross-fold sign STABILITY
  (single-split ranking churned 50% across seeds: half the kills were coin flips);
- prune only CONFIDENT deaths (helped in zero folds); ambiguous ones get another round;
- feed the LM the REASON each pruned hypothesis failed, plus confusion HOT SPOTS
  (grouped errors; batches force pattern-level proposals, single examples invite overfit);
- stop on held-out plateau (generation saturates ~round 2: refill hit-rate decays
  100% -> 20% -> 0%);
- instrument every round: held-out accuracy, and each refill's standalone AUC on the
  hot spot it targeted (once the head interpolates train, marginal importance cannot
  see a good new feature; standalone alignment can).
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import PoolConfig
from .data import Bundle, labeled_examples, stratified_indices
from .dedup import Deduper
from .encoder import EntailmentScorer
from .proposer import Proposer


@dataclass
class Ranking:
    """Per-HYPOTHESIS ranking aggregated over its two feature columns."""

    order: np.ndarray  # hypothesis indices, most useful first
    importance: np.ndarray  # mean CV permutation importance, summed over the 2 columns
    stability: np.ndarray  # fraction of folds where either column helped
    errors: list[tuple[int, int]]  # (local text index, predicted class) over all held-out folds

    @property
    def heldout_accuracy(self) -> float:
        return 1.0 - len(self.errors) / self._n

    _n: int = 0


def rank_hypotheses(
    x: np.ndarray, y: np.ndarray, m: int, seed: int, folds: int = 4, lex: np.ndarray | None = None
) -> Ranking:
    """x is the (n, 2m) NLI feature matrix; every sample is held out in exactly one fold.

    `lex` (n, d), if given, is the cheap lexical channel as a FIXED baseline: it joins every
    fold's model and drives the errors/hot spots, but is never ranked or pruned. Each NLI
    hypothesis's importance then measures its MARGINAL value ON TOP OF lexical — a hypothesis
    whose signal TF-IDF already carries scores ~0 and dies. Since an NLI feature costs a
    cross-encoder forward pass at inference and TF-IDF is ~free, this minimizes the NLI pool
    (and thus per-prediction cost) down to hypotheses that add semantics lexical cannot."""
    xx = x if lex is None else np.concatenate([x, lex], axis=1)
    imps = np.zeros((folds, xx.shape[1]))
    errors: list[tuple[int, int]] = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for f, (itr, ihe) in enumerate(skf.split(xx, y)):
        clf = HistGradientBoostingClassifier(max_iter=100, random_state=seed)
        clf.fit(xx[itr], y[itr])
        imps[f] = permutation_importance(
            clf, xx[ihe], y[ihe], n_repeats=3, random_state=seed
        ).importances_mean
        preds = clf.predict(xx[ihe])
        errors += [(int(i), int(p)) for i, p in zip(ihe, preds) if p != y[i]]

    # rank/prune only the NLI columns (first 2m: [entail | contradict]); lexical columns are baseline
    col_stability = (imps > 0).mean(axis=0)
    mean_imp = imps.mean(axis=0)
    hyp_importance = mean_imp[:m] + mean_imp[m : 2 * m]
    hyp_stability = np.maximum(col_stability[:m], col_stability[m : 2 * m])
    order = np.lexsort((-hyp_stability, -hyp_importance))
    r = Ranking(order=order, importance=hyp_importance, stability=hyp_stability, errors=errors)
    r._n = len(y)
    return r


def hotspots(
    errors: list[tuple[int, int]], y: np.ndarray, n_classes: int, min_rate: float = 0.04
) -> list[list[int]]:
    """Groups of mutually-confused classes: connected components of the symmetrized
    confusion graph, thresholded on confusion rate. Pairwise views miss block
    structure — a clique of confused classes needs carving as a group."""
    counts = np.zeros((n_classes, n_classes))
    for i, p in errors:
        counts[y[i], p] += 1
    sym = counts + counts.T
    support = np.bincount(y, minlength=n_classes).astype(float)

    adj: list[list[int]] = [[] for _ in range(n_classes)]
    for a in range(n_classes):
        for b in range(a + 1, n_classes):
            if sym[a, b] / max(1.0, min(support[a], support[b])) >= min_rate:
                adj[a].append(b)
                adj[b].append(a)

    seen: set[int] = set()
    groups = []
    for start in range(n_classes):
        if start in seen or not adj[start]:
            continue
        stack, comp = [start], []
        while stack:
            c = stack.pop()
            if c not in seen:
                seen.add(c)
                comp.append(c)
                stack.extend(adj[c])
        if len(comp) >= 2:
            groups.append(sorted(comp))
    groups.sort(key=lambda g: -sum(sym[a, b] for a in g for b in g))
    return groups


def _failure_reason(col_e: np.ndarray, col_c: np.ndarray, survivor_cols: np.ndarray) -> str:
    """Why did a pruned hypothesis fail? Tells the LM whether to abandon the
    concept (undetectable) or just this angle on it."""
    if np.std(col_e) < 0.02 and np.std(col_c) < 0.02:
        return "the NLI encoder scores every text identically on this — undetectable property"
    if survivor_cols.shape[1]:
        best = 0.0
        for col in (col_e, col_c):
            cc = col - col.mean()
            sc = survivor_cols - survivor_cols.mean(axis=0)
            denom = np.linalg.norm(cc) * np.linalg.norm(sc, axis=0)
            best = max(best, float(np.max(np.abs((cc @ sc) / np.where(denom == 0, np.inf, denom)))))
        if best > 0.9:
            return "redundant — signal nearly identical to a kept hypothesis"
    return "detectable but carries no held-out predictive value for these classes"


def _confusion_evidence(
    bundle: Bundle,
    sub: np.ndarray,
    errors: list[tuple[int, int]],
    groups: list[list[int]],
    rng: np.random.Generator,
) -> list[str]:
    """Hot spots as grouped example batches; scattered errors as counts only."""
    names = bundle.class_names
    global_errors = [(int(sub[i]), p) for i, p in errors]
    rng.shuffle(global_errors)
    evidence, used = [], set()
    for g in groups[:3]:
        gset = set(g)
        in_group = [(i, p) for i, p in global_errors if bundle.y_train[i] in gset and p in gset][:8]
        if not in_group:
            continue
        lines = [
            f"HOT SPOT — the classes {{{', '.join(names[c] for c in g)}}} are mutually "
            f"confused ({len(in_group)}+ errors shown). Write hypotheses for what these "
            f"errors SHARE that would carve the classes apart:"
        ]
        lines += [
            f"  [true: {names[bundle.y_train[i]]}, predicted: {names[p]}] {bundle.train_texts[i][:220]}"
            for i, p in in_group
        ]
        evidence.append("\n".join(lines))
        used.update(i for i, _ in in_group)
    scattered = Counter((bundle.y_train[i], p) for i, p in global_errors if i not in used)
    if scattered:
        evidence.append(
            "Scattered confusions outside hot spots (counts only): "
            + "; ".join(f"{names[t]}→{names[p]}: {c}" for (t, p), c in scattered.most_common(5))
        )
    return evidence


def _target_aucs(
    x: np.ndarray, pool: list[str], refills: list[str], m: int, target: tuple[int, int], y: np.ndarray
) -> list[float]:
    """Standalone AUC of each surviving refill on the class pair it targeted,
    taking the better of its two feature columns."""
    a, b = target
    mask = (y == a) | (y == b)
    yy = (y[mask] == a).astype(int)
    aucs = []
    for s in refills:
        if s not in pool:
            continue
        j = pool.index(s)
        best = 0.5
        for col in (x[mask, j], x[mask, m + j]):
            if np.std(col) > 1e-9:
                auc = roc_auc_score(yy, col)
                best = max(best, auc, 1 - auc)
        aucs.append(round(float(best), 3))
    return aucs


def evolve(
    bundle: Bundle,
    pool: list[str],
    scorer: EntailmentScorer,
    proposer: Proposer,
    deduper: Deduper,
    cfg: PoolConfig,
    seed: int,
    lex_train: np.ndarray | None = None,
) -> tuple[list[str], list[dict]]:
    """Returns (final pool, per-round history). History is the audit trail:
    every prune with its reason, every refill with its later target-AUC.

    `lex_train` (n_train, d), if given, is the lexical channel over ALL train texts; the
    ranking sees it as a fixed baseline so NLI hypotheses are pruned by MARGINAL value over
    lexical (redundant-with-TF-IDF hypotheses die), and refill targets the confusions lexical
    leaves behind."""
    rng = np.random.default_rng(seed)
    if cfg.rank_sample and cfg.rank_sample < len(bundle.train_texts):
        sub = stratified_indices(bundle.y_train, cfg.rank_sample, rng)
    else:
        sub = np.arange(len(bundle.train_texts))
    sub_texts = [bundle.train_texts[i] for i in sub]
    sub_y = bundle.y_train[sub]
    lex_sub = lex_train[sub] if lex_train is not None else None
    examples = labeled_examples(bundle, per_class=3, rng=rng)

    seen = {s.casefold() for s in pool}
    history: list[dict] = []
    best_acc, since_best = -1.0, 0
    prev_refills: list[str] = []
    prev_target: tuple[int, int] | None = None

    for round_i in range(cfg.rounds):
        m = len(pool)
        x = scorer.features(sub_texts, pool)
        # fixed fold seed across rounds: the plateau check compares round-over-round
        # held-out accuracy, which is only meaningful on the SAME fold splits
        ranking = rank_hypotheses(x, sub_y, m, seed, lex=lex_sub)

        # instrumentation: did last round's refills hit their assigned hot spot?
        refill_aucs: list[float] = []
        if prev_refills and prev_target is not None:
            refill_aucs = _target_aucs(x, pool, prev_refills, m, prev_target, sub_y)
            if history:
                history[-1]["refill_target_aucs"] = refill_aucs
                history[-1]["refill_hit_rate"] = round(
                    float(np.mean([a >= 0.75 for a in refill_aucs])) if refill_aucs else 0.0, 3
                )

        # prune confident deaths only, capped so one round never guts the pool
        max_prune = m - max(4, int(m * cfg.min_keep_frac))
        dead = [int(i) for i in ranking.order[::-1] if ranking.stability[i] == 0.0][:max_prune]
        dead_set = set(dead)
        keep = [int(i) for i in ranking.order if i not in dead_set]
        survivors = [pool[i] for i in keep]
        survivor_cols = x[:, keep]

        failed = [f"{pool[i]} ({_failure_reason(x[:, i], x[:, m + i], survivor_cols)})" for i in dead]

        groups = hotspots(ranking.errors, sub_y, bundle.n_classes)
        evidence = _confusion_evidence(bundle, sub, ranking.errors, groups, rng)

        refills: list[str] = []
        if dead:
            proposed = proposer.refill(
                bundle.task, bundle.class_descriptions, examples, survivors, failed, evidence, n=len(dead)
            )
            refills, _ = deduper.filter(proposed, against=survivors, seen=seen)

        acc = ranking.heldout_accuracy
        history.append(
            {
                "round": round_i,
                "heldout_acc": round(acc, 4),
                "survivors": survivors,
                "failed": failed,
                "refills": refills,
            }
        )
        print(
            f"--- evolve round {round_i}: heldout {acc:.4f}, kept {len(survivors)}, "
            f"pruned {len(failed)}, refilled {len(refills)}",
            flush=True,
        )

        pool = survivors + refills
        prev_refills = refills
        prev_target = tuple(groups[0][:2]) if groups else None

        if acc > best_acc + 1e-4:
            best_acc, since_best = acc, 0
        else:
            since_best += 1
            if since_best >= cfg.patience:
                print(f"--- evolve stop: no held-out improvement for {cfg.patience} rounds", flush=True)
                break
    return pool, history
