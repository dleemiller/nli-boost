"""Orchestrates the method end to end (METHOD.md stages 1-4) and writes artifacts.

Artifacts per run in runs/<run_name>/:
  config.yaml   resolved config snapshot
  model.json    the pool (the model IS a list of English sentences) + head params
  log.jsonl     evolution audit trail: every prune with reason, refill with target-AUC
  metrics.json  pool_cv on test — the ONLY reported head (honest protocol)
  costs.json    LM spend, encoder pairs, abnormal finishes, wall time

`scorer` and `proposer` are injectable so the full pipeline is testable without
a GPU or an LM key.
"""

import json

import numpy as np

from . import data
from ..cache import ScoreCache
from ..config import RunConfig
from ..costs import CostTracker
from ..dedup import Deduper
from ..encoder import EntailmentScorer
from .evolve import evolve
from .head import cv_selected_head, evaluate
from .proposer import Proposer, generate_pool


def _phase(msg: str) -> None:
    print(f"--- {msg}", flush=True)


def build_matrices(cfg: RunConfig, scorer, bundle, pool: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Train/test feature matrices for a pool: hypothesis features plus the optional
    lexical channel (fit on TRAIN ONLY). Shared by the runner and `compare` so both
    reconstruct a run's exact representation identically."""
    x_train = scorer.features(bundle.train_texts, pool)
    x_test = scorer.features(bundle.test_texts, pool)
    if cfg.lexical.kind != "none":
        from .lexical import LexicalFeaturizer

        lex = LexicalFeaturizer(cfg.lexical, cfg.seed).fit(bundle.train_texts)
        x_train = np.concatenate([x_train, lex.transform(bundle.train_texts)], axis=1)
        x_test = np.concatenate([x_test, lex.transform(bundle.test_texts)], axis=1)
    return x_train, x_test


def run(cfg: RunConfig, scorer=None, proposer=None, deduper=None, bundle=None) -> dict:
    out_dir = cfg.runs_dir / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(out_dir / "config.yaml")

    costs = CostTracker()
    bundle = bundle or data.load(cfg.data, cfg.seed)
    scorer = scorer or EntailmentScorer(cfg.encoder, ScoreCache(cfg.cache_dir / "nli_scores.sqlite"), costs)
    proposer = proposer or Proposer(cfg.lm, costs)
    rng = np.random.default_rng(cfg.seed)
    if deduper is None:
        if cfg.dedup.kind == "sts":  # text-space, data-free (the low-data choice)
            from ..dedup import STSDeduper

            deduper = STSDeduper(model=cfg.dedup.model, threshold=cfg.dedup.threshold)
        else:  # covariance: correlate candidate score vectors on a train subsample
            sub = data.stratified_indices(bundle.y_train, min(cfg.dedup.ref_size, len(bundle.y_train)), rng)
            deduper = Deduper(
                scorer, [bundle.train_texts[i] for i in sub], cfg.dedup.threshold, cfg.dedup.min_std
            )

    # STAGE 1 — pool: generate, or reuse a previous run's (encoder finalization)
    if cfg.pool.from_run:
        pool = json.loads((cfg.runs_dir / cfg.pool.from_run / "model.json").read_text())["hypotheses"]
        _phase(f"reusing pool of {len(pool)} from {cfg.pool.from_run}")
        history: list[dict] = []
    else:
        fixed = list(cfg.pool.fixed_hypotheses)
        _phase(f"generating pool of {cfg.pool.size}" + (f" (+{len(fixed)} fixed)" if fixed else ""))
        examples = data.labeled_examples(
            bundle.train_texts, bundle.y_train, bundle.class_names, per_class=3, rng=rng
        )
        pool = generate_pool(
            proposer, deduper, bundle.task, bundle.class_descriptions, examples, cfg.pool.size, fixed=fixed
        )
        # STAGE 2 — evolve. FIXED baseline blocks (never pruned; generated hypotheses must add
        # marginal value over them): the optional TF-IDF channel, and the user's fixed hypotheses'
        # own entail/contradict features.
        blocks = []
        if cfg.lexical.kind != "none":
            from .lexical import LexicalFeaturizer

            feat = LexicalFeaturizer(cfg.lexical, cfg.seed).fit(bundle.train_texts)
            blocks.append(feat.transform(bundle.train_texts))
        if fixed:
            blocks.append(scorer.features(bundle.train_texts, fixed))
        baseline = np.concatenate(blocks, axis=1) if blocks else None
        if cfg.pool.method == "tree":
            from .tree_evolve import tree_evolve

            _phase(f"tree-evolving (up to {cfg.pool.tree.rounds} rounds, {cfg.pool.tree.strategy})")
            pool, history = tree_evolve(
                bundle, pool, scorer, proposer, deduper, cfg.pool, cfg.seed, baseline_train=baseline
            )
        else:
            _phase(f"evolving (cap {cfg.pool.rounds} rounds, patience {cfg.pool.patience})")
            pool, history = evolve(
                bundle, pool, scorer, proposer, deduper, cfg.pool, cfg.seed, baseline_train=baseline
            )
        pool = fixed + pool  # fixed hypotheses are part of the model, ahead of the evolved ones

    # STAGE 3 — CV-selected head on the full train split; the optional lexical
    # channel (fit on TRAIN ONLY) is concatenated here and only here
    _phase(f"fitting CV-selected head on {len(bundle.train_texts)} texts x {len(pool)} hypotheses")
    if cfg.lexical.kind != "none":
        _phase(f"concatenating lexical channel: {cfg.lexical.kind} ({cfg.lexical.dims} dims)")
    x_train, x_test = build_matrices(cfg, scorer, bundle, pool)
    head, head_params, cv_acc = cv_selected_head(x_train, bundle.y_train, cfg.seed)

    # STAGE 4 — one test evaluation; pool_cv is the only headline
    _phase("evaluating on test (once)")
    results = {
        "pool_cv": evaluate(bundle.y_test, head.predict_proba(x_test), bundle.n_classes),
        "cv_train_accuracy": round(cv_acc, 4),
    }

    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_name": cfg.run_name,
                "dataset": cfg.data.name,
                "seed": cfg.seed,
                "encoder": cfg.encoder.model,
                "results": results,
            },
            indent=2,
        )
    )
    (out_dir / "model.json").write_text(
        json.dumps(
            {"type": "nli_pool", "encoder": cfg.encoder.model, "hypotheses": pool, "head": head_params},
            indent=2,
        )
    )
    with open(out_dir / "log.jsonl", "w") as f:
        for e in history:
            f.write(json.dumps(e) + "\n")
    (out_dir / "costs.json").write_text(json.dumps(costs.to_dict(), indent=2))
    return results
