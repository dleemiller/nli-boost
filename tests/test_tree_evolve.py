"""Tree-guided evolve, exercised without a GPU or an LM.

FakeScorer maps hypothesis 'f<i> ...' to feature column i of 'v0|v1|...' texts, so a proposer that
returns 'f0 ...' hands tree_evolve a column that actually separates the synthetic classes — the same
mechanism the other fake-driven tests use.
"""

import numpy as np
from conftest import FakeScorer, TextOnlyDeduper, encode

from hypothesis_vectorizer.config import PoolConfig, TreeConfig
from hypothesis_vectorizer.train.data import Bundle
from hypothesis_vectorizer.train.tree_evolve import (
    _best_split_gain,
    _entropy,
    _pick_leaf,
    tree_evolve,
)


class TreeFakeProposer:
    """split_leaf always returns the same statement, scored via the real evaluate_fn."""

    def __init__(self, hyp):
        self.hyp = hyp
        self.calls = 0
        self.last_related = None

    def split_leaf(self, *, evaluate_fn, related_hypotheses=None, **kwargs):
        self.calls += 1
        self.last_related = related_hypotheses
        return self.hyp, float(evaluate_fn(self.hyp)["score"])


def _binary_bundle(n=80, seed=0):
    """col0 cleanly separates the two classes; cols 1..3 are noise."""
    rng = np.random.default_rng(seed)
    x = rng.random((n, 4))
    y = (rng.random(n) > 0.5).astype(np.int64)
    x[:, 0] = np.clip(np.where(y == 1, 0.85, 0.15) + rng.normal(0, 0.03, n), 0, 1)
    return Bundle(
        name="t",
        task="separate A from B",
        class_names=["A", "B"],
        class_descriptions=["A: class zero", "B: class one"],
        train_texts=encode(x),
        y_train=y,
        val_texts=[],
        y_val=np.array([], dtype=np.int64),
        test_texts=[],
        y_test=np.array([], dtype=np.int64),
    )


def _cfg(**tree_kwargs):
    defaults = dict(
        rounds=5,
        refine_attempts=1,
        leaf_shots=6,
        leaf_min_samples=15,
        max_depth=3,
        min_samples_leaf=20,
        patience=2,
    )
    defaults.update(tree_kwargs)
    return PoolConfig(method="tree", size=2, tree=TreeConfig(**defaults))


def test_entropy_and_split_gain():
    assert _entropy(np.array([0, 0, 0, 0])) == 0.0
    assert abs(_entropy(np.array([0, 0, 1, 1])) - 1.0) < 1e-9
    y = np.array([0] * 10 + [1] * 10)
    perfect = np.array([0.1] * 10 + [0.9] * 10)  # threshold at 0.5 splits cleanly
    assert _best_split_gain(perfect, y, _entropy(y)) > 0.99
    assert _best_split_gain(np.full(20, 0.5), y, _entropy(y)) == 0.0  # constant -> no gain


def test_pick_leaf_targets_impurity_and_skips_pure():
    y = np.array([0, 0, 1, 1] * 5)  # 20 samples
    leaf = np.array([1] * 10 + [2] * 10)
    # leaf 1 pure (all class 0), leaf 2 impure -> pick leaf 2
    y2 = np.array([0] * 10 + [0, 1] * 5)
    assert _pick_leaf(leaf, y2, min_samples=5) == 2
    assert _pick_leaf(leaf, np.zeros(20, dtype=int), min_samples=5) is None  # all pure
    assert _pick_leaf(leaf, y, min_samples=50) is None  # none big enough


def test_tree_evolve_grows_pool_with_separating_hypothesis():
    bundle = _binary_bundle()
    proposer = TreeFakeProposer("f0 separates the classes")
    pool, history = tree_evolve(
        bundle,
        ["f1 noise one", "f2 noise two"],  # initial pool reads only noise columns
        FakeScorer(),
        proposer,
        TextOnlyDeduper(),
        _cfg(),
        seed=0,
    )
    # the informative hypothesis was proposed, rewarded, and added exactly once
    assert "f0 separates the classes" in pool
    assert len(pool) == 3
    added = [h for h in history if h["added"]]
    assert len(added) == 1 and added[0]["info_gain"] > 0.5
    # once f0 is in, the tree is pure -> it stops instead of proposing again
    assert proposer.calls == 1


def test_tree_evolve_dedups_and_stops_on_patience():
    bundle = _binary_bundle()
    # proposer keeps returning a hypothesis already in the pool -> always deduped, never added
    proposer = TreeFakeProposer("f1 noise one")
    pool, history = tree_evolve(
        bundle, ["f1 noise one"], FakeScorer(), proposer, TextOnlyDeduper(), _cfg(patience=2), seed=0
    )
    assert pool == ["f1 noise one"]  # nothing added
    assert all(not h["added"] for h in history)
    assert proposer.calls == 2  # stopped after `patience` no-add rounds


def test_runner_from_run_tree_evolves_reused_pool(tmp_path, fast_models, monkeypatch):
    """from_run + method='tree': the reused (truncated) pool is the STARTING pool and grows."""
    import json

    from conftest import FakeProposer, make_bundle
    from hypothesis_vectorizer.config import DataConfig, RunConfig
    from hypothesis_vectorizer.train import runner as runner_mod
    from hypothesis_vectorizer.train.runner import run

    bundle = make_bundle()
    cfg0 = RunConfig(
        run_name="src",
        data=DataConfig(name="trec"),
        pool={"size": 4, "rounds": 0},
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
    )
    # first two are NOISE columns so the reused pool leaves the tree impure -> growth happens
    run(
        cfg0,
        scorer=FakeScorer(),
        proposer=FakeProposer([["f4 noise", "f5 noise", "f0 a", "f1 b"]]),
        deduper=TextOnlyDeduper(),
        bundle=bundle,
    )

    # tree run reuses the first 2 hypotheses and grows via split_leaf
    tree_proposer = TreeFakeProposer("f1 b grows back")
    monkeypatch.setattr(runner_mod, "Proposer", lambda *a, **k: tree_proposer)
    cfg1 = RunConfig(
        run_name="treed",
        data=DataConfig(name="trec"),
        pool={
            "from_run": "src",
            "from_run_top": 2,
            "method": "tree",
            # shallow tree: noise splits at depth 6 shred 100 samples below leaf_min_samples
            "tree": {
                "rounds": 2,
                "max_depth": 2,
                "leaf_min_samples": 10,
                "min_samples_leaf": 10,
                "patience": 1,
            },
        },
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
    )
    run(cfg1, scorer=FakeScorer(), proposer=tree_proposer, deduper=TextOnlyDeduper(), bundle=bundle)
    model = json.loads((tmp_path / "runs" / "treed" / "model.json").read_text())
    assert model["hypotheses"][:2] == ["f4 noise", "f5 noise"]  # truncation kept the first 2
    assert tree_proposer.calls >= 1  # tree evolution actually ran on the reused pool


def test_evaluator_zeroes_covariant_candidates_and_names_the_culprit():
    from hypothesis_vectorizer.train.tree_evolve import _make_evaluator

    bundle = _binary_bundle()
    pool = ["f0 already reads the signal"]
    scorer = FakeScorer()
    x_leaf = scorer.features(bundle.train_texts, pool)  # existing pool features on the "leaf"
    ev = _make_evaluator(scorer, bundle.train_texts, bundle.y_train, pool, x_leaf)
    # a candidate reading the SAME column: high gain but zero novelty -> score ~0, culprit named
    r = ev("f0 same column different words")
    assert r["gain"] > 0.5 and r["novelty"] < 0.05 and r["score"] < 0.05
    assert r["covariant_with"] == "f0 already reads the signal"
    # an unrelated noisy column: novel but useless -> low gain dominates
    r2 = ev("f2 noise")
    assert r2["novelty"] > 0.5 and r2["gain"] < 0.3


def test_related_hypotheses_ranks_informative_first():
    from hypothesis_vectorizer.train.tree_evolve import _related_hypotheses

    bundle = _binary_bundle()
    pool = ["f1 noise", "f0 informative"]
    x_leaf = FakeScorer().features(bundle.train_texts, pool)
    rel = _related_hypotheses(x_leaf, bundle.y_train, pool, top=2)
    assert "f0 informative" in rel[0] and rel[0].startswith("resolves")
