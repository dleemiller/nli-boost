"""Tree-guided evolve, exercised without a GPU or an LM.

FakeScorer maps hypothesis 'f<i> ...' to feature column i of 'v0|v1|...' texts, so a proposer that
returns 'f0 ...' hands tree_evolve a column that actually separates the synthetic classes — the same
mechanism the other fake-driven tests use.
"""

from types import SimpleNamespace

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
    """split_leaf always returns the same statement, scored via the real reward_fn (info gain)."""

    def __init__(self, hyp):
        self.hyp = hyp
        self.calls = 0

    def split_leaf(self, *, reward_fn, **kwargs):
        self.calls += 1
        return self.hyp, float(reward_fn(kwargs, SimpleNamespace(hypothesis=self.hyp)))


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
