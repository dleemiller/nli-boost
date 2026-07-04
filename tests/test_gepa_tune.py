import numpy as np
from conftest import make_bundle

from nli_boost.gepa_tune import PoolRewardMetric


class _StubScorer:
    """Returns deterministic (n, 2m) features; hypothesis text seeds each column."""

    def features(self, texts, pool):
        n = len(texts)
        cols = []
        for h in pool:
            rng = np.random.default_rng(abs(hash(h)) % (2**32))
            cols.append(rng.random(n))
        e = np.column_stack(cols) if cols else np.empty((n, 0))
        return np.concatenate([e, 1 - e], axis=1)


class _Pred:
    def __init__(self, statements):
        self.hypotheses = [type("H", (), {"statement": s})() for s in statements]


class _Gold(dict):
    __getattr__ = dict.get


def _gold(bundle):
    return _Gold(
        dataset="fake",
        seed=0,
        task=bundle.task,
        class_definitions=bundle.class_descriptions,
        sub=list(range(len(bundle.train_texts))),
    )


def test_metric_returns_scored_feedback():
    bundle = make_bundle()
    metric = PoolRewardMetric(_StubScorer(), {("fake", 0): bundle})
    out = metric(_gold(bundle), _Pred([f"hypothesis {i}" for i in range(8)]))
    assert 0.0 <= out.score <= 1.0
    assert "fake" in out.feedback and "dataset-agnostic" in out.feedback


def test_metric_handles_empty_pool():
    bundle = make_bundle()
    metric = PoolRewardMetric(_StubScorer(), {("fake", 0): bundle})
    out = metric(_gold(bundle), _Pred([]))
    assert out.score == 0.0


def test_metric_dedups_before_scoring():
    bundle = make_bundle()
    metric = PoolRewardMetric(_StubScorer(), {("fake", 0): bundle})
    # duplicates collapse; a single unique statement still scores without error
    out = metric(_gold(bundle), _Pred(["same", "same", "same"]))
    assert 0.0 <= out.score <= 1.0
