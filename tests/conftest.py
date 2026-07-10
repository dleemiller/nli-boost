"""Fakes for GPU/LM-free testing of the full pipeline.

FakeScorer: texts encode feature values as 'v0|v1|...'; hypothesis 'f<i>'
reads column i. features() mirrors EntailmentScorer's (n, 2m) layout with an
uninformative contradiction half.
"""

import numpy as np

from hypothesis_vectorizer.train.data import Bundle


class FakeScorer:
    def __init__(self):
        self.pairs_scored = 0

    def features(self, texts, pool):
        vals = np.array([[float(v) for v in t.split("|")] for t in texts])
        cols = [int(h.split()[0][1:]) for h in pool]  # "f3 ..." -> column 3
        e = vals[:, cols]
        self.pairs_scored += e.size
        return np.concatenate([e, np.full_like(e, 0.5)], axis=1)


class FakeProposer:
    """Returns queued responses; records every call for assertions."""

    def __init__(self, generate_batches=None, refill_batches=None):
        self.generate_batches = list(generate_batches or [])
        self.refill_batches = list(refill_batches or [])
        self.generate_calls = []
        self.refill_calls = []

    def generate(self, task, class_definitions, examples, n, avoid, opening_hints=()):
        self.generate_calls.append(dict(n=n, avoid=list(avoid), opening_hints=list(opening_hints)))
        return self.generate_batches.pop(0) if self.generate_batches else []

    def refill(
        self, task, class_definitions, examples, survivors, failed, confusion_evidence, n, opening_hints=()
    ):
        self.refill_calls.append(
            dict(
                n=n,
                survivors=list(survivors),
                failed=list(failed),
                confusion_evidence=list(confusion_evidence),
                opening_hints=list(opening_hints),
            )
        )
        return self.refill_batches.pop(0) if self.refill_batches else []


class TextOnlyDeduper:
    """Real textual dedup behavior without the STS model."""

    def filter(self, candidates, against, seen):
        from hypothesis_vectorizer.dedup import norm_statement

        kept, rejected = [], []
        for c in candidates:
            key = norm_statement(c)
            if key and key not in seen:
                seen.add(key)
                kept.append(c)
            else:
                rejected.append(c)
        return kept, rejected


def encode(x: np.ndarray) -> list[str]:
    return ["|".join(f"{v:.6f}" for v in row) for row in np.atleast_2d(x)]


def make_bundle(n=200, seed=0, n_features=8, n_classes=4):
    """f0, f1 informative (define the label); remaining features are noise;
    the LAST feature column is constant (guaranteed confident-dead)."""
    rng = np.random.default_rng(seed)
    x = rng.random((n, n_features))
    x[:, -1] = 0.5  # constant -> undetectable by construction
    y = (x[:, 0] > 0.5).astype(int) * 2 + (x[:, 1] > 0.5).astype(int)
    y = y % n_classes
    names = [f"C{i}" for i in range(n_classes)]
    half = n // 2
    return Bundle(
        name="fake",
        task="separate the classes",
        class_names=names,
        class_descriptions=[f"{c}: fake definition" for c in names],
        train_texts=encode(x[:half]),
        y_train=y[:half],
        val_texts=encode(x[half : half + n // 4]),
        y_val=y[half : half + n // 4],
        test_texts=encode(x[half + n // 4 :]),
        y_test=y[half + n // 4 :],
    )


import pytest  # noqa: E402


@pytest.fixture
def fast_models(monkeypatch):
    """Shrink every estimator-size knob for LOGIC tests: same code paths, same wiring, same
    assertions — tiny estimator budgets. Head/evolve behavior on separable fake data is
    identical at 20 trees vs 300; the prod sizes are a quality knob, not a logic branch.
    The 0.964 reproduction test (marked slow, opt-in) is the only one needing prod sizes."""
    from hypothesis_vectorizer.train import evolve, head

    monkeypatch.setattr(head, "_RF_TREES", 20)
    monkeypatch.setattr(head, "_HGB_ITERS", 20)
    monkeypatch.setattr(evolve, "_RANK_ITERS", 20)
    monkeypatch.setattr(evolve, "_PERM_REPEATS", 1)
    monkeypatch.setattr(
        head,
        "_GRID",
        [
            dict(kind="rf", min_samples_leaf=5, max_features=0.6),
            dict(kind="hgb", learning_rate=0.12, l2_regularization=0.01),
        ],
    )
