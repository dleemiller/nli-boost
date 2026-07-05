"""HypothesisVectorizer: sklearn-transformer contract, composition, and persistence.

A class-level fake scorer stands in for the GPU encoder so these run on CPU and survive
sklearn clone() (Pipeline / ColumnTransformer clone their steps on fit)."""

import json
import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from conftest import FakeProposer, TextOnlyDeduper

from nli_boost.vectorizer import HypothesisVectorizer


class _FakeScorer:
    """features(texts, pool) -> (n, 2m) = [entail | contradict], text-dependent so a head can learn."""

    def features(self, texts, pool):
        m = len(pool)
        rows = []
        for t in texts:
            base = (len(str(t)) % 5) / 10.0
            e = np.clip(np.linspace(0.1, 0.9, m) + base, 0.0, 1.0)
            rows.append(np.concatenate([e, 1.0 - e]))
        return np.asarray(rows, dtype=np.float32)


@pytest.fixture(autouse=True)
def _fake_scorer(monkeypatch):
    monkeypatch.setattr(HypothesisVectorizer, "_get_scorer", lambda self: _FakeScorer())


HYPS = ["The text is about sports.", "The text asks for a number.", "The text names a place."]


def test_fit_transform_shape_and_feature_names():
    v = HypothesisVectorizer(HYPS).fit()
    X = v.transform(["hello world", "another example text"])
    assert X.shape == (2, 2 * len(HYPS))  # entail | contradict
    names = list(v.get_feature_names_out())
    assert names[: len(HYPS)] == [f"entail: {h}" for h in HYPS]
    assert names[len(HYPS) :] == [f"contradict: {h}" for h in HYPS]


def test_score_modes():
    e = HypothesisVectorizer(HYPS, score_mode="entail").fit().transform(["a", "bb"])
    c = HypothesisVectorizer(HYPS, score_mode="contrast").fit().transform(["a", "bb"])
    both = HypothesisVectorizer(HYPS, score_mode="entail_contradict").fit().transform(["a", "bb"])
    m = len(HYPS)
    assert e.shape == (2, m) and c.shape == (2, m)
    np.testing.assert_allclose(e, both[:, :m], rtol=1e-5)
    np.testing.assert_allclose(c, both[:, :m] - both[:, m:], rtol=1e-5)


def test_bad_score_mode_and_missing_hypotheses():
    with pytest.raises(ValueError):
        HypothesisVectorizer(None).fit()
    with pytest.raises(ValueError):
        HypothesisVectorizer(HYPS, score_mode="nonsense").fit()


def test_accepts_text_column_shapes():
    v = HypothesisVectorizer(HYPS).fit()
    a = v.transform(["x", "yy"])  # 1-D list
    b = v.transform(np.array([["x"], ["yy"]], dtype=object))  # (n,1) column, as ColumnTransformer hands over
    np.testing.assert_allclose(a, b)
    with pytest.raises(ValueError):  # more than one column is ambiguous
        v.transform(np.array([["x", "z"], ["yy", "w"]], dtype=object))


def test_sklearn_clone_preserves_params():
    v = HypothesisVectorizer(HYPS, score_mode="entail", batch_size=64)
    w = clone(v)
    assert w.get_params() == v.get_params()


def test_pipeline_and_column_transformer_compose():
    y = [0, 1, 0, 1]
    texts = ["short", "a much longer piece of text here", "tiny", "medium length text"]
    # Pipeline: text -> hypotheses -> classifier
    pipe = Pipeline([("hyp", HypothesisVectorizer(HYPS)), ("clf", LogisticRegression())])
    pipe.fit(texts, y)
    assert pipe.predict(texts).shape == (4,)

    # ColumnTransformer: one text column scored, a numeric column passed through
    X = np.array([[t, float(i)] for i, t in enumerate(texts)], dtype=object)
    ct = ColumnTransformer([("hyp", HypothesisVectorizer(HYPS), [0]), ("num", "passthrough", [1])])
    out = ct.fit_transform(X)
    assert out.shape == (4, 2 * len(HYPS) + 1)  # nli features + the passthrough numeric column


def test_save_load_roundtrip(tmp_path):
    v = HypothesisVectorizer(HYPS, score_mode="entail", encoder="some/encoder").fit()
    p = tmp_path / "vec.json"
    v.save(p)
    w = HypothesisVectorizer.load(p)
    assert w.hypotheses_ == HYPS and w.score_mode == "entail" and w.encoder == "some/encoder"
    np.testing.assert_allclose(w.transform(["a", "bb"]), v.transform(["a", "bb"]))


def test_from_config_yaml(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "encoder: {model: dleemiller/finecat-nli-l, device: cpu, batch_size: 16}\n"
        "score_mode: entail\n"
        f"hypotheses: [{', '.join(repr(h) for h in HYPS)}]\n"
    )
    v = HypothesisVectorizer.from_config(cfg)
    assert v.encoder == "dleemiller/finecat-nli-l" and v.batch_size == 16 and v.device == "cpu"
    assert v.hypotheses_ == HYPS  # present -> fitted
    assert v.transform(["a"]).shape == (1, len(HYPS))


def test_from_run(tmp_path):
    run = tmp_path / "runs" / "myrun"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text("encoder: {model: enc/x, device: cpu}\nscore_mode: entail\n")
    (run / "model.json").write_text(json.dumps({"hypotheses": HYPS}))
    v = HypothesisVectorizer.from_run(run)  # config.yaml encoder + model.json pool -> fitted
    assert v.hypotheses_ == HYPS and v.encoder == "enc/x" and v.score_mode == "entail"
    assert v.transform(["a", "bb"]).shape == (2, len(HYPS))


def test_fit_generates_when_no_hypotheses(monkeypatch):
    import nli_boost.dedup as dedup_mod
    import nli_boost.proposer as prop_mod

    fake = FakeProposer(generate_batches=[["hypothesis A", "hypothesis B", "hypothesis C"]])
    monkeypatch.setattr(prop_mod, "Proposer", lambda *a, **k: fake)
    monkeypatch.setattr(dedup_mod, "Deduper", lambda *a, **k: TextOnlyDeduper())

    v = HypothesisVectorizer(task="classify", class_definitions=["A: x", "B: y"], n_hypotheses=3)
    v.fit(["t one", "t two", "t three", "t four"], [0, 1, 0, 1])  # no hypotheses -> generate
    assert v.hypotheses_ == ["hypothesis A", "hypothesis B", "hypothesis C"]
    assert fake.generate_calls  # the LM proposer was actually invoked
    assert v.transform(["z"]).shape == (1, 2 * 3)


def test_fit_evolves_when_enabled(monkeypatch):
    import nli_boost.dedup as dedup_mod
    import nli_boost.proposer as prop_mod

    fake = FakeProposer(
        generate_batches=[[f"gen {i}" for i in range(8)]],
        refill_batches=[[f"ref {r}-{i}" for i in range(4)] for r in range(6)],
    )
    monkeypatch.setattr(prop_mod, "Proposer", lambda *a, **k: fake)
    monkeypatch.setattr(dedup_mod, "Deduper", lambda *a, **k: TextOnlyDeduper())

    texts = [f"text sample number {i}" for i in range(40)]
    y = np.array([0, 1] * 20)
    v = HypothesisVectorizer(task="classify", class_definitions=["A: x", "B: y"], n_hypotheses=8, evolve=True)
    v.fit(texts, y)  # generate -> evolve (CV-prune/refill)
    assert v.hypotheses_
    assert fake.generate_calls and fake.refill_calls  # both generation AND evolution ran
    assert v.transform(["z"]).shape == (1, 2 * len(v.hypotheses_))


def test_fit_generate_requires_data_and_metadata():
    with pytest.raises(ValueError):
        HypothesisVectorizer().fit()  # no hypotheses and no (X, y)
    with pytest.raises(ValueError):
        HypothesisVectorizer(class_definitions=["A"]).fit(["a"], [0])  # missing task


def test_pickle_drops_live_scorer():
    v = HypothesisVectorizer(HYPS).fit()
    blob = pickle.dumps(v)  # must not choke on a live sqlite/encoder handle
    w = pickle.loads(blob)
    assert w.hypotheses_ == HYPS
    assert w.transform(["a", "bb"]).shape == (2, 2 * len(HYPS))
