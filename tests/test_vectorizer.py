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

from hypothesis_vectorizer.vectorizer import HypothesisVectorizer


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


def test_sklearn_tags_declare_string_input():
    tags = HypothesisVectorizer(HYPS).__sklearn_tags__()
    assert tags.input_tags.string and tags.input_tags.one_d_array
    assert not tags.input_tags.two_d_array  # not a numeric-array transformer
    assert not tags.target_tags.required  # no y needed to transform


def test_set_output_pandas():
    pd = pytest.importorskip("pandas")
    v = HypothesisVectorizer(HYPS).fit().set_output(transform="pandas")
    df = v.transform(["hello", "world there"])
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == list(v.get_feature_names_out())


def test_sklearn_check_estimator():
    # sklearn skips its array-based checks for string-input transformers (as for TfidfVectorizer);
    # the checks it does run (cloneability, param invariance, ...) must pass — none may fail.
    from sklearn.utils.estimator_checks import check_estimator

    results = check_estimator(HypothesisVectorizer(HYPS), on_fail=None)
    failed = [(r["check_name"], str(r["exception"])[:200]) for r in results if r["status"] == "failed"]
    assert not failed, failed


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


def test_save_requires_fitted(tmp_path):
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        HypothesisVectorizer(HYPS).save(tmp_path / "nope.json")  # save is the FITTED artifact


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


def test_from_config_carries_generation_params_and_ignores_unknowns():
    v = HypothesisVectorizer.from_config(
        {"task": "classify", "class_definitions": ["A: x"], "n_hypotheses": 8, "run_name": "junk"}
    )
    assert v.task == "classify" and v.class_definitions == ["A: x"] and v.n_hypotheses == 8
    assert not hasattr(v, "hypotheses_")  # no hypotheses -> unfitted, ready for fit(X, y)


def test_from_run(tmp_path):
    run = tmp_path / "runs" / "myrun"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text("encoder: {model: enc/x, device: cpu}\nscore_mode: entail\n")
    (run / "model.json").write_text(json.dumps({"hypotheses": HYPS}))
    v = HypothesisVectorizer.from_run(run)  # config.yaml encoder + model.json pool -> fitted
    assert v.hypotheses_ == HYPS and v.encoder == "enc/x" and v.score_mode == "entail"
    assert v.transform(["a", "bb"]).shape == (2, len(HYPS))


def test_fit_generates_when_no_hypotheses(monkeypatch):
    import hypothesis_vectorizer.dedup as dedup_mod
    import hypothesis_vectorizer.train.proposer as prop_mod

    fake = FakeProposer(generate_batches=[["hypothesis A", "hypothesis B", "hypothesis C"]])
    monkeypatch.setattr(prop_mod, "Proposer", lambda *a, **k: fake)
    monkeypatch.setattr(dedup_mod, "Deduper", lambda *a, **k: TextOnlyDeduper())

    v = HypothesisVectorizer(task="classify", class_definitions=["A: x", "B: y"], n_hypotheses=3)
    v.fit(["t one", "t two", "t three", "t four"], [0, 1, 0, 1])  # no hypotheses -> generate
    assert v.hypotheses_ == ["hypothesis A", "hypothesis B", "hypothesis C"]
    assert fake.generate_calls  # the LM proposer was actually invoked
    assert v.transform(["z"]).shape == (1, 2 * 3)


def test_sts_deduper_drops_near_duplicates(monkeypatch):
    from hypothesis_vectorizer.dedup import STSDeduper

    # fake embeddings: first word one-hot -> same first word = cosine 1.0
    def fake_embed(self, texts):
        words = sorted({t.split()[0] for t in texts})
        return np.array([np.eye(len(words))[words.index(t.split()[0])] for t in texts])

    monkeypatch.setattr(STSDeduper, "_embed", fake_embed)
    d = STSDeduper(threshold=0.9)
    seen: set[str] = set()
    kept, rejected = d.filter(
        ["alpha statement one", "alpha statement two", "beta statement"], against=[], seen=seen
    )
    assert kept == ["alpha statement one", "beta statement"]  # 2nd alpha = near-duplicate
    assert any("sts" in r for r in rejected)


def test_fit_dedup_accepts_custom_object_and_rejects_unknown(monkeypatch):
    import hypothesis_vectorizer.train.proposer as prop_mod

    fake = FakeProposer(generate_batches=[["h one", "h two"]])
    monkeypatch.setattr(prop_mod, "Proposer", lambda *a, **k: fake)
    custom = TextOnlyDeduper()
    v = HypothesisVectorizer(task="t", class_definitions=["A: x"], n_hypotheses=2, dedup=custom)
    v.fit(["a", "b"], [0, 1])
    assert v.hypotheses_ == ["h one", "h two"]  # went through the custom deduper

    with pytest.raises(ValueError):
        HypothesisVectorizer(task="t", class_definitions=["A: x"], dedup="nonsense").fit(["a"], [0])


def test_fit_passes_baseline_features_to_evolution(monkeypatch):
    import hypothesis_vectorizer.train.evolve as evolve_mod
    import hypothesis_vectorizer.train.proposer as prop_mod

    fake = FakeProposer(generate_batches=[["h one", "h two"]])
    monkeypatch.setattr(prop_mod, "Proposer", lambda *a, **k: fake)
    captured = {}

    def fake_evolve(bundle, pool, scorer, proposer, deduper, cfg, seed, baseline_train=None):
        captured["baseline"] = baseline_train
        return pool, [{"round": 0, "heldout_acc": 0.9, "pool": list(pool)}]

    monkeypatch.setattr(evolve_mod, "evolve", fake_evolve)

    Z = np.random.default_rng(0).random((4, 3))  # e.g. other tabular columns
    v = HypothesisVectorizer(
        task="t", class_definitions=["A: x"], n_hypotheses=2, evolve=True, dedup=TextOnlyDeduper()
    )
    v.fit(["a", "b", "c", "d"], [0, 1, 0, 1], baseline_features=Z)
    assert captured["baseline"] is not None and captured["baseline"].shape == (4, 3)
    assert v.evolution_history_[0]["pool"]  # per-round pools saved on the instance

    with pytest.raises(ValueError):  # misaligned baseline is refused
        v.fit(["a", "b"], [0, 1], baseline_features=np.zeros((3, 2)))


def test_fit_evolves_when_enabled(monkeypatch):
    import hypothesis_vectorizer.dedup as dedup_mod
    import hypothesis_vectorizer.train.proposer as prop_mod

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


def test_fixed_hypotheses_prepended_and_protected(monkeypatch):
    import hypothesis_vectorizer.train.proposer as prop_mod

    # provided-hypotheses path: fixed come first, duplicates collapsed
    v = HypothesisVectorizer(["h a", "FIX one"], fixed_hypotheses=["FIX one", "FIX two"]).fit()
    assert v.hypotheses_ == ["FIX one", "FIX two", "h a"]

    # generation path: fixed are in the LM's avoid list and dedup baseline, and lead the pool
    fake = FakeProposer(generate_batches=[["gen one", "gen two"]])
    monkeypatch.setattr(prop_mod, "Proposer", lambda *a, **k: fake)
    w = HypothesisVectorizer(
        task="t",
        class_definitions=["A: x"],
        n_hypotheses=2,
        fixed_hypotheses=["FIX one"],
        dedup=TextOnlyDeduper(),
    )
    w.fit(["a", "b"], [0, 1])
    assert w.hypotheses_ == ["FIX one", "gen one", "gen two"]
    assert "FIX one" in fake.generate_calls[0]["avoid"]  # LM told not to duplicate it


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
