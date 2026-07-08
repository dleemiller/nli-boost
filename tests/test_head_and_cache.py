import numpy as np
from conftest import FakeScorer, make_bundle

from hypothesis_vectorizer.cache import ScoreCache
from hypothesis_vectorizer.encoder import digest, normalize
from hypothesis_vectorizer.train.head import cv_selected_head, evaluate


def test_cv_selected_head_learns_separable_data(fast_models):
    bundle = make_bundle(n=600)
    scorer = FakeScorer()
    pool = [f"f{i}" for i in range(8)]
    x = scorer.features(bundle.train_texts, pool)
    head, params, cv_acc = cv_selected_head(x, bundle.y_train, seed=0)
    assert cv_acc > 0.9  # f0/f1 define the label
    assert params["kind"] in ("rf", "hgb")
    x_test = scorer.features(bundle.test_texts, pool)
    m = evaluate(bundle.y_test, head.predict_proba(x_test), bundle.n_classes)
    assert m["accuracy"] > 0.9
    assert set(m) == {"accuracy", "macro_f1", "logloss"}


def test_cache_roundtrip_and_model_isolation(tmp_path):
    cache = ScoreCache(tmp_path / "c.sqlite")
    hh = digest("The text is about sports.")
    rows = [(digest(f"t{i}"), f"t{i}", np.array([i, 0.5, -1.0], dtype=np.float32)) for i in range(700)]
    cache.put_logits("m", hh, "The text is about sports.", rows)
    got = cache.get_logits("m", hh, [r[0] for r in rows] + [digest("missing")])
    assert len(got) == 700
    np.testing.assert_allclose(got[rows[3][0]], [3.0, 0.5, -1.0])
    assert cache.get_logits("other-model", hh, [rows[0][0]]) == {}


def test_normalize_is_cache_key_stable():
    assert normalize("Hello   world\n\tfoo", 1200) == normalize("Hello world foo", 1200)
    assert len(normalize("x" * 5000, 1200)) == 1200
