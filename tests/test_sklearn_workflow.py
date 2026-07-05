"""Integration: the standard sklearn workflow reproduces the headline 0.964 TF-IDF run.

Real-data regression guard — skipped unless the trained run and its score cache are present
(so CI / fake-only environments skip it). When present it runs on cached -l scores (no GPU)."""

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not (Path("runs/trec_best_l_max/model.json").exists() and Path("cache/nli_scores.sqlite").exists()),
    reason="needs the trained best_l_max run + its score cache",
)

RUN = "runs/trec_best_l_max"


def _pipeline():
    from nli_boost import HypothesisVectorizer
    from nli_boost.config import RunConfig
    from sklearn.decomposition import TruncatedSVD
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline

    cfg = RunConfig.from_yaml(f"{RUN}/config.yaml")
    head = json.load(open(f"{RUN}/model.json"))["head"]
    features = FeatureUnion(
        [
            ("nli", HypothesisVectorizer.from_run(RUN)),
            (
                "tfidf",
                make_pipeline(
                    TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                    TruncatedSVD(n_components=cfg.lexical.dims, random_state=cfg.seed),
                ),
            ),
        ]
    )
    clf = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=head["learning_rate"],
        l2_regularization=head["l2_regularization"],
        random_state=cfg.seed,
    )
    return Pipeline([("features", features), ("clf", clf)]), cfg


def test_sklearn_pipeline_reproduces_0964():
    from nli_boost.data import load
    from sklearn.base import clone

    pipe, cfg = _pipeline()
    clone(pipe)  # params must round-trip through FeatureUnion + vectorizer
    b = load(cfg.data, cfg.seed)
    pipe.fit(b.train_texts, b.y_train)
    acc = pipe.score(b.test_texts, b.y_test)
    assert acc == pytest.approx(0.964, abs=1e-9)  # exact reproduction of the run's pool_cv
    names = pipe.named_steps["features"].get_feature_names_out()
    assert len(names) == 2 * 64 + cfg.lexical.dims  # entail|contradict + tfidf-svd
    assert np.asarray(names)[0].startswith("nli__entail: ")
