import numpy as np
from conftest import FakeProposer, FakeScorer, TextOnlyDeduper, make_bundle

from hypothesis_vectorizer.config import DataConfig, LexicalConfig, PoolConfig, RunConfig
from hypothesis_vectorizer.train.lexical import LexicalFeaturizer
from hypothesis_vectorizer.train.runner import run


def test_tfidf_svd_shapes_and_train_only_fit():
    train = [f"alpha beta doc {i} gamma" for i in range(40)] + [
        f"delta epsilon doc {i} zeta" for i in range(40)
    ]
    lex = LexicalFeaturizer(LexicalConfig(kind="tfidf_svd", dims=8), seed=0).fit(train)
    xt = lex.transform(train)
    assert xt.shape == (80, 8) and xt.dtype == np.float32
    # transform of unseen text works without refitting (no leakage by construction)
    xu = lex.transform(["alpha unseen zeta"])
    assert xu.shape == (1, 8)


def test_tfidf_svd_clamps_dims_to_vocabulary():
    tiny = ["aa bb", "aa bb", "aa cc", "aa cc"]  # vocabulary smaller than requested dims
    lex = LexicalFeaturizer(LexicalConfig(kind="tfidf_svd", dims=64), seed=0).fit(tiny)
    assert lex.transform(tiny).shape[1] < 64


def test_runner_concatenates_lexical_channel(tmp_path, fast_models):
    cfg = RunConfig(
        run_name="lex",
        data=DataConfig(name="trec"),
        pool=PoolConfig(size=8, rounds=1, patience=2, rank_sample=0),
        lexical=LexicalConfig(kind="tfidf_svd", dims=4),
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
    )
    proposer = FakeProposer(generate_batches=[[f"f{i}" for i in range(8)]], refill_batches=[[]])
    results = run(
        cfg, scorer=FakeScorer(), proposer=proposer, deduper=TextOnlyDeduper(), bundle=make_bundle()
    )
    # fake texts ("0.1|0.5|...") still vectorize; the run completes with the wider matrix
    assert set(results) == {"pool_cv", "cv_train_accuracy"}
    assert results["pool_cv"]["accuracy"] > 0.8  # informative hypothesis features still dominate
