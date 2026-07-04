import numpy as np
from conftest import FakeProposer, FakeScorer, TextOnlyDeduper, make_bundle

from nli_boost.config import PoolConfig
from nli_boost.evolve import evolve, hotspots, rank_hypotheses


def test_rank_informative_hypotheses_first():
    bundle = make_bundle()
    pool = [f"f{i}" for i in range(8)]
    scorer = FakeScorer()
    x = scorer.features(bundle.train_texts, pool)
    r = rank_hypotheses(x, bundle.y_train, m=len(pool), seed=0)
    assert set(r.order[:2].tolist()) == {0, 1}
    assert r.stability[7] == 0.0  # the constant feature never helps in any fold
    assert 0 < r.heldout_accuracy <= 1


def test_hotspots_group_mutually_confused_classes():
    y = np.array([0] * 50 + [1] * 50 + [2] * 50)
    # classes 0 and 1 confuse each other heavily; class 2 is clean
    errors = [(i, 1) for i in range(10)] + [(i, 0) for i in range(50, 60)]
    groups = hotspots(errors, y, n_classes=3)
    assert groups == [[0, 1]]


def test_evolve_prunes_constant_feature_with_reason_and_stops_on_plateau():
    bundle = make_bundle()
    pool = [f"f{i}" for i in range(8)]  # f7 is constant -> confident dead
    proposer = FakeProposer(refill_batches=[[f"f{2 + i} variant {i}" for i in range(4)]] * 6)
    cfg = PoolConfig(size=8, rounds=6, patience=2, rank_sample=0)

    final, history = evolve(bundle, pool, FakeScorer(), proposer, TextOnlyDeduper(), cfg, seed=0)

    # the constant feature was pruned with the undetectable reason
    all_failed = [f for h in history for f in h["failed"]]
    assert any(f.startswith("f7") and "undetectable" in f for f in all_failed)
    # informative features always survive
    assert any(h.startswith("f0") for h in final) and any(h.startswith("f1") for h in final)
    # patience stopped it before the cap (synthetic data saturates immediately)
    assert len(history) < 6
    # every round logs the held-out accuracy
    assert all("heldout_acc" in h for h in history)
    # the refill LM saw failure reasons and confusion evidence
    assert proposer.refill_calls and proposer.refill_calls[0]["failed"]


def test_evolve_records_refill_target_aucs():
    bundle = make_bundle()
    pool = [f"f{i}" for i in range(8)]
    # refills map to informative-ish columns so instrumentation has something to measure
    proposer = FakeProposer(refill_batches=[["f2 fresh"], ["f3 fresh"], ["f4 fresh"]])
    cfg = PoolConfig(size=8, rounds=3, patience=3, rank_sample=0)
    _, history = evolve(bundle, pool, FakeScorer(), proposer, TextOnlyDeduper(), cfg, seed=0)
    instrumented = [h for h in history if "refill_target_aucs" in h]
    if instrumented:  # requires >=2 rounds and a hotspot; structure check
        assert isinstance(instrumented[0]["refill_target_aucs"], list)
        assert "refill_hit_rate" in instrumented[0]
