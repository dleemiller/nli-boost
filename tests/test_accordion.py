"""Accordion loop, exercised without GPU/LM (FakeScorer maps 'f<i>' -> feature column i)."""

import numpy as np
from conftest import FakeProposer, FakeScorer, TextOnlyDeduper, make_bundle

from hypothesis_vectorizer.train.accordion import accordion, effective_rank, _keep_representatives


def test_effective_rank_counts_independent_directions():
    rng = np.random.default_rng(0)
    indep = rng.standard_normal((200, 6))  # 6 independent columns -> rank ~6 at 90%
    assert effective_rank(indep, 0.90) >= 5
    base = rng.standard_normal((200, 1))
    corr = np.hstack([base + 0.01 * rng.standard_normal((200, 1)) for _ in range(6)])  # 1 real direction
    assert effective_rank(corr, 0.90) == 1


def test_keep_representatives_returns_k_real_hypotheses():
    rng = np.random.default_rng(0)
    pool = [f"f{i} h" for i in range(8)]
    E = rng.standard_normal((100, 8))
    keep = _keep_representatives(pool, E, (rng.random(100) > 0.5).astype(int), 3, seed=0)
    assert len(keep) == 3 and all(h in pool for h in keep)


def test_accordion_runs_and_plateaus_on_redundant_generation():
    bundle = make_bundle(n=200, n_features=8, n_classes=4)
    # round 0 generates 8 distinct feature-reading hyps; refills return DUPLICATES -> no new
    # directions -> kept plateaus -> patience stop
    gen = [[f"f{i} distinct" for i in range(8)]]
    refills = [[f"f{i} again" for i in range(8)] for _ in range(5)]  # same columns, new wording
    proposer = FakeProposer(generate_batches=gen, refill_batches=refills)
    kept, history = accordion(
        bundle,
        FakeScorer(),
        proposer,
        seed=0,
        gen_size=8,
        rounds=5,
        var_threshold=0.9,
        patience=2,
        min_keep=2,
        sample=100,
        deduper=TextOnlyDeduper(),
    )
    assert history and all(h["kept"] >= 1 for h in history)
    assert all("f" in k for k in kept)  # kept are real generated hypotheses
    # plateaued before exhausting all rounds (redundant refills add no new directions)
    assert len(history) < 5
    assert history[-1]["eff_rank"] <= history[-1]["deduped"]
