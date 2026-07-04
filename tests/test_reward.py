import numpy as np

from nli_boost.reward import RewardConfig, effective_rank, geometric_mean, pool_reward


def test_effective_rank_spans_collapsed_to_orthogonal():
    rng = np.random.default_rng(0)
    base = rng.random((200, 1))
    collapsed = np.hstack([base + 1e-6 * rng.random((200, 1)) for _ in range(8)])
    orthogonal = rng.random((200, 8))
    assert effective_rank(collapsed) < 1.5
    assert effective_rank(orthogonal) > 6.0


def test_geometric_mean_craters_on_a_zero():
    assert geometric_mean([0.9, 0.9, 0.9]) > 0.85
    assert geometric_mean([0.9, 0.9, 0.0]) < 0.05  # one tanked dataset vetoes the aggregate


def test_reward_prefers_informative_diverse_pool():
    rng = np.random.default_rng(0)
    n, m = 300, 6
    y = (rng.random(n) > 0.5).astype(int)
    # informative+diverse: each entail col separates y with independent noise
    ent_good = np.column_stack([y * 0.6 + 0.2 + 0.15 * rng.random(n) for _ in range(m)])
    ent_good += 0.3 * rng.random((n, m))  # decorrelate
    x_good = np.hstack([ent_good, 1 - ent_good])
    # collapsed: one signal copied across all columns
    sig = (y * 0.6 + 0.2 + 0.1 * rng.random(n))[:, None]
    ent_bad = np.repeat(sig, m, axis=1) + 1e-3 * rng.random((n, m))
    x_bad = np.hstack([ent_bad, 1 - ent_bad])
    texts = ["a text here"] * n
    pool = [f"hypothesis {i}" for i in range(m)]
    cfg = RewardConfig(cv_seeds=2)
    r_good = pool_reward(x_good, y, pool, texts, cfg)
    r_bad = pool_reward(x_bad, y, pool, texts, cfg)
    assert r_good["score"] > r_bad["score"]
    assert r_good["effective_rank"] > r_bad["effective_rank"]


def test_reward_flags_length_artifact():
    rng = np.random.default_rng(1)
    n = 200
    y = (rng.random(n) > 0.5).astype(int)
    texts = ["x" * int(v) for v in (50 + 100 * rng.random(n))]
    lengths = np.array([len(t) for t in texts], dtype=float)
    ent = (lengths / lengths.max())[:, None] + 0.01 * rng.random((n, 1))  # tracks length
    x = np.hstack([ent, 1 - ent])
    r = pool_reward(x, y, ["hypothesis 0"], texts, RewardConfig(cv_seeds=2))
    assert r["n_length_artifacts"] == 1
    assert r["components"]["anti_hack"] < 1.0
