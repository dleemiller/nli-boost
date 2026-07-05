"""Paired significance test between two runs on the SAME test set.

Seed bands (report/METHOD.md) capture generation + training variance across
independent fits. This is the complementary axis: given ONE fixed test set, do
two models' predictions differ by more than chance? On ~500-2000 test examples a
1-point accuracy gap is often inside one standard error — McNemar on the paired
predictions says whether a single-run A/B (e.g. lexical on/off, evolved vs
static pool) is real or noise.

Predictions are reconstructed from the NLI cache (features free) and the
deterministic CV-selected head, so any two existing runs are comparable for free.
Comparing runs with different test sets is refused, not silently averaged.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from .cache import ScoreCache
from .config import RunConfig
from .costs import CostTracker
from .data import load
from .encoder import EntailmentScorer
from .head import cv_selected_head, fit_head
from .runner import build_matrices


def _predictions(run_dir: Path):
    """Reconstruct a run's (dataset, seed, y_test, test predictions) from cache.

    Refits the run's SAVED head params (deterministic → same head), not the whole
    CV grid — features come free from the NLI cache, so this is a few seconds.
    """
    cfg = RunConfig.from_yaml(run_dir / "config.yaml")
    bundle = load(cfg.data, cfg.seed)
    scorer = EntailmentScorer(cfg.encoder, ScoreCache(cfg.cache_dir / "nli_scores.sqlite"), CostTracker())
    model = json.loads((run_dir / "model.json").read_text())
    x_train, x_test = build_matrices(cfg, scorer, bundle, model["hypotheses"])
    if "head" in model:
        head = fit_head(model["head"], x_train, bundle.y_train, cfg.seed)
    else:  # runs predating head-saving: reproduce it via the (deterministic) CV grid
        head, _, _ = cv_selected_head(x_train, bundle.y_train, cfg.seed)
    return cfg, bundle.y_test, head.predict(x_test)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion — honest near 0/1 and small n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - half) / d, (c + half) / d)


def mcnemar(y: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> dict:
    """Exact (binomial) McNemar test on the discordant pairs.

    b = A correct where B wrong; c = A wrong where B correct. Under H0 the
    discordant pairs split 50/50; the exact two-sided binomial avoids the
    chi-square approximation's failure when b+c is small (our usual regime).
    """
    a_ok, b_ok = (pa == y), (pb == y)
    b = int((a_ok & ~b_ok).sum())  # A fixes what B misses
    c = int((~a_ok & b_ok).sum())  # B fixes what A misses
    n_disc = b + c
    p = binomtest(b, n_disc, 0.5).pvalue if n_disc else 1.0
    return {"b_only_A": b, "c_only_B": c, "discordant": n_disc, "p_value": float(p)}


def compare_runs(run_a: Path, run_b: Path) -> dict:
    cfg_a, y_a, pa = _predictions(run_a)
    cfg_b, y_b, pb = _predictions(run_b)
    if not (cfg_a.data.name == cfg_b.data.name and cfg_a.seed == cfg_b.seed and np.array_equal(y_a, y_b)):
        raise ValueError(
            f"refusing to compare mismatched test sets: "
            f"{cfg_a.data.name}/seed{cfg_a.seed} vs {cfg_b.data.name}/seed{cfg_b.seed}"
        )
    y = y_a
    n = len(y)
    acc_a, acc_b = float((pa == y).mean()), float((pb == y).mean())
    mc = mcnemar(y, pa, pb)
    return {
        "run_a": run_a.name,
        "run_b": run_b.name,
        "dataset": cfg_a.data.name,
        "seed": cfg_a.seed,
        "n_test": n,
        "acc_a": round(acc_a, 4),
        "acc_b": round(acc_b, 4),
        "delta_b_minus_a": round(acc_b - acc_a, 4),
        "ci_a": [round(v, 4) for v in _wilson(int((pa == y).sum()), n)],
        "ci_b": [round(v, 4) for v in _wilson(int((pb == y).sum()), n)],
        "mcnemar": mc,
        "significant_at_05": mc["p_value"] < 0.05,
    }
