#!/usr/bin/env python
"""GoEmotions prior-rewording experiment: does a better-worded prior restore the low-N win?

Motivation (closes the loop with the tau=4 result): the prior-schedule (sched_blend, tau=4)
gave a big low-N lift on TREC (strong prior, prior_fixed=0.59) but only a marginal one on
GoEmotions (weak prior, prior_fixed=0.32). Thesis: the payoff is gated by PRIOR QUALITY, and
GoEmotions' prior is weak partly because the Ekman `joy` bucket collapses ~12 fine emotions
into too few hypotheses. This runs the SAME systems on two pools that differ ONLY in wording:

    v1 = hvexp.hypotheses.expert_pool('goemotions')     (17 hyps, the current pool)
    v2 = data/goemotions_pool_v2.json                    (~31 hyps; count per class scaled to
                                                          bucket heterogeneity, Reddit-register cues)

If v2 lifts prior_fixed and the low-N sched_blend delta jumps back up, that PROVES the thesis
end-to-end on the exact dataset where the effect is currently weak. If prior_fixed barely
moves, the ceiling is the NLI encoder's emotion sense, not the wording — also a real finding.

Usage:
    uv run python experiments/scripts/probe_goemotions_rewording.py --device cuda --seeds 5
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from hvexp import datasets, hypotheses, metrics  # noqa: E402
from hvexp import systems as S  # noqa: E402
from hvexp.features import NLIFeaturizer  # noqa: E402

RESULTS_ROOT = pathlib.Path(__file__).resolve().parents[1] / "results" / "raw"
ENCODER = "dleemiller/finecat-nli-l"
V2_POOL = pathlib.Path(__file__).resolve().parents[1] / "data" / "goemotions_pool_v2.json"


def load_v2_pool():
    obj = json.loads(V2_POOL.read_text())
    return [h["text"] for h in obj], [h["intended_class"] for h in obj]


def make_systems(variant, n, pool, tags, class_names, fz, seed):
    """The tau=4 machinery, identical across pool variants (variant only tags the rows)."""
    tag = lambda base: f"{base}__{variant}"
    return [
        S.PriorAggregation(n, pool, tags, class_names, fz, mode="fixed", name=tag("hv_prior_fixed")),
        S.PriorAggregation(n, pool, tags, class_names, fz, mode="reweight", name=tag("hv_prior_reweight")),
        S.PriorScheduleBlend(n, pool, tags, class_names, fz, tau=4.0, learned="rf", seed=seed,
                             name=tag("hv_prior_sched_blend")),
        S.HVHead(n, pool, fz, head="rf", seed=seed, name=tag("hv_expert_rf")),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shots", nargs="+", default=["1", "2", "3", "5", "10", "20"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--test-size", type=int, default=2000)
    ap.add_argument("--test-seed", type=int, default=7)
    ap.add_argument("--run-id", default="probe_goemotions_rewording")
    args = ap.parse_args()
    shots = [s if s == "all" else int(s) for s in args.shots]

    run_dir = RESULTS_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "results.jsonl"
    out.unlink(missing_ok=True)

    raw = datasets.load_raw("goemotions", test_size=args.test_size, test_seed=args.test_seed)
    n = raw.n_classes
    fz = NLIFeaturizer(encoder=ENCODER, device=args.device, verbose=True)

    variants = {
        "v1": hypotheses.expert_pool("goemotions"),
        "v2": load_v2_pool(),
    }
    for name, (pool, tags) in variants.items():
        assert all(t in raw.class_names for t in tags), f"{name}: bad class tag"
        print(f"[pool] {name}: {len(pool)} hyps")

    # warm NLI cache once, on the union of touched rows + test, for BOTH pools
    y_full = raw.y_train
    used: set[int] = set()
    for k in shots:
        for seed in range(args.seeds):
            used.update(int(i) for i in datasets.subsample_indices(y_full, k, seed))
    warm = [raw.train_texts[i] for i in sorted(used)] + list(raw.test_texts)
    t0 = time.time()
    for name, (pool, _t) in variants.items():
        print(f"[nli] warming {name} ({len(pool)} hyps x {len(warm)} texts)...")
        fz.features(warm, pool, "entail")             # prior heads use entail
        fz.features(warm, pool, "entail_contradict")  # HV/blend learned head uses both
    print(f"[nli] cache warm {time.time()-t0:.0f}s  cost={fz.cost_summary()}")

    rows = []
    with out.open("w") as fh:
        for k in shots:
            for seed in range(args.seeds):
                idx = datasets.subsample_indices(y_full, k, seed)
                tr_texts = [raw.train_texts[i] for i in idx]
                tr_y = y_full[idx]
                for variant, (pool, tags) in variants.items():
                    for sysobj in make_systems(variant, n, pool, tags, raw.class_names, fz, seed):
                        t = time.time()
                        try:
                            proba = sysobj.run(tr_texts, tr_y, raw.test_texts)
                            m = metrics.compute_metrics(raw.y_test, proba.argmax(1), proba,
                                                        n_classes=n)
                            err = None
                        except Exception as e:
                            m, err = {}, f"{type(e).__name__}: {e}"
                        row = {"system": sysobj.name, "variant": variant, "shots": k, "seed": seed,
                               "error": err, **m}
                        fh.write(json.dumps(row) + "\n"); fh.flush()
                        rows.append(row)
                        status = err or f"acc={m.get('accuracy', float('nan')):.3f}"
                        print(f"  k={k!s:>3} seed={seed} {sysobj.name:<28} {status}")

    # summary: prior_fixed lift, and sched_blend v1 vs v2 by k
    by = {}
    for r in rows:
        if r["error"] or "accuracy" not in r:
            continue
        by.setdefault(r["system"], {}).setdefault(r["shots"], []).append(r["accuracy"])
    mean = lambda s, k: statistics.mean(by[s][k]) if by.get(s, {}).get(k) else float("nan")
    print("\nprior_fixed (label-free) — does wording lift the prior?")
    for k in shots:
        print(f"  k={k!s:>3}: v1 {mean('hv_prior_fixed__v1', k):.3f}  ->  "
              f"v2 {mean('hv_prior_fixed__v2', k):.3f}")
    print("\nsched_blend (tau=4) — does a better prior restore the low-N win?")
    for k in shots:
        print(f"  k={k!s:>3}: v1 {mean('hv_prior_sched_blend__v1', k):.3f}  ->  "
              f"v2 {mean('hv_prior_sched_blend__v2', k):.3f}   "
              f"(rf ref v2 {mean('hv_expert_rf__v2', k):.3f})")
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
