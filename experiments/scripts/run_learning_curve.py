#!/usr/bin/env python
"""Learning-curve harness: sweep examples/class x seeds x systems on a fixed test set.

Protocol (docs/low-n-plan.md): ONE fixed test set; resample k examples/class from the full
train pool across seeds; every NLI corpus is scored once (cached) so the whole sweep is cheap
CPU after the first GPU pass. Emits one JSONL row per (system, k, seed) with metrics, plus a
manifest and a per-class table at a reference point.

Usage:
    uv run python experiments/scripts/run_learning_curve.py --config experiments/configs/runs/trec_lown.yaml
    uv run python experiments/scripts/run_learning_curve.py --dataset trec --encoder dleemiller/finecat-nli-l \
        --shots 1 2 3 5 10 20 50 --seeds 5 --systems baselines
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # experiments/
from hvexp import datasets, hypotheses, metrics, repro  # noqa: E402
from hvexp.features import NLIFeaturizer  # noqa: E402
from hvexp import systems as S  # noqa: E402

RESULTS_ROOT = pathlib.Path(__file__).resolve().parents[1] / "results" / "raw"


def build_systems(which: str, raw, fz: NLIFeaturizer, pool, tags, seed: int,
                  gen_pools: dict | None = None) -> list:
    """Instantiate the requested system set.

    `pool`/`tags` are the hand-written fixed-expert HV pool. `gen_pools` maps a short name to a
    (pool, tags) pair for each LLM-generated pool (Phase 2), each added as hv_<name>_* systems —
    the ONLY part of the study that needs an LM, and it plugs in here with no other change.
    """
    n = raw.n_classes
    zshot = hypotheses.ZEROSHOT_TEMPLATES[raw.name]
    out: list = []
    if which in ("finetuned", "all"):
        out.append(S.FineTunedEncoder(n, device=fz.cfg.device, seed=seed))
    if which in ("baselines", "all"):
        out += [
            S.TfidfLogReg(n, analyzer="word", ngram_range=(1, 2)),
            S.TfidfLogReg(n, analyzer="char_wb", ngram_range=(2, 5), name="tfidf_char+logreg"),
            S.TfidfUnionLogReg(n),
            S.EmbeddingLogReg(n, device=fz.cfg.device),
            S.ZeroShotNLI(n, zshot, fz),
        ]
    if which in ("hv", "all", "baselines"):
        out += [
            S.HVHead(n, pool, fz, head="rf", seed=seed, name="hv_expert_rf"),
            S.HVHead(n, pool, fz, head="logreg_cv", seed=seed, name="hv_expert_logreg"),
            S.PriorAggregation(n, pool, tags, raw.class_names, fz, mode="fixed",
                               name="hv_prior_fixed"),
            S.PriorAggregation(n, pool, tags, raw.class_names, fz, mode="reweight",
                               name="hv_prior_reweight"),
        ]
    for gname, (gpool, gtags) in (gen_pools or {}).items():
        out += [S.HVHead(n, gpool, fz, head="rf", seed=seed, name=f"hv_{gname}_rf"),
                S.HVHead(n, gpool, fz, head="logreg_cv", seed=seed, name=f"hv_{gname}_logreg")]
        if gtags and all(t in raw.class_names for t in gtags):
            out += [S.PriorAggregation(n, gpool, gtags, raw.class_names, fz, mode="fixed",
                                       name=f"hv_{gname}_prior_fixed")]
    return out


def load_generated_pool(path) -> tuple[list[str], list[str] | None]:
    obj = json.loads(pathlib.Path(path).read_text())
    if obj and isinstance(obj[0], dict):
        return [h["text"] for h in obj], [h.get("intended_class") for h in obj]
    return list(obj), None


def parse_gen_pools(specs: list[str] | None) -> dict:
    """Each spec is 'name=path/to/pool.json' -> {name: (pool, tags)}."""
    out = {}
    for spec in specs or []:
        name, _, path = spec.partition("=")
        out[name] = load_generated_pool(path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=pathlib.Path)
    ap.add_argument("--dataset", default="trec")
    ap.add_argument("--encoder", default="dleemiller/finecat-nli-l")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shots", nargs="+", default=["1", "2", "3", "5", "10", "20", "50", "all"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--test-size", type=int, default=2000)
    ap.add_argument("--test-seed", type=int, default=7)
    ap.add_argument("--systems", default="baselines")
    ap.add_argument("--generated-pool", nargs="*", default=None,
                    help="one or more 'name=pool.json' — adds hv_<name>_* systems (Phase 2)")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    if args.config:
        cfg = yaml.safe_load(args.config.read_text())
        for k, v in cfg.items():
            setattr(args, k.replace("-", "_"), v)
    shots = [s if s == "all" else int(s) for s in args.shots]

    run_id = args.run_id or f"lc_{args.dataset}_{args.systems}_{repro.git_commit()}"
    run_dir = RESULTS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    results_path.unlink(missing_ok=True)

    print(f"[load] {args.dataset} (test_size={args.test_size}, test_seed={args.test_seed})")
    raw = datasets.load_raw(args.dataset, test_size=args.test_size, test_seed=args.test_seed)
    print(f"[load] full train={len(raw.train_texts)}  fixed test={len(raw.test_texts)}  "
          f"classes={raw.class_names}")

    fz = NLIFeaturizer(encoder=args.encoder, device=args.device, verbose=True)
    pool, tags = hypotheses.expert_pool(args.dataset)
    zshot = hypotheses.ZEROSHOT_TEMPLATES[args.dataset]
    gen_pools = parse_gen_pools(args.generated_pool)

    # ---- pre-compute the exact train rows the sweep will touch (union over k, seed) ------
    y_full = raw.y_train
    used = set()
    for k in shots:
        for seed in range(args.seeds):
            used.update(int(i) for i in datasets.subsample_indices(y_full, k, seed))
    used_idx = sorted(used)
    warm_texts = [raw.train_texts[i] for i in used_idx] + list(raw.test_texts)

    # ---- warm the NLI cache ONCE on those texts (the only GPU pass) ---------------------
    t0 = time.time()
    print(f"[nli] warming cache: {len(warm_texts)} texts x {len(pool)} expert hyps "
          f"+ {len(zshot)} templates on {args.encoder} ...")
    fz.features(warm_texts, pool, score_mode="entail_contradict")
    fz.features(warm_texts, zshot, score_mode="entail")
    for _gname, (gpool, _gt) in gen_pools.items():
        fz.features(warm_texts, gpool, score_mode="entail_contradict")
    print(f"[nli] cache warm in {time.time()-t0:.0f}s  cost={fz.cost_summary()}")

    repro.Manifest(
        run_id=run_id,
        config={"dataset": args.dataset, "encoder": args.encoder, "shots": shots,
                "seeds": args.seeds, "test_size": args.test_size, "test_seed": args.test_seed,
                "systems": args.systems, "expert_pool_size": len(pool)},
        seed=args.test_seed, dataset=args.dataset, encoder=args.encoder,
        pool_id="expert", extra={"expert_pool": pool, "expert_tags": tags},
    ).write(run_dir)

    n_rows = 0
    with results_path.open("w") as fh:
        for k in shots:
            for seed in range(args.seeds):
                idx = datasets.subsample_indices(y_full, k, seed)
                tr_texts = [raw.train_texts[i] for i in idx]
                tr_y = y_full[idx]
                for sysobj in build_systems(args.systems, raw, fz, pool, tags, seed,
                                            gen_pools=gen_pools):
                    t = time.time()
                    try:
                        proba = sysobj.run(tr_texts, tr_y, raw.test_texts)
                        pred = proba.argmax(axis=1)
                        m = metrics.compute_metrics(raw.y_test, pred, proba, n_classes=raw.n_classes)
                        err = None
                    except Exception as e:  # keep the sweep going; record the failure
                        m, err = {}, f"{type(e).__name__}: {e}"
                    row = {"dataset": args.dataset, "encoder": args.encoder, "system": sysobj.name,
                           "shots": k, "seed": seed, "n_train": int(len(tr_y)),
                           "fit_seconds": round(time.time() - t, 3), "error": err, **m}
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    n_rows += 1
                    tag = err or f"acc={m.get('accuracy', float('nan')):.3f}"
                    print(f"  k={k!s:>3} seed={seed} {sysobj.name:<22} {tag}")

    print(f"[done] {n_rows} rows -> {results_path}")


if __name__ == "__main__":
    main()
