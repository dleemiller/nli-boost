#!/usr/bin/env python
"""RQ4 ablation harness: toggle ONE HV knob at a time on a fixed pool/dataset/train size.

Each ablation axis varies a single knob away from the reference HV config
(head='auto', score_mode='entail_contradict', full expert pool) and reports accuracy /
macro-F1 on the ONE fixed test set. Like the learning-curve harness, every NLI corpus is
scored once (cached) per (encoder, pool) on the union of train+test texts, so the sweep is
cheap CPU after the first GPU pass. No LLM is ever called.

Axes (`--axis`, default `all`):
  * score_mode : entail_contradict vs entail vs contrast   (HVHead.score_mode)
  * pool_size  : first-k of the pool at sizes --pool-sizes (default 8/16/24) — saturation curve
  * head       : auto (CV-selected RF/HGB) vs logreg_cv     (HVHead.head)
  * encoder    : compare several NLI encoders (--encoders); this is the only axis that
                 RE-SCORES the corpus (one NLIFeaturizer + cache pass per encoder).

The pool defaults to hvexp.hypotheses.expert_pool(dataset); pass --pool-json (a JSON list of
hypothesis strings, e.g. an LM-generated pool) to swap it. Only HVHead is used, so the pool's
class tags are not needed.

Usage:
    uv run python experiments/scripts/run_ablation.py --config experiments/configs/runs/trec_ablation.yaml
    uv run python experiments/scripts/run_ablation.py --dataset trec --axis all --seeds 3 \
        --encoders dleemiller/finecat-nli-m dleemiller/finecat-nli-l
    uv run python experiments/scripts/run_ablation.py --dataset sst2 --axis pool_size \
        --pool-json experiments/pools/sst2_lm.json --pool-sizes 8 16 24 32
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
from hvexp import systems as S  # noqa: E402
from hvexp.features import NLIFeaturizer  # noqa: E402

RESULTS_ROOT = pathlib.Path(__file__).resolve().parents[1] / "results" / "raw"
ALL_AXES = ["score_mode", "pool_size", "head", "encoder"]


def load_pool(dataset: str, pool_json: pathlib.Path | None) -> tuple[list[str], str]:
    """(pool texts, pool_id). Either an LM/JSON pool or the hand-written expert pool."""
    if pool_json:
        obj = json.loads(pathlib.Path(pool_json).read_text())
        if obj and isinstance(obj[0], dict):  # [{"text":..,"intended_class":..}] -> texts
            pool = [h["text"] for h in obj]
        elif isinstance(obj, list) and all(isinstance(h, str) for h in obj):
            pool = obj
        else:
            raise ValueError(f"--pool-json {pool_json} must be a JSON list of strings or {{text,...}}")
        return pool, pathlib.Path(pool_json).stem
    pool, _tags = hypotheses.expert_pool(dataset)
    return pool, "expert"


def variants_for_axis(axis: str, *, pool: list[str], featurizers: dict[str, NLIFeaturizer],
                      primary: str, encoders: list[str], n_classes: int, seed: int,
                      pool_sizes: list[int]) -> list[tuple[str, str, S.HVHead]]:
    """List of (variant_label, encoder_id, system) for one ablation axis at one seed.

    Every axis holds the reference config (head='auto', score_mode='entail_contradict', full
    pool) fixed except the single knob it sweeps.
    """
    out: list[tuple[str, str, S.HVHead]] = []
    if axis == "score_mode":
        fz = featurizers[primary]
        for mode in ("entail_contradict", "entail", "contrast"):
            out.append((mode, primary,
                        S.HVHead(n_classes, pool, fz, head="auto", score_mode=mode, seed=seed)))
    elif axis == "head":
        fz = featurizers[primary]
        for head in ("auto", "logreg_cv"):
            out.append((head, primary,
                        S.HVHead(n_classes, pool, fz, head=head,
                                 score_mode="entail_contradict", seed=seed)))
    elif axis == "pool_size":
        fz = featurizers[primary]
        # single RF head (the paper's flexible-head line): the pool-size scaling curve should
        # vary the pool, not the head, and a CV grid on many classes is needlessly slow here.
        for k in [k for k in pool_sizes if k <= len(pool)]:
            out.append((str(k), primary,
                        S.HVHead(n_classes, pool[:k], fz, head="rf",
                                 score_mode="entail_contradict", seed=seed)))
    elif axis == "encoder":
        for enc in encoders:
            out.append((enc, enc,
                        S.HVHead(n_classes, pool, featurizers[enc], head="auto",
                                 score_mode="entail_contradict", seed=seed)))
    else:
        raise ValueError(f"unknown axis {axis!r}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=pathlib.Path)
    ap.add_argument("--dataset", default="trec")
    ap.add_argument("--encoder", default="dleemiller/finecat-nli-l")
    ap.add_argument("--encoders", nargs="+", default=None,
                    help="encoder ids for the 'encoder' axis (defaults to [--encoder])")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--axis", default="all", choices=[*ALL_AXES, "all"])
    ap.add_argument("--pool-json", type=pathlib.Path, default=None)
    ap.add_argument("--pool-sizes", nargs="+", type=int, default=[8, 16, 24])
    ap.add_argument("--train-size", default="all", help="examples/class, or 'all' (full pool)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--test-size", type=int, default=2000)
    ap.add_argument("--test-seed", type=int, default=7)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    if args.config:
        cfg = yaml.safe_load(args.config.read_text())
        for k, v in cfg.items():
            setattr(args, k.replace("-", "_"), v)

    selected = ALL_AXES if args.axis == "all" else [args.axis]
    train_size = args.train_size if args.train_size == "all" else int(args.train_size)
    encoders = list(args.encoders) if args.encoders else [args.encoder]
    primary = args.encoder if args.encoder in encoders else encoders[0]

    run_id = args.run_id or f"abl_{args.dataset}_{args.axis}_{repro.git_commit()}"
    run_dir = RESULTS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    results_path.unlink(missing_ok=True)

    print(f"[load] {args.dataset} (test_size={args.test_size}, test_seed={args.test_seed})")
    raw = datasets.load_raw(args.dataset, test_size=args.test_size, test_seed=args.test_seed)
    print(f"[load] full train={len(raw.train_texts)}  fixed test={len(raw.test_texts)}  "
          f"classes={raw.class_names}")

    pool, pool_id = load_pool(args.dataset, args.pool_json)
    print(f"[pool] id={pool_id} size={len(pool)} axes={selected}")

    # ---- which encoders must we build/warm? primary for local axes, all for the encoder axis ----
    enc_needed: set[str] = set()
    if any(a != "encoder" for a in selected):
        enc_needed.add(primary)
    if "encoder" in selected:
        enc_needed.update(encoders)

    # ---- the exact train rows the sweep touches (union over seeds at the fixed train size) ------
    y_full = raw.y_train
    used: set[int] = set()
    for seed in range(args.seeds):
        used.update(int(i) for i in datasets.subsample_indices(y_full, train_size, seed))
    used_idx = sorted(used)
    warm_texts = [raw.train_texts[i] for i in used_idx] + list(raw.test_texts)

    # ---- build one featurizer per encoder and warm its cache ONCE on the whole pool ------------
    featurizers: dict[str, NLIFeaturizer] = {}
    for enc in sorted(enc_needed):
        fz = NLIFeaturizer(encoder=enc, device=args.device, verbose=True)
        t0 = time.time()
        print(f"[nli] warming cache: {len(warm_texts)} texts x {len(pool)} hyps on {enc} ...")
        fz.features(warm_texts, pool, score_mode="entail_contradict")  # caches all score_modes
        print(f"[nli] {enc} warm in {time.time()-t0:.0f}s  cost={fz.cost_summary()}")
        featurizers[enc] = fz

    repro.Manifest(
        run_id=run_id,
        config={"dataset": args.dataset, "encoder": args.encoder, "encoders": encoders,
                "axes": selected, "pool_id": pool_id, "pool_size": len(pool),
                "pool_sizes": args.pool_sizes, "train_size": train_size, "seeds": args.seeds,
                "test_size": args.test_size, "test_seed": args.test_seed},
        seed=args.test_seed, dataset=args.dataset, encoder=args.encoder,
        pool_id=pool_id, extra={"pool": pool},
    ).write(run_dir)

    n_rows = 0
    with results_path.open("w") as fh:
        for seed in range(args.seeds):
            idx = datasets.subsample_indices(y_full, train_size, seed)
            tr_texts = [raw.train_texts[i] for i in idx]
            tr_y = y_full[idx]
            for axis in selected:
                variants = variants_for_axis(
                    axis, pool=pool, featurizers=featurizers, primary=primary,
                    encoders=encoders, n_classes=raw.n_classes, seed=seed,
                    pool_sizes=args.pool_sizes,
                )
                for variant, enc, sysobj in variants:
                    t = time.time()
                    try:
                        proba = sysobj.run(tr_texts, tr_y, raw.test_texts)
                        pred = proba.argmax(axis=1)
                        m = metrics.compute_metrics(raw.y_test, pred, proba,
                                                    n_classes=raw.n_classes)
                        err = None
                    except Exception as e:  # keep the sweep going; record the failure
                        m, err = {}, f"{type(e).__name__}: {e}"
                    n_pool = len(sysobj.pool)
                    row = {"dataset": args.dataset, "encoder": enc, "axis": axis,
                           "variant": variant, "seed": seed, "n_train": int(len(tr_y)),
                           "n_pool": n_pool, "fit_seconds": round(time.time() - t, 3),
                           "error": err, **m}
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    n_rows += 1
                    tag = err or f"acc={m.get('accuracy', float('nan')):.3f}"
                    print(f"  seed={seed} {axis:<11} {variant:<24} {tag}")

    print(f"[done] {n_rows} rows -> {results_path}")


if __name__ == "__main__":
    main()
