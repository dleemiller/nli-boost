#!/usr/bin/env python
"""Generate an LLM hypothesis pool for a dataset and save it as {text,intended_class} JSON.

This is the only step that spends LM budget. It uses the library's proposer (DeepSeek-v4-flash
via OpenRouter) through `HypothesisVectorizer.fit`, from a stratified sample of the TRAIN split
only (strict: never the test set). The proposer returns untagged statements; we optionally derive
an `intended_class` per hypothesis from the training sample (arg-max mean entailment on each
class), so the prior-aggregation head can consume the generated pool too.

Usage:
    # cheap validation
    uv run python experiments/scripts/generate_pool.py --dataset trec --n 8 --sample-per-class 3 \
        --no-evolve --dedup sts --out experiments/results/pools/trec_gen_smoke.json
    # a real static pool
    uv run python experiments/scripts/generate_pool.py --dataset trec --n 64 --sample-per-class 3 \
        --out experiments/results/pools/trec_gen_static.json
    # evolved pool (more LM calls)
    uv run python experiments/scripts/generate_pool.py --dataset trec --n 64 --evolve \
        --out experiments/results/pools/trec_gen_evolved.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from hvexp import datasets, repro  # noqa: E402
from hvexp.features import NLIFeaturizer  # noqa: E402


def derive_tags(pool: list[str], train_texts, y_train, class_names, fz) -> list[str]:
    """Tag each hypothesis with the class whose train texts most entail it (arg-max mean P(entail))."""
    e = fz.features(train_texts, pool, score_mode="entail")  # (n, m)
    tags = []
    y = np.asarray(y_train)
    for j in range(len(pool)):
        means = [e[y == c, j].mean() if (y == c).any() else -1 for c in range(len(class_names))]
        tags.append(class_names[int(np.argmax(means))])
    return tags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="trec")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--sample-per-class", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--encoder", default="dleemiller/finecat-nli-l")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lm", default="openrouter/deepseek/deepseek-v4-flash")
    ap.add_argument("--dedup", default="sts")
    ap.add_argument("--evolve", action="store_true")
    ap.add_argument("--no-evolve", dest="evolve", action="store_false")
    ap.add_argument("--no-tag", dest="tag", action="store_false", help="skip deriving class tags")
    ap.add_argument("--out", required=True)
    ap.set_defaults(evolve=False, tag=True)
    args = ap.parse_args()

    from hypothesis_vectorizer import HypothesisVectorizer

    raw = datasets.load_raw(args.dataset, test_size=10, test_seed=args.seed)  # test unused here
    # stratified train sample for generation (strict: train only)
    idx = datasets.subsample_indices(raw.y_train, args.sample_per_class, args.seed)
    tr_texts = [raw.train_texts[i] for i in idx]
    tr_y = raw.y_train[idx]
    print(f"[gen] {args.dataset}: generating n={args.n} from {len(tr_texts)} train texts "
          f"({args.sample_per_class}/class), evolve={args.evolve}, dedup={args.dedup}, lm={args.lm}")

    class_defs = list(raw.class_descriptions)
    t0 = time.time()
    vec = HypothesisVectorizer(
        task=raw.task, class_definitions=class_defs, class_names=raw.class_names,
        n_hypotheses=args.n, encoder=args.encoder, device=args.device, lm=args.lm,
        dedup=args.dedup, evolve=args.evolve, random_state=args.seed, verbose=True,
    )
    vec.fit(tr_texts, tr_y)
    pool = list(vec.hypotheses_)
    dt = time.time() - t0
    print(f"[gen] got {len(pool)} hypotheses in {dt:.0f}s")

    tags = None
    if args.tag and pool:
        fz = NLIFeaturizer(encoder=args.encoder, device=args.device, verbose=False)
        tags = derive_tags(pool, tr_texts, tr_y, raw.class_names, fz)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = [{"text": h, "intended_class": (tags[j] if tags else None)} for j, h in enumerate(pool)]
    out.write_text(json.dumps(records, indent=2))
    # provenance next to the pool
    repro.Manifest(
        run_id=out.stem, config=vars(args), seed=args.seed, dataset=args.dataset,
        encoder=args.encoder, pool_id=out.stem,
        extra={"n_generated": len(pool), "gen_seconds": round(dt, 1),
               "sample_texts": len(tr_texts), "lm": args.lm, "evolve": args.evolve},
    ).write(out.parent)
    print(f"[gen] wrote {out} ({len(pool)} hypotheses; tagged={bool(tags)})")


if __name__ == "__main__":
    main()
