#!/usr/bin/env python
"""Prepare the CFPB text+tabular benchmark: build a CSV + generate an HV pool (train-only).

Monetary-relief prediction (Wang, Zhu & Chen 2026): predict whether a consumer complaint closes
with monetary relief, from the free-text narrative + categorical metadata. This script:
  1. streams the CFPB mirror, keeps closed complaints WITH a narrative, and takes a balanced
     per-class sample (monetary relief is ~8% of complaints, so balance stabilizes AUC);
  2. writes a CSV (narrative + Product/Company/State/Submitted via + relief + date);
  3. generates an HV hypothesis pool from the TRAIN narratives only (matching the random split
     run_text_tabular will use), saved as a JSON list of strings.

Then run the 6-config comparison:
    HV_CACHE_DIR=/tmp/hv_cache uv run python experiments/scripts/run_text_tabular.py \
        --csv experiments/results/cfpb/cfpb.csv --text-col narrative \
        --cat-cols Product Company State "Submitted via" --label-col relief \
        --split random --test-frac 0.2 --max-text-chars 512 \
        --pool-json experiments/results/cfpb/cfpb_pool.json

Needs OPENROUTER_API_KEY + GPU.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from hvexp import repro  # noqa: E402

_CLOSED = {
    "Closed with monetary relief", "Closed with non-monetary relief",
    "Closed with explanation", "Closed without relief", "Closed",
}
_CATS = ["Product", "Company", "State", "Submitted via"]


def load_frame(per_class: int | None = None, limit: int | None = None):
    """Stream closed complaints that have a narrative.

    `per_class`: balanced — up to this many of EACH relief class (stable AUC; artificial rate).
    `limit`: NATURAL rate — the first `limit` closed+narrative complaints, keeping the real ~8%
    monetary-relief base rate (the benchmark setting; pair with a temporal split).
    """
    import pandas as pd
    from datasets import load_dataset

    ds = load_dataset("BEE-spoke-data/consumer-finance-complaints", split="train", streaming=True)
    rows, kept = [], {0: 0, 1: 0}
    for r in ds:
        narrative = (r.get("Consumer complaint narrative") or "").strip()
        resp = r.get("Company response to consumer")
        if not (narrative and resp in _CLOSED):
            continue
        y = int(resp == "Closed with monetary relief")
        if per_class is not None and kept[y] >= per_class:
            continue
        kept[y] += 1
        rows.append({"date": r["Date received"], "narrative": narrative,
                     "Product": r.get("Product") or "unknown", "Company": r.get("Company") or "unknown",
                     "State": r.get("State") or "unknown",
                     "Submitted via": r.get("Submitted via") or "unknown", "relief": y})
        if per_class is not None and kept[0] >= per_class and kept[1] >= per_class:
            break
        if limit is not None and len(rows) >= limit:
            break
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    # rare companies -> "other" (matches examples/cfpb.py; keeps the one-hot manageable)
    top = set(df["Company"].value_counts().head(200).index)
    df["Company"] = df["Company"].where(df["Company"].isin(top), "other")
    return df.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=None,
                    help="balanced: up to N of each relief class (random split)")
    ap.add_argument("--limit", type=int, default=None,
                    help="natural-rate: first N closed+narrative complaints (temporal split)")
    ap.add_argument("--split", default=None, choices=["random", "temporal"],
                    help="pool-gen train selection; default random for --per-class, temporal for --limit")
    ap.add_argument("--n-hypotheses", type=int, default=64)
    ap.add_argument("--evolve", action="store_true",
                    help="prune hypotheses by MARGINAL value over the tabular block (slower)")
    ap.add_argument("--max-text-chars", type=int, default=512)
    ap.add_argument("--encoder", default="dleemiller/finecat-nli-l")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lm", default="openrouter/deepseek/deepseek-v4-flash")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--outdir", default="experiments/results/cfpb")
    args = ap.parse_args()

    from sklearn.preprocessing import OneHotEncoder

    from hypothesis_vectorizer import HypothesisVectorizer

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.per_class is None and args.limit is None:
        args.per_class = 2500  # default to the balanced setting
    split = args.split or ("temporal" if args.limit is not None else "random")
    mode = f"natural-rate limit={args.limit}" if args.limit is not None else f"balanced {args.per_class}/class"
    print(f"[cfpb] streaming {mode}, split={split} ...")
    df = load_frame(per_class=args.per_class, limit=args.limit)
    outfile = "cfpb_temporal.csv" if split == "temporal" else "cfpb.csv"
    csv_path = outdir / outfile
    df.to_csv(csv_path, index=False)
    print(f"[cfpb] {len(df)} rows -> {csv_path}  (relief rate {df['relief'].mean():.3f})")

    # select the pool-gen TRAIN rows to match the split run_text_tabular will use (no test leakage)
    n = len(df)
    n_test = max(1, int(round(args.test_frac * n)))
    if split == "temporal":  # df is date-sorted -> oldest (n - n_test) are train
        tr_idx = np.arange(n - n_test)
    else:  # random split (seed, test_frac), matching run_text_tabular's rng.permutation
        tr_idx = np.random.default_rng(args.seed).permutation(n)[n_test:]
    tr = df.iloc[tr_idx]
    print(f"[cfpb] pool generated from {len(tr)} {split}-train narratives "
          f"(train relief {tr['relief'].mean():.3f}; test {n_test} held out)")

    baseline = None
    if args.evolve:
        baseline = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(tr[_CATS])

    t0 = time.time()
    vec = HypothesisVectorizer(
        task="Predict whether a consumer's financial complaint will be resolved with monetary relief.",
        class_definitions=[
            "no relief: the complaint is closed without monetary compensation to the consumer",
            "monetary relief: the company resolves the complaint by paying the consumer money",
        ],
        class_names=["no relief", "monetary relief"],
        encoder=args.encoder, device=args.device, lm=args.lm,
        max_text_chars=args.max_text_chars, n_hypotheses=args.n_hypotheses,
        dedup="sts", evolve=args.evolve, random_state=args.seed, verbose=True,
    )
    vec.fit(tr["narrative"].tolist(), tr["relief"].to_numpy(), baseline_features=baseline)
    pool = list(vec.hypotheses_)
    dt = time.time() - t0

    pool_path = outdir / (f"cfpb_pool_{split}.json")
    pool_path.write_text(json.dumps(pool, indent=2))  # list of strings for run_text_tabular
    repro.Manifest(run_id=f"cfpb_pool_{split}", config=vars(args), seed=args.seed, dataset="cfpb",
                   encoder=args.encoder, pool_id=f"cfpb_pool_{split}",
                   extra={"n_hypotheses": len(pool), "gen_seconds": round(dt, 1),
                          "evolve": args.evolve, "relief_rate": float(df["relief"].mean())}).write(outdir)
    print(f"[cfpb] {len(pool)} hypotheses in {dt:.0f}s -> {pool_path}")
    for h in pool[:8]:
        print("   -", h)


if __name__ == "__main__":
    main()
