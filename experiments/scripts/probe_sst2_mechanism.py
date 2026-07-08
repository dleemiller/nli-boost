#!/usr/bin/env python
"""SST-2 mechanism probe: prove *why* HV obliterates MiniLM at low N.

Claim: SST-2 sentiment is a ~1-D, lexically-cued axis. HV *injects* that axis directly
(P(entail | "expresses positive sentiment")), so it needs ~0 labels; a generic sentence
embedding must *discover* the axis from labels, which is slow at low N and — because a
similarity-tuned embedding entangles sentiment with topic — never fully catches up.

The confound in the baseline learning curve: HV uses `finecat-nli-l`, the MiniLM baseline
uses `all-MiniLM-L6-v2`, so "HV wins" conflates *hypothesis injection* with *encoder choice*.
This probe adds the decisive control — POOLED EMBEDDINGS FROM THE SAME finecat encoder + logreg.

Reading the output:
  * hv_entail_zshot / hv_entail_pool  : entailment scoring (the mechanism). Flat & high => 1-axis, label-free.
  * emb:finecat-pooled               : SAME encoder, pooled embedding + logreg. If this CRAWLS up from
                                       chance like MiniLM, the win is the MECHANISM, not the encoder.
  * emb:all-MiniLM                   : honest bi-encoder baseline (reference).
Gap(entail - finecat-pooled) isolates entailment-vs-embedding with the encoder held fixed.

Usage:
    uv run python experiments/scripts/probe_sst2_mechanism.py --device cuda --seeds 5
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shots", nargs="+", default=["1", "2", "3", "5", "10", "20", "50", "100", "all"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--test-size", type=int, default=2000)
    ap.add_argument("--test-seed", type=int, default=7)
    ap.add_argument("--run-id", default="probe_sst2_mechanism")
    args = ap.parse_args()
    shots = [s if s == "all" else int(s) for s in args.shots]

    run_dir = RESULTS_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "results.jsonl"
    out.unlink(missing_ok=True)

    raw = datasets.load_raw("sst2", test_size=args.test_size, test_seed=args.test_seed)
    n = raw.n_classes
    fz = NLIFeaturizer(encoder=ENCODER, device=args.device, verbose=True)
    pool, tags = hypotheses.expert_pool("sst2")
    zshot = hypotheses.ZEROSHOT_TEMPLATES["sst2"]

    # warm NLI cache once on the union of touched train rows + test (same protocol as the harness)
    y_full = raw.y_train
    used: set[int] = set()
    for k in shots:
        for seed in range(args.seeds):
            used.update(int(i) for i in datasets.subsample_indices(y_full, k, seed))
    warm = [raw.train_texts[i] for i in sorted(used)] + list(raw.test_texts)
    t0 = time.time()
    fz.features(warm, pool, "entail_contradict")
    fz.features(warm, zshot, "entail")
    print(f"[nli] cache warm {time.time()-t0:.0f}s")

    def make_systems(seed: int) -> list:
        return [
            # --- the mechanism: entailment scoring ---
            S.ZeroShotNLI(n, zshot, fz, name="hv_entail_zshot"),          # 1 template/class (minimal)
            S.HVHead(n, pool, fz, head="rf", seed=seed, name="hv_entail_pool"),  # 8-hyp pool
            S.PriorAggregation(n, pool, tags, raw.class_names, fz, mode="fixed",
                               name="hv_prior_fixed"),
            # --- the controls: embeddings + logreg ---
            S.EmbeddingLogReg(n, model_name=ENCODER, device=args.device,
                              name="emb:finecat-pooled"),                  # SAME encoder, pooled
            S.EmbeddingLogReg(n, model_name="sentence-transformers/all-MiniLM-L6-v2",
                              device=args.device, name="emb:all-MiniLM"),  # honest bi-encoder
        ]

    rows = []
    with out.open("w") as fh:
        for k in shots:
            for seed in range(args.seeds):
                idx = datasets.subsample_indices(y_full, k, seed)
                tr_texts = [raw.train_texts[i] for i in idx]
                tr_y = y_full[idx]
                for sysobj in make_systems(seed):
                    t = time.time()
                    try:
                        proba = sysobj.run(tr_texts, tr_y, raw.test_texts)
                        m = metrics.compute_metrics(raw.y_test, proba.argmax(1), proba,
                                                    n_classes=n)
                        err = None
                    except Exception as e:
                        m, err = {}, f"{type(e).__name__}: {e}"
                    row = {"system": sysobj.name, "shots": k, "seed": seed,
                           "fit_seconds": round(time.time() - t, 3), "error": err, **m}
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    rows.append(row)
                    status = err or f"acc={m.get('accuracy', float('nan')):.3f}"
                    print(f"  k={k!s:>3} seed={seed} {sysobj.name:<20} {status}")

    # quick summary table
    by = {}
    for r in rows:
        if r["error"] or "accuracy" not in r:
            continue
        by.setdefault(r["system"], {}).setdefault(r["shots"], []).append(r["accuracy"])
    print("\nmean accuracy — SST-2 mechanism probe")
    hdr = f"{'system':<20}" + "".join(f"{str(k):>8}" for k in shots)
    print(hdr)
    for s in ["hv_entail_zshot", "hv_entail_pool", "hv_prior_fixed",
              "emb:finecat-pooled", "emb:all-MiniLM"]:
        if s in by:
            print(f"{s:<20}" + "".join(
                f"{statistics.mean(by[s][k]):>8.3f}" if by[s].get(k) else f"{'-':>8}"
                for k in shots))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
