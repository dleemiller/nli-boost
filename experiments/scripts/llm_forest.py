#!/usr/bin/env python
"""llm-trees adaptation (arXiv 2409.18594): zero-shot LLM decision-tree forest, INDUCTION vs
EMBEDDING, in the HV text/NLI regime.

Per dataset: generate (or load) a forest of K trees, then compare on a fixed test set across low-N
budgets —
  * label-free anchors: zeroshot_nli, prior_fixed (expert pool)
  * label-free NEW:      llm_forest_induction (soft + hard routing)
  * learned:             llm_forest_embed (HVHead over the forest's node conditions)
  * learned control:     flat_pool_embed (HVHead over a size-matched flat generated pool) — isolates
                         whether the tree STRUCTURE of generation matters vs just generating N hyps.

Pre-registered expectation (see paper/experiment_notes.md): embedding >> induction; forest_embed ~
flat_pool_embed; induction beats single-template zeroshot but loses to the prior head.

    # zero-spend smoke test of the whole pipeline (no GPU, no API, synthetic data):
    uv run python experiments/scripts/llm_forest.py --dry-run

    # real run — GATED on approval (LM spend + GPU). Model fixed at deepseek-v4-flash:
    uv run python experiments/scripts/llm_forest.py --datasets trec sst2 goemotions \
        --n-trees 5 --flat-pool experiments/results/pools/{ds}_gen_static.json \
        --out experiments/results/raw/llm_forest.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from hvexp import datasets, forest as F, hypotheses, metrics  # noqa: E402
from hvexp.features import NLIFeaturizer  # noqa: E402
from hvexp.systems import HVHead, LLMForestInduction, PriorAggregation, ZeroShotNLI  # noqa: E402

FOREST_DIR = pathlib.Path(__file__).resolve().parents[1] / "results" / "forests"


def _examples(raw, per_class: int, seed: int) -> list[str]:
    """'[class] text' strings from a stratified TRAIN sample (strict: never the test set)."""
    idx = datasets.subsample_indices(raw.y_train, per_class, seed)
    return [f"[{raw.class_names[int(raw.y_train[i])]}] {raw.train_texts[i]}" for i in idx]


def get_forest(raw, args) -> list:
    """Load a cached forest if present, else generate one (LM spend — the gated step)."""
    cache = FOREST_DIR / f"{raw.name}_k{args.n_trees}_s{args.seed}.json"
    if cache.exists() and not args.regen:
        print(f"[forest] {raw.name}: loading cached {cache.name}")
        return F.load_forest(cache)

    from hypothesis_vectorizer.config import LMConfig
    from hypothesis_vectorizer.costs import CostTracker
    from hypothesis_vectorizer.train.proposer import Proposer

    print(f"[forest] {raw.name}: GENERATING {args.n_trees} trees via {args.lm} (LM spend)")
    proposer = Proposer(LMConfig(model=args.lm), CostTracker())
    examples = _examples(raw, args.sample_per_class, args.seed)
    forest = F.build_forest(proposer, raw.task, list(raw.class_descriptions), examples, args.n_trees)
    F.save_forest(forest, cache)
    conds, leaves = F.flatten_conditions(forest), F.leaf_classes(forest)
    missing = set(raw.class_names) - leaves
    print(f"[forest] {raw.name}: {len(forest)} trees, {len(conds)} unique conditions, "
          f"leaves cover {len(leaves)}/{raw.n_classes} classes"
          + (f" (MISSING: {sorted(missing)})" if missing else ""))
    return forest


def build_systems(raw, forest, flat_pool, fz, anchors=None):
    """Every system exposes run(train_texts, y_train, test_texts) -> proba. Label-free systems
    ignore the labels, so they are evaluated once (budget-independent). `anchors`, when given,
    supplies (templates, expert_pool, expert_tags) directly (dry-run); else the dataset dicts."""
    if anchors is not None:
        templates, exp_pool, exp_tags = anchors
    else:
        templates = hypotheses.ZEROSHOT_TEMPLATES[raw.name]
        exp_pool, exp_tags = hypotheses.expert_pool(raw.name)
    conditions = F.flatten_conditions(forest)
    labelfree = {
        "zeroshot_nli": ZeroShotNLI(raw.n_classes, templates, fz),
        "prior_fixed": PriorAggregation(raw.n_classes, exp_pool, exp_tags, raw.class_names, fz, mode="fixed"),
        "forest_induction_soft": LLMForestInduction(raw.n_classes, forest, raw.class_names, fz, "soft"),
        "forest_induction_hard": LLMForestInduction(raw.n_classes, forest, raw.class_names, fz, "hard"),
    }
    learned = {"forest_embed": HVHead(raw.n_classes, conditions, fz, head="auto")}
    if flat_pool:
        learned["flat_pool_embed"] = HVHead(raw.n_classes, flat_pool, fz, head="auto")
    return labelfree, learned


def evaluate(raw, forest, flat_pool, fz, budgets, seeds, anchors=None) -> dict:
    labelfree, learned = build_systems(raw, forest, flat_pool, fz, anchors)
    out: dict = {"n_classes": raw.n_classes, "n_conditions": len(F.flatten_conditions(forest)),
                 "label_free": {}, "learned": {}}
    # label-free: budget-independent, compute once
    for name, sysm in labelfree.items():
        proba = sysm.run([], np.array([]), list(raw.test_texts))
        m = metrics.compute_metrics(raw.y_test, proba.argmax(1), proba, n_classes=raw.n_classes)
        out["label_free"][name] = {"acc": m["accuracy"], "f1": m["macro_f1"]}
        print(f"    [{raw.name:10} {name:22}] acc={m['accuracy']:.3f} f1={m['macro_f1']:.3f}")
    # learned heads: budget x seed
    for name, sysm in learned.items():
        out["learned"][name] = {}
        for k in budgets:
            accs = []
            for s in seeds:
                idx = datasets.subsample_indices(raw.y_train, k, s)
                tr_texts = [raw.train_texts[i] for i in idx]
                proba = sysm.run(tr_texts, raw.y_train[idx], list(raw.test_texts))
                accs.append(metrics.compute_metrics(
                    raw.y_test, proba.argmax(1), proba, n_classes=raw.n_classes)["accuracy"])
            out["learned"][name][str(k)] = {"acc_mean": float(np.mean(accs)),
                                            "acc_std": float(np.std(accs)), "seeds": len(seeds)}
            print(f"    [{raw.name:10} {name:16} k={str(k):>4}] "
                  f"acc={np.mean(accs):.3f}±{np.std(accs):.3f}")
    # paired McNemar on the fixed test set at the top budget, single seed (deterministic)
    sig_budget = "all" if "all" in budgets else max(b for b in budgets if b != "all")
    out["significance"] = paired_significance(raw, labelfree, learned, sig_budget, seeds[0])
    for pair, r in out["significance"].items():
        print(f"    [{raw.name:10} McNemar {pair:20}] p={r['p_value']:.3f} "
              f"(discordant {r['discordant']}: A {r['b_only_A']} / B {r['c_only_B']}) @k={sig_budget}")
    return out


def paired_significance(raw, labelfree, learned, budget, seed) -> dict:
    """Exact McNemar on the fixed test set for the decisive pairs, at one (budget, seed) so both
    heads are deterministic. Reuses the library's `mcnemar` (train/compare.py)."""
    from hypothesis_vectorizer.train.compare import mcnemar

    idx = datasets.subsample_indices(raw.y_train, budget, seed)
    tr_texts, tr_y = [raw.train_texts[i] for i in idx], raw.y_train[idx]
    pred = {name: sysm.run(tr_texts, tr_y, list(raw.test_texts)).argmax(1)
            for name, sysm in learned.items()}
    pred["induction_soft"] = labelfree["forest_induction_soft"].run(
        [], np.array([]), list(raw.test_texts)).argmax(1)
    out: dict = {"embed_vs_induction": mcnemar(raw.y_test, pred["forest_embed"], pred["induction_soft"])}
    if "flat_pool_embed" in pred:  # the promotion test
        out["forest_vs_flat"] = mcnemar(raw.y_test, pred["forest_embed"], pred["flat_pool_embed"])
    return out


# --------------------------------------------------------------------------- dry-run (no GPU/API)
class _FakeFeaturizer:
    def features(self, texts, pool, score_mode="entail_contradict"):
        vals = np.array([[float(v) for v in t.split("|")] for t in texts])
        cols = [int(h.split()[0][1:]) for h in pool]
        e = vals[:, cols]
        if score_mode == "entail":
            return e
        c = 1.0 - e
        return e - c if score_mode == "contrast" else np.concatenate([e, c], axis=1)


def _synthetic_raw():
    from hvexp.datasets import RawData
    rng = np.random.default_rng(0)
    x = rng.random((600, 4))
    y = ((x[:, 0] > 0.5).astype(int) * 2 + (x[:, 1] > 0.5).astype(int))
    enc = ["|".join(f"{v:.6f}" for v in row) for row in x]
    return RawData(name="synthetic", task="separate", class_names=["A", "B", "C", "D"],
                   class_descriptions=["A: x", "B: y", "C: z", "D: w"],
                   train_texts=enc[:400], y_train=y[:400],
                   test_texts=enc[400:], y_test=y[400:], test_seed=7)


def _synthetic_forest():
    from hypothesis_vectorizer.train.proposer import TreeNode
    lo = TreeNode(condition="f1 second bit", yes=TreeNode(leaf_class="B"), no=TreeNode(leaf_class="A"))
    hi = TreeNode(condition="f1 second bit", yes=TreeNode(leaf_class="D"), no=TreeNode(leaf_class="C"))
    t1 = TreeNode(condition="f0 first bit", yes=hi, no=lo)
    t2 = TreeNode(condition="f0 first bit high",
                  yes=TreeNode(condition="f1 second high", yes=TreeNode(leaf_class="D"), no=TreeNode(leaf_class="C")),
                  no=TreeNode(condition="f1 second low", yes=TreeNode(leaf_class="B"), no=TreeNode(leaf_class="A")))
    return [t1, t2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["trec", "sst2", "goemotions"])
    ap.add_argument("--n-trees", type=int, default=5)
    ap.add_argument("--budgets", nargs="+", default=["5", "20", "100", "all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 23])
    ap.add_argument("--sample-per-class", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7, help="forest-generation / example seed")
    ap.add_argument("--encoder", default="dleemiller/finecat-nli-l")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lm", default="openrouter/deepseek/deepseek-v4-flash")
    ap.add_argument("--flat-pool", default=None,
                    help="path template to a generated flat pool json, '{ds}' substituted per dataset")
    ap.add_argument("--regen", action="store_true", help="regenerate forests even if cached (LM spend)")
    ap.add_argument("--dry-run", action="store_true", help="synthetic data + fake featurizer; no GPU/API")
    ap.add_argument("--out", default="experiments/results/raw/llm_forest.json")
    args = ap.parse_args()

    budgets = [b if b == "all" else int(b) for b in args.budgets]

    if args.dry_run:
        print("[dry-run] synthetic data, fake featurizer — validating wiring, zero spend")
        raw = _synthetic_raw()
        # anchors must be 'f<i> ...' so the fake featurizer can read a column from each
        anchors = (
            [f"f{i} is class {c}" for i, c in enumerate(raw.class_names)],  # zero-shot templates
            [f"f{i} is class {c}" for i, c in enumerate(raw.class_names)],  # expert pool texts
            list(raw.class_names),  # expert tags
        )
        res = evaluate(raw, _synthetic_forest(), ["f0 first bit", "f1 second bit"],
                       _FakeFeaturizer(), budgets=[5, "all"], seeds=[0, 1], anchors=anchors)
        print("\n[dry-run] OK — label-free:", {k: round(v["acc"], 3) for k, v in res["label_free"].items()})
        return

    results: dict = {}
    for ds in args.datasets:
        raw = datasets.load_raw(ds, test_size=2000, test_seed=args.seed)
        fz = NLIFeaturizer(encoder=args.encoder, device=args.device, verbose=False)
        forest = get_forest(raw, args)
        flat_pool = None
        if args.flat_pool:
            p = pathlib.Path(args.flat_pool.replace("{ds}", ds))
            if p.exists():
                flat_pool = [r["text"] for r in json.loads(p.read_text())]
                print(f"[flat] {ds}: loaded {len(flat_pool)} hypotheses from {p.name}")
            else:
                print(f"[flat] {ds}: {p} not found — skipping flat_pool_embed control")
        t0 = time.time()
        results[ds] = evaluate(raw, forest, flat_pool, fz, budgets, args.seeds)
        results[ds]["seconds"] = round(time.time() - t0, 1)
        results[ds]["cost"] = fz.cost_summary()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\n[done] wrote {out}")
    _print_verdict_table(results, budgets)


def _print_verdict_table(results: dict, budgets) -> None:
    print("\n===== INDUCTION vs EMBEDDING (test accuracy) =====")
    for ds, r in results.items():
        lf = r["label_free"]
        print(f"\n{ds} ({r['n_classes']} classes, {r['n_conditions']} forest conditions)")
        print(f"  label-free: zeroshot={lf['zeroshot_nli']['acc']:.3f}  "
              f"prior={lf['prior_fixed']['acc']:.3f}  "
              f"induction(soft)={lf['forest_induction_soft']['acc']:.3f}  "
              f"induction(hard)={lf['forest_induction_hard']['acc']:.3f}")
        for name, curve in r["learned"].items():
            cells = "  ".join(f"k={k}:{curve[str(k)]['acc_mean']:.3f}" for k in budgets if str(k) in curve)
            print(f"  {name:16} {cells}")
        for pair, s in r.get("significance", {}).items():
            print(f"  McNemar {pair:20} p={s['p_value']:.3f}  "
                  f"discordant {s['discordant']} (forest {s['b_only_A']} / other {s['c_only_B']})")


if __name__ == "__main__":
    main()
