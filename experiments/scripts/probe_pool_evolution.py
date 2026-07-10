#!/usr/bin/env python
"""Did generating/evolving hypotheses move the NLI per-head profile the "good" way?

Question (Bradley): as we go from hand-written -> LM-generated -> selected/evolved hypotheses (and
round-by-round inside a tree-evolve run), do the pairs move toward better predictions — MATCHED
pairs gaining entailment, and OFF-CLASS (mismatched) pairs moving OUT of contradict INTO neutral
(the pool learning to ABSTAIN on texts it doesn't apply to instead of over-claiming a contradiction)?

Same per-head lens as probe_head_channels (entail=0, neutral=1, contradict=2; softmax e+n+c=1) but
we sweep the POOL, not the encoder, at fixed -l over the fixed test set (2000, seed 7).

Tagging is held CONSTANT across every pool so the movement reflects the HYPOTHESES, not the tags:
every hypothesis is tagged by arg-max mean train-entailment (generate_pool.derive_tags — exactly how
the LM pools were tagged). matched(text, hyp) = text's gold class == hyp's derived tag.

Part A — GENERATION PIPELINE (static pools):
    expert (hand) -> gen256 (raw LM, deepseek-v4-flash, STS-dedup, no evolve) -> gen_static -> gen_evolved
Part B — TREE-EVOLVE TRAJECTORY (round-by-round): reconstruct each round's pool from runs/*/log.jsonl
    (seed 8 hyps + accepted hyps up to round k), across seeds 7/17/27, and track the per-head profile
    and RF accuracy as a function of round.

Per pool/round we report mean P(entail/neutral/contradict) on matched & mismatched pairs, the entail
separation 2*|AUC-0.5|, the off-class abstain ratio neutral/(neutral+contradict), and downstream RF
accuracy (entail_contradict, fixed 100/class train). Pure local scoring, no LLM spend.

    HV_CACHE_DIR=/tmp uv run python experiments/scripts/probe_pool_evolution.py --dataset trec
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from hvexp import datasets, hypotheses, metrics  # noqa: E402
from hvexp.features import NLIFeaturizer  # noqa: E402

RESULTS_ROOT = pathlib.Path(__file__).resolve().parents[1] / "results" / "raw"
FIGDIR = pathlib.Path(__file__).resolve().parents[1] / "results" / "figures"
POOLDIR = pathlib.Path(__file__).resolve().parents[1] / "results" / "pools"
RUNSDIR = pathlib.Path(__file__).resolve().parents[2] / "runs"
ENCODER = "dleemiller/finecat-nli-l"

PIPELINE = {
    "trec": [
        ("expert (hand)", "expert"),
        ("gen256 (raw LM)", POOLDIR / "trec_gen256.json"),
        ("gen_static", POOLDIR / "trec_gen_static.json"),
        ("gen_evolved", POOLDIR / "trec_gen_evolved.json"),
    ],
}
TREE_RUNS = {"trec": ["trec_tree_scratch2", "trec_tree_scratch_s17", "trec_tree_scratch_s27"]}
COL = {"entail": "#2166ac", "neutral": "#7f7f7f", "contradict": "#b2182b"}


def load_pool_texts(dataset: str, source) -> list[str]:
    if source == "expert":
        pool, _ = hypotheses.expert_pool(dataset)
        return pool
    obj = json.loads(pathlib.Path(source).read_text())
    return [h["text"] for h in obj]


def derive_tags(pool, train_texts, y_train, class_names, fz) -> list[str]:
    """Each hyp -> class whose train texts most entail it (arg-max mean P(entail)). generate_pool.py."""
    e = fz.features(train_texts, pool, score_mode="entail")
    y = np.asarray(y_train)
    return [class_names[int(np.argmax([e[y == c, j].mean() if (y == c).any() else -1
                                       for c in range(len(class_names))]))]
            for j in range(len(pool))]


def profile(probs: np.ndarray, matched: np.ndarray) -> dict:
    e, n, c = probs[:, :, 0].ravel(), probs[:, :, 1].ravel(), probs[:, :, 2].ravel()
    m = matched.ravel()
    def means(mask):
        return {"entail": float(e[mask].mean()), "neutral": float(n[mask].mean()),
                "contradict": float(c[mask].mean())}
    mt, mm = means(m), means(~m)
    try:
        auc = float(roc_auc_score(m.astype(int), e))
    except ValueError:
        auc = float("nan")
    abstain = mm["neutral"] / max(mm["neutral"] + mm["contradict"], 1e-9)
    return {"matched": mt, "mismatched": mm,
            "entail_separation": float(abs(auc - 0.5) * 2) if auc == auc else float("nan"),
            "abstain_ratio_mismatched": float(abstain)}


def rf_metrics(fz, pool, raw, tr_idx) -> dict:
    """Downstream RF (entail_contradict, fixed train) — accuracy AND calibration (logloss/ECE)."""
    Xtr = fz.features([raw.train_texts[i] for i in tr_idx], pool, "entail_contradict")
    Xte = fz.features(list(raw.test_texts), pool, "entail_contradict")
    head = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=4).fit(Xtr, raw.y_train[tr_idx])
    proba = head.predict_proba(Xte)
    return metrics.compute_metrics(raw.y_test, proba.argmax(1), proba, n_classes=raw.n_classes)


def eval_pool(pool, fz, raw, tr_idx, *, with_acc=True) -> dict:
    tags = np.asarray(derive_tags(pool, [raw.train_texts[i] for i in tr_idx], raw.y_train[tr_idx],
                                  raw.class_names, fz))
    probs = fz.probs(list(raw.test_texts), pool)
    test_names = np.asarray([raw.class_names[y] for y in raw.y_test])
    prof = profile(probs, test_names[:, None] == tags[None, :])
    prof["n_hyp"] = len(pool)
    if with_acc:
        m = rf_metrics(fz, pool, raw, tr_idx)
        prof["acc"] = m["accuracy"]
        prof["logloss"] = m["log_loss"]
        prof["ece"] = m["ece"]
    return prof


# ---------------------------------------------------------------- Part B: tree round reconstruction
def round_pools(run: str) -> list[list[str]]:
    """pool snapshot after each round: seed(8) + accepted hyps up to that round."""
    mo = json.loads((RUNSDIR / run / "model.json").read_text())
    final = mo["hypotheses"]
    log = [json.loads(l) for l in (RUNSDIR / run / "log.jsonl").read_text().splitlines() if l.strip()]
    added = [r["hypothesis"] for r in log if r["added"]]
    seed = [h for h in final if h not in added]
    snaps = []
    for k in range(len(log)):
        acc_upto = [r["hypothesis"] for r in log[:k + 1] if r["added"]]
        snaps.append(seed + acc_upto)
    return snaps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="trec")
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-trajectory", action="store_true")
    args = ap.parse_args()
    ds = args.dataset

    run_dir = RESULTS_ROOT / f"probe_pool_evolution_{ds}"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw = datasets.load_raw(ds, test_size=2000, test_seed=7)
    fz = NLIFeaturizer(encoder=ENCODER, device=args.device, verbose=True)
    tr_idx = datasets.subsample_indices(raw.y_train, args.n_per_class, seed=0)

    # ---- Part A: generation pipeline ----
    labels, res = [], []
    for label, source in PIPELINE[ds]:
        t0 = time.time()
        r = eval_pool(load_pool_texts(ds, source), fz, raw, tr_idx)
        labels.append(label); res.append(r)
        print(f"[A {label:<16} n={r['n_hyp']:>3}] acc={r['acc']:.3f}  E|match={r['matched']['entail']:.3f}"
              f"  C|mis={r['mismatched']['contradict']:.3f} N|mis={r['mismatched']['neutral']:.3f}"
              f"  abstain={r['abstain_ratio_mismatched']:.3f} sep={r['entail_separation']:.3f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
    (run_dir / "pipeline.json").write_text(json.dumps(
        {"dataset": ds, "encoder": ENCODER, "stages": [{"label": l, **r} for l, r in zip(labels, res)]},
        indent=2))
    print(f"\n===== {ds}: per-head profile across generation/evolution stages =====")
    print(f"{'stage':<16}{'n':>5}{'acc':>7}{'logloss':>9}{'ece':>7}{'E|match':>9}{'C|mis':>8}"
          f"{'N|mis':>8}{'abstain':>9}{'ent_sep':>9}")
    for l, r in zip(labels, res):
        print(f"{l:<16}{r['n_hyp']:>5}{r['acc']:>7.3f}{r['logloss']:>9.3f}{r['ece']:>7.3f}"
              f"{r['matched']['entail']:>9.3f}{r['mismatched']['contradict']:>8.3f}"
              f"{r['mismatched']['neutral']:>8.3f}{r['abstain_ratio_mismatched']:>9.3f}"
              f"{r['entail_separation']:>9.3f}")
    print("  → calibration check: does logloss/ECE DROP across stages while accuracy stays flat?")

    # ---- Part B: tree-evolve round trajectory (per seed) ----
    traj = {}
    if not args.skip_trajectory and ds in TREE_RUNS:
        for run in TREE_RUNS[ds]:
            if not (RUNSDIR / run / "log.jsonl").exists():
                print(f"[B] {run}: missing, skipped"); continue
            snaps = round_pools(run)
            rows = []
            for k, pool in enumerate(snaps):
                r = eval_pool(pool, fz, raw, tr_idx)
                rows.append({"round": k, **r})
            traj[run] = rows
            last = rows[-1]
            print(f"[B {run:<22}] rounds={len(rows)} final n={last['n_hyp']} acc={last['acc']:.3f} "
                  f"E|match={last['matched']['entail']:.3f} abstain={last['abstain_ratio_mismatched']:.3f}",
                  flush=True)
        (run_dir / "trajectory.json").write_text(json.dumps(traj, indent=2))

    _figures(ds, labels, res, traj)
    print(f"[done] -> {run_dir}")


def _figures(ds, labels, res, traj) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGDIR.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(labels)); w = 0.26

    # Figure 1 — pipeline stages
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, pane in zip(axes[:2], ["matched", "mismatched"]):
        for kk, head in enumerate(["entail", "neutral", "contradict"]):
            ax.bar(x + (kk - 1) * w, [r[pane][head] for r in res], w, color=COL[head], label=head)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.set_ylim(0, 1); ax.set_ylabel("mean P(head)")
        ax.set_title("MATCHED pairs  (want entail ↑)" if pane == "matched"
                     else "MISMATCHED pairs  (want contradict → neutral)", fontsize=10)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend(fontsize=8, loc="upper right")
    ax = axes[2]
    ax.plot(x, [r["entail_separation"] for r in res], "o-", color=COL["entail"], label="entail separation")
    ax.plot(x, [r["abstain_ratio_mismatched"] for r in res], "s-", color=COL["neutral"],
            label="off-class abstain\nneutral/(neutral+contra)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.25); ax.set_title("summary", fontsize=10)
    ax2 = ax.twinx(); ax2.plot(x, [r["acc"] for r in res], "^--", color="black", label="RF accuracy")
    ax2.set_ylabel("RF accuracy")
    ln = ax.get_lines() + ax2.get_lines(); ax.legend(ln, [l.get_label() for l in ln], fontsize=7)
    fig.suptitle(f"{ds}: NLI per-head profile as hypotheses are generated & evolved "
                 f"(fixed {ENCODER.split('/')[-1]}, uniform arg-max-entail tags)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"pool_evolution_{ds}.{ext}", dpi=160)
    plt.close(fig)
    print(f"[figure] pool_evolution_{ds}.(png|pdf)")

    # Figure 2 — tree-evolve round trajectory (mean over seeds, per-seed thin)
    if not traj:
        return
    metrics_spec = [("matched.entail", "E | matched", COL["entail"]),
                    ("mismatched.contradict", "C | mismatched", COL["contradict"]),
                    ("mismatched.neutral", "N | mismatched", COL["neutral"]),
                    ("abstain_ratio_mismatched", "abstain ratio", "#1b7837")]
    def get(row, path):
        cur = row
        for p in path.split("."):
            cur = cur[p]
        return cur
    maxr = max(len(v) for v in traj.values())
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for key, lab, col in metrics_spec:
        # align by round; carry each seed's last value to maxr
        series = []
        for rows in traj.values():
            vals = [get(r, key) for r in rows]
            vals += [vals[-1]] * (maxr - len(vals))
            series.append(vals)
        arr = np.array(series)
        ax.plot(range(maxr), arr.mean(0), "-", color=col, lw=2.4, label=lab)
        for s in arr:
            ax.plot(range(maxr), s, "-", color=col, lw=0.5, alpha=0.25)
    ax.set_ylim(0, 1); ax.set_xlabel("tree-evolve round"); ax.set_ylabel("mean P(head) / ratio")
    ax.grid(True, alpha=0.25)
    ax2 = ax.twinx()
    accs = []
    for rows in traj.values():
        v = [r["acc"] for r in rows]; v += [v[-1]] * (maxr - len(v)); accs.append(v)
    ax2.plot(range(maxr), np.array(accs).mean(0), "^--", color="black", lw=2.2, label="RF accuracy")
    ax2.set_ylabel("RF accuracy")
    handles = [plt.Line2D([], [], color=c, lw=2.4, label=l) for _, l, c in metrics_spec]
    handles.append(plt.Line2D([], [], color="black", lw=2.2, ls="--", marker="^", label="RF accuracy"))
    ax.legend(handles=handles, fontsize=8, loc="center right")
    ax.set_title(f"{ds}: per-head profile across tree-evolve rounds "
                 f"(mean of {len(traj)} seeds; thin = per seed)", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"pool_trajectory_{ds}.{ext}", dpi=160)
    plt.close(fig)
    print(f"[figure] pool_trajectory_{ds}.(png|pdf)")


if __name__ == "__main__":
    main()
