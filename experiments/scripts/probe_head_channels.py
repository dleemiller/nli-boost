#!/usr/bin/env python
"""Per-channel (per-head) signal probe across the 5 finecat sizes.

finecat is a 3-head NLI cross-encoder (entail=0, neutral=1, contradict=2). Every HV feature is
one of these heads (or a combination: `contrast` = P(entail) - P(contradict)). This probe asks a
simple, label-grounded question of each head *in isolation*:

    For a (text, hypothesis) pair, how well does head h separate MATCHED pairs (the text's true
    class == the hypothesis's intended class) from MISMATCHED pairs?

That separation is the raw discriminative signal the downstream head (RF/logreg) gets to work
with, measured BEFORE any training — it depends only on the encoder + the fixed expert pool. We
sweep it across finecat {xxs, xs, s, m, l} to see WHICH channel's signal grows with capacity, and
overlay the downstream accuracy at each size (the encoder resource curve, the size-analog of the
learning curve) so the mechanism can be read against the outcome.

Why per-head, why now: Lee's contrastive/framing probe (NOTES 2135-2151) hand-framed hypotheses on
the stuck TREC DESC/ENTY leaf and found the `entail` column AND the `neutral` column BOTH cleared
the split bar — "the 3-class push validated at the leaf level." This generalises that single-leaf,
single-encoder (-l) observation to a systematic per-head distribution across the whole size family:
is neutral a real third channel, or does it just absorb uncertainty mass? Does contradict add
signal orthogonal to entail (the `contrast` score_mode's premise)? And does whichever channel
carries the signal sharpen with capacity in step with accuracy, or is the family already saturated?

Per (dataset, size) we emit, for each channel in {entail, neutral, contradict, contrast}:
  * matched / mismatched distribution (mean, std, fixed-bin histograms — for violins/hists)
  * separation = 2*|AUC-0.5|  (direction-agnostic channel informativeness, 0..1)
  * signed roc_auc and Cohen's d (sign shows which side "matched" sits on; contradict is < 0.5)
plus the downstream RF accuracy/macro-F1 on the fixed test set (mirrors encoder_size_sweep so the
two resource curves are directly comparable).

Protocol: fixed test set (size 2000, seed 7 — same as the learning curve), full expert pool, pure
local scoring (no LLM). Separation uses the ENTIRE test set (label-free of training); accuracy uses
a fixed, seed-pinned train subsample so it is identical across sizes.

    HV_CACHE_DIR=/tmp uv run python experiments/scripts/probe_head_channels.py \
        --datasets trec sst2 goemotions ag_news --device cuda

Then the figures land in experiments/results/figures/head_channel_*.{png,pdf}.
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

SIZES = ["xxs", "xs", "s", "m", "l"]
# entail=0, neutral=1, contradict=2; `contrast` is the entail-contradict score_mode (features.py).
HEADS = ["entail", "neutral", "contradict"]
CHANNELS = ["entail", "neutral", "contradict", "contrast"]
# datasets whose expert-pool tags are REAL class names (matched-set is well defined). banking77 /
# clinc150 tags are only representative classes, so their matched set is degenerate — opt in
# explicitly if you want them.
DEFAULT_DATASETS = ["trec", "sst2", "goemotions", "ag_news"]

PROB_BINS = np.linspace(0.0, 1.0, 21)      # entail / neutral / contradict live in [0, 1]
CONTRAST_BINS = np.linspace(-1.0, 1.0, 21)  # entail - contradict lives in [-1, 1]


def _channel_values(probs: np.ndarray, channel: str) -> np.ndarray:
    """(n_text, n_hyp) values for one channel from the (n_text, n_hyp, 3) prob tensor."""
    e, neu, c = probs[:, :, 0], probs[:, :, 1], probs[:, :, 2]
    return {"entail": e, "neutral": neu, "contradict": c, "contrast": e - c}[channel]


def _separation_stats(vals: np.ndarray, matched: np.ndarray) -> dict:
    """Distribution + separation of one channel between matched and mismatched flattened pairs."""
    m, mm = vals[matched], vals[~matched]
    bins = CONTRAST_BINS if vals.min() < -1e-9 else PROB_BINS
    pooled = np.sqrt(((m.var() * len(m)) + (mm.var() * len(mm))) / max(len(m) + len(mm), 1))
    d = float((m.mean() - mm.mean()) / pooled) if pooled > 0 else 0.0
    try:
        auc = float(roc_auc_score(matched.astype(int), vals))  # matched=1
    except ValueError:
        auc = float("nan")
    return {
        "n_matched": int(matched.sum()), "n_mismatched": int((~matched).sum()),
        "mean_matched": float(m.mean()), "mean_mismatched": float(mm.mean()),
        "std_matched": float(m.std()), "std_mismatched": float(mm.std()),
        "cohens_d": d,
        "roc_auc": auc,                                   # signed; contradict sits < 0.5
        "separation": float(abs(auc - 0.5) * 2) if auc == auc else float("nan"),  # 0..1
        "hist_bins": bins.tolist(),
        "hist_matched": np.histogram(m, bins=bins)[0].tolist(),
        "hist_mismatched": np.histogram(mm, bins=bins)[0].tolist(),
    }


def eval_one(dataset: str, size: str, n_per_class: int, device: str) -> dict:
    raw = datasets.load_raw(dataset, test_size=2000, test_seed=7)
    pool, tags = hypotheses.expert_pool(dataset)
    fz = NLIFeaturizer(encoder=f"dleemiller/finecat-nli-{size}", device=device, verbose=False)

    # --- per-head separation on the FULL fixed test set (label-free of training) ---
    t0 = time.time()
    probs = fz.probs(list(raw.test_texts), pool)  # (n_test, n_hyp, 3)
    # matched[i, j] = the test text's true class name == hypothesis j's intended class
    test_names = np.asarray([raw.class_names[y] for y in raw.y_test])
    matched = (test_names[:, None] == np.asarray(tags)[None, :])  # (n_test, n_hyp) bool
    channels = {
        ch: _separation_stats(_channel_values(probs, ch).ravel(), matched.ravel())
        for ch in CHANNELS
    }

    # --- downstream accuracy at this size (mirrors encoder_size_sweep, entail_contradict + RF) ---
    idx = datasets.subsample_indices(raw.y_train, n_per_class, seed=0)
    tr_texts = [raw.train_texts[i] for i in idx]
    tr_y = raw.y_train[idx]
    Xtr = fz.features(tr_texts, pool, "entail_contradict")
    Xte = fz.features(list(raw.test_texts), pool, "entail_contradict")
    head = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=4).fit(Xtr, tr_y)
    proba = head.predict_proba(Xte)
    m = metrics.compute_metrics(raw.y_test, proba.argmax(1), proba, n_classes=raw.n_classes)

    return {
        "dataset": dataset, "size": size, "n_hyp": len(pool), "n_test": len(raw.y_test),
        "n_train": int(len(tr_y)), "acc": m["accuracy"], "macro_f1": m["macro_f1"],
        "secs": round(time.time() - t0, 1), "channels": channels,
    }


def _figures(dataset: str, per_size: dict, sizes: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = [s for s in sizes if s in per_size]
    xs = list(range(len(sizes)))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    colors = {"entail": "#2166ac", "neutral": "#7f7f7f", "contradict": "#b2182b",
              "contrast": "#1b7837"}

    # --- Figure A: per-head separation vs encoder size, with accuracy overlaid ---
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for ch in CHANNELS:
        ys = [per_size[s]["channels"][ch]["separation"] for s in sizes]
        ax.plot(xs, ys, marker="o", ms=5, lw=2.2 if ch != "contrast" else 1.8,
                ls="-" if ch != "contrast" else "--", color=colors[ch], label=f"{ch} sep")
    ax.set_ylabel("channel separation  2·|AUC−0.5|  (matched vs mismatched)")
    ax.set_xticks(xs); ax.set_xticklabels([f"-{s}" for s in sizes])
    ax.set_xlabel("finecat encoder size")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax2 = ax.twinx()
    accs = [per_size[s]["acc"] for s in sizes]
    ax2.plot(xs, accs, marker="s", ms=5, lw=2.4, color="black", label="RF accuracy")
    ax2.set_ylabel("downstream RF accuracy")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=7, loc="best", ncol=2)
    ax.set_title(f"{dataset}: per-head signal separation vs encoder size (accuracy overlaid)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"head_channel_separation_{dataset}.{ext}", dpi=160)
    plt.close(fig)

    # --- Figure B: distribution by head — rows = sizes, cols = 3 heads, matched vs mismatched ---
    nr, nc = len(sizes), len(HEADS)
    fig, axes = plt.subplots(nr, nc, figsize=(3.0 * nc, 1.9 * nr), squeeze=False, sharex="col")
    for r, s in enumerate(sizes):
        for c, head in enumerate(HEADS):
            ax = axes[r][c]
            st = per_size[s]["channels"][head]
            edges = np.asarray(st["hist_bins"])
            ctr = (edges[:-1] + edges[1:]) / 2
            mm = np.asarray(st["hist_mismatched"], float); mmn = mm / max(mm.sum(), 1)
            mt = np.asarray(st["hist_matched"], float); mtn = mt / max(mt.sum(), 1)
            w = edges[1] - edges[0]
            ax.bar(ctr, mmn, width=w, color="#bbbbbb", alpha=0.7, label="mismatched")
            ax.step(ctr, mtn, where="mid", color=colors[head], lw=1.8, label="matched")
            ax.set_yticks([])
            if r == 0:
                ax.set_title(head, fontsize=10, color=colors[head])
            if c == 0:
                ax.set_ylabel(f"-{s}", rotation=0, ha="right", va="center", fontsize=10)
            if r == nr - 1:
                ax.set_xlabel("P(head)")
            ax.text(0.97, 0.85, f"sep={st['separation']:.2f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7)
    axes[0][nc - 1].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"{dataset}: per-head probability distribution, matched vs mismatched pairs "
                 f"(rows = finecat size)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"head_channel_dist_{dataset}.{ext}", dpi=160)
    plt.close(fig)
    print(f"[figure] head_channel_(separation|dist)_{dataset}.(png|pdf)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--sizes", nargs="+", default=SIZES)
    ap.add_argument("--n-per-class", type=int, default=100,
                    help="train examples/class for the downstream accuracy overlay (fixed across sizes)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--run-id", default="probe_head_channels")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    run_dir = RESULTS_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    scalar = run_dir / "results.jsonl"
    scalar.unlink(missing_ok=True)

    results: dict = {d: {} for d in args.datasets}
    with scalar.open("w") as fh:
        for d in args.datasets:
            for s in args.sizes:
                try:
                    r = eval_one(d, s, args.n_per_class, args.device)
                except Exception as e:  # noqa: BLE001
                    print(f"[{d:10} -{s:>3}] FAILED {type(e).__name__}: {e}", flush=True)
                    continue
                results[d][s] = r
                # full record (with histograms) per (dataset, size)
                (run_dir / f"{d}__{s}.json").write_text(json.dumps(r, indent=2))
                # one flat scalar row per (dataset, size, channel) for easy tabulation
                for ch in CHANNELS:
                    st = r["channels"][ch]
                    fh.write(json.dumps({
                        "dataset": d, "size": s, "channel": ch, "acc": r["acc"],
                        "macro_f1": r["macro_f1"], "n_hyp": r["n_hyp"],
                        **{k: v for k, v in st.items() if not k.startswith("hist")},
                    }) + "\n")
                fh.flush()
                sep = " ".join(f"{ch[:4]}={r['channels'][ch]['separation']:.2f}" for ch in CHANNELS)
                print(f"[{d:10} -{s:>3}] acc={r['acc']:.3f} f1={r['macro_f1']:.3f} | sep: {sep} "
                      f"({r['secs']}s)", flush=True)

    # summary: per dataset, separation of each channel across sizes + accuracy row
    for d in args.datasets:
        if not results[d]:
            continue
        sizes = [s for s in args.sizes if s in results[d]]
        print(f"\n===== {d}: channel separation (2·|AUC−0.5|) vs encoder size =====")
        print(f"{'channel':<12}" + "".join(f"{('-' + s):>8}" for s in sizes))
        for ch in CHANNELS:
            print(f"{ch:<12}" + "".join(
                f"{results[d][s]['channels'][ch]['separation']:>8.3f}" for s in sizes))
        print(f"{'RF acc':<12}" + "".join(f"{results[d][s]['acc']:>8.3f}" for s in sizes))
        if not args.no_figures:
            _figures(d, results[d], args.sizes)

    print("\nread: which channel's separation climbs with size = where capacity buys signal;")
    print("      neutral separation >~0.1 = a real third channel (Lee's 3-class push), not just")
    print("      uncertainty mass; separation flat-but-accuracy-flat = family saturated on the task.")
    print(f"[done] -> {scalar}")


if __name__ == "__main__":
    main()
