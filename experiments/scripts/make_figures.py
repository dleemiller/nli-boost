#!/usr/bin/env python
"""Publication figures from learning-curve results. Saves both PDF (paper) and PNG (preview).

Figure 1 — learning curves: x = examples/class (log), y = accuracy or macro-F1, one line per
system with a bootstrap 95% CI band. The 'all' point is placed at the right with a gap and a
tick label.

Usage:
    uv run python experiments/scripts/make_figures.py experiments/results/raw/lc_trec_baselines \
        --title "TREC-6 low-label learning curve" --out trec_learning_curve
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from summarize_results import aggregate, load_rows  # noqa: E402

FIGDIR = pathlib.Path(__file__).resolve().parents[1] / "results" / "figures"

# stable, readable ordering + colors; systems not listed still plot (default cycle)
_ORDER = [
    "hv_prior_fixed", "hv_prior_reweight", "hv_expert_rf", "hv_expert_logreg",
    "hv_static_rf", "hv_static_prior_fixed", "hv_static_logreg",
    "hv_evolved_rf", "hv_evolved_prior_fixed", "hv_evolved_logreg",
    "zeroshot_nli", "tfidf_word+logreg", "tfidf_char+logreg", "tfidf_word+char+logreg",
    "emb:all-MiniLM-L6-v2+logreg", "finetune:distilbert-base-uncased",
]
# baselines drawn dashed to separate them from the HV method lines
_DASHED = ("tfidf", "emb", "zeroshot", "finetune")


def _xpos(shots, all_x):
    return all_x if shots == "all" else int(shots)


def plot(rows, metric: str, title: str, out: str) -> None:
    agg = aggregate(rows, metric)
    systems = sorted({s for s, _ in agg}, key=lambda s: (_ORDER.index(s) if s in _ORDER else 999, s))
    numeric_shots = sorted({int(k) for _, k in agg if k != "all"})
    has_all = any(k == "all" for _, k in agg)
    all_x = (max(numeric_shots) * 2) if numeric_shots else 100

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for sysname in systems:
        xs, ys, los, his = [], [], [], []
        for sh in numeric_shots + (["all"] if has_all else []):
            if (sysname, sh) not in agg:
                continue
            m, lo, hi, _n = agg[(sysname, sh)]
            xs.append(_xpos(sh, all_x))
            ys.append(m)
            los.append(lo)
            his.append(hi)
        if not xs:
            continue
        style = "--" if sysname.startswith(_DASHED) else "-"
        lw = 2.4 if sysname.startswith("hv_prior") else 1.6
        (line,) = ax.plot(xs, ys, style, marker="o", ms=4, lw=lw, label=sysname)
        ax.fill_between(xs, los, his, alpha=0.12, color=line.get_color())

    ax.set_xscale("log")
    ticks = numeric_shots + ([all_x] if has_all else [])
    labels = [str(s) for s in numeric_shots] + (["all"] if has_all else [])
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel("training examples per class")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = FIGDIR / f"{out}_{metric}.{ext}"
        fig.savefig(path, dpi=160)
    print(f"[figure] {FIGDIR / f'{out}_{metric}'}.(pdf|png)")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=pathlib.Path)
    ap.add_argument("--title", default="Learning curve")
    ap.add_argument("--out", default="learning_curve")
    ap.add_argument("--metrics", nargs="+", default=["accuracy", "macro_f1"])
    args = ap.parse_args()
    rows = load_rows(args.runs)
    for metric in args.metrics:
        plot(rows, metric, args.title, args.out)


if __name__ == "__main__":
    main()
