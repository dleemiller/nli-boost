"""Run the accordion (expand<->compact) on a dataset and save the resulting compact pool.

GPU is used only to score each round's NEW hypotheses; the final kept pool is saved as
runs/<save>/model.json so a from_run config (rounds:0) can eval it. Needs train extras + a GPU.

    uv run python experiments/scripts/run_accordion.py --data trec --gen-size 64 --rounds 6 --save trec_accordion
"""

import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="trec")
    ap.add_argument("--gen-size", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--var-threshold", type=float, default=0.90)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--encoder", default="dleemiller/finecat-nli-l")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save", default="trec_accordion")
    ap.add_argument("--cache", default="cache/nli_scores.sqlite")
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    if os.environ.get("APIKEY") and not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = os.environ["APIKEY"]

    from hypothesis_vectorizer.cache import ScoreCache
    from hypothesis_vectorizer.config import DataConfig, EncoderConfig, LMConfig
    from hypothesis_vectorizer.costs import CostTracker
    from hypothesis_vectorizer.encoder import EntailmentScorer
    from hypothesis_vectorizer.train.accordion import accordion
    from hypothesis_vectorizer.train.data import load
    from hypothesis_vectorizer.train.proposer import Proposer

    costs = CostTracker()
    bundle = load(DataConfig(name=args.data, train_size=5452, val_size=0, test_size=2000), seed=args.seed)
    scorer = EntailmentScorer(EncoderConfig(model=args.encoder), ScoreCache(args.cache), costs)
    proposer = Proposer(LMConfig(), costs)

    kept, history = accordion(
        bundle,
        scorer,
        proposer,
        args.seed,
        gen_size=args.gen_size,
        rounds=args.rounds,
        var_threshold=args.var_threshold,
        patience=args.patience,
    )

    out = Path("runs") / args.save
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(
        json.dumps(
            {
                "type": "nli_pool",
                "encoder": args.encoder,
                "hypotheses": kept,
                "head": None,
                "provenance": f"accordion gen={args.gen_size} rounds={len(history)} of {args.data}",
            },
            indent=2,
        )
    )
    (out / "config.yaml").write_text(f"run_name: {args.save}  # synthetic: accordion kept pool\n")
    with open(out / "accordion_history.jsonl", "w") as f:
        for h in history:
            f.write(json.dumps(h) + "\n")
    print(f"\naccordion done: {len(kept)} kept hypotheses over {len(history)} rounds -> {out}/model.json")
    print("costs:", costs.to_dict())


if __name__ == "__main__":
    main()
