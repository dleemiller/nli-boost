"""CFPB Consumer Complaints: monetary-relief prediction from narrative + tabular metadata.

Benchmark (Wang, Zhu & Chen 2026, arXiv:2606.22664): binary "monetary relief" from narrative + LDA
topics + engineered features + categorical (company, state), temporal split. Reported AUC-ROC 0.78
(their hybrid GBM) vs 0.69 (TF-IDF baseline). See docs/cfpb-benchmark.md.

Here: interpretable NLI hypotheses on the narrative (HypothesisVectorizer) fused with the tabular
metadata via ColumnTransformer, and — the distinctive lever — the tabular block passed as
`baseline_features` so generated hypotheses are pruned by MARGINAL value over the metadata.

Needs the `train` extras + OPENROUTER_API_KEY + a GPU. Large; --limit subsamples for a first pass.
    uv run python examples/cfpb.py --limit 20000 --evolve
"""

import argparse
import os

_CLOSED = {
    "Closed with monetary relief",
    "Closed with non-monetary relief",
    "Closed with explanation",
    "Closed without relief",
    "Closed",
}
_CATS = ["Product", "Company", "State", "Submitted via"]


def load_frame(limit: int, per_class: int | None = None):
    """Stream the CFPB mirror, keeping closed complaints that have a narrative.

    `per_class` (recommended): collect up to this many of EACH relief class — monetary relief is
    only ~8% of complaints, so a plain `limit` subset has too few positives for a stable AUC.
    Balanced sampling makes the base rate artificial but AUC (a rank metric) stays comparable."""
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
        if per_class is not None:
            if kept[y] >= per_class:
                continue
            kept[y] += 1
        rows.append(
            {
                "date": r["Date received"],
                "narrative": narrative,
                "Product": r.get("Product") or "unknown",
                "Company": r.get("Company") or "unknown",
                "State": r.get("State") or "unknown",
                "Submitted via": r.get("Submitted via") or "unknown",
                "relief": y,
            }
        )
        if per_class is not None:
            if kept[0] >= per_class and kept[1] >= per_class:
                break
        elif len(rows) >= limit:
            break
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20000, help="closed+narrative complaints (natural rate)")
    ap.add_argument("--per-class", type=int, default=None, help="balanced: this many of each relief class")
    ap.add_argument("--encoder", default="dleemiller/finecat-nli-l")
    ap.add_argument("--n-hypotheses", type=int, default=64)
    ap.add_argument("--evolve", action="store_true", help="run the CV-prune/refill loop in fit")
    ap.add_argument("--cache", default="cache/nli_scores.sqlite")
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    if os.environ.get("APIKEY") and not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = os.environ["APIKEY"]

    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    from hypothesis_vectorizer import HypothesisVectorizer

    df = load_frame(args.limit, per_class=args.per_class)
    if args.per_class is not None:
        # balanced subset -> random STRATIFIED split (a temporal split would shift the class
        # balance: rare positives are older, so they'd pile into train and starve the test set)
        from sklearn.model_selection import train_test_split

        tr, te = train_test_split(df, test_size=0.2, random_state=7, stratify=df["relief"])
        tr, te = tr.copy(), te.copy()
        split = "random stratified (balanced subset)"
    else:
        cut = int(0.8 * len(df))  # natural-rate: temporal split (older train, newer test)
        tr, te = df.iloc[:cut].copy(), df.iloc[cut:].copy()
        split = "temporal (natural rate)"
    top = set(tr["Company"].value_counts().head(200).index)  # rare companies -> "other"
    for d in (tr, te):
        d["Company"] = d["Company"].where(d["Company"].isin(top), "other")
    print(
        f"{len(tr)} train / {len(te)} test | split: {split} | "
        f"train relief {tr['relief'].mean():.3f}, test {te['relief'].mean():.3f}"
    )

    # tabular block (dense one-hot), fit on train — the head sees it AND it prunes the hypotheses
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(tr[_CATS])
    baseline_tr = ohe.transform(tr[_CATS])

    # 1) GENERATE the hypothesis pool from the narratives, pruning by marginal value over the tabular
    gen = HypothesisVectorizer(
        task="Predict whether a consumer's financial complaint will be resolved with monetary relief.",
        class_definitions=[
            "no relief: the complaint is closed without monetary compensation to the consumer",
            "monetary relief: the company resolves the complaint by paying the consumer money",
        ],
        class_names=["no relief", "monetary relief"],
        encoder=args.encoder,
        n_hypotheses=args.n_hypotheses,
        evolve=args.evolve,
        cache_path=args.cache,
        verbose=True,
    )
    gen.fit(tr["narrative"].tolist(), tr["relief"].to_numpy(), baseline_features=baseline_tr)

    # 2) SERVE the fixed pool (ColumnTransformer clones its steps on fit, which would otherwise
    #    re-generate) alongside the same tabular block -> classifier
    served = HypothesisVectorizer(hypotheses=gen.hypotheses_, encoder=args.encoder, cache_path=args.cache)
    model = Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        ("hyp", served, "narrative"),
                        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), _CATS),
                    ]
                ),
            ),
            ("clf", HistGradientBoostingClassifier(max_iter=300)),
        ]
    )
    model.fit(tr, tr["relief"].to_numpy())
    proba = model.predict_proba(te)[:, 1]
    auc = roc_auc_score(te["relief"].to_numpy(), proba)
    print(f"\nCFPB monetary-relief test AUC-ROC = {auc:.4f}   (benchmark: 0.78 hybrid, 0.69 TF-IDF)")
    print(f"pool size: {len(gen.hypotheses_)} hypotheses")


if __name__ == "__main__":
    main()
