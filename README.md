# nli-boost

Text classification from **LM-written NLI hypotheses**: a frozen NLI cross-encoder
([finecat](https://huggingface.co/dleemiller/finecat-nli-m)) scores whether each text entails
each of ~64 English sentences written by an LLM; those scores are features for a CV-disciplined
classical head. No fine-tuning anywhere — task adaptation lives in the sentences.

**See [METHOD.md](METHOD.md)** for the full process and the measurement behind every design
choice. TREC-6 with 2k training examples: 0.916–0.938 test accuracy across seeds, ~7 minutes and
under $0.01 per fit; 0.946 with the larger encoder.

## Setup

```bash
uv sync
echo 'OPENROUTER_API_KEY=sk-or-...' > .env   # the hypothesis proposer LM
uv run pre-commit install
```

## Usage

```bash
uv run nli-boost run configs/trec.yaml            # full method: generate -> evolve -> head -> test
uv run nli-boost run configs/trec_finalize_l.yaml # reuse a fitted pool, re-score with -l encoder
uv run nli-boost report                           # pool_cv results across runs
uv run nli-boost diagnose runs/trec               # error decomposition + reward-hacking flags
```

Artifacts per run in `runs/<run_name>/`: the pool itself (`model.json` — the model is a list of
English sentences), the evolution audit trail (`log.jsonl`: every prune with its reason, every
refill with its target-AUC), `metrics.json` (the single honest headline), and `costs.json`.
All NLI scores are cached in `cache/nli_scores.sqlite`; reruns and post-hoc analyses are ~free.

## Development

```bash
uv run pytest          # full pipeline runs under fakes — no GPU or LM key needed
uv run ruff check .    # also enforced via pre-commit
```

The pre-rewrite exploratory code (trees, boosting, GEPA proposer tuning, and the experiments
that selected this method) is archived untracked in `src-bak/`; the experiment log lives in
`NOTES.md`.
