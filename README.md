# nli-boost

Text classification from **LM-written NLI hypotheses**: a frozen NLI cross-encoder
([finecat](https://huggingface.co/dleemiller/finecat-nli-m)) scores whether each text entails
each of ~64 English sentences written by an LLM; those scores are features for a CV-disciplined
classical head. No fine-tuning anywhere — task adaptation lives in the sentences.

**See [METHOD.md](METHOD.md)** for the full process and the measurement behind every design
choice. TREC-6 with 2k training examples, current recipe (seed 7): **0.934** test accuracy at `-m`
and **0.954** at `-l`, ~7 minutes and under $0.01 per fit. (The original instruction spanned
0.916–0.938 across seeds at `-m`; the answer-oriented instruction below is what raised the `-m`
number.)

## Current best recipe

The configuration that wins on TREC today (`configs/trec_best_l.yaml`):

- **Encoder `finecat-nli-l`** — the one lever that reliably moves accuracy (`-m`→`-l` ≈ +5 pts,
  p=0.024). Everything else below is within noise at `-l`; the encoder is where the accuracy is.
- **Hand-written answer-oriented instruction** (the code default) — each hypothesis describes both
  the question and the *answer form* it implies (e.g. *"equivalent to asking someone to name a
  person"* / *"can be answered with a short proper name"*). Automated tuning of this instruction
  was neutral (McNemar p≈1.0), so it stays hand-written.
- **Covariance dedup** — reject a candidate whose entail-score vector correlates >0.95 with a kept
  hypothesis (removes *behavioral* duplicates that text-similarity dedup misses).
- **Pool of 64, evolved** — generate → rank by CV permutation-importance + cross-fold stability →
  prune confident deaths, refill against confusion hot-spots → repeat to a held-out plateau.
- **CV-selected classical head** (RF / HistGBM) over the entail+contradict features.

Result (seed 7): **0.934** at `-m`, **0.954** at `-l`. The answer-oriented instruction is what
lifts `-m` — the original instruction scored 0.920 at the same seed/dedup (`trec`), the
answer-oriented one 0.934 (`trec_newinstr`, +0.014). At `-l` that gain washes into the ~0.95
saturation band (`baseline_l` 0.952), so the instruction is only measured to help at `-m`. The new
instruction is currently validated at seed 7 only. ~7 min and <$0.01 per fit, and the model is a
human-readable list of ~64 English sentences.

**Add the lexical channel when inference cost matters** (`configs/trec_best_l_max.yaml`:
`lexical: {kind: tfidf_svd, dims: 128}`). TF-IDF is ~free at prediction time, while every NLI
hypothesis is a cross-encoder forward pass. So the lexical block joins evolution as a **fixed
baseline** and NLI hypotheses are pruned by their **marginal value over TF-IDF** — a hypothesis
whose signal TF-IDF already carries dies. The NLI pool (and thus per-prediction cost) shrinks to
only the hypotheses carrying semantics lexical can't reach, in the same accuracy band. Best point
estimate to date: **0.964** at `-l` (seed 7), though not yet significantly above the plain-TF-IDF
run (0.956, McNemar p=0.42) — treat as promising, not established.

> This recipe targets the **data-rich** regime. The method's expected edge is at **low-N**
> (2–5 examples/class), where a different pipeline applies (evolution off, prior-selected
> hypotheses, STS dedup, light head) — see [docs/low-n-plan.md](docs/low-n-plan.md).

## Setup

```bash
uv sync
echo 'OPENROUTER_API_KEY=sk-or-...' > .env   # the hypothesis proposer LM
uv run pre-commit install
```

## Usage

Training (produces a pool) needs the `train` extras (`dspy`, dataset loading, CLI):

```bash
uv run nli-boost run configs/trec.yaml            # full method: generate -> evolve -> head -> test
uv run nli-boost run configs/trec_finalize_l.yaml # reuse a fitted pool, re-score with -l encoder
uv run nli-boost report                           # pool_cv results across runs
uv run nli-boost compare runs/a runs/b            # paired McNemar: is a delta real or noise?
```

### Inference: `HypothesisVectorizer`

Inference is just NLI scoring against a fixed hypothesis list — **no LM, no dspy**. `HypothesisVectorizer`
is a scikit-learn transformer: it turns a column of text into features by scoring, for each text, how
strongly it entails (and contradicts) each hypothesis. Install inference-only with `pip install
nli-boost` (core deps); the `train` dependency-group adds what's needed to *generate* pools.

```python
from nli_boost import HypothesisVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

vec = HypothesisVectorizer.from_run("runs/trec_best_l")     # encoder (config.yaml) + pool (model.json)
clf = Pipeline([("hyp", vec), ("clf", HistGradientBoostingClassifier())]).fit(texts, y)
clf.predict(new_texts)
```

**Constructor** — `HypothesisVectorizer(hypotheses, *, encoder="dleemiller/finecat-nli-l",
score_mode="entail_contradict", device="cuda", batch_size=128, max_text_chars=1200, cache_path=None)`.
Standard sklearn params (introspectable via `get_params`/`set_params`, works with `clone`/`GridSearchCV`).

**Input / output** — `transform(X)` accepts a 1-D sequence of strings or a single text column (as
`ColumnTransformer` hands over). Output columns per `score_mode`: `entail_contradict` → `2·len(hypotheses)`
(`[P(entail) | P(contradict)]`), `entail` → `len(hypotheses)`, `contrast` → `len(hypotheses)`
(`P(entail) − P(contradict)`). `get_feature_names_out()` returns the hypotheses themselves, so feature
importances stay readable.

**Construct / persist** — `from_run(dir)` (a trained run's `config.yaml` + `model.json`),
`from_config(dict_or_yaml)`, `save(path)` / `load(path)` (JSON: hypotheses + encoder config, no weights).
A fitted vectorizer pickles cleanly (the live encoder/cache is dropped and rebuilt on demand).

**Compose** — it's a plain transformer, so the usual sklearn machinery applies:

```python
# one text column alongside other tabular features:
ColumnTransformer([("hyp", HypothesisVectorizer(hyps), "text"), ("num", StandardScaler(), num_cols)])

# optional TF-IDF channel — plain sklearn, not baked in:
FeatureUnion([("nli", HypothesisVectorizer(hyps)),
              ("tfidf", make_pipeline(TfidfVectorizer(), TruncatedSVD(128)))])
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

The pre-rewrite exploratory code (trees, boosting, and the experiments that selected this method)
is archived untracked in `src-bak/`; the experiment log lives in `NOTES.md`.
