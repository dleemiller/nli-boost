# nli-boost

Interpretable text classification from **LM-written NLI hypotheses**. An LLM writes ~64 short English
sentences ("hypotheses"); a frozen NLI cross-encoder
([finecat](https://huggingface.co/dleemiller/finecat-nli-l)) scores, for each input text, how strongly
it **entails** and **contradicts** each hypothesis; those scores are the features for a
CV-disciplined classical head (RandomForest / HistGradientBoosting). Nothing is fine-tuned — task
adaptation lives entirely in the sentences, so the model *is* a readable list of hypotheses.

Two clean halves:

- **Inference** is just NLI scoring against a fixed hypothesis list → a scikit-learn transformer,
  [`HypothesisVectorizer`](#inference-hypothesisvectorizer). No LM, no `dspy`.
- **Training** generates + evolves the hypothesis list from labeled data (needs an LLM). Kept in a
  separate `train` dependency group so inference installs stay light.

```python
from nli_boost import HypothesisVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

vec = HypothesisVectorizer.from_run("runs/trec_best_l")   # load a trained pool + its encoder
clf = Pipeline([("hyp", vec), ("clf", HistGradientBoostingClassifier())]).fit(texts, y)
clf.predict(new_texts)
```

## Install

```bash
pip install nli-boost           # inference only: encoder + sklearn, no dspy
pip install "nli-boost[train]"  # + hypothesis generation/evolution (dspy), dataset loading, CLI
```

Developing from source (uv installs the `train` group by default):

```bash
uv sync
echo 'OPENROUTER_API_KEY=sk-or-...' > .env   # the hypothesis-proposer LM (training only)
uv run pre-commit install
```

## Inference: `HypothesisVectorizer`

A plain scikit-learn transformer that turns a column of text into NLI-entailment features against a
fixed hypothesis set. Its "model" is the hypothesis list + encoder name, so it needs **no LM and no
dspy** — only the encoder.

**Constructor** — `HypothesisVectorizer(hypotheses=None, *, encoder="dleemiller/finecat-nli-l",
score_mode="entail_contradict", device="cuda", batch_size=128, max_text_chars=1200, cache_path=None,
task=None, class_definitions=None, class_names=None, n_hypotheses=64, lm=..., dedup_corr=0.95)`.
Standard sklearn params (stored verbatim; `get_params`/`set_params`/`clone`/`GridSearchCV` all work).
The last six are generation knobs, used only by `fit` when `hypotheses` is None (see below).

**`transform(X)`** — accepts a 1-D sequence of strings *or* a single text column (as
`ColumnTransformer` hands over). Output columns per `score_mode`:

| `score_mode` | columns | meaning |
|---|---|---|
| `entail_contradict` (default) | `2·len(hypotheses)` | `[P(entail) ‖ P(contradict)]` |
| `entail` | `len(hypotheses)` | `P(entail)` |
| `contrast` | `len(hypotheses)` | `P(entail) − P(contradict)` |

`get_feature_names_out()` returns the hypotheses themselves, so feature importances stay readable.

**`fit(X, y)`** — fixes the hypothesis set. If you passed `hypotheses`, they're used as-is (pure
transformer, no LM). If not, the pool is **generated from `(X, y)`** via the proposer — which requires
the `train` extras and `task` + `class_definitions` set (clear error otherwise). So an inference-only
install can score with a supplied/loaded pool but cannot generate one.

**Construct / persist** — `from_run(dir)` (a trained run's `config.yaml` encoder + `model.json` pool),
`from_config(dict_or_yaml)`, and `save(path)` / `load(path)` (JSON: hypotheses + encoder config, no
weights). A fitted vectorizer pickles cleanly — the live encoder/cache handle is dropped and rebuilt
on demand.

**Compose** — it's a transformer, so the usual sklearn machinery applies:

```python
# score one text column alongside other tabular features
ColumnTransformer([("hyp", HypothesisVectorizer(hyps), "text"),
                   ("num", StandardScaler(), ["price", "age"])])

# optional TF-IDF channel — plain sklearn, not baked in
FeatureUnion([("nli", HypothesisVectorizer(hyps)),
              ("tfidf", make_pipeline(TfidfVectorizer(), TruncatedSVD(128)))])
```

`cache_path` points at a sqlite score cache (raw logits keyed by text+hypothesis+model); a shared path
makes repeat scoring across runs ~free. `None` uses an in-process cache for the instance's lifetime.

## Training: producing a pool

Needs the `train` extras. Either drive it from a YAML config with the CLI, or let the vectorizer's
`fit` generate a static pool.

```bash
uv run nli-boost run configs/trec.yaml     # generate -> evolve -> CV head -> one test eval
uv run nli-boost report                    # pool_cv results across runs/
uv run nli-boost compare runs/a runs/b     # paired McNemar: is a delta real or noise?
```

The full pipeline (`nli-boost run`) is:

1. **Generate** — the LLM proposes hypotheses from the task + class definitions + sampled examples.
2. **Dedup (covariance)** — reject a candidate whose entail-score vector correlates > `dedup_corr`
   with a kept one (removes *behavioral* duplicates, not just paraphrases).
3. **Evolve** — rank hypotheses by CV-fold permutation importance + cross-fold stability, prune
   confident deaths, refill against confusion hot-spots, repeat to a held-out plateau. With a TF-IDF
   channel configured, ranking is *marginal over TF-IDF*, so hypotheses whose signal the (free)
   lexical channel already carries are dropped — shrinking the per-prediction NLI pool.
4. **Head** — a CV-selected classical head over the entail+contradict (+ optional TF-IDF) features;
   one held-out test evaluation (`pool_cv`) is the only reported number.

**Train sklearn-native.** With no `hypotheses`, `fit(X, y)` generates a static pool (steps 1–2) from
the data — a standard sklearn `fit`, just needing the `train` extras and the task metadata:

```python
from nli_boost import HypothesisVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

vec = HypothesisVectorizer(
    task="Classify the type of answer a question is asking for.",
    class_definitions=["ABBR: abbreviation", "ENTY: entity", "DESC: description",
                       "HUM: person", "LOC: location", "NUM: number"],
    class_names=["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"],
    n_hypotheses=64,
    encoder="dleemiller/finecat-nli-l",
    evolve=True,   # optional: also run the CV-prune/refill loop (stronger pool, more LM calls)
)
clf = Pipeline([("hyp", vec), ("clf", HistGradientBoostingClassifier())])
clf.fit(train_texts, y_train)   # vec.fit generates (+ evolves) the pool via the LM, then the head fits
clf.score(test_texts, y_test)

vec.save("my_pool.json")        # persist hypotheses+encoder for later dspy-free inference
#   later: HypothesisVectorizer.load("my_pool.json")
```

`evolve=False` (default) stops at a static generated pool (fast, one LM pass): ~0.956 on TREC through
the sklearn workflow. `evolve=True` runs the full CV-prune/refill loop inside `fit` for the strongest
pool (~0.964), at the cost of more LM calls and encoder passes. The CLI `nli-boost run` does the same
training end-to-end from a YAML config; either way, serve the result with
`HypothesisVectorizer.from_run(...)` / `load(...)`.

Artifacts per run in `runs/<run_name>/`: `model.json` (the pool — the model is a list of English
sentences — plus head params), `log.jsonl` (evolution audit trail: every prune with its reason),
`metrics.json` (the single honest headline), `costs.json` (LM spend, encoder pairs, wall time).

## What's measured

Full audit and per-decision measurements are in `NOTES.md`. Headlines on TREC-6 (2k train, seed 7,
honest `pool_cv` protocol):

- **The encoder is the one reliable accuracy lever** — `-m → -l` ≈ **+5 pts** (p=0.024). Instruction
  wording, proposer model, pool size, and tree-structured prompting all wash out within noise at `-l`.
- **Best single pool:** **0.934** at `-m`, **0.954** at `-l`. Adding the TF-IDF channel reaches
  **0.964** (best point estimate; not significantly above the 0.956 plain-TF-IDF run, p=0.42).
- **Averaging independent pools (a committee) reaches 0.964 robustly** — beating any single pool
  without having to pick the lucky one.
- **vs baselines** (ag_news / sst2): the method **beats TF-IDF decisively** (+4 to +24 pts) but only
  **ties zero-shot NLI** — its edge is interpretable features that beat bag-of-words, concentrated on
  multi-class carving (TREC, ag_news); on binary sentiment a single zero-shot hypothesis suffices.

The expected frontier is the **low-N** regime (2–5 examples/class), where transfer knowledge should
matter most and a different pipeline applies — see [docs/low-n-plan.md](docs/low-n-plan.md).

## Development

```bash
uv run pytest          # full pipeline + vectorizer under fakes — no GPU or LM key needed
uv run ruff check .    # also enforced via pre-commit
```

The pre-rewrite exploratory code (trees, boosting, and the experiments that selected this method) is
archived untracked in `src-bak/`; the running experiment log lives in `NOTES.md`.
