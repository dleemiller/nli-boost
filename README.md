# nli-boost

Interpretable text classification from **LM-written NLI hypotheses**. An LLM writes ~64 short English
sentences ("hypotheses"); a frozen NLI cross-encoder
([finecat](https://huggingface.co/dleemiller/finecat-nli-l)) scores, for each input text, how strongly
it **entails** and **contradicts** each hypothesis; those scores are the features for a
CV-disciplined classical head (RandomForest / HistGradientBoosting). Nothing is fine-tuned — task
adaptation lives entirely in the sentences, so the model *is* a readable list of hypotheses.

Two clean halves:

- **Inference** is just NLI scoring against a fixed hypothesis list → a scikit-learn transformer,
  [`HypothesisVectorizer`](#hypothesisvectorizer--api-reference). No LM, no `dspy`.
- **Training** generates + evolves the hypothesis list from labeled data (needs an LLM). Kept in a
  separate `train` extra so inference installs stay light.

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

Developing from source (`uv sync` installs the dev group, which includes the `train` extra):

```bash
uv sync
echo 'OPENROUTER_API_KEY=sk-or-...' > .env   # the hypothesis-proposer LM (training only)
uv run pre-commit install
```

## `HypothesisVectorizer` — API reference

A scikit-learn transformer that turns a column of text into NLI-entailment features against a fixed
hypothesis set. Its "model" is the hypothesis list + encoder name, so inference needs **no LM and no
dspy** — only the encoder. With no `hypotheses`, `fit(X, y)` *generates* the set from your data
(`train` extra).

```python
HypothesisVectorizer(
    hypotheses=None, *, encoder="dleemiller/finecat-nli-l", score_mode="entail_contradict",
    device="cuda", batch_size=128, max_text_chars=1200, cache_path=None, verbose=False,
    task=None, class_definitions=None, class_names=None, n_hypotheses=64,
    lm="openrouter/deepseek/deepseek-v4-flash", dedup="covariance", dedup_threshold=0.95,
    evolve=False, random_state=0,
)
```

### Parameters

| parameter | type / default | description |
|---|---|---|
| `hypotheses` | list[str], default None | The feature vocabulary. If given, `fit` uses it as-is (no LM anywhere). If None, `fit(X, y)` generates it (needs the `train` extra + `task`/`class_definitions`). |
| `fixed_hypotheses` | list[str], default None | **Your hand-written hypotheses, always kept.** Scored and fit alongside the rest, but never pruned — during generation/evolution they act as a fixed baseline (like the TF-IDF block), so generated hypotheses must add marginal value over them. Prepended to `hypotheses_`. |
| `encoder` | str, `"dleemiller/finecat-nli-l"` | HF cross-encoder id (labels: entail=0, neutral=1, contradict=2). The method's capacity knob (`-m`→`-l` ≈ +5 pts). |
| `score_mode` | `{"entail_contradict", "entail", "contrast"}` | Columns per hypothesis: both probabilities (2), P(entail) only (1), or P(entail)−P(contradict) (1). |
| `device` | str, `"cuda"` | Encoder (and sts-dedup) device. |
| `batch_size` | int, 128 | Encoder inference batch size. |
| `max_text_chars` | int, 1200 | Texts are whitespace-normalized and truncated to this before scoring/caching (stable cache keys). |
| `cache_path` | str \| Path \| None | sqlite score cache. A path persists raw logits across processes (repeat scoring ~free); None = in-process only. |
| `verbose` | bool, False | Progress lines during long encoder passes. |
| `task` | str, None | *(generation)* One-line task description shown to the proposer LM. |
| `class_definitions` | list[str], None | *(generation)* `"NAME: one-line definition"` per class. |
| `class_names` | list[str], None | *(generation)* Class display names; default `class 0..K`. |
| `n_hypotheses` | int, 64 | *(generation)* Pool size to generate (and evolution's target size). |
| `lm` | str | *(generation)* litellm model id for the proposer. |
| `dedup` | `{"covariance", "sts"}` or object | *(generation)* Candidate dedup. `"covariance"`: reject candidates whose entail-score vectors are ~collinear with a kept one — behaviorally exact, needs enough data. `"sts"`: bi-encoder cosine on the hypothesis *texts* — data-free, the right choice at ~3–5 examples/class. Or any object with `.filter(candidates, against, seen)`. |
| `dedup_threshold` | float, 0.95 | Rejection threshold (\|Pearson\| for covariance; cosine for sts, ~0.9 sensible). |
| `evolve` | bool, False | *(generation)* Also run the CV-prune/refill loop inside `fit` — strongest pool, more LM calls. Avoid at very low N (CV over a handful of examples is noise). |
| `random_state` | int, 0 | Seeds example sampling, dedup sampling, and evolution. |

### Attributes

| attribute | description |
|---|---|
| `hypotheses_` | The fitted hypothesis list (feature vocabulary). |
| `evolution_history_` | After `fit` with `evolve=True`: one dict per round with the exact `pool` scored and its `heldout_acc` — **every round is saved and recoverable**, not just the last. |

### Methods

| method | description |
|---|---|
| `fit(X=None, y=None, baseline_features=None)` | Fix (or generate) the hypothesis set. `baseline_features` (n, d): any extra features the downstream head will also see — other tabular columns, TF-IDF, embeddings; with `evolve=True`, hypotheses are pruned by **marginal value over that fixed block**. In a Pipeline: `pipe.fit(X, y, hyp__baseline_features=Z)`. |
| `transform(X)` | 1-D sequence of strings, or a single text column (as `ColumnTransformer` hands over) → `(n, k)` float array; `k` per `score_mode` (see table above ×`len(hypotheses_)`). |
| `fit_transform(X, y=None, **fit_params)` | Standard `TransformerMixin`. |
| `get_feature_names_out()` | The hypotheses themselves (`"entail: The text asks for a number."`) — importances stay readable. |
| `save(path)` / `load(path)` | Persist / restore the fitted inference artifact (hypotheses + encoder config, JSON — no weights). |
| `from_run(run_dir)` | Load a trained CLI run (`config.yaml` encoder + `model.json` pool) into a fitted vectorizer sharing the run's score cache. |
| `from_config(dict_or_yaml)` | Build from constructor params as a dict/YAML; nested `encoder: {model, device, ...}` accepted; unknown keys ignored; fitted if `hypotheses` present. |
| `get_params` / `set_params` / `set_output(transform="pandas")` | Standard sklearn surface; string-input estimator tags declared (like `TfidfVectorizer`); pickles/clones cleanly; `check_estimator` passes its applicable checks. |

### Use cases

**Text column + tabular features.** Compose with `ColumnTransformer`; during training, pass the
tabular block as the evolution baseline so hypotheses that just re-encode it are pruned:

```python
Z = df[["price", "age"]].to_numpy()                       # features the head will also see
vec = HypothesisVectorizer(task=..., class_definitions=..., evolve=True)
vec.fit(df["text"], y, baseline_features=Z)               # prune by MARGINAL value over Z
ct = ColumnTransformer([("hyp", vec, "text"), ("num", StandardScaler(), ["price", "age"])])
```

**TF-IDF channel.** Same mechanism — fit your TF-IDF pipeline, pass its output as
`baseline_features`, then serve both via `FeatureUnion` (this is how the 0.964 TREC recipe works):

```python
tfidf = make_pipeline(TfidfVectorizer(), TruncatedSVD(128)).fit(texts)
vec = HypothesisVectorizer(task=..., class_definitions=..., evolve=True)
vec.fit(texts, y, baseline_features=tfidf.transform(texts))
features = FeatureUnion([("nli", vec), ("tfidf", tfidf)])
```

**Low data (~3–5 examples/class).** Covariance dedup and CV evolution both need data they don't
have; use text-space dedup and skip evolution — the pool is then a pure LM prior:

```python
vec = HypothesisVectorizer(task=..., class_definitions=...,
                           dedup="sts", dedup_threshold=0.9, evolve=False)
```

(See [docs/low-n-plan.md](docs/low-n-plan.md) for the fuller low-N methodology.)

**Your own hypotheses, protected.** Domain knowledge you already trust goes in `fixed_hypotheses` —
always in the model, never pruned; the LM fills in *around* them (told to avoid duplicating them,
and evolution only keeps generated hypotheses that add marginal value over them):

```python
vec = HypothesisVectorizer(
    fixed_hypotheses=["The text asks what an abbreviation stands for.",
                      "The text can be answered with a number."],
    task=..., class_definitions=..., evolve=True,
)
```

## Training: producing a pool

Needs the `train` extra (`pip install "nli-boost[train]"`). Either drive it from a YAML config with the CLI, or let the vectorizer's
`fit` generate a static pool.

```bash
uv run nli-boost run configs/trec.yaml     # generate -> evolve -> CV head -> one test eval
uv run nli-boost report                    # pool_cv results across runs/
uv run nli-boost compare runs/a runs/b     # paired McNemar: is a delta real or noise?
```

[`configs/example.yaml`](configs/example.yaml) documents every config knob (encoder / proposer-LM /
sts-dedup models, dedup backend + threshold, pool size/rounds, `fixed_hypotheses`, lexical channel);
the other configs are the maintained recipes (`trec`, `trec_best_l`, `trec_best_l_max` = the 0.964
recipe, `trec_finalize_l` = re-score a fitted pool with a bigger encoder, `ag_news`, `sst2`).

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
the data — a standard sklearn `fit`, just needing the `train` extra and the task metadata:

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
