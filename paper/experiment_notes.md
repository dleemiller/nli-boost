# Paper experiment notes

Running log for the paper-production effort. Started 2026-07-06. This file records the
repository audit, the experiment plan, and per-phase results as they land. It is the
paper-oriented companion to the engineering log in `NOTES.md`.

---

## Phase 0 — Repository audit (2026-07-06)

### Environment as checked out

- **Machine:** RTX 5090 (32 GB), CUDA available and **idle** — the GPU that was blocked by an
  external training job in the last `NOTES.md` entry is now free.
- **Toolchain:** Python 3.13.9, `uv` 0.9.5. `uv sync` succeeds; `torch==2.12.1`,
  `transformers==5.13.0`, `scikit-learn==1.9.0`, `sentence-transformers==5.6.0`, `dspy==3.2.1`.
- **Tests:** `uv run pytest` → **40 passed, 1 skipped** (the skip is `check_estimator`'s
  string-input case, expected). The suite runs entirely under fakes — no GPU or LM key needed.
- **Prerequisites verified working (no API key needed):**
  - HF dataset loading via `data.load` (TREC downloads and splits correctly).
  - The finecat NLI cross-encoder (`dleemiller/finecat-nli-m`) loads on CUDA and scores pairs
    through `EntailmentScorer.probs/.features` (label order entail=0, neutral=1, contradict=2).

### What is ABSENT from this checkout (reproducibility gaps)

The following are gitignored (`runs/`, `cache/`, `src-bak/`, `models/`, `.env`) and therefore **not
present** — this is effectively a fresh checkout, and all prior experimental artifacts must be
regenerated:

- **No `runs/` directory.** Every result cited in `README.md`/`METHOD.md`/`NOTES.md`
  (`runs/trec_best_l`, the 0.964 recipe, etc.) is gone. None of the headline numbers can be
  reproduced without re-running.
- **No NLI score cache.** Every experiment starts cold; first scoring pass per (encoder, corpus) is
  full price, cached thereafter.
- **No `.env` / `OPENROUTER_API_KEY`.** **This is the one hard blocker.** Hypothesis *generation*
  (HV-static, HV-evolved, evolution refill, GEPA) is impossible without an LM. Everything else —
  NLI scoring against a *given* pool, all baselines, zero-shot NLI, hand-written fixed-expert pools —
  runs without it.

### Current capabilities (what the code does well)

- **Method core is solid and tested.** Frozen NLI cross-encoder → P(entail)/P(contradict) features →
  CV-selected classical head (`EntailmentScorer`, `head.cv_selected_head`). Sklearn-native
  `HypothesisVectorizer` transformer; `check_estimator` passes its applicable checks.
- **Deterministic, leakage-safe splits.** `data.load` draws the **test split from its own fresh RNG
  keyed on `seed` only** (`data.py:188`), so test is held fixed independent of train size — exactly
  the property the low-N learning curve needs. K-shot sampling (`per_class_indices`) picks exactly
  k/class.
- **Honest evaluation protocol.** Head family + regularization chosen by 4-fold CV on train only,
  then one test evaluation (`pool_cv`). This is the discipline the paper should foreground.
- **Significance machinery.** `compare` does paired McNemar + Wilson CIs on a shared test set and
  refuses mismatched test sets. This is paper-grade and rare in prompt-classification work.
- **Evolution audit trail.** `log.jsonl` records every round's pool, held-out accuracy, each pruned
  hypothesis with a human-readable failure reason, and refill target-AUCs. Strong raw material for
  the interpretability section.
- **Cost accounting.** `costs.json` tracks LM calls/tokens/USD, encoder pairs requested vs GPU-run
  vs cache hits, wall time — directly feeds the cost/latency table.
- **Built-in datasets:** TREC-6, AG News, SST-2, 20 Newsgroups (`_SPECS` in `data.py`), each with a
  task string and one-line class descriptions.

### Gaps vs. a publishable paper (prioritized)

**Blocking the central thesis:**
1. **No LM access (API key).** Blocks all LLM-generated pools. *Owner: user.*

**Core research gaps (highest paper value):**
2. **No low-N prior-aggregation / low-capacity head.** The only heads are a 300-tree RF and a
   200-iter HGB grid (`head._GRID`) — run on ~30 rows at 5-shot, which `README.md` already flags as
   overfitting (cv-train 0.80 ≫ test 0.67). **This is RQ1's central research task**, not just
   plumbing: implement a class-tagged prior-aggregation head (N=0 → zero-shot NLI ensemble) and a
   strongly-regularized linear head, and a shrinkage head interpolating between them
   (`docs/low-n-plan.md`).
3. **No learning-curve harness.** A run is single-seed / single-size. Need a sweep over
   {1,2,3,5,10,20,50,100,all} examples/class × ≥5 seeds with **mean ± bootstrap/Wilson CIs**, fixed
   test set, features cached once and reused across all points (cheap after the first scoring pass).

**Missing baselines (reviewers will demand these):**
4. **No runnable/reported baselines.** TF-IDF+logreg (word & char), sentence-embeddings+logreg,
   and zero-shot NLI (class-template argmax) exist only as numbers quoted in docstrings. All three
   are LM-free and buildable now. TF-IDF exists only as a *concatenated channel*, never standalone.

**Missing experiment infrastructure:**
5. **No ablation runner** (pool size, NLI encoder, evolve on/off, score_mode, dedup, head).
6. **No config-driven text+tabular runner.** Capability exists only in `examples/cfpb.py` via the
   sklearn `baseline_features` API; not wired into configs/CLI.
7. **No results summarizer** beyond `report`'s flat `pool_cv` table — no seed aggregation, no
   per-dataset roll-up, no LaTeX/markdown export.
8. **No figure generation** (no matplotlib anywhere in `src/`).
9. **No per-class metrics.** `evaluate` emits only accuracy/macro-F1/logloss — no per-class P/R/F1,
   confusion matrix, or calibration (ECE).
10. **Reproducibility artifacts:** no `manifest.json` (git hash, lib versions, dataset revision), no
    saved split indices, no saved predictions/feature-importances. Splits are only *re-derivable*
    from seed — fine until the HF data or sampling code drifts.

**Missing datasets:** Banking77, CLINC150, GoEmotions have no loader; CFPB exists only as the
`examples/cfpb.py` script (not integrated into `data.load`/CLI).

### Opportunities to simplify / things to preserve

- **Preserve:** the honest `pool_cv` protocol, McNemar `compare`, deterministic splits, cost
  accounting, evolution audit trail. These are the project's methodological strengths and should be
  foregrounded in the paper's Experiments section.
- **The existing `docs/low-n-plan.md` is excellent** and already frames RQ1 correctly (the crossover
  study: "the optimal configuration is a function of N"). The paper's low-N section should be built
  directly on it. Do not re-derive.
- **Version inconsistency to fix:** `__version__ = "0.3.0"` (`__init__.py`) vs packaging bumped to
  0.4.0; encoder default differs between surfaces (`EncoderConfig` → `-m`, `HypothesisVectorizer` →
  `-l`). Minor, worth aligning before release.

### Where paper artifacts live (established this session)

```
paper/            draft.md, related_work.md, method.md, experiments.md, limitations.md,
                  research_report.md, experiment_notes.md (this file), figures/, tables/
experiments/      configs/{datasets,models,runs}/  scripts/  results/{raw,processed,tables,figures}/
                  notebooks/
```

---

## Phase 1 — Reproduce existing behavior + build infrastructure (2026-07-06)

### Environment fixes

- **Real bug fixed: SQLite cache fails on ZFS.** `ScoreCache` hard-coded `PRAGMA
  journal_mode=WAL`; this machine's filesystem is **ZFS**, which accepts the WAL pragma but then
  throws `disk I/O error` on the first write — *and that poisons the connection and the
  half-created DB file*. Any real run on this machine would have crashed on cache creation. Fixed
  `src/hypothesis_vectorizer/cache.py` to probe WAL support on a throwaway file, then open the
  real cache directly in WAL (if supported) or DELETE (durable fallback) / MEMORY. `journal_mode`
  is now an attribute. Full test suite still green (40 passed / 1 skipped).

### Infrastructure built (LM-free, all runnable now)

Package `experiments/hvexp/` + scripts `experiments/scripts/`:

- `hvexp/repro.py` — `Manifest` (git commit, lib versions, dataset, seed, config hash),
  `save_split` (persist exact train/test indices). **Closes the manifest/provenance gap.**
- `hvexp/metrics.py` — accuracy, macro/weighted-F1, per-class P/R/F1, top-label **ECE
  calibration**, bootstrap CIs. **Closes the per-class + calibration gap.**
- `hvexp/features.py` — `NLIFeaturizer` reusing the library's cached `EntailmentScorer`; scores
  a corpus once, every subsample is a cache hit.
- `hvexp/hypotheses.py` — hand-written **class-tagged expert pools** + zero-shot class templates
  for TREC / AG News / SST-2. Powers HV-fixed-expert, the prior head, and zero-shot NLI, all
  LM-free.
- `hvexp/datasets.py` — learning-curve protocol: full train pool + **one fixed test set**,
  resample k/class across seeds. Reuses the library's samplers so splits match the CLI.
- `hvexp/systems.py` — uniform pluggable systems: TF-IDF(word/char/union)+logreg,
  embeddings+logreg, zero-shot NLI, `HVHead` (auto RF/HGB grid | L2-logreg-CV), and
  **`PriorAggregation`** — the low-N prior head (class-tagged mean entailment; `fixed` = N=0
  zero-shot ensemble, `reweight` = strong-L2). **Closes the baselines + low-N-head gaps.**
- `scripts/run_learning_curve.py` — the harness (k × seeds × systems, one GPU pass, JSONL +
  manifest). Ready for LLM pools via `--generated-pool` (adds `hv_generated_*` systems).
- `scripts/summarize_results.py` — seed aggregation → mean+bootstrap-CI → markdown + **LaTeX** +
  CSV tables.
- `scripts/make_figures.py` — learning-curve figures with CI bands, **PDF + PNG**.
- `scripts/inspect_hypotheses.py` — RQ2 interpretability: global permutation importance,
  per-class hypotheses, redundancy clusters, cross-fold stability, per-hypothesis exemplars,
  error cases with top-activating hypotheses → markdown report + CSV.
- `scripts/run_ablation.py`, `scripts/run_text_tabular.py` — RQ4 / RQ5 runners (scaffolded).

### Smoke validation (—m encoder, tiny sweep)

Harness → summarizer → figures verified end-to-end. Even at smoke scale the low-N thesis shows:
at 2/class, `hv_prior_fixed` (label-free) = 0.655 beat TF-IDF (~0.29–0.55), embeddings (~0.43),
zero-shot NLI (0.39) **and** the RF/HGB `auto` head (~0.50–0.57) — the RF/HGB overfitting at low
N is real and the prior head is the fix, exactly as `docs/low-n-plan.md` predicted.

### RESULT: full TREC `-l` baseline learning curve (10 seeds, fixed 500-test)

`experiments/configs/runs/trec_lown_baselines.yaml`; `lc_trec_baselines_l`. Test **accuracy**,
mean over 10 seeds. Bold = best per column.

| system | 1 | 2 | 3 | 5 | 10 | 20 | 50 | 100 | all |
|---|---|---|---|---|---|---|---|---|---|
| **hv_prior_fixed** (label-free) | **0.594** | 0.594 | 0.594 | 0.594 | 0.594 | 0.594 | 0.594 | 0.594 | 0.594 |
| **hv_prior_reweight** | 0.556 | **0.629** | **0.642** | 0.660 | 0.680 | 0.679 | 0.686 | 0.700 | 0.862 |
| hv_expert_rf | 0.474 | 0.597 | 0.642 | **0.682** | **0.753** | **0.802** | **0.849** | **0.892** | **0.954** |
| hv_expert_logreg | 0.364 | 0.517 | 0.547 | 0.604 | 0.712 | 0.725 | 0.746 | 0.784 | 0.912 |
| zeroshot_nli | 0.428 | 0.428 | 0.428 | 0.428 | 0.428 | 0.428 | 0.428 | 0.428 | 0.428 |
| emb MiniLM+logreg | 0.301 | 0.400 | 0.431 | 0.486 | 0.574 | 0.649 | 0.703 | 0.742 | 0.856 |
| tfidf word+char+logreg | 0.307 | 0.374 | 0.402 | 0.426 | 0.486 | 0.572 | 0.634 | 0.713 | 0.870 |
| tfidf word+logreg | 0.315 | 0.362 | 0.374 | 0.414 | 0.475 | 0.542 | 0.560 | 0.623 | 0.852 |

Figures: `experiments/results/figures/trec_learning_curve_{accuracy,macro_f1}.{pdf,png}`.
Tables (md/LaTeX/CSV): `experiments/results/tables/trec_lown_*`.

**Findings (all with the hand-written expert pool — NO LLM yet):**

1. **The crossover is textbook and validates the whole low-N thesis.** At 1/class the *label-free*
   prior-aggregation head (0.594) is best; at 2–3/class the reweighted prior (0.629/0.642) leads;
   from ~5/class the flexible RF head takes over and climbs to **0.954** at full data. *The optimal
   configuration is a function of N* — exactly the claim of `docs/low-n-plan.md`, now measured.
2. **HV dominates both baselines at every N on TREC.** TF-IDF and MiniLM embeddings trail HV
   throughout and only reach ~0.85–0.87 at full data vs HV's 0.954. (TREC is semantically clean and
   NLI-favorable; do not over-generalize — the paper frames this as a favorable Pareto point, and
   AG News / 20NG are expected to be harder for HV, per prior NOTES.)
3. **Multi-hypothesis prior ≫ single-template zero-shot with zero labels:** 0.594 vs 0.428 (+16.6
   pts). A standalone finding: an ensemble of readable hypotheses beats one class template per class.
4. **The flexible-head low-N failure is real but graceful with RF.** A single HGB head *collapsed to
   a constant class* (0.018) below 10/class — a degenerate artifact, not the documented failure —
   so the sweep uses a RandomForest, which overfits gracefully (0.474 at 1/class, still > chance)
   and reproduces the README's ~0.95 at full data. Recorded so we don't mis-report the HGB collapse.
5. **Interpretability (RQ2) works out of the box** (`inspect_hypotheses.py`,
   `results/processed/trec_expert_inspect_l/`): top hypotheses by permutation importance read as
   plain English with class tags (LOC "asks where something is located" = 0.094; NUM "answered with
   a numeric value" = 0.073), cross-fold-stable, and the redundancy detector flags the one near-dup
   pair (person/identity, corr 0.92).

**Compute note:** the RF/HGB CV-grid head (`cv_selected_head`, 52 fits/point) is too slow for a
90-point × 10-seed sweep; the harness uses a single fast RF/HGB (`head="rf"`) for the curve and
keeps the exact library grid available as `head="auto_full"` for headline single points.

---

## Phase 2 — LLM hypothesis generation (DONE for TREC; 2026-07-06)

Key auth-validated; pools generated with DeepSeek-v4-flash via `generate_pool.py` (strict: train
sample only; test never seen). Two pools:
- **static** (`trec_gen_static.json`): n=64 from 5/class, no evolution, STS dedup, $≈0.
- **evolved** (`trec_gen_evolved.json`): n=64→**62** from 50/class, CV-prune/refill evolution —
  which **stopped after 2 rounds on a held-out plateau**, independently reproducing the METHOD.md
  "generation saturates ~2 rounds" finding.

The proposer returns *untagged* statements (and its tree hypotheses are deliberately multi-class),
so `intended_class` for the prior head is **derived** from the train sample (arg-max mean
entailment per class) — noisier than hand tags, recorded as a caveat.

### RESULT: generated vs expert pools (TREC-6, `-l`, 10 seeds, shots 1–50)

`lc_trec_generated_l`; figure `trec_generated_accuracy.pdf`; table `trec_generated_accuracy.md`.
Test accuracy, mean over seeds (best learned system per column in **bold**):

| system | 1 | 2 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|---|
| hv_evolved_rf | 0.449 | 0.624 | **0.703** | **0.784** | **0.834** | **0.873** | **0.894** |
| hv_static_rf | 0.482 | 0.628 | 0.692 | 0.726 | 0.788 | 0.856 | 0.886 |
| hv_expert_rf | 0.474 | 0.597 | 0.642 | 0.682 | 0.753 | 0.802 | 0.849 |
| hv_evolved_logreg | 0.384 | 0.589 | 0.664 | 0.740 | 0.810 | 0.851 | 0.893 |
| hv_static_logreg | 0.350 | 0.486 | 0.576 | 0.628 | 0.773 | 0.837 | 0.890 |
| hv_prior_fixed (expert, 0 labels) | **0.594** | 0.594 | 0.594 | 0.594 | 0.594 | 0.594 | 0.594 |
| hv_evolved_prior_fixed | 0.544 | 0.544 | 0.544 | 0.544 | 0.544 | 0.544 | 0.544 |
| hv_static_prior_fixed | 0.484 | 0.484 | 0.484 | 0.484 | 0.484 | 0.484 | 0.484 |

**Findings (RQ4):**
1. **LLM-generated pools BEAT the hand-written expert pool as a feature basis for learned heads.**
   Evolved+RF leads everywhere from 3/class up (0.703→0.894 vs expert 0.642→0.849); the richer
   64-hyp generated basis gives RF/logreg more to work with than the 24-hyp expert pool. The core
   thesis — *LLM-generated hypotheses are a strong semantic basis* — holds, not just anecdotally.
2. **Evolution gives a small, consistent lift over static** (evolved ≥ static on RF and logreg at
   nearly every N; clearest for logreg at low N: 0.664 vs 0.576 at 3/class), and it stopped at 2
   rounds — consistent with prior saturation findings. Evolution is a *refinement*, not the lever.
3. **The zero-label prior head still favors the hand-written expert pool** (0.594 > evolved 0.544 >
   static 0.484): clean class tags matter when there are no labels to reweight them; evolution
   improves the generated pool's tag-ability (evolved > static). Motivates either expert tags or a
   better tag-derivation step for the N=0 setting.
4. At **1/class the label-free expert prior (0.594) is still the single best system** — no learned
   head beats it until ≥2 labels exist.

Net: the generated-pool result strengthens the paper — the method works from *generated* hypotheses
(the actual claim), generated ≳ expert for learned heads, evolution is a minor refinement, and the
expert pool's edge is confined to the zero-label prior via clean tags.

## Phase 3 — Additional datasets (STARTED)

- **Wired Banking77 (77 classes) + CLINC150 (151, incl. oos)** into `data.py`/`config.py` and added
  intent-family expert pools + programmatic zero-shot templates to `hvexp/hypotheses.py`. Verified
  loadable; tests green (40/1 skip). Caveat: the config-less CLINC parquet-convert triplicates its
  test split — fine for the stratified sampler but note before citing CLINC test numbers.
- AG News / SST-2 already had loaders + expert pools.

### RESULT: generality — AG News + Banking77 (2026-07-06, `-l`, 10 seeds)

Runs `lc_agnews_baselines_l` (shots 1–100) and `lc_banking77_baselines_l` (shots 1–20). Three
datasets now show **three distinct patterns** — the honest generality story:

| task type | dataset | low-N winner | HV verdict |
|---|---|---|---|
| small clean taxonomy | TREC-6 | HV (prior→RF) | **HV dominates at all N** |
| broad topics | AG News | zero-shot NLI (.892) / HV prior (.858) | NLI prior near-ceiling; HV learned heads add little |
| fine-grained 77-way | Banking77 | MiniLM emb (.535→.886) | **HV loses**; 24-hyp pool too thin for 77 intents |

- **AG News:** zero-shot NLI is best at *every* budget (0.892) — broad topics are near-trivial for
  the NLI prior; TF-IDF weak-but-climbing (0.32→0.81 by 100/class). HV's value is the label-free
  prior, not the learned head, here.
- **Banking77:** dense embeddings win everywhere; HV's hand-written 24-hypothesis pool structurally
  can't span 77 intents (prior head floors at 0.231, covering only its ~24 tagged classes). **Key
  implication: pool size must scale with the label space** — motivates a large *generated* pool +
  the pool-size ablation for many-class tasks.

**Efficiency fix landed:** `NLIFeaturizer.probs` now memoizes full prob tensors by (texts, pool)
content, so the fixed test matrix is scored once per sweep instead of re-read from SQLite on every
(k, seed). Cut the Banking77 sweep from a ~55-min projection to a few minutes (warm stays cached).

### RESULT: Banking77 pool-scaling test (2026-07-07) — hypothesis CONFIRMED

Generated a 256-hyp Banking77 pool (`banking77_gen256.json`, DeepSeek-v4-flash, 3/class; covers
**76/77 classes**) and re-ran the curve (`lc_banking77_scaling_l`, 5 seeds, shots 1–20). Scaling
24→256 hypotheses **doubled** low-N accuracy (0.31→0.53 at 1/class) and **closed almost the entire
gap to dense embeddings** — `hv_gen256_logreg` tracks MiniLM within ~0.004–0.03 at every budget
(0.530 vs 0.534 @1; 0.859 vs 0.886 @20) and beats TF-IDF throughout. Table `banking77_scaling_*`,
figure `banking77_scaling_accuracy.pdf`. **Banking77's earlier loss was a thin-pool artifact, not a
method limit — the pool must scale with the label space.** The prior head improved (0.231→0.392)
but the learned head is where the large pool pays off (256 auto-tagged hyps → 77 classes is noisy
for pure averaging).

**Infra fix — 9P filesystem / cache location.** The workspace is a 9P-mounted ZFS share; SQLite
per-lookup RPCs stalled the all-miss gen256 warm (GPU sat at 0% in `D`/`p9_client_rpc`). Made the
NLI cache path env-configurable (`HV_CACHE_DIR`, `experiments/hvexp/features.py`) and pointed runs
at local ext `/tmp/hv_cache` — GPU went to 99% immediately (1.88M pairs scored ~1090/s). **Run
future scoring passes with `HV_CACHE_DIR=/tmp/hv_cache`.** The canonical cache remains committed
under `cache/` (gitignored) — copy it to the local dir before a big run.

## Phase 5 — CFPB text+tabular (RQ5) — DONE (2026-07-07)

Monetary-relief prediction; balanced 4,000-row sample, random 80/20 split, HGB head; 64-hyp pool
generated from train narratives only (`experiments/scripts/prep_cfpb.py` → `run_text_tabular.py`,
run `tt_cfpb_random`). ROC-AUC by feature config:

| config | AUC | marginal |
|---|---|---|
| tabular_only | 0.914 | — |
| tfidf_only | 0.895 | — |
| hv_only | 0.856 | — |
| tabular+tfidf | 0.938 | +0.024 vs tabular |
| tabular+hv | 0.935 | **+0.021 vs tabular** |
| tabular+tfidf+hv | **0.945** | **+0.007 over tabular+tfidf** |

**HV features add interpretable marginal value over structured metadata** — +0.021 AUC over tabular
(≈ TF-IDF's own +0.024), and +0.007 even on top of tabular+TF-IDF (best config 0.945). HV alone is
the weakest single channel (0.856), but it is complementary and, unlike TF-IDF/embeddings,
**auditable** — the contributing hypotheses are readable ("mentions a specific dollar amount the
consumer lost or is owed", "demand for a refund or compensation", "emotional distress rather than a
specific financial loss"). This is the applied/regulatory claim, confirmed.

Caveats: balanced/random (AUC ~0.91–0.94) is NOT the temporal natural-rate benchmark (0.78/0.69).
Pool is static; the `--evolve` marginal-over-tabular pruning path is the next refinement.

## Phase 6 — RQ4 pool-size ablation (2026-07-07)

Subsampled generated pools to 8…256 (RF head, 20/class, 5 seeds; `abl_{trec,banking77}_poolsize`,
figure `poolsize_scaling.pdf`). Accuracy vs pool size:

| # hyps | 8 | 16 | 32 | 64 | 128 | 192 | 256 |
|---|---|---|---|---|---|---|---|
| TREC-6 (6 cls) | .622 | .714 | .824 | .829 | .877 | .884 | .877 |
| Banking77 (77 cls) | .543 | .678 | .733 | .780 | .831 | .841 | .845 |

**The useful pool size scales with the label space.** TREC saturates by ~32–64 (≈30 useful
directions, as METHOD.md found); Banking77 keeps climbing to ~128–192 before plateauing near 256.
Confirms pool size is a task-dependent knob and that Banking77's earlier loss was an under-sized
pool, not a method limit. (Used a single RF head, not the CV grid — the grid on 77 classes was
~2 min/fit; RF is ~1s and is the paper's flexible-head line anyway.)

## Phase 7 — fine-tuned encoder baseline (2026-07-07)

Fine-tune DistilBERT per training subsample (`FineTunedEncoder`, systems=finetuned; 5 seeds,
`lc_trec_finetuned`; combined figure `trec_with_finetune_accuracy.pdf`). TREC-6 accuracy:

| shots | 1 | 2 | 3 | 5 | 10 | 20 | 50 | 100 | all |
|---|---|---|---|---|---|---|---|---|---|
| fine-tune | .263 | .311 | .419 | .528 | .739 | .822 | .904 | .921 | **.964** |
| HV expert-RF | .474 | .597 | .642 | .682 | .753 | .802 | .849 | .892 | .954 |

Textbook Pareto boundary: fine-tuning is **catastrophic at low N** (0.263 @1/class, below every
baseline), **crosses HV at ~5–10/class**, and **wins the data-rich regime** (0.964 @all vs HV
0.954). Confirms the paper's positioning — HV owns 1–10/class + interpretability + LLM-free serving;
a fine-tuned encoder owns ≥20/class but is opaque and data-hungry. Recipe: AdamW lr 2e-5, 20 epochs
capped at 1000 steps, max_len 128; ~15–40s/fit.

## Phase 8 — optional: SST-2 + method ablations (2026-07-07)

**SST-2 (binary sentiment, `lc_sst2_baselines_l`, -l, 5 seeds).** The NLI prior near-solves it:
label-free HV prior head = **0.953 at every budget**, > zero-shot NLI 0.947, while TF-IDF/embeddings
sit near chance at low N (~0.49) and reach only ~0.82 at the full 67k train set. HV wins at *every*
budget incl. full data — the most extreme "prior alone wins" case (fourth generality pattern,
reinforcing AG News). Figure `sst2_learning_curve_accuracy.pdf`.

**Method ablations (TREC, 100/class, RF head, 5 seeds; `abl_trec_{scoremode,encoder}`).**
- Encoder is the dominant lever: finecat-nli-m 0.798 → -l **0.892 (+0.094)** on the same pool.
- Score channel within seed noise: entail 0.886 · contrast 0.888 · entail+contradict 0.892
  (default kept). Capacity lives in the encoder; other knobs are second-order — confirms METHOD.md.
- (Switched the ablation's score_mode/encoder axes to a single RF head; the CV grid was needlessly
  slow and the comparison is head-independent.)

## Phase 9 — optional: GoEmotions (Ekman-7 emotion) (2026-07-07)

Wired `goemotions` (single-label Ekman-7 mirror `Jsevisal/go_emotions_ekman_unilabel`, 7 classes:
anger/disgust/fear/joy/neutral/sadness/surprise; 39,575 train). Baseline+expert curve
`lc_goemotions_baselines_l` (-l, 5 seeds). Accuracy:

| shots | 1 | 5 | 20 | 100 | all (39k) |
|---|---|---|---|---|---|
| HV expert-RF | .314 | .406 | .490 | .523 | .636 |
| HV prior-reweight | .343 | .415 | .486 | .514 | .594 |
| MiniLM emb | .180 | .267 | .341 | .368 | .611 |
| TF-IDF (w+c) | .137 | .196 | .268 | .388 | **.674** |

**The cleanest low-N crossover in the study.** Emotion is hard (absolute acc 0.32–0.67, neutral-heavy),
but the shape is textbook: HV roughly doubles the baselines at 1/class and leads through 100/class,
then TF-IDF overtakes at the full 39k train set. First non-fine-tune case where a simple baseline
clearly wins data-rich — "ordinary models catch up as N grows." Fifth generality pattern. Figure
`goemotions_learning_curve_accuracy.pdf`. Tests green after wiring (config Literal + data spec + pool).

## Phase 10 — CFPB temporal natural-rate run (2026-07-07)

Natural rate (3.6% relief), temporal split (oldest 24k train → newest 6k test; drift 3.0%→5.9%),
64-hyp pool from temporal-train narratives only (`tt_cfpb_temporal`). At 3.6% base rate accuracy is
majority-dominated (~0.94); AUC + macro-F1 are the metrics.

| config | AUC | macro-F1 |
|---|---|---|
| tabular | .875 | .497 |
| tfidf | .875 | .548 |
| hv | .867 | .513 |
| tabular+tfidf | .891 | .550 |
| tabular+hv | .891 | .551 |
| tabular+tfidf+hv | **.899** | **.587** |

**Marginal-value story holds under the realistic temporal/natural-rate setting**: HV adds +0.016 AUC
over tabular, and +0.008 AUC / +0.037 macro-F1 on top of tabular+TF-IDF — the macro-F1 gain shows HV
improves rare-class (relief) detection under imbalance + drift. **NOT comparable to the published
0.78/0.69 benchmark** (recent 30k window at 3.6% vs full-history ~8%; plain one-hot vs LDA+engineered
tabular; our tabular baseline alone 0.875 already exceeds their numbers) — we claim only HV's marginal
contribution, not beating the benchmark.

Ops note: the 1.92M-pair -l scoring pass on 512-char narratives is slow (~300 pairs/s); paused
mid-run for a GPU thermal concern and resumed cleanly from the local cache (797k pairs were cache
hits). Added `--limit`/`--split temporal` to prep_cfpb.py. Extended `prep_cfpb.load_frame` for
natural-rate streaming; `run_text_tabular.py` already supported the temporal split.

## Phase 11 — CFPB evolve marginal-pruning refinement (2026-07-07) — NEGATIVE

Regenerated the balanced CFPB pool with evolution pruning by marginal value over a **holistic
tabular+TF-IDF baseline** (388 features), so survivors must beat both metadata AND lexical channels
(`prep_cfpb.py --evolve --baseline tabular_tfidf`; pool `cfpb_pool_random_evolved_tabular_tfidf`,
run `tt_cfpb_evolved`). 6 rounds, internal held-out 0.783→0.798 (the pruning mechanism works — it
found more-complementary hypotheses like "charge after a dispute that was not refunded", "explicitly
requests the company pay a specific amount").

**But it did NOT help on test — an honest negative:**

| config | static AUC | evolved AUC |
|---|---|---|
| hv_only | 0.856 | 0.849 |
| tabular+hv | 0.935 | 0.932 |
| tabular+tfidf+hv | **0.945** | 0.942 |

Marginal of HV over tabular+tfidf: static +0.007 → evolved +0.004 (both within single-split noise).
Pruning-for-marginal even *weakened* standalone HV (removed individually-predictive-but-redundant
probes). Verdict: the marginal signal is real but thin; evolution does not reliably sharpen it here.
Static generated pool stays the default. Reinforces "evolution is a minor lever" across datasets.

The "holistic hv+tfidf" pass = the `tabular+tfidf+hv` config itself (0.945 static / 0.942 evolved) —
the best config on this task, i.e. HV + TF-IDF + metadata together.

prep_cfpb.py gained `--baseline {tabular,tabular_tfidf}` and `--from-csv` (regenerate a pool on
existing rows without re-streaming).

## Phase 12 — llm-trees forest: zero-shot LLM decision-tree induction vs embedding (2026-07-10)

Adapts "Oh LLM, I'm Asking Thee, Please Give Me a Decision Tree" (Knauer/Koddenbrock et al., KDD
'25; arXiv 2409.18594) to the text/NLI regime. An LLM writes a forest of K=5 traversable trees
(internal node = an NLI hypothesis, leaf = a class; `deepseek-v4-flash`, temp 1.0, avoid-seeded).
Two uses, mirroring the paper: **induction** (route text through the trees via NLI, label-free) and
**embedding** (flatten node conditions into a pool → learned head). Code: `experiments/hvexp/
forest.py`, `LLMForestInduction` in `systems.py`, script `experiments/scripts/llm_forest.py`.
Encoder finecat-nli-l; learned heads mean over seeds {7,17,23}; McNemar at k=all, seed 7.

**Controls are size-matched:** the flat_pool_embed control is a flat generated pool at the SAME
target size as the forest's conditions (TREC 17, SST-2 6, GoEmotions 30), same train sample/seed —
so forest-vs-flat isolates the *structure of generation*, not pool size.

| dataset | zeroshot | prior | induction (soft/hard) | forest_embed (k=all) | flat_pool_embed (k=all) |
|---|---|---|---|---|---|
| TREC (6c, 17 conds) | .428 | .594 | .486 / .472 | .945 | .948 |
| SST-2 (2c, 6 conds) | .947 | .953 | .936 / .931 | .949 | .952 |
| GoEmotions (7c, 30 conds) | .319 | .320 | .317 / .331 | .645 | .671 |

**Verdict (vs pre-registration):**
- ✅ **Embedding ≫ induction, significant on all three** — McNemar p=0.000 (TREC, disc 234/5),
  p=0.024 (SST-2, 21/8), p=0.000 (GoEmotions, 753/93). The llm-trees headline replicates in
  text/NLI: routing text through the LLM trees is near the zero-shot floor, while using their node
  conditions as a feature basis climbs far above.
- ❌ **forest_embed never beats flat_pool_embed** at matched size — TREC p=0.839 (tie), SST-2
  p=1.000 (tie), GoEmotions **p=0.047 in FLAT's favor** (flat 129 vs forest 164... i.e. flat fixes
  more). The **promotion criterion (forest > flat on ≥2/3) is NOT met** → no production
  `pool.method="forest"`. The earlier "compression" read (forest 17 conds ≈ flat 64) dissolves
  once sizes are matched: a flat pool at size 17 does equally well or better.
- Induction beats single-template zeroshot only on TREC (.486 vs .428) and loses to the prior head
  (.594); on prior-saturated SST-2 and hard GoEmotions the label-free anchors tie.

**Conclusion — reinforces the paper thesis:** the LLM tree *structure* (as classifier or as a
generation scaffold) is not the lever; its hypotheses as a frozen-NLI feature basis are. Filed as a
replicated cross-domain negative + a §5.1 label-free induction row + a §5.2 note. On the hard 7-way
task, forcing hierarchical group-splits is mildly *worse* than free-form flat generation (GoEmotions
p=0.047), consistent with generation diversity mattering more than imposed structure. Assets:
`experiments/results/forests/*.json`, `pools/{ds}_gen_matched.json`, `raw/llm_forest_matched.json`.

_(Further phases appended as they run.)_
