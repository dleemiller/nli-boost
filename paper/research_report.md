# Hypothesis Vectorization — Results Report

**Method.** An LLM writes a pool of short natural-language *hypotheses*; a **frozen** NLI
cross-encoder (`dleemiller/finecat-nli-l`) scores each input's entailment/contradiction against each
hypothesis; a classical sklearn head learns the task over those interpretable features. Nothing is
fine-tuned and **no LLM runs at inference**.

**Protocol (all tables).** TREC-6 question classification. One **fixed** 500-example test set,
never subsampled and never seen during hypothesis generation. Training set subsampled to *k*
examples/class, **resampled over 10 seeds**; cells are mean test accuracy. Head/regularization
selected by CV on train only. Baselines are LM-free. Encoder held fixed at `finecat-nli-l`.

**Systems.** `hv_prior_fixed` = label-free prior head (class score = mean entailment of that class's
hypotheses, arg-max; at N=0 a zero-shot ensemble). `hv_prior_reweight` = strong-L2 reweighting of
those class scores. `hv_*_rf` / `hv_*_logreg` = RandomForest / L2-logistic head over the full
entail+contradict feature matrix. `expert` = 24 hand-written hypotheses; `static` / `evolved` = 64 /
62 **LLM-generated** hypotheses (DeepSeek-v4-flash, from train samples only; evolved = +CV
prune/refill, which self-stopped after 2 rounds).

---

## Table 1 — RQ1: Low-label learning curve, HV vs standard baselines

Test accuracy on TREC-6 (10 seeds). **Bold** = best in column. Hand-written expert pool.

| System | 1 | 2 | 3 | 5 | 10 | 20 | 50 | 100 | all |
|---|---|---|---|---|---|---|---|---|---|
| **HV prior (0 labels)** | **.594** | .594 | .594 | .594 | .594 | .594 | .594 | .594 | .594 |
| **HV prior (reweighted)** | .556 | **.629** | **.642** | .660 | .680 | .679 | .686 | .700 | .862 |
| **HV + RF head** | .474 | .597 | .642 | **.682** | **.753** | **.802** | **.849** | **.892** | **.954** |
| HV + L2-logreg head | .364 | .517 | .547 | .604 | .712 | .725 | .746 | .784 | .912 |
| Zero-shot NLI (templates) | .428 | .428 | .428 | .428 | .428 | .428 | .428 | .428 | .428 |
| MiniLM embeddings + logreg | .301 | .400 | .431 | .486 | .574 | .649 | .703 | .742 | .856 |
| TF-IDF (word+char) + logreg | .307 | .374 | .402 | .426 | .486 | .572 | .634 | .713 | .870 |
| TF-IDF (word) + logreg | .315 | .362 | .374 | .414 | .475 | .542 | .560 | .623 | .852 |
| TF-IDF (char) + logreg | .277 | .351 | .372 | .410 | .458 | .540 | .627 | .718 | .846 |

**Read:** a clean regime crossover — the *label-free* prior head is best at 1/class, the reweighted
prior at 2–3/class, the flexible RF head from ~5/class up to **0.954** at full data. HV beats TF-IDF
and sentence embeddings at **every** budget on this dataset. The multi-hypothesis prior beats
single-template zero-shot NLI by **+16.6 pts using zero labels** (0.594 vs 0.428).

---

## Table 2 — RQ4: LLM-generated pools vs hand-written expert pool

Test accuracy on TREC-6 (10 seeds), low-N region. **Bold** = best learned head in column.

| System | 1 | 2 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|---|
| HV **evolved** + RF | .449 | .624 | **.703** | **.784** | **.834** | **.873** | **.894** |
| HV **static** + RF | .482 | .628 | .692 | .726 | .788 | .856 | .886 |
| HV expert + RF | .474 | .597 | .642 | .682 | .753 | .802 | .849 |
| HV **evolved** + logreg | .384 | .589 | .664 | .740 | .810 | .851 | .893 |
| HV **static** + logreg | .350 | .486 | .576 | .628 | .773 | .837 | .890 |
| HV expert + logreg | .364 | .517 | .547 | .604 | .712 | .725 | .746 |
| HV prior — expert (0 labels) | **.594** | .594 | .594 | .594 | .594 | .594 | .594 |
| HV prior — evolved (0 labels) | .544 | .544 | .544 | .544 | .544 | .544 | .544 |
| HV prior — static (0 labels) | .484 | .484 | .484 | .484 | .484 | .484 | .484 |

**Read:** **LLM-generated pools beat the hand-written expert pool as a feature basis** for learned
heads — evolved+RF leads from 3/class up (the richer 62–64-hypothesis basis gives the head more to
exploit than 24 expert hypotheses). **Evolution adds a small, consistent lift** over static and
**saturated at 2 rounds** (reproducing prior findings). The expert pool keeps its edge **only** for
the zero-label prior head, because its class tags are hand-clean while the generated pools' tags are
auto-derived from the train sample.

---

## Table 3 — Headline comparison at representative budgets (accuracy)

| Budget | Best HV system | HV acc | Best baseline | Baseline acc | HV advantage |
|---|---|---|---|---|---|
| 1/class | prior, 0 labels | **0.594** | zero-shot NLI | 0.428 | **+0.166** |
| 5/class | evolved + RF | **0.784** | MiniLM emb | 0.486 | **+0.298** |
| 50/class | evolved + RF | **0.894** | MiniLM emb | 0.703 | **+0.191** |
| all (5,452) | expert + RF | **0.954** | TF-IDF w+c | 0.870 | **+0.084** |

---

## Table 4 — RQ2: Interpretability (top hypotheses by global importance)

Permutation importance of each hypothesis for the trained head (TREC, expert pool, 2k train).
Each feature is a readable sentence; `±` is cross-fold std (stability).

| Rank | Importance | ± | Hypothesis | Class |
|---|---|---|---|---|
| 1 | 0.094 | .005 | The text asks where something is located. | LOC |
| 2 | 0.073 | .009 | The text can be answered with a numeric value. | NUM |
| 3 | 0.037 | .004 | The text asks for a date, year, or period of time. | NUM |
| 4 | 0.031 | .004 | The text asks for the definition of a term. | DESC |
| 5 | 0.020 | .006 | The text asks for a description of something's meaning or purpose. | DESC |
| 6 | 0.014 | .005 | The text asks for the name of a thing or object. | ENTY |
| 7 | 0.006 | .005 | The text asks what an abbreviation or acronym stands for. | ABBR |
| 8 | 0.004 | .008 | The text asks for the name of a person. | HUM |

Redundancy check flagged the one near-duplicate pair ("name of a person" ⟷ "identity of an
individual", r = 0.92). Full per-class tables, exemplars, and error cases:
`experiments/results/processed/trec_expert_inspect_l/inspection.md`.

---

## Table 5 — Status of the research questions

| RQ | Question | Status | Verdict so far |
|---|---|---|---|
| RQ1 | Low-label performance | **Done (TREC)** | Strong: clean crossover; HV wins at every N on TREC |
| RQ2 | Interpretability | **Done (TREC)** | Readable, stable, class-aligned importances |
| RQ3 | vs standard representations | Partial | TF-IDF/emb/zero-shot done; fine-tuned encoder pending |
| RQ4 | Generation & evolution | **Done (TREC)** | Generated ≥ expert for learned heads; evolution minor lift |
| RQ5 | Text + tabular (CFPB) | Pending | Runner built (`run_text_tabular.py`), not yet run |
| — | Beyond TREC (AG News, Banking77, …) | Wired, not run | Datasets loadable; curves not yet run |

---

## Assessment

**Core thesis — supported (on TREC).** LLM-generated NLI-hypothesis features give an interpretable,
low-label, LLM-free-at-inference representation whose optimal head is a function of the label budget.
Both the low-N crossover (RQ1) and the generated-basis result (RQ4) came out as predicted.

**Strongest angle:** the **low-N regime story** — "the best configuration is a function of N," with a
label-free prior head owning the 1–3/class band and generated pools + a flexible head taking over as
labels grow — combined with full interpretability. This is a defensible Pareto point, not a
beat-everything claim.

**Weakest results / caveats to flag to the reader:**
- Results are **TREC-only** so far. TREC is semantically clean and NLI-favorable; prior notes expect
  TF-IDF to be much more competitive on topic tasks (AG News) and HV to struggle on long documents
  (20NG). **Do not generalize the "HV wins everywhere" pattern** until AG News/intent/CFPB are in.
- The zero-label prior head depends on **clean class tags**; generated pools need a better tagging
  step to match the expert pool there.
- Single-HGB head collapses to a constant class at <10 rows (a degenerate artifact); we use
  RandomForest for the flexible-head line, which overfits *gracefully*. Worth stating explicitly.
- Absolute numbers use one NLI encoder and one proposer; encoder size is the known capacity lever.

**Recommended next experiments (in order):** (1) AG News + Banking77 curves — test generality on
topic and intent tasks; (2) CFPB text+tabular marginal-value study (the applied/auditability story);
(3) ablations (pool size, `-m` vs `-l` encoder, score-mode, dedup); (4) replicate generated-vs-expert
on a second dataset; (5) add a fine-tuned small-encoder baseline for full-data positioning.

**Venue fit:** an interpretability / efficient-NLP / low-resource workshop is the natural first
target (EMNLP/ACL/NeurIPS workshops); the interpretable-low-label-Pareto framing suits that scope.

---

*Artifacts: figures in `experiments/results/figures/` (PDF+PNG); tables in
`experiments/results/tables/` (Markdown + LaTeX); raw per-run rows in
`experiments/results/raw/<run>/results.jsonl`; live log in `paper/experiment_notes.md`. Branch:
`paper-experiments`.*
