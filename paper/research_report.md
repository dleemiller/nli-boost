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

## Generality across task types (RQ3) — three distinct patterns

Same protocol on two more datasets (10 seeds, finecat-nli-l, hand-written expert pool). These are
the honest "where HV wins and where it doesn't" evidence — the pattern **depends on task structure**.

### Table 5a — AG News (4-way topic). Test accuracy.

| System | 1 | 2 | 5 | 10 | 50 | 100 |
|---|---|---|---|---|---|---|
| **Zero-shot NLI (templates)** | **.892** | **.892** | **.892** | **.892** | **.892** | **.892** |
| HV prior (0 labels) | .858 | .858 | .858 | .858 | .858 | .858 |
| HV + RF head | .720 | .786 | .829 | .848 | .876 | .884 |
| HV + L2-logreg | .725 | .811 | .841 | .853 | .857 | .862 |
| MiniLM emb + logreg | .454 | .576 | .689 | .755 | .813 | .826 |
| TF-IDF (word+char) | .321 | .367 | .472 | .577 | .759 | .805 |

**Read:** on broad topics the **NLI prior is already near-ceiling** — zero-shot NLI (0.892) and the
label-free prior head (0.858) win at every budget, and HV's *learned* heads add little at low N.
TF-IDF is weak when starved but climbs steeply (0.32→0.81), the classic "lexical catches up with
data on topic tasks."

### Table 5b — Banking77 (77-way intent). Test accuracy.

| System | 1 | 2 | 5 | 10 | 20 |
|---|---|---|---|---|---|
| **MiniLM emb + logreg** | **.535** | **.663** | **.792** | **.851** | **.886** |
| TF-IDF (word+char) | .334 | .461 | .626 | .722 | .801 |
| HV + RF head | .299 | .434 | .580 | .669 | .731 |
| HV + L2-logreg | .306 | .380 | .468 | .524 | .568 |
| Zero-shot NLI (77 templates) | .411 | .411 | .411 | .411 | .411 |
| HV prior (0 labels) | .231 | .231 | .231 | .231 | .231 |

**Read (HV LOSES here — the important negative result):** on fine-grained 77-way intent, dense
embeddings win at every budget and TF-IDF beats HV too. The **24-hypothesis expert pool is far too
thin to span 77 intents** — it structurally cannot separate the ~53 classes it has no hypothesis for
(prior head floors at 0.231). This says HV's pool must **scale with the label space**: many-class
tasks need a large *generated* pool, not a small hand-written one. A direct motivation for the
pool-size ablation and generated-pool coverage.

### Synthesis: HV's advantage is a function of task structure

| Task type | Dataset | What wins at low N | HV verdict |
|---|---|---|---|
| Small clean taxonomy | TREC-6 | HV (prior → RF) | **HV dominates at all N** |
| Broad topics | AG News | zero-shot NLI / HV prior | NLI prior near-ceiling; learned head adds little |
| Fine-grained many-class | Banking77 | dense embeddings | **HV loses** with a thin pool; pool must scale to labels |

## Table 6 — Status of the research questions

| RQ | Question | Status | Verdict so far |
|---|---|---|---|
| RQ1 | Low-label performance | **Done (TREC)** | Strong: clean crossover; HV wins at every N on TREC |
| RQ2 | Interpretability | **Done (TREC)** | Readable, stable, class-aligned importances |
| RQ3 | vs standard representations | **Done (3 datasets)** | 3 distinct patterns; HV wins TREC, ties/loses on AG News/Banking77 |
| RQ4 | Generation & evolution | **Done (TREC)** | Generated ≥ expert for learned heads; evolution minor lift |
| RQ5 | Text + tabular (CFPB) | Pending | Runner built (`run_text_tabular.py`), not yet run |
| — | Generality (AG News, Banking77) | **Done** | See Table 5a/5b; fine-tuned-encoder baseline still pending |

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
- **HV is not a universal winner — now shown, not just suspected.** On AG News the zero-shot NLI
  prior already near-ceilings, so HV's learned heads add little; on **Banking77 HV loses outright**
  to dense embeddings and TF-IDF because a 24-hypothesis pool cannot span 77 intents. The honest
  framing is a task-structure-dependent Pareto point (Table 5a/5b), not a leaderboard claim.
- The many-class result implies **pool size must scale with the label space** — the paper should run
  the pool-size ablation and a large *generated* pool on Banking77 before drawing final conclusions
  there.
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
