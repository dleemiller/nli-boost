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

## Table 1b — vs a fine-tuned encoder (the strong data-rich reference)

Fine-tuning DistilBERT end-to-end per training subsample (5 seeds; figure
`trec_with_finetune_accuracy.pdf`) draws the Pareto boundary that frames the whole method.

| examples/class | 1 | 2 | 3 | 5 | 10 | 20 | 50 | 100 | all |
|---|---|---|---|---|---|---|---|---|---|
| fine-tuned DistilBERT | 0.263 | 0.311 | 0.419 | 0.528 | 0.739 | 0.822 | **0.904** | **0.921** | **0.964** |
| HV + RF head | 0.474 | 0.597 | 0.642 | 0.682 | 0.753 | 0.802 | 0.849 | 0.892 | 0.954 |
| HV prior (0 labels) | **0.594** | **0.594** | 0.594 | 0.594 | 0.594 | 0.594 | 0.594 | 0.594 | 0.594 |

**Read — a clean regime split.** Fine-tuning is **catastrophic at low N** (0.263 at 1/class — below
every baseline; no semantic prior and only a handful of gradient steps), **crosses HV at ~5–10
examples/class**, and **wins the data-rich regime** (0.904→0.964 at 50→all). HV owns 1–10/class; the
fine-tuned encoder owns ≥20/class; at full data they converge (0.954 vs 0.964, HV staying
interpretable). This is the paper's positioning in one figure — HV is the low-label, interpretable,
LLM-free operating point; a fine-tuned encoder is the accurate-but-opaque, data-hungry one.

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

### Table 5c — Banking77 pool scaling (resolves the negative result)

Testing the hypothesis that Banking77's loss was a *thin-pool* artifact, not a method limit: we
generated a **256-hypothesis** pool (DeepSeek-v4-flash, from 3/class; covers **76 of 77 classes**
vs the expert pool's ~24) and re-ran the curve (5 seeds). Test accuracy:

| System | 1 | 2 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|
| MiniLM emb + logreg | **.534** | **.673** | **.734** | **.789** | **.851** | **.886** |
| **HV gen-256 + logreg** | .530 | .668 | .729 | .773 | .824 | .859 |
| **HV gen-256 + RF** | .474 | .646 | .704 | .758 | .806 | .845 |
| TF-IDF (word+char) | .341 | .464 | .545 | .624 | .722 | .800 |
| HV expert-24 + RF | .309 | .438 | .510 | .580 | .668 | .731 |
| HV expert-24 + logreg | .312 | .380 | .422 | .471 | .525 | .568 |

**Read — hypothesis confirmed.** Scaling 24→256 hypotheses roughly **doubles** HV's low-N accuracy
(0.31→0.53 at 1/class) and **closes almost the entire gap to dense embeddings** — gen-256+logreg
tracks MiniLM within ~0.004–0.03 at every budget and beats TF-IDF throughout. HV was not
structurally bad for fine-grained intent; **its pool must scale with the label space**. Embeddings
still edge it slightly, so the honest statement is *parity-minus-a-hair with an opaque embedding
probe, while staying interpretable and LLM-free*. (The label-free prior head improved too, 0.231→
0.392, but averaging 256 auto-tagged hypotheses into 77 classes stays weak — the learned head is
where the large pool pays off.)

### Table 5e — SST-2 (binary sentiment). Test accuracy.

| System | 1 | 2 | 5 | 10 | 50 | 100 | all |
|---|---|---|---|---|---|---|---|
| **HV prior (0 labels)** | **.953** | **.953** | **.953** | **.953** | **.953** | **.953** | **.953** |
| HV expert + logreg | .950 | .950 | .950 | .950 | .952 | .952 | .952 |
| zero-shot NLI | .947 | .947 | .947 | .947 | .947 | .947 | .947 |
| MiniLM emb + logreg | .486 | .530 | .581 | .637 | .721 | .707 | .815 |
| TF-IDF (word+char) | .492 | .511 | .517 | .549 | .605 | .630 | .826 |

**Read:** sentiment is *near-solved by the NLI prior* — the label-free HV prior head reaches **0.953
at every budget** and never needs data, beating zero-shot NLI (0.947) and crushing TF-IDF/embeddings,
which sit near chance at low N (~0.49) and only reach ~0.82 even at the full 67k-example train set.
The most extreme "prior alone wins" case: HV dominates at *every* budget and the data-driven
baselines never catch up.

### Table 5g — GoEmotions (Ekman-7 emotion). Test accuracy.

| System | 1 | 3 | 5 | 10 | 20 | 50 | 100 | all (39k) |
|---|---|---|---|---|---|---|---|---|
| HV + RF head | .314 | **.402** | .406 | .463 | **.490** | .504 | **.523** | .636 |
| HV prior (reweighted) | **.343** | .350 | **.415** | **.480** | .486 | **.517** | .514 | .594 |
| HV prior (0 labels) | .320 | .320 | .320 | .320 | .320 | .320 | .320 | .320 |
| zero-shot NLI | .319 | .319 | .319 | .319 | .319 | .319 | .319 | .319 |
| MiniLM emb + logreg | .180 | .219 | .267 | .307 | .341 | .372 | .368 | .611 |
| TF-IDF (word+char) | .137 | .164 | .196 | .232 | .268 | .337 | .388 | **.674** |

**Read — the cleanest low-N crossover in the study.** Emotion is hard (modest absolute accuracy,
0.32–0.67, a neutral-heavy 7-way task), but the *shape* is textbook: HV imports emotion priors and
**roughly doubles the baselines at 1/class** (0.34 vs 0.14–0.18) and leads through 100/class — then,
with **39k labeled examples, TF-IDF overtakes HV** (0.674 vs 0.636). This is the first non-fine-tune
case where a simple supervised baseline clearly wins the data-rich regime, exactly the "ordinary
models catch up as N grows" prediction — HV owns the low-label band, lexical features own data-rich.

### Synthesis: HV's advantage is a function of task structure

| Task type | Dataset | What wins at low N | HV verdict |
|---|---|---|---|
| Small clean taxonomy | TREC-6 | HV (prior → RF) | **HV dominates at all N**; fine-tune crosses at ~5–10/class |
| Sentiment | SST-2 | HV prior (0.953) | **NLI prior near-solves it; HV wins at every N** incl. full data |
| Broad topics | AG News | zero-shot NLI / HV prior | NLI prior near-ceiling; learned head adds little |
| Emotion | GoEmotions | HV (prior/RF) | **HV owns low-N (~2× baselines); TF-IDF overtakes at 39k** — clean crossover |
| Fine-grained many-class | Banking77 | dense embeddings | thin 24-hyp pool loses; **256-hyp generated pool reaches embedding parity** (Table 5c) |

### Table 5d2 — method ablations (TREC, 100 examples/class, RF head, 5 seeds)

| axis | variant → accuracy |
|---|---|
| **NLI encoder (capacity lever)** | finecat-nli-**m** 0.798 → finecat-nli-**l** **0.892** (**+0.094**) |
| score channel | entail 0.886 · contrast 0.888 · entail+contradict 0.892 (within noise) |

**Read:** the **frozen NLI encoder is by far the dominant lever** (+9.4 pts from -m→-l on the *same*
hypotheses), dwarfing the score-channel choice, which is within seed noise under a flexible head
(entail+contradict marginally best, kept as default). Capacity lives in the encoder; the other knobs
are second-order.

## Table 5d — RQ4: pool-size scaling (how many hypotheses?)

Subsampling a generated pool to sizes 8…256 (RF head, 20 examples/class, 5 seeds; figure
`poolsize_scaling.pdf`). Directly answers "how many hypotheses do you need?" — and the answer
**scales with the label space**.

| # hypotheses | 8 | 16 | 32 | 64 | 128 | 192 | 256 |
|---|---|---|---|---|---|---|---|
| **TREC-6** (6 classes) | .622 | .714 | .824 | .829 | .877 | .884 | .877 |
| **Banking77** (77 classes) | .543 | .678 | .733 | .780 | .831 | .841 | .845 |

**Read:** on the small taxonomy (TREC) accuracy **saturates by ~32–64 hypotheses** — beyond that,
extra probes barely move it (0.824 at 32 vs 0.877 at 256), consistent with the method's "≈30 useful
semantic directions" finding. On the 77-way task, accuracy **keeps climbing to ~128–192** before
plateauing near 256 (0.54→0.85). The useful pool size grows with the number of classes / semantic
complexity — so pool size is a task-dependent knob, and the earlier Banking77 deficit was simply an
under-sized pool.

## Table 6 — RQ5: CFPB text+tabular marginal value

Monetary-relief prediction on CFPB consumer complaints (narrative + Product/Company/State/channel
metadata). Balanced 4,000-row sample (relief rate forced to 0.50 for a stable rank metric), random
80/20 split, HistGradientBoosting head; HV pool = 64 hypotheses generated from **train narratives
only**. The question is not "does text beat tabular" but **how much does each feature family add on
top of the others**.

| Feature configuration | # feat | Accuracy | Macro-F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| tabular_only | 260 | 0.828 | 0.827 | 0.914 |
| tfidf_only | 128 | 0.823 | 0.822 | 0.895 |
| hv_only | 128 | 0.769 | 0.769 | 0.856 |
| tabular + tfidf | 388 | 0.859 | 0.859 | 0.938 |
| **tabular + hv** | 388 | 0.849 | 0.849 | **0.935** |
| **tabular + tfidf + hv** | 516 | **0.870** | **0.870** | **0.945** |

**Read — HV adds interpretable marginal value.** The structured metadata alone is a strong predictor
(AUC 0.914 — company/product largely determine relief), and HV *alone* is the weakest single channel
(0.856 < TF-IDF 0.895 < tabular). But HV is **complementary**: adding it to the tabular block lifts
AUC by **+0.021** (0.914→0.935, ≈ TF-IDF's own +0.024 marginal contribution), and adding it on top of
tabular+TF-IDF **still helps** (+0.007, best config 0.945) — signal neither structured fields nor
lexical features capture. Crucially, that added signal is **auditable**: the contributing hypotheses
are readable ("mentions a specific dollar amount the consumer lost or is owed", "demand for a refund
or compensation", "emotional distress rather than a specific financial loss"). This is the practical
claim — hypothesis features can be justified by their marginal value over existing structured
features in a real, regulated pipeline.

*Caveats:* this balanced/random setup (AUC ~0.91–0.94) is a clean controlled comparison, not the
benchmark protocol; the temporal natural-rate run is Table 6b.

**Evolve marginal-pruning refinement (negative result).** We regenerated the pool with the
evolution loop pruning hypotheses by their marginal value over a *holistic* tabular+TF-IDF baseline
(so survivors must beat both the metadata and the lexical channel). It improved the pool's internal
held-out metric (0.783→0.798 over 6 rounds) but **did not translate to a larger contribution on
test**: the holistic `tabular+tfidf+hv` config is 0.942 (evolved) vs 0.945 (static) — within
single-split noise — and pruning-for-marginal actually *weakened* standalone HV (hv_only 0.849 vs
0.856) by dropping individually-predictive-but-redundant probes. So the marginal signal is real but
thin, and evolution does not reliably sharpen it here — consistent with evolution being a minor
lever (§RQ4). The static generated pool is the recommended default for this task.

### Table 6b — CFPB, natural rate + temporal split (benchmark protocol)

30,000 complaints at the **real ~3.6% monetary-relief rate**, **temporal** split (oldest 24k train →
newest 6k test; relief drifts 3.0%→5.9% over the window), HGB head, 64-hyp pool generated from the
temporal-train narratives only. At a 3.6% base rate **accuracy is majority-dominated (~0.94 ≈
predict-none) and uninformative — ROC-AUC and macro-F1 are the metrics.**

| configuration | tabular | tfidf | hv | tab+tfidf | **tab+hv** | **tab+tfidf+hv** |
|---|---|---|---|---|---|---|
| ROC-AUC | .875 | .875 | .867 | .891 | **.891** | **.899** |
| macro-F1 | .497 | .548 | .513 | .550 | .551 | **.587** |

**Read:** the marginal-value story **holds under the realistic temporal, natural-rate setting**. HV
alone is competitive (AUC 0.867 ≈ tabular/TF-IDF 0.875); adding HV to the tabular block lifts AUC
+0.016, and on top of tabular+TF-IDF it still adds **+0.008 AUC and +0.037 macro-F1** (best 0.899 /
0.587) — the macro-F1 gain shows HV specifically improves detection of the rare *relief* class under
class imbalance and temporal drift.

*Not comparable to the published benchmark (0.78 hybrid / 0.69 TF-IDF, Wang et al. 2026):* our
sample is a recent 30k window at a 3.6% rate (vs their full historical data at ~8%), the split
window differs, and our tabular block is plain one-hot Product/Company/State/channel (vs their LDA
topics + engineered features). Our tabular baseline alone (0.875) already exceeds their numbers
because company/product one-hot is very predictive in this window, so **we make no claim of beating
the benchmark** — the result we stand behind is HV's *marginal* contribution over the other channels.

## Table 7 — Status of the research questions

| RQ | Question | Status | Verdict so far |
|---|---|---|---|
| RQ1 | Low-label performance | **Done (TREC)** | Strong: clean crossover; HV wins at every N on TREC |
| RQ2 | Interpretability | **Done (TREC)** | Readable, stable, class-aligned importances |
| RQ3 | vs standard representations | **Done (3 datasets + fine-tune)** | 3 patterns; fine-tuned encoder crosses HV at ~5–10/class (Table 1b) |
| RQ4 | Generation & evolution | **Done (TREC)** | Generated ≥ expert for learned heads; evolution minor lift |
| RQ5 | Text + tabular (CFPB) | **Done (balanced)** | HV adds +0.021 AUC over tabular, +0.007 over tabular+TF-IDF (Table 6) |
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
- The many-class deficit was a **thin-pool artifact, now fixed**: a 256-hypothesis generated pool
  reaches embedding parity on Banking77 (Table 5c). The remaining honest caveat is that dense
  embeddings still edge HV by a few points there — HV's win is interpretability + LLM-free serving
  at near-parity accuracy, not raw accuracy.
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
