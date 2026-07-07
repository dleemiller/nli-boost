# Hypothesis Vectorization: Interpretable Low-Label Text Classification with LLM-Generated NLI Features

*Working draft — 2026-07-07. All numbers are from the runs in `experiments/results/`; the companion
files `paper/method.md`, `paper/related_work.md`, and the live log `paper/experiment_notes.md` hold
the expanded method, full related work, and per-run provenance.*

## Abstract

We introduce **Hypothesis Vectorization (HV)**, a method that represents a text by its
natural-language *entailment profile*. An LLM proposes a compact pool of short, readable hypotheses
about the input; a **frozen** natural-language-inference (NLI) cross-encoder scores how strongly each
input entails or contradicts each hypothesis; and a classical model learns the task over the
resulting feature matrix. The representation imports semantic prior knowledge from the proposer and
the NLI model without any fine-tuning, so every feature is a plain English sentence whose learned
weight is directly auditable, and **inference uses no LLM** — only a frozen encoder. Across
learning-curve experiments on three tasks we find a clean regime structure: at 1–3 labeled
examples per class a *label-free* prior-aggregation head over the hypotheses is strongest; as labels
grow a flexible classical head takes over; and a fully fine-tuned encoder — catastrophic at low N —
overtakes HV only at ~5–10 examples per class and wins the data-rich regime. We show that
LLM-*generated* hypotheses form a stronger feature basis than a hand-written expert pool for learned
heads, that evolution adds a small consistent lift and saturates in two rounds, and that the number
of useful hypotheses scales with the label space (≈32 for a 6-way task, ≈128–192 for a 77-way one).
On a text+tabular regulatory task (CFPB consumer complaints) HV features add interpretable marginal
value over structured metadata. We position HV not as a state-of-the-art accuracy claim but as a
compelling Pareto point: low-label-strong, interpretable, cheap to adapt, LLM-free to serve, and
compatible with classical and text+tabular pipelines.

## 1. Introduction

Organizations routinely need text classifiers for narrow, evolving taxonomies with few labeled
examples, in settings where the model must be auditable and cheap to serve — triage queues, intent
routers, compliance flags. The dominant options each give something up. Fine-tuning an encoder is
data-hungry and yields an opaque model. Dense-embedding probes are cheap but their dimensions carry
no readable meaning. Direct LLM classification is strong but pays a per-inference cost, drifts with
prompt changes, and resists audit. Classical lexical models, though strong, key on surface forms and
miss semantics.

Hypothesis Vectorization occupies the middle. An LLM writes a pool of short declarative
**hypotheses** about the input — "The text asks where something is located.", "The complaint demands
a refund of a specific dollar amount." — and a frozen NLI model measures each input against each
hypothesis. A classical head then estimates the label from those entailment scores. The LLM's
semantic prior is captured *once*, at authoring time, and frozen into a readable feature basis. This
buys three properties a practitioner cares about: (i) every feature is a sentence, so importances and
errors are explainable and a domain expert can add or veto probes; (ii) the semantic prior
substitutes for labeled data, which matters most when labels are scarce; and (iii) serving needs no
LLM — the per-example cost is a fixed number of cached NLI forward passes.

We frame HV as representation learning with an explicit factorization that organizes the paper: the
**LLM proposes a semantic basis**, the **NLI model is a fixed measurement operator**, the
**classical head is the task estimator**, and pruning/evolution **searches over semantic features
rather than neural weights**.

**Contributions.**
1. A method for turning LLM-generated natural-language hypotheses into NLI-scored, interpretable
   semantic features, with a reproducible scikit-learn-compatible implementation that uses no LLM at
   inference (§3).
2. A **low-label learning-curve study** establishing a regime structure — a label-free
   prior-aggregation head best at the smallest budgets, a flexible head as labels grow — and locating
   the crossover against a fine-tuned encoder at ~5–10 examples per class (§5.1).
3. **Ablations** over hypothesis generation, evolution, and pool size, including honest negatives:
   generation saturates in two rounds, and the useful pool size scales with the label space (§5.2).
4. A **generality study** across question-type, topic, and intent tasks that reports, without
   inflation, where HV wins, ties, and loses (§5.3).
5. A **text+tabular** study (CFPB) evaluating hypothesis features by their *marginal* value over
   structured metadata, for real regulated pipelines (§5.4).
6. An **interpretability analysis** using the readable features: global and per-class importances,
   redundancy, stability, and qualitative error explanations (§5.5).

## 2. Related Work

*(Full treatment in `paper/related_work.md`.)* HV generalizes **zero-shot NLI classification**
(Yin et al., 2019) from one template per class to a *learned head over a pool of many LLM-generated
probes*; empirically the multi-hypothesis ensemble beats single-template zero-shot by 16.6 points
using zero labels (§5.1). It is a **concept bottleneck** (Koh et al., 2020) whose concepts are
generated in natural language and scored by a frozen NLI model rather than hand-annotated and
learned, and are *searched* by pruning/evolution. It relates to **weak supervision** (Ratner et al.,
2017) but replaces brittle code/regex labeling functions with semantic NLI probes fed to a
discriminative head. Unlike **direct LLM classification** it keeps the semantic prior while removing
the LLM from the inference path, and unlike **dense-embedding probes** (Sentence-BERT, E5, BGE) its
dimensions are readable. It is complementary to prompt/program optimizers (**DSPy**, **GEPA**), whose
optimization discipline we borrow but apply to a different search space — which hypotheses form the
best semantic basis.

## 3. Method

*(Formal treatment in `paper/method.md`.)* Let a labeled dataset be $D=\{(x_i,y_i)\}_{i=1}^n$ with
$y_i\in\{1,\dots,K\}$ and a hypothesis pool $H=\{h_1,\dots,h_m\}$ of short declarative statements
about the input. A frozen NLI model maps a (premise, hypothesis) pair to a distribution over
{entail, neutral, contradict}, $f_\theta(x,h)=(p_e,p_n,p_c)$. The feature map scores $x$ against the
whole pool, $\phi(x)=[g(f_\theta(x,h_1)),\dots,g(f_\theta(x,h_m))]$, where the **score channel** $g$
selects the entailment probability, both entailment and contradiction (the default, two columns per
hypothesis), or their contrast $p_e-p_c$. A classical estimator predicts $\hat y=g_\psi(\phi(x))$.
Because each dimension is a named hypothesis, $g_\psi$'s feature importances attach to English
sentences.

**Generation.** The proposer sees the task description, class names with one-line *definitions*, and
a small stratified sample of *training* examples only. It returns hypotheses as structured JSON,
`{text, intended_class, rationale}`; only `text` becomes a feature. Prompt rules enforce short,
atomic, affirmative statements about the text, verifiable from the text alone, class-relevant and
diverse, never bare label names or dataset-leaking strings. Hand-written **fixed hypotheses** from a
domain expert can be supplied and are always scored and never pruned. Candidates are deduplicated by
text-space similarity (data-free, the right default at low N) or by behavioral covariance of their
score vectors (once enough data exists), with a variance floor rejecting vacuous near-constant
probes.

**Evolution (optional).** For data-rich settings a loop ranks hypotheses by cross-fold permutation
importance *and* cross-fold sign stability (a single split churned ~50% of prune decisions across
seeds), prunes only *confident deaths*, and refills against confusion hot-spots, told the failure
reason for each pruned probe. With a baseline feature block configured (TF-IDF or tabular columns),
ranking is *marginal over that block*.

**The low-N head.** At 1–5 examples per class a flexible RF/HGB head overfits and is compute-heavy;
we introduce a **prior-aggregation head** that scores class $k$ as the mean entailment of the
hypotheses tagged for $k$ and predicts the arg-max — at $N=0$ exactly a zero-shot NLI *ensemble* —
with a strongly-regularized reweighting as $N$ grows.

**Inference & reproducibility.** At inference the pool is fixed, the NLI model frozen, and no LLM
runs; the per-example cost is $m$ cached NLI passes. Head family and regularization are chosen by CV
on train only and evaluated on test **once** (`pool_cv`); two-run comparisons use paired McNemar with
Wilson intervals. Every (text, hypothesis, encoder) score is cached, so the entire seed/train-size
sweep costs a single GPU scoring pass, and each run records git commit, library versions, dataset
revision, seed, split indices, config, pool, metrics, and cost in a manifest.

## 4. Experimental setup

**Datasets.** TREC-6 (question-type, 6 classes), AG News (topic, 4), and Banking77 (banking intent,
77); plus CFPB consumer complaints for the text+tabular study. **Protocol.** For each dataset we draw
**one fixed test set** (keyed on a single seed) and subsample the training set to $k$ examples per
class for $k\in\{1,2,3,5,10,20,50,100,\text{all}\}$, resampled over 5–10 seeds; test is never
subsampled and never inspected during hypothesis generation, pruning, or head selection. NLI features
for the whole corpus are scored once and cached, so each learning curve is one GPU pass plus cheap
CPU refits. **Baselines.** TF-IDF (word / char / union) + logistic regression; sentence-embeddings
(all-MiniLM-L6-v2) + logistic regression; zero-shot NLI with one class template per class; and a
**fine-tuned DistilBERT** trained end-to-end per subsample. **Models.** Frozen NLI encoder
`finecat-nli-l`; proposer DeepSeek-v4-flash via OpenRouter. **Metrics.** Accuracy, macro/weighted-F1,
per-class P/R/F1, ECE calibration, bootstrap CIs over seeds; ROC-AUC for the binary CFPB task; zero
inference-time LLM calls by construction.

## 5. Results

### 5.1 Low-label learning curves and the Pareto boundary (RQ1, RQ3)

On TREC-6 (finecat-nli-l, 10 seeds, fixed 500-example test) the method shows a clean regime crossover
(test accuracy):

| system | 1 | 2 | 3 | 5 | 10 | 20 | 50 | 100 | all |
|---|---|---|---|---|---|---|---|---|---|
| HV prior (0 labels) | **.594** | .594 | .594 | .594 | .594 | .594 | .594 | .594 | .594 |
| HV prior (reweighted) | .556 | **.629** | **.642** | .660 | .680 | .679 | .686 | .700 | .862 |
| HV + RF head | .474 | .597 | .642 | **.682** | **.753** | .802 | .849 | .892 | .954 |
| HV + L2-logreg head | .364 | .517 | .547 | .604 | .712 | .725 | .746 | .784 | .912 |
| zero-shot NLI | .428 | .428 | .428 | .428 | .428 | .428 | .428 | .428 | .428 |
| MiniLM emb + logreg | .301 | .400 | .431 | .486 | .574 | .649 | .703 | .742 | .856 |
| TF-IDF (word+char) | .307 | .374 | .402 | .426 | .486 | .572 | .634 | .713 | .870 |
| **fine-tuned DistilBERT** | .263 | .311 | .419 | .528 | .739 | **.822** | **.904** | **.921** | **.964** |

Three observations. First, the *label-free* prior-aggregation head is the single best system at
1/class (0.594) and the reweighted prior leads at 2–3/class, after which the flexible RF head takes
over and climbs to 0.954 at full data — *the optimal head is a function of the label budget*. Second,
the multi-hypothesis prior beats single-template zero-shot NLI by **+16.6 points using zero labels**
(0.594 vs 0.428), isolating the value of a *pool* of probes over one template per class. Third, the
fine-tuned encoder draws the Pareto boundary: it is **catastrophic at low N** (0.263 at 1/class,
below every baseline — no semantic prior and only a handful of gradient steps), **crosses HV at ~5–10
examples per class**, and **wins the data-rich regime** (0.904→0.964 at 50→all). At full data HV and
fine-tuning converge (0.954 vs 0.964), with HV remaining interpretable. HV owns the 1–10/class band;
a fine-tuned encoder owns ≥20/class. On this semantically clean dataset HV also beats the TF-IDF and
embedding baselines at *every* budget — a favorable case we do not over-generalize (§5.3).

### 5.2 Generation, evolution, and pool size (RQ4)

**Generated vs expert pools.** We generate a static pool (64 hypotheses, from 5 examples/class) and
an evolved pool (62 hypotheses, from 50/class; evolution self-stopped at two rounds on a held-out
plateau) and drop both onto the TREC curve. LLM-generated pools are a **stronger feature basis than
the hand-written 24-hypothesis expert pool** for learned heads: the evolved-pool RF head leads from
3/class upward (0.703→0.894 over 3–50/class vs the expert pool's 0.642→0.849), because the richer
basis gives the classifier more to exploit. **Evolution adds a small, consistent lift** over the
static pool (clearest for the linear head at low N) and saturates in two rounds — a refinement, not
the capacity lever, consistent with the encoder being the dominant lever in prior analysis. The
zero-label prior head still favors the expert pool (0.594 vs 0.544 evolved / 0.484 static), because
its class tags are hand-clean whereas the generated pools' tags are derived from the training sample.

**Pool size.** Subsampling a generated pool to 8…256 hypotheses (RF head, 20 examples/class) shows
the useful pool size **scales with the label space**:

| # hypotheses | 8 | 16 | 32 | 64 | 128 | 192 | 256 |
|---|---|---|---|---|---|---|---|
| TREC-6 (6 classes) | .622 | .714 | .824 | .829 | .877 | .884 | .877 |
| Banking77 (77 classes) | .543 | .678 | .733 | .780 | .831 | .841 | .845 |

The 6-way task saturates by ~32–64 hypotheses (its "≈30 useful semantic directions"), while the
77-way task keeps improving to ~128–192 before plateauing near 256. Pool size is therefore a
task-dependent knob, not a fixed hyperparameter.

### 5.3 Generality: three task structures (RQ3)

Running the same protocol on two further datasets yields three distinct patterns — the honest "where
HV helps and where it doesn't" evidence.

*Broad topics (AG News, 4 classes).* The NLI prior is already near-ceiling: zero-shot NLI (0.892) and
the label-free prior head (0.858) win at every budget, HV's learned heads add little at low N, and
TF-IDF is weak-but-climbing (0.32→0.81 by 100/class). On easy topic tasks the *prior alone* suffices.

*Fine-grained intent (Banking77, 77 classes).* With a thin 24-hypothesis expert pool HV **loses** to
dense embeddings (0.53→0.89) and TF-IDF — the pool cannot span 77 intents (the prior head floors at
0.231, covering only its tagged classes). But this is a *pool-size artifact, not a method limit*:
scaling to a **256-hypothesis generated pool** (covering 76/77 classes) roughly doubles low-N
accuracy (0.31→0.53 at 1/class) and **closes almost the entire gap to dense embeddings** — the
generated-pool logistic head tracks MiniLM within ~0.004–0.03 at every budget and beats TF-IDF
throughout. HV's advantage is thus a function of task structure: it dominates on small clean
taxonomies, defers to the bare NLI prior on broad topics, and needs a pool scaled to the label space
on many-class tasks, where it reaches embedding parity while staying interpretable.

### 5.4 Text + tabular marginal value (RQ5)

On CFPB monetary-relief prediction (narrative + Product/Company/State/channel; balanced 4,000-row
sample, random split, HistGradientBoosting head, 64-hypothesis pool generated from train narratives
only), we measure how much each feature family adds on top of the others (ROC-AUC):

| configuration | tabular | tfidf | hv | tab+tfidf | **tab+hv** | **tab+tfidf+hv** |
|---|---|---|---|---|---|---|
| ROC-AUC | .914 | .895 | .856 | .938 | **.935** | **.945** |

Structured metadata alone is a strong predictor (0.914) and HV *alone* is the weakest single channel
(0.856), but HV is **complementary**: adding it to the tabular block lifts AUC by **+0.021** (≈
TF-IDF's own +0.024 marginal contribution), and adding it on top of tabular+TF-IDF *still* helps
(+0.007, best 0.945) — signal neither structured fields nor lexical features capture. Crucially that
signal is **auditable**: the contributing hypotheses read as plain English a compliance reviewer can
inspect ("mentions a specific dollar amount the consumer lost or is owed", "demand for a refund or
compensation", "emotional distress rather than a specific financial loss"). This is the practical
claim — hypothesis features can be justified by their marginal value over existing structured
features in a regulated pipeline. (This balanced/random setup, AUC ~0.91–0.94, is not the temporal
natural-rate benchmark of 0.78/0.69 and is not directly comparable to it.)

### 5.5 Interpretability (RQ2)

Because each feature is a sentence, the trained head is directly inspectable. On TREC the top
hypotheses by permutation importance are readable and class-aligned — "The text asks where something
is located." (LOC, importance 0.094), "The text can be answered with a numeric value." (NUM, 0.073),
"The text asks for the definition of a term." (DESC, 0.031) — with low cross-fold variance, so the
explanations are stable, not artifacts of one split. A redundancy scan surfaces near-duplicate probes
(e.g. "asks for the name of a person" vs "asks for the identity of an individual", correlation 0.92),
which a maintainer can prune. Full per-class tables, per-hypothesis exemplars, and error cases with
their top-activating hypotheses are in `experiments/results/processed/`.

## 6. Analysis

**Where HV wins, and why.** HV is strongest exactly where its assumptions hold: a semantically clean
taxonomy the NLI model can measure, and a low label budget where an imported prior beats a
data-hungry estimate. The prior-aggregation head is the clearest expression of this — at $N=0$ it is
a zero-shot ensemble, and it dominates the 1–3/class band on TREC. **Where it fails.** On broad-topic
data the bare NLI prior already near-ceilings, so the learned head is redundant; on fine-grained
many-class data a thin pool cannot cover the label space. Both are diagnosable and, for the latter,
fixable by scaling the pool. **The flexible-head low-N failure** is real but must be reported
carefully: a single gradient-boosting head *collapses to a constant class* below ~10 rows (a
degenerate artifact), whereas a random forest overfits *gracefully* and still beats chance — we use
the latter and flag the former. **Generation, not selection, is the lever within the encoder's
reach**: evolution and larger pools help only up to a task-dependent saturation, and the frozen
encoder's capacity bounds the achievable accuracy.

## 7. Limitations

NLI-model quality bounds accuracy, and the frozen encoder is the dominant capacity lever. Generation
can produce redundant or spurious probes (we detect but do not yet automatically prune redundancy
outside evolution). Large pools raise inference cost linearly. Low-N evaluation is high-variance; we
use 5–10 seeds with CIs but individual points remain noisy. Some tasks — broad topics at scale, or
lexically-easy problems — are served as well or better by simpler baselines, and a fine-tuned encoder
wins once labels are abundant. The zero-label prior head depends on clean class tags, which generated
pools approximate only noisily. Hypotheses can inherit biases from the proposer or the NLI model, and
importances are associational, not causal. Long documents need chunking, and non-English performance
requires separate evaluation. Our strongest single-dataset result (TREC) is on a semantically clean,
NLI-favorable task; the generality study is the corrective.

## 8. Conclusion

Hypothesis Vectorization turns LLM-generated natural-language hypotheses into an auditable, reusable,
low-label semantic feature space, measured by a frozen NLI model and consumed by classical ML. Its
value is a **Pareto point**: interpretable, low-label-strong, cheap to adapt, LLM-free at inference,
and compatible with existing scikit-learn and text+tabular pipelines — with the honest caveat that in
data-rich, lexically-easy regimes simpler baselines and fine-tuned encoders close or reverse the gap.
The method gives a practitioner a dial they can read: a list of English hypotheses they can inspect,
edit, extend with domain knowledge, and scale to the taxonomy at hand.
