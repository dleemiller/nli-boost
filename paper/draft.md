# Hypothesis Vectorization: Interpretable Low-Label Text Classification with LLM-Generated NLI Features

**Status:** working draft. Numbers marked _[prelim]_ are from the current experiment run and will
be finalized; sections marked _[pending]_ await experiments in progress. See
`paper/experiment_notes.md` for the live experiment log and `paper/method.md`,
`paper/related_work.md` for the expanded Method and Related Work.

## Abstract

We introduce **Hypothesis Vectorization**, a method that represents a text by its natural-language
*entailment profile*: an LLM proposes a compact pool of readable hypotheses, a frozen NLI
cross-encoder scores how strongly each input entails or contradicts each hypothesis, and a
classical model learns the task over the resulting interpretable feature matrix. The representation
imports semantic prior knowledge from the proposer and the NLI model without any fine-tuning, so
each feature is a plain English sentence whose learned weight is directly auditable, and inference
requires **no LLM** — only the frozen encoder. We study the method through learning curves across
label budgets and find a clean regime structure: at 1–3 examples/class a **label-free
prior-aggregation head** over the hypotheses is strongest; as labels grow a flexible classical head
takes over; and the best configuration is a function of the label budget. On TREC-6 the method
dominates TF-IDF and sentence-embedding baselines at every budget and reproduces strong full-data
accuracy, while remaining fully interpretable. We position Hypothesis Vectorization not as a
state-of-the-art accuracy claim but as a compelling Pareto point among low-label performance,
interpretability, cheap adaptation, no-inference-LLM deployment, and compatibility with classical
and text+tabular pipelines.

## 1. Introduction

Organizations routinely need text classifiers for narrow, evolving taxonomies with few labeled
examples, in settings where the model must be auditable and cheap to serve. The dominant options
each sacrifice something: fine-tuning an encoder is data-hungry and produces an opaque model;
dense-embedding probes are cheap but their dimensions are uninterpretable; direct LLM
classification is strong but incurs per-inference cost and latency, drifts with prompt changes, and
is hard to audit; and classical lexical models, though strong, key on surface forms and miss
semantics.

Hypothesis Vectorization occupies the middle. An LLM writes a pool of short declarative
**hypotheses** about the input ("The text asks where something is located."); a frozen NLI model
measures each input against each hypothesis; and a classical head estimates the label from the
scores. The LLM's semantic prior is captured *once*, at authoring time, and then frozen into a
readable feature basis. This yields three properties a practitioner cares about: (i) every feature
is a sentence, so importances and errors are explainable and experts can add or veto probes; (ii)
the semantic prior substitutes for labeled data, which helps most when labels are scarce; and (iii)
serving needs no LLM.

We frame the method as representation learning with an explicit factorization — the **LLM proposes a
semantic basis**, the **NLI model is a fixed measurement operator**, the **classical head is the
task estimator**, and pruning/evolution **searches over semantic features rather than neural
weights**. This abstraction organizes the paper.

**Contributions.**
1. A method for turning LLM-generated natural-language hypotheses into NLI-scored, interpretable
   semantic features for text classification, with a reproducible sklearn-compatible implementation
   that uses no LLM at inference.
2. A **low-label learning-curve study** showing a regime structure in which a label-free
   prior-aggregation head over the hypotheses is best at the smallest budgets and a flexible head
   overtakes it as labels grow — *the optimal configuration is a function of N*.
3. **Ablations** over hypothesis generation, pool size, NLI encoder, deduplication, score channel,
   and classifier head, including honest negative results (generation saturates quickly; the
   encoder is the capacity lever). _[pending]_
4. **Text+tabular** experiments evaluating hypothesis features by their *marginal* value over
   structured metadata, for practical pipelines. _[pending]_
5. An **interpretability analysis** using the readable hypothesis features: global and per-class
   importances, redundancy, stability, and qualitative error explanations.

## 2. Related Work

See `paper/related_work.md` for the full treatment. In brief, HV generalizes **zero-shot NLI
classification** (Yin et al., 2019) from one template per class to a *learned head over a pool of
many LLM-generated probes*; it is a **concept bottleneck** (Koh et al., 2020) whose concepts are
generated in natural language and scored by a frozen NLI model rather than hand-annotated and
learned; it relates to **weak supervision** (Ratner et al., 2017) but replaces brittle
code/regex labeling functions with semantic NLI probes; and it keeps the semantic prior of **direct
LLM classification** while removing the LLM from the inference path. Unlike dense-embedding probes,
its dimensions are readable.

## 3. Method

See `paper/method.md` for the formal treatment. Let $D=\{(x_i,y_i)\}$ and a hypothesis pool
$H=\{h_j\}$. A frozen NLI model $f_\theta(x,h)=(p_e,p_n,p_c)$ scores each pair, and the feature map
$\phi(x)=[g(f_\theta(x,h_1)),\dots]$ selects a **score channel** $g$ (entailment; entailment and
contradiction, the default; or their contrast). A classical estimator predicts
$\hat y=g_\psi(\phi(x))$. Because each dimension is a named hypothesis, $g_\psi$'s weights are
interpretable. Hypotheses are LLM-generated from the task, class definitions, and a *training-only*
sample, optionally anchored by fixed expert hypotheses, deduplicated (behavioral covariance at high
$N$, text-space STS at low $N$), and optionally evolved by CV-importance pruning + confusion-driven
refill. At inference the pool is fixed, the NLI model frozen, and no LLM runs. Head family and
regularization are chosen by CV on train only and evaluated on test once (`pool_cv`); two-run
deltas use paired McNemar with Wilson intervals.

**The low-N head.** At 1–5 examples/class a flexible RF/HGB head overfits; we introduce a
**prior-aggregation head** that scores class $k$ as the mean entailment of the hypotheses tagged for
$k$ and predicts the arg-max — at $N=0$ exactly a zero-shot NLI *ensemble*, richer than a single
template — with a strongly-regularized reweighting as $N$ grows.

## 4. Experiments

**Datasets.** TREC-6 (question type); AG News (topic); SST-2 (sentiment); intent classification
(Banking77 / CLINC150); and CFPB consumer complaints for text+tabular. _[TREC complete; others in
progress.]_ **Protocol.** One fixed test set per dataset; the training set is subsampled to $k$
examples/class for $k\in\{1,2,3,5,10,20,50,100,\text{all}\}$, resampled over 10 seeds; test is never
subsampled and never inspected during hypothesis generation, pruning, or head selection.
**Baselines.** TF-IDF (word/char/union)+logreg; sentence-embedding (MiniLM)+logreg; zero-shot NLI
(class templates). **Metrics.** accuracy, macro/weighted-F1, per-class P/R/F1, ECE calibration, with
bootstrap CIs over seeds; NLI-call counts and wall time; zero inference-time LLM calls by
construction. **Models.** frozen NLI encoder `finecat-nli-l`; proposer DeepSeek-v4-flash via
OpenRouter.

## 5. Results

### 5.1 Low-label learning curves (RQ1)

On TREC-6 (finecat-nli-l, 10 seeds, fixed 500-example test), test accuracy: _[prelim]_

| examples/class | 1 | 2 | 3 | 5 | 10 | 20 | 50 | 100 | all |
|---|---|---|---|---|---|---|---|---|---|
| HV prior (label-free) | **0.594** | 0.594 | 0.594 | 0.594 | 0.594 | 0.594 | 0.594 | 0.594 | 0.594 |
| HV prior (reweighted) | 0.556 | **0.629** | **0.642** | 0.660 | 0.680 | 0.679 | 0.686 | 0.700 | 0.862 |
| HV + RF head | 0.474 | 0.597 | 0.642 | **0.682** | **0.753** | **0.802** | **0.849** | **0.892** | **0.954** |
| HV + L2-logreg head | 0.364 | 0.517 | 0.547 | 0.604 | 0.712 | 0.725 | 0.746 | 0.784 | 0.912 |
| zero-shot NLI (templates) | 0.428 | 0.428 | 0.428 | 0.428 | 0.428 | 0.428 | 0.428 | 0.428 | 0.428 |
| MiniLM embeddings + logreg | 0.301 | 0.400 | 0.431 | 0.486 | 0.574 | 0.649 | 0.703 | 0.742 | 0.856 |
| TF-IDF (word+char) + logreg | 0.307 | 0.374 | 0.402 | 0.426 | 0.486 | 0.572 | 0.634 | 0.713 | 0.870 |

(Figure: `experiments/results/figures/trec_learning_curve_accuracy.pdf`.) Three observations. (i)
The **regime crossover**: the label-free prior head is best at 1/class, the reweighted prior at
2–3/class, and the flexible RF head from ~5/class up. (ii) HV **dominates the lexical and embedding
baselines at every budget** on this dataset, and still leads at full data (0.954 vs ~0.86–0.87).
(iii) The **multi-hypothesis prior beats single-template zero-shot NLI by +16.6 points using zero
labels** (0.594 vs 0.428), isolating the value of a *pool* of probes over one template per class.
These use a **hand-written expert pool**; the LLM-generated-pool comparison is §5.2.

### 5.2 Generated vs expert pools, static vs evolved (RQ4) _[prelim]_

We generate a static pool (64 hypotheses, from 5 examples/class) and an evolved pool (62
hypotheses, from 50/class; evolution stopped at 2 rounds on a held-out plateau) with
DeepSeek-v4-flash and drop both onto the TREC learning curve (figure
`trec_generated_accuracy.pdf`). Three findings. (i) **LLM-generated pools are a stronger feature
basis than the hand-written expert pool for learned heads**: evolved-pool + RF leads from 3/class
upward (0.703→0.894 over 3–50/class vs the expert pool's 0.642→0.849), as the richer 64-hypothesis
basis gives the classifier more to exploit than 24 expert hypotheses — the central claim that
*LLM-generated hypotheses form a useful semantic basis* holds directly. (ii) **Evolution provides a
small, consistent lift over the static pool** (evolved ≥ static at nearly every budget, clearest for
the linear head at low N) and saturates in two rounds, matching the method's known
generation-saturation behavior — evolution is a refinement, not the capacity lever. (iii) **The
zero-label prior head still favors the hand-written expert pool** (0.594 vs 0.544 evolved / 0.484
static), because its class tags are clean whereas the generated pools' tags are derived from the
training sample; evolution improves the generated pool's tag-ability. This isolates *where*
hand-authoring still helps — supplying clean class tags for the no-label prior — versus where
generation wins (a richer basis for any learned head).

### 5.3 Full-data comparison and other datasets (RQ3) _[in progress]_

### 5.4 Text + tabular marginal value (RQ5) _[pending, CFPB]_

### 5.5 Cost and latency

HV issues **zero LLM calls at inference**; per-example cost is $m$ cached NLI forward passes. Pool
generation is a one-time LLM cost (~\$0.01–0.07 per pool). _[table pending]_

### 5.6 Interpretability (RQ2)

On TREC, the top hypotheses by permutation importance are readable and class-aligned, e.g. "The
text asks where something is located." (LOC, importance 0.094), "The text can be answered with a
numeric value." (NUM, 0.073), "The text asks for the definition of a term." (DESC, 0.031), with low
cross-fold variance. The redundancy detector flags near-duplicate probes (e.g. "asks for the name of
a person" vs "asks for the identity of an individual", $r=0.92$). Full per-class tables, exemplars,
and error explanations: `experiments/results/processed/trec_expert_inspect_l/`.

## 6. Analysis _[in progress]_

Where HV works best (semantically clean taxonomies, low labels), where it should struggle
(TF-IDF-favorable topic tasks at high N; pragmatics-heavy sentiment), the flexible-head low-N
overfitting failure and its fix, hypothesis redundancy, and the encoder-as-capacity-lever finding.

## 7. Limitations

NLI-model quality bounds accuracy; generation can produce redundant or spurious probes; large pools
raise inference cost; low-N evaluation is high variance (we use 10 seeds + CIs); some tasks are
better served by lexical features; hypotheses can inherit proposer/encoder biases; feature
importances are associational, not causal; long documents need chunking; non-English performance
needs separate evaluation. The current strong TREC result is on a semantically clean, NLI-favorable
dataset — we explicitly avoid over-generalizing and report harder datasets alongside.

## 8. Conclusion

Hypothesis Vectorization turns LLM-generated natural-language hypotheses into an auditable,
reusable, low-label semantic feature space measured by a frozen NLI model and consumed by classical
ML. Its value is a **Pareto point**: interpretable, low-label-strong, cheap to adapt, LLM-free at
inference, and compatible with existing sklearn and text+tabular pipelines — with the honest caveat
that in data-rich, lexically-easy regimes simpler baselines close the gap.
