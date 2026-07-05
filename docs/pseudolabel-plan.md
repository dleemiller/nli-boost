# Low-N via LLM pseudo-labeling

## Problem (measured)

At 5 examples/class the committed method collapses to **0.666** on TREC (`trec_lown`), barely above
zero-shot NLI (0.632) and far below full-data (0.964). Two coupled causes:

- **The head overfits.** 64 hypotheses → 128 features on 30 rows; `cv_train 0.804 ≫ test 0.666`. RF /
  HistGBM cannot regularize that ratio.
- **Evolution can't run.** CV permutation-importance over ~5 rows/class is noise, so we disable it
  (`rounds: 0`) — the pool is a raw LM prior, never pruned or refilled against confusion hot-spots.

Both are *data-volume* problems. We have plenty of **unlabelled** text (in the benchmark, the rest of
the train set with labels hidden; in practice, the deployment corpus) — just not labels.

## Approach

Use an **LLM to pseudo-label the unlabelled pool**, then run the *normal* pipeline (evolution + head)
on the K real + M pseudo examples. The frozen encoder scores are cheap and cached; the only new cost
is the one-time LLM labeling pass. Net effect:

- **De-overfit the head:** M ≫ features, so RF/HGB has rows to spare — `cv_train ≈ test`.
- **Unlock evolution:** CV ranking + hot-spot refill now run on enough labelled rows to be stable.
- **Framing:** this is **distillation** of the LLM's task knowledge into a cheap, frozen-encoder,
  *interpretable* model. The ceiling is the LLM labeler's own accuracy; the win is cost / latency /
  interpretability, plus a model that runs without the LLM at inference.

## The validity risk (we have been burned by this exact shape)

Evolution optimizes hypotheses to predict *the labels you hand it*. Pseudo labels are the LLM's — and
the hypotheses are also the LLM's. Optimizing hypotheses against LLM labels can just relearn the LLM's
self-consistent biases and **not transfer to real test** — the same failure mode as grow-then-select
overfitting the CV proxy (NOTES 2026-07-05: held-out went anti-correlated with test). Guardrails:

1. **Test is real and untouched.** Always. Report `pool_cv` on the real held-out only.
2. **Accept/validate on REAL labels.** Pseudo labels supply *volume* (features + head fitting), but
   the evolution keep/prune decision and any head selection should lean on the K real labels (or a
   small real held-out), not the pseudo CV. Volume from pseudo, signal from real.
3. **Weight real ≫ pseudo** in the head (sklearn `sample_weight`), and/or **confidence-filter** the
   pseudo set (keep only high-margin LLM labels) to cap label noise.
4. **Measure the pseudo-label accuracy** against the K real labels (and, for the benchmark only, the
   hidden true labels) so we know the noise floor we are training against.

## Pipeline

1. **Seed:** K real labelled/class (`shots_per_class`).
2. **Label:** LLM classifies the unlabelled pool given task + class definitions + the K real examples
   as in-context anchors; returns label (+ a confidence / logprob if available).
3. **Filter:** keep the top-confidence pseudo labels (threshold or top-M per class to stay balanced).
4. **Featurize:** frozen encoder scores the generated pool on real + pseudo (cached).
5. **Evolve:** CV-prune / hot-spot-refill on real + pseudo, with the accept gate on the real K-shot.
6. **Head:** fit RF/HGB on real + pseudo with `sample_weight` (real ≫ pseudo).
7. **Evaluate:** once, on the real test set.

## Experiment (TREC first — cheap, features cached from `trec_full`)

- 5 real/class; pseudo-label the remaining ~5,400 real TREC train texts (labels hidden).
- **Labeler comparison** (tells us where signal comes from): (a) LLM zero-shot — LLMs are strong on
  question-type, likely ≫ the encoder's 0.63; (b) the 5-shot pool+head self-training (weak,
  circular-ish) as a lower reference.
- **Compare against:** static low-N **0.666** (floor), full-real-label **0.964** (ceiling), the LLM
  labeler's *own* zero-shot accuracy (the distillation ceiling), zero-shot NLI 0.632.
- **Success:** move 0.666 up toward the labeler's accuracy with `cv_train ≈ test` (overfit gone); a
  further bump from turning evolution back on would show pseudo-volume genuinely unlocks it.

### Ablations
- pseudo volume M (100 → all unlabelled) — learning curve;
- confidence threshold (noise vs volume trade);
- real/pseudo `sample_weight` ratio;
- evolution on vs off on the pseudo-augmented set;
- accept gate on real-K vs on pseudo-CV (does the guardrail matter?).

## Implementation (deferred)

The labeler will be a **simple dspy program** — a `Classify(dspy.Signature)` (task +
class_definitions + the K real examples as demos → predicted label, optionally with
self-consistency for a confidence estimate) — living on the training side alongside the proposer, so
inference stays dspy-free. Small and self-contained; **build later** once we commit to the approach.

## Open decisions
- **Labeler model** — reuse the proposer LM (`lm.model`) or a stronger judge? Cost vs quality.
- **Unlabelled source** — benchmark: rest of train; real use: a user-supplied corpus (a new config /
  fit param, e.g. `fit(X, y, unlabeled=...)`).
- **Confidence signal** — LLM self-reported confidence is unreliable; consider self-consistency
  (sample k times, majority + agreement) or encoder-margin agreement as the filter.
- **Where it lives** — a training-side step (CLI `data` + runner) and/or a `HypothesisVectorizer`
  `fit(..., unlabeled=...)` path; keep inference untouched (dspy-free).
