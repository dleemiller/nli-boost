# hypothesis-vectorizer: the method we landed at

Text classification with a **frozen NLI cross-encoder** and **LM-written natural-language
hypotheses as features**. No gradient touches any neural weight; task adaptation lives in
~64 English sentences, an evolutionary refinement loop, and a CV-disciplined classical head.

Every design choice below carries the measurement that forced it (from NOTES.md, 2026-07-03).

```python
# ============================================================================
# INPUTS
# ============================================================================
# D_train, D_val, D_test : labeled texts (val/test never touched until the end)
# classes                : names + ONE-LINE DEFINITIONS   # +TREC clarity; ~free
# encoder                : frozen NLI cross-encoder       # capacity knob: -m -> -l = +5 pts
# proposer_lm            : cheap LM (deepseek-flash)      # ~$0.01 per full fit
# (dedup)                : covariance of score vectors    # drop collinear features, |corr|>0.95

# score(text, hypothesis) -> P(entail), P(contradict)     # 1 forward pass, 2 features
# every (text, hypothesis) score is cached forever — reruns/resumes are ~free


# ============================================================================
# STAGE 1 — GENERATE the initial pool
# ============================================================================
pool = proposer_lm.generate(
    task, class_definitions,                  # definitions, not bare names
    examples = stratified_sample(D_train, per_class=3),
    n = 64,                                   # ~30 useful directions exist; 64 gives slack
    rules = "single declarative sentence about 'the text'; verifiable from the
             text alone; affirmative; vary specificity",
)
pool = covariance_dedup(textual_dedup(pool))   # drop collinear features (redundant score vectors)


# ============================================================================
# STAGE 2 — EVOLVE the pool  (saturates fast: ~2 useful rounds, measured)
# ============================================================================
sub = stratified_sample(D_train, 800)          # ranking needs no full-matrix precision
best_heldout, patience = -inf, 2

for round in range(6):                                          # cap; patience exits sooner
    X = score_matrix(sub, pool)                                 # cache-through

    # ---- rank by STABILITY, not by one split -------------------------------
    # single-split importance churned 50% across seeds: half the kills were
    # coin flips. k-fold permutation importance + cross-fold sign agreement.
    order, stability, heldout_errors = cv_importance(X, y_sub, folds=4)
    heldout_acc = 1 - len(heldout_errors) / len(sub)            # free per-round metric

    # ---- prune CONFIDENT deaths only ---------------------------------------
    # ambiguous hypotheses get another round instead of dying to split noise
    dead = [h for h in pool if stability[h] == 0.0][:len(pool)//2]
    survivors = pool - dead

    # ---- tell the LM WHY each one died (so it fixes concept vs phrasing) ----
    for h in dead:
        h.why = ("encoder cannot detect this property"     if scores_constant(h)
            else "helps on some splits, not others"        if 0 < stability[h] < .5
            else "redundant with a kept hypothesis"        if max_corr(h, survivors) > .9
            else "detectable but no held-out predictive value")

    # ---- confusion HOT SPOTS, not single examples --------------------------
    # single examples invite hypotheses overfit to one text. Group mutually
    # confused classes (connected components of the thresholded confusion
    # graph), show a BATCH of errors per group, counts-only for the rest.
    hotspots = connected_components(confusion_graph(heldout_errors))
    evidence = [batch_of_errors(g, k=8) for g in hotspots[:3]] + counts_only(rest)

    # ---- refill ------------------------------------------------------------
    refills = proposer_lm.refill(survivors, dead_with_reasons, evidence,
                                 n=len(dead))
    refills = covariance_dedup(textual_dedup(refills))
    pool = survivors + refills

    # instrumentation: next round, measure each refill's STANDALONE AUC on the
    # hot-spot pair it targeted. (Once the head interpolates train, marginal
    # importance cannot see a good new feature; standalone alignment can.
    # Measured decay 100% -> 20% -> 0%: generation saturates by round 2.)

    # ---- stop on plateau ---------------------------------------------------
    if heldout_acc <= best_heldout: patience -= 1
    else:                           best_heldout, patience = heldout_acc, 2
    if patience == 0: break


# ============================================================================
# STAGE 3 — HEAD, chosen honestly  (the systemic overfitting fix: +2 pts)
# ============================================================================
X_train = features(D_train, pool)              # P(entail) + P(contradict) per hypothesis
if lexical.kind != "none":                     # OPTIONAL lexical channel (when macro-F1 matters):
    X_train += tfidf_svd(D_train, dims=128)    # fit on TRAIN ONLY, concat at head stage only.
                                               # Measured: acc within noise (−0.2 TREC, +0.8 AGN)
                                               # but macro-F1 +1.4/+0.7 — helps rare classes.
                                               # wordllama variant dominated by tfidf on short texts.
head = argmax_over_grid(                       # RF / HistGBM x regularization settings
    cv_accuracy(model, X_train, y_train, folds=4)   # CV ON TRAIN ONLY.
)                                              # never pick the head by val or test —
head.fit(X_train, y_train)                     # best-of-6-on-test inflated us +2.2 pts


# ============================================================================
# STAGE 4 — EVALUATE ONCE, REPORT HONESTLY
# ============================================================================
report(head.score(features(D_test, pool)))     # one test evaluation, pool_cv is the headline
audit(pool, head)        # reward-hacking checks: val-gain collapse, length correlation
diagnose(pool, head)     # error decomposition: coverage gap / redundancy / fit gap / label noise
# Any two runs on the SAME test set: paired McNemar (exact binomial on discordant
# pairs) + Wilson CIs. Seed bands catch across-fit variance; this catches whether a
# single-run A/B delta clears the noise floor. Measured caution: on TREC-500 the
# evolve loop's +0.8 over the static pool is p=0.644 — inside the noise.


# ============================================================================
# STAGE 5 (optional) — FINALIZE WITH A BIGGER ENCODER
# ============================================================================
# Hypotheses transfer across encoders; only re-score and refit the head.
# Measured on TREC: same 58 hypotheses, -m -> -l = 0.896 -> 0.946.
X_l = score_matrix_with(bigger_encoder, D_train, pool)
head_l = cv_selected_head(X_l);  report_once(test)
```

## Where budget scales (measured)

| lever | effect | status |
|---|---|---|
| more hypotheses / rounds / selection search | ~0 beyond round 2 | **falsified** — don't spend here |
| encoder size (`-m`→`-l`, same hypotheses) | **+5 pts** | the capacity knob |
| training data (2k→4.4k) | +2.4 | bounded by dataset |
| CV-selected head (vs fixed defaults) | +2 | in the method |
| honest reporting (vs best-head-on-test) | −2.2 of illusion | in the method |
| lexical channel (TF-IDF→SVD concat at head) | ~0 acc, **+0.7–1.4 macro-F1** | optional — when rare classes matter |
| multi-pool / multi-encoder ensembling | untested, est. +0.5–1.5 | next candidates |

## Results under this method (TREC-6, 2k train, seeds 7/17)

pool_cv **0.916 / 0.938** test (mean 0.927), macro-F1 0.88–0.93, ~7 min and <$0.01 per fit,
0 abnormal LM calls. With `-l` finalization: **0.946** (measured at 2k, old protocol).
Reference points: TF-IDF 0.828, zero-shot NLI (`-m`) 0.356, fine-tuned BERT ~96–97.

## What the artifacts let you check

- `runs/<name>/log.jsonl` — every hypothesis tried/pruned/refilled per round, with reasons,
  held-out accuracy, refill target-AUCs
- `hypothesis-vectorizer audit` — reward-hacking signatures (val-gain collapse, length correlation)
- `hypothesis-vectorizer diagnose` — error decomposition (coverage / redundancy / fit gaps / label noise)
- the NLI score cache — every rerun and post-hoc analysis is ~free
