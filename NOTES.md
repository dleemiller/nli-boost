# Experiment notes

Running log of results reviews, qualitative reward-hacking checks, and config decisions.
(Appended by the 15-minute review cron and by hand.)

## 2026-07-03 — setup decisions

- **NLI model for prototyping switched to `dleemiller/finecat-nli-m`** (0.1B). Benchmark on
  AG News (500 texts × 4 hypotheses, GPU shared with a running job):
  `-m` 361 pairs/s / 0.868 zero-shot acc; `-s` 950 pairs/s / 0.746; `-xs` 2133 pairs/s / 0.560.
  `-m` keeps near `-l` quality (~0.87-0.89) at several× effective speed. `-l` (0.4B) reserved for
  final evals; NLI-model-size is itself an ablation axis. Cache keys include the model, so results
  never mix.
- `ag_news_pool` ran with `-l` (in flight before the switch); all later runs use `-m`.
  `*_baselines_m.yaml` configs exist so every comparison table can be single-model.
- NLI scoring now chunk-commits to cache every 8192 pairs (kill-safe) and prints progress.
- Live per-node/per-round progress: `runs/<name>/progress.jsonl` + `nli-boost status`.
- Reward-hacking guardrails: `nli-boost audit <run_dir>` (val-gain collapse, length-correlation,
  degenerate thresholds, branch sample texts); GEPA metric logs every eval to `*.evals.jsonl`.

## GEPA paper review (arXiv 2507.19457) — ideas borrowed/adapted

GEPA's four mechanisms and how they map here:

1. **Minibatch-then-full evaluation** → *borrowed* as two-stage candidate screening
   (`screen_size=300` stratified texts estimate every candidate's gain; only `screen_top=3` get
   full node scoring; screened-out candidates stay visible to reflection with estimated gains).
   ~3× NLI savings at large nodes.
2. **Acceptance test before a mutation enters the pool** → *borrowed* as the boost `val_accept`
   gate: a stump is kept only if validation loss does not rise; rejected stumps are permanently
   blocked and fed back to the proposer with a REJECTED note (also breaks the LM-cache determinism
   deadlock). Doubles as a direct reward-hacking guard: train-gain artifacts die at the val gate.
3. **System-aware Merge (crossover of complementary lessons)** → *adapted* as the tree's
   `reuse_bank`: hypotheses with gain ≥ 2×min_gain discovered at any node are re-offered as free
   candidates at every other node (mostly cache hits). Cross-branch recombination without LM cost.
4. **Instance-level Pareto frontier** (candidate kept if best on ≥1 instance; sampling ∝ instances
   won) → *partially present*: multiclass boosting's shared pool with per-class stump selection is
   exactly per-instance (per-class) frontier selection. NOT yet adopted: a global frontier over
   texts. Future ablation: per-class-pair frontier at tree nodes; per-text hardness tracking fed
   into proposal contexts.

GEPA ablation notes relevant to us: Pareto selection beats select-best mainly by preserving
diversity — our analogue is keeping multiple high-gain hypotheses alive in the bank rather than
only the node winner. Feedback engineering matters more than optimizer machinery — our
`format_tried` already gives per-candidate gains + entailed-branch histograms; consider adding
explicit "classes still mixed: X vs Y" phrasing.

## Reviews

### 2026-07-04 — GEPA instruction tuning, redesigned + PRE-REGISTERED (feasibility probe)

Rebuilt GEPA around the committed pool method (the first attempt was for the tree proposer and
overfit one dataset: +5.4 TREC / −3.3 AG News). New pieces committed: reward.py (composite
pool-quality reward) + gepa_tune.py (optimize the GeneratePool INSTRUCTION, flash student, pro
reflection+judge) + `nli-boost gepa-tune`. Reward design and validation:

- Terms (each ties to a finding): noise-averaged held-out CV skill (primary; averaging beats the
  ~0.003 HGB jitter so GEPA can't hack it), effective-rank diversity (anti-collapse — the lever
  Lee identified), length/vacuity anti-hack penalties (from diagnostics.py), optional pro judge.
  Cross-dataset aggregation = GEOMETRIC mean (craters if any dataset tanks → generalization pressure).
- Reward VALIDATED on cached pools (no GPU): ranks diverse > collapsed on trec/ag_news/sst2
  (+0.014/+0.043/+0.064). Sharpest check — sst2 collapsed sub-pool has EQUAL raw CV acc (0.9353 vs
  0.9338) but scores lower (0.770 vs 0.834) via eff rank 8.3 vs 16 + 2 length artifacts. The reward
  penalizes fragile collapse before it costs accuracy; pure-accuracy optimization would miss it.

Pre-registration (feasibility probe, before launch):
- Setup: tune on trec:7 + sst2:7 (question-type + sentiment, maximally different tasks); HOLD OUT
  ag_news for the accept gate. pool 28, sub 400, -m encoder, max_calls 40, pro judge (reasoning off)
  + pro reflection, flash student.
- Bar: baseline = current instruction's reward geo-mean (printed at run start).
- DECISION RULE: PASS if tuned geo-mean > baseline + 0.01 AND the evolved instruction reads as
  dataset-agnostic + sensible (inspect models/proposer_instruction.evals.jsonl + the printed
  instruction). Then scale up + full-method McNemar accept gate on held-out ag_news at -l (adopt
  only on a significant, transferring gain). FAIL if no reward gain or dataset-specific overfit →
  GEPA instruction tuning refuted for this method; document and stop.
- Honest upside: bounded. Encoder is the capacity ceiling (only significant win, p=0.024) and flash
  pools ≈ pro pools (review #46); this optimizes pool quality up to that ceiling. Value is a
  reusable, noise-robust, anti-hacking instruction-optimization loop + any non-obvious gains.
- Cost/safety: est. ~30–45 min shared GPU (32×400 pairs/eval × 40 on -m), <$1 LM, trivial RAM
  (serial CV, 400×~56 matrices). Justification for >40-min soft cap: one-time methodology gate.
  GPU scoring lock-serialized, CV serial (n_jobs=1), GEPA num_threads=2 — no process forks, no OOM.
- Interruptible: stops on max_calls OR timeout_min (default 40) OR `touch <out>.stop` OR Ctrl-C,
  all keeping best-so-far; log_dir checkpoints every iteration so re-running the command resumes.
  So it never needs to run to 100% — safe to stop when the GPU is needed for training.

### 2026-07-04 — hourly: plateau-epsilon tune REFUTED by trajectory calibration (no code change)

Pre-registered expectation (from the backlog): held-out crept +1e-4/round forever, so the plateau
epsilon (hardcoded 1e-4 at evolve.py:264) should be raised above the round noise (~0.003) to stop
the loop wasting rounds. **Tested offline on all 10 evolved runs' log.jsonl held-out trajectories
(free, no GPU) — and the premise is wrong.** Decision rule set before looking at candidate epsilons:
adopt only if a raised epsilon stops early WITHOUT sacrificing >0.003 held-out on any run.

- **Round noise floor, measured for free:** trec rounds 2-4 have IDENTICAL pools (0 prunes, 0
  refills) yet held-out reads 0.8639 → 0.8627 → 0.8652 — a ±0.003 wobble from HistGBM OpenMP
  thread-nondeterminism (float reduction order), not signal. So 1e-4 is 30× below the noise; the
  plateau check treats jitter as improvement and runs to the round cap. That part of the premise holds.
- **But improvements are non-monotonic with real LATE JUMPS, not a creep.** Simulating evolve()'s
  stop logic over the logged accs:
  - trec_pro_l (-l, the run that genuinely climbs): +0.0012, +0.0013, **+0.010 (round 3)**, ... —
    eps=0.003/patience=2 stops after round 2 at 0.8864, cutting the jump to 0.8964. **−0.0087 lost.**
  - ag_news: +0.0013, **−0.0063 (dip)**, **+0.0125 (round 3)**, ... — eps=0.003 stops after round 2
    at 0.8625, missing the climb to 0.8788. **−0.0163 lost.**
  The real gains arrive at round 3 AFTER a stagnant/declining round 2. Any epsilon big enough to
  beat the noise (≥0.002) + patience 2 cuts these jumps. Raising epsilon HURTS exactly the runs
  evolution helps. **Refuted; epsilon stays 1e-4 / run-to-cap.**
- The only strictly-safe stop is "pool static (0 prunes AND 0 refills) for patience rounds" — a
  frozen pool cannot jump, so remaining variation is provably noise. But that triggers only on trec
  (saves rounds 2-4, which are cache-hit CPU re-ranking, ~seconds) and correctly never fires on
  ag_news/trec_pro_l (pool changing every round). Savings are the cheap rounds; the expensive
  refill rounds are exactly the productive ones. Not worth the added branch + validation run.
- **Refines the "evolution not significant (p=0.644)" finding:** evolution's gains are real but
  small and concentrated in one or two rounds, swamped by 500-example test noise at -m. The place
  it clearly pays is -l (trec_pro_l round-3 +0.010 held-out) — consistent with encoder-relative
  saturation. Backlog item closed as refuted; script kept at scratchpad/plateau.py.

### 2026-07-04 — peer method-doc (bsmith) cross-read + significance testing added

Read an independent method doc on the same problem (frozen NLI encoder + LM-written
hypotheses). Extracted what's additive, what we already have, and what we jointly falsified.

- **Style-partitioned generation (their headline +2.2 pts): targets a deficiency we don't have.**
  Their claim is that ONE general prompt mode-collapses onto label paraphrases (effective rank
  8.3/64, 10 near-dup feature pairs |corr|>0.9); splitting the same call budget across 5 style
  prompts de-collapses it (10.3/64, 5 pairs) for +2.2. Measured OUR pools (P(entail) columns,
  cache-only): trec_pool **22.9/64 (3 near-dups)**, ag_news_pool **27.1/64 (0)**, sst2_pool
  15.6/64 (0). Our `GeneratePool` prompt already bakes in the diversity instruction their general
  prompt lacked ("cover every class from multiple angles… contrastive hypotheses… vary
  specificity"), so we sit ~2.8× less collapsed than the baseline their win recovers from. Their
  mechanism doesn't apply to us. NOT adopting; measurement recorded so we don't chase it later.
- **Significance discipline (McNemar + discordant counts + CIs): the real import — adopted.** We
  reported seed bands (across-fit variance) but never a paired test on a FIXED test set. Built
  `nli-boost compare runA runB` (src/compare.py): reconstructs both runs' test predictions from
  the NLI cache + saved head params (falls back to the CV grid for pre-head-saving runs), exact
  binomial McNemar on discordant pairs, Wilson CIs, refuses mismatched test sets. Applied it to
  three A/Bs we'd never significance-checked:

  | comparison | Δacc | discordant (A-only / B-only) | McNemar p | verdict |
  |---|---|---|---|---|
  | evolution vs static pool (TREC, n=500) | +0.008 | 42 (19 / 23) | 0.644 | **not significant** |
  | lexical on/off (AG News, n=2000) | +0.0075 | 55 (20 / 35) | 0.058 | not significant (borderline) |
  | lexical on/off (TREC frozen pool, n=500) | −0.002 | 23 (12 / 11) | 1.000 | not significant |
  | maxed vs baseline (trec_pro_l vs trec, n=500) | +0.026 | 29 (8 / 21) | **0.024** | **SIGNIFICANT** |

  The maxed comparison is the first clean win under McNemar: `-l` encoder + pro proposer + fresh
  pool beats the `-m`/flash baseline decisively (fixes 21, breaks 8). Confirms the METHOD.md
  scaling story — capacity is the lever, selection is not. Caveat: it bundles 3 changes; prior
  evidence (review #46: pro pool ≈ flash pool) attributes the gain to the encoder. Clean
  attribution would need a frozen-pool -m→-l swap.

  - **The evolution result is the one that stings:** on TREC-500 the whole evolve loop (prune +
    refill) does NOT beat the raw generated pool at p<0.05 — it fixes 23 test items and breaks 19.
    Consistent with the measured saturation (-m saturates ~round 2) and with our own finding that
    generation, not selection, is the lever. Does NOT mean evolution is worthless (it earns its
    keep at -l where the encoder can measure deeper, and on harder datasets), but on easy/small
    test sets its headline gain is inside the noise floor. This is exactly the sub-standard-error
    "win" the peer doc warns about, and we now have the instrument to catch it going forward.
  - **Lexical AG News p=0.058 refines last cycle's verdict:** the +0.75 pt gain is borderline, not
    clean — 35 items fixed vs 20 broken, just shy of significance on 2000 test examples. Strengthens
    "optional macro-F1 channel, not a default." The TREC lexical A/B is flatly null (p=1.0).
- **Jointly falsified (independent confirmation):** their overgenerate-2N + MMR/diversity selection
  is dead at 2× cost — matches our mRMR≈importance-only finding. Their reflection saturates ~round
  2 with a val-gate — matches our plateau/patience early-stop. Two labs, same negatives.

Action items surfaced: (1) going forward, every single-run A/B verdict should carry a McNemar p,
not just a point delta — retro-applying it already downgraded two "wins" to noise. (2) The
evolution-not-significant-on-TREC result argues for testing the loop where it should matter (harder
datasets, -l encoder) rather than assuming the +0.8 is real. (3) `report` could optionally show
Wilson CIs next to each accuracy so the noise floor is always visible.

### 2026-07-04 — lexical verdicts: tfidf mixed (F1 +1.4), wordllama dominated

- trec_lex_wordllama: 0.9100 / F1 0.8895 — worse than tfidf_svd on every axis. Mechanism:
  TREC's signal is specific short-question vocabulary, which TF-IDF keeps sharp and pooled
  static embeddings blur. Dropped for short-text datasets; may deserve a retry on 20NG
  (long documents) if the channel survives at all.
- Adoption rule status: neither variant clears ≥ +1 accuracy. tfidf's +1.4 F1 / +1.0 cv_train
  earns the AG News check (running, frozen pool, LM-free): if F1 gain replicates there, the
  channel becomes a "when macro-F1 matters" option in METHOD.md rather than a default.

### 2026-07-04 — lexical channel CLOSED: F1 option, not an accuracy lever

- **ag_news_lex: pool_cv 0.8985 / F1 0.8985 vs baseline 0.8910 / 0.8911** (+0.75 acc, +0.74 F1,
  cv_train 0.8965 vs 0.8935 — CV predicted the gain honestly). Notable: the first thing to move
  the AG News number past the presumed 0.890 label-noise wall, though still under the +1 pt bar.
- **Verdict across both datasets:** adoption rule (≥ +1 pt accuracy on 2+) FAILS (−0.2 TREC,
  +0.75 AGN). But the macro-F1 gain REPLICATES (+1.4 TREC, +0.74 AGN) and cv_train rose on both —
  per the registered follow-up, tfidf_svd enters METHOD.md as an *optional* channel "when
  macro-F1 / rare classes matter", not a default. wordllama dropped for short texts (dominated
  by tfidf on every axis: 0.9100/0.8895 on TREC); possible retry only on 20NG long docs.
- Cost of the option: ~0 (sklearn fit, no LM, no encoder pairs). Question closed; follow-up
  ideas (wordllama clustering, lexical-aware evolution folds) YAGNI'd unless a dataset demands it.

### 2026-07-04 review #47 (cron) — first lexical verdict

- **trec_lex_tfidf_svd: pool_cv 0.9180 / F1 0.8965 vs baseline 0.9200 / 0.8822** — accuracy in
  the "subsumed" band (−0.2, noise), but macro-F1 +1.4 and cv_train +1.0 (0.888 vs 0.878): the
  lexical channel helps rare classes and the honest CV estimate without moving headline
  accuracy. Mixed — per the adoption rule (≥ +1 pt accuracy) this does NOT qualify, but the F1
  gain earns the AG News/SST-2 check before closing the question. wordllama variant running.

### 2026-07-04 — lexical channel experiments (Lee's idea), pre-registered

Static lexical features concatenated with hypothesis features at the head stage (flag-gated,
fit-on-train-only, evolution untouched). Both runs REUSE runs/trec's exact pool — the only
variable is the channel. Motivation: TF-IDF standalone = 0.828 TREC / 0.565 20NG; if
complementary to NLI features, concat pays.
- trec_lex_tfidf_svd (TF-IDF → SVD 128) and trec_lex_wordllama (static embeddings, 128 dims)
- Baseline: trec pool_cv 0.9200. Registered: complementary → 0.925-0.94; subsumed → ~0.92.
  Adoption rule: ≥ +1 pt on 2+ datasets before it enters METHOD.md; AG News (0.890 wall) is
  the negative control to run next if TREC is positive.
- Follow-up idea noted: include the lexical channel in evolution's fold models so confusion
  evidence targets what lexical can't already solve.

### 2026-07-04 review #46 (cron) — idle; qualitative check of trec_pro_l pool

- Nothing running. trec_pro_l pool read: the pro proposer produces the same semantic+wh-word
  mix as flash ("The text asks for the full form of an initialism.", "begins with 'Who'") —
  consistent with proposer quality not being the binding variable; the -l gain came from the
  encoder. Head chosen: HistGBM lr=0.12 l2=0.3 (heavier regularization at -l — CV adapting).
- All work complete and judged. The 15-minute review cron is now redundant with the hourly
  improvement cron and can be deleted.

### 2026-07-04 — ALL REGISTERED BANDS HIT: method cross-validated + maxed run

- Cross-dataset (committed code, honest protocol): ag_news 0.8910/0.8895 (band 0.885-0.895 ✓,
  seed spread 0.15 pts), sst2 0.9404/0.9404 (band 0.94-0.95 ✓, identical across seeds),
  trec 0.9200 (band ✓; pre-rewrite seeds 0.916/0.938). External validity CONFIRMED.
- **trec_pro_l (pro proposer / large STS / -l encoder): 0.9460 / F1 0.9248** (band 0.93-0.96 ✓),
  26 min, $0.07, 0 abnormal. Matches the old -l re-score (0.946) with a fully fresh native pool.
- **New scaling-law observation:** with -l, evolution did NOT saturate — held-out climbed through
  round 4 (0.884→0.898) with productive prune/refill every round, vs -m's collapse by round 3.
  The generation well is as deep as the encoder can measure: saturation is encoder-relative.
  This upgrades the METHOD.md scaling story and motivates the two-encoder union experiment.
- Day complete: idea → method → measured defects → fixed loop → pre-registered validation →
  replication → clean commit (f5e205f) → cross-dataset + maxed confirmation.

### 2026-07-03 review #45 (cron) — first cross-dataset verdict

- **ag_news s7: pool_cv 0.8910 / F1 0.8911 — INSIDE the registered 0.885-0.895 band.** The
  AG News label-noise ceiling holds under the committed code; the old protocol was not masking
  headroom. cv_train 0.8935 consistent. ag_news_s17 running; sst2 pair + trec_pro_l behind.

### 2026-07-03 review #44 (cron)

- Cross-dataset batch on committed code: ag_news (s7) mid-evolution, heldout ~0.867-0.869,
  small confident-death counts (6/1/6) — behaving like the pre-rewrite runs. Three runs +
  trec_pro_l chained behind. No stalls, nothing completed to judge yet.

### 2026-07-03 — trec_pro_l queued (Lee: "the test with pro / large / large")

Maxed configuration on the committed method: deepseek-v4-pro proposer, ModernCE-large STS,
finecat-nli-l encoder, TREC-6 seed 7. Pre-registered: band 0.93-0.96 (deliberately stacks
encoder + proposer levers — a "what does the method deliver maxed" number, not a lever
isolation; the -l re-score of an -m pool previously measured 0.946). Est. 60-90 min (-l is
~5x slower per pair, GPU shared with Lee's training). Chained behind the cross-dataset batch.
Hourly improvement cron armed (8dd3e10a, minute :43, session-only, 7-day expiry).

### 2026-07-03 — COMMITTED PACKAGE VALIDATED: pool_cv 0.920 on TREC

- Fresh pool, fresh LM calls, committed code: **0.920 / F1 0.882** — inside the 0.916-0.938
  pre-rewrite band. $0.01, 10 min, 0 abnormal. Head chosen: HistGBM lr=0.06 l2=0.01 (not RF —
  the CV grid picks per-pool). Port is behaviorally faithful.
- Qualitative: fresh pool mixes semantic and wh-word lexical hypotheses as before; "The text is
  a question." survived (vacuous on TREC — everything is a question — should be
  scores-constant... verify: likely near-constant but not < 0.02 std; diagnostics will show if
  it carries weight). Two tuning notes for later, not churned now: (1) plateau epsilon 1e-4 is
  finer than round-to-round noise — held-out crept +0.0087 over 6 rounds and patience never
  fired (rounds 2-4 pruned/refilled 0, so the extra rounds were nearly free but pointless);
  (2) consider dropping always-true statements at generation time via a variance check.
- Next: re-run cross-dataset validation (ag_news, sst2 × seeds) on the committed package.

### 2026-07-03 review #43 (cron) — post-commit validation run in flight

- Rewrite committed and pushed (f5e205f, 29 files): clean METHOD.md implementation, 11 fake-based
  tests, ruff + pre-commit, minimal configs; old code archived untracked in src-bak/.
- Real end-to-end validation of the committed package running on TREC (fresh LM calls — new
  prompts don't hit the old cache). Process healthy: GPU-resident, active LM connections; output
  silence is grep block-buffering on sparse phase prints (noted footgun; next launches go to a
  log file directly). Expected band per prior seeds: pool_cv 0.91-0.94.

### 2026-07-03 review #42 (cron)

- ag_news_evolve3_s7 mid-run: heldout 0.881 → 0.868 → 0.891 (noisier trajectory than TREC),
  and notably MORE confident deaths per round (6/19/15 vs TREC's 4-5) — on AG News many
  hypotheses are decisively useless, consistent with the label-noise wall. Held-out sits inside
  the pre-registered 0.885-0.895 band so far. Queue healthy, 3 runs + pro test behind it.

### 2026-07-03 — pro-proposer test queued (deconfounds the saturation claim)

trec_evolve3_pro: identical to trec_evolve3 except proposer = deepseek-v4-pro. Chained behind
the cross-dataset queue. Pre-registered: pool_cv within ±1.5 of flash's 0.916 + same hit-rate
decay → saturation is ENCODER-shaped (claim upgraded, "cheap proposer suffices" stands);
pool_cv ≥ ~0.93 or persistent round-2+ hit-rate → proposer quality is a real lever (retract).
Secondary: round-0 heldout vs flash's 0.825.

### 2026-07-03 — cross-dataset validation launched (the external-validity test)

Fixed loop + honest protocol on AG News + SST-2, seeds 7/17 (4 runs, ~40 min, ~$0.04).
Pre-registered expectations: (1) saturation ≤ round 2 both datasets (SST-2 maybe r1);
(2) pool_cv ≥ best fixed head on each; (3) refill hit-rate decay replicates;
(4) AG News caps at 0.885-0.895 (label-noise wall — exceeding 0.90 would mean the old
protocol was masking headroom), SST-2 at 0.94-0.95 (pragmatics ceiling).
All four hold → method is cross-dataset validated, write-up ready. Deviations = next finding.

### 2026-07-03 review #41 (cron) — idle; s17 qualitative check

- Nothing running. s17 evolution log read: same decay shape (hit-rate 0.33→0.125→0.167→0,
  held-out peak r3, patience fired r5); refills semantically sound ("The text asks for the name
  of a person who is quoted or attributed with a statement.") with the usual dataset-relative
  lexical entry ("begins with the word 'Name'"). Consistent with all conclusions. Cron can be
  removed.

### 2026-07-03 — REPLICATED: pool_cv 0.916 (s7) / 0.938 (s17) — fixed loop validated

- Seed-17 replicate: **pool_cv 0.9380 / F1 0.9261**. Honest-protocol mean across seeds
  0.927 ± 1.1 — above the old inflated best-head number (0.902) and even above trec_full's
  0.926 (which used 2.2x data under the old protocol). Both runs: 2 useful evolution rounds,
  patience-stopped, ~7 min, <$0.01, 0 abnormal finishes.
- **The day's method conclusions, all measured:** hypotheses saturate (~2 rounds, ~30 useful
  directions); prune only confident deaths (churn was 50% under single-split); CV-select the
  head (worth +2, systemic); report honestly (best-of-6-on-test inflates +2.2); encoder size is
  the real capacity knob (+5 from -m→-l); data +2.4/2.2x. Scaling budget goes to encoders and
  ensembling, never to more hypothesis generation.
- M6 closed. Remaining (Lee's call): -l finalization + 3-seed pool ensemble (~40 min, ~$0.03)
  for the final table; two-encoder feature union for the scaling-law story; write-up.

### 2026-07-03 — trec_evolve3 COMPLETE: pool_cv 0.9160 honest-protocol headline

- **pool_cv (CV-selected head, no test peeking): 0.9160 / F1 0.8818** — beats every fixed head in
  the same run by ≥2 points (best fixed 0.896) AND beats evolve2's optimism-inflated 0.902,
  at 2k train, 7 min, $0.008, 0 abnormal finishes. Lee's CV-systemic-fix position: measured
  correct — honest head selection was worth more than rounds 2-4 of hypothesis generation.
- All pre-registered rules resolved: saturation at r2 ✓, patience fired ✓, confident-death
  pruning stabilizes pool ✓, refill hit-rate decays 100→20→0 ✓ (generation budget = 2 rounds).
- Per rule: seed-17 replicate launched (trec_evolve3_s17) — believe the number only if it holds.

### 2026-07-03 review #40 (cron) — trec_evolve3 mid-run readout vs pre-registered rules

- Held-out CV: 0.839 → **0.8464 (r2 peak)** → 0.840 → 0.824; patience fired at r4 ✓ (saturation
  at round ~2, exactly as the orthogonality analysis predicted).
- Confident deaths: 5 → 4 → 0 → 4 — the noise-safe pruning stabilizes the pool almost
  immediately; there is nearly nothing confidently dead after round 1.
- **Refill hit-rate decays 100% → 20% → 0%** across rounds — the generation well dries fast,
  though round 2 still surfaced one gem (0.916 AUC on ABBR/ENTY, its assigned target).
- Maps to pre-registered rule 1: plateau + decaying hit-rate → generation saturates by round 2;
  beyond that is encoder ceiling. Loop mechanics (patience, confident pruning, target
  instrumentation) all behaved as designed. CV peak vs evolve2's honest 0.8435: +0.3, noise.
- pool_cv headline pending final scoring.

### 2026-07-03 — trec_evolve3 launched: instrumented validation of the fixed loop

Pre-registered measurements → decisions:
1. per-round held-out CV acc (does evolution add value; does patience fire ~round 2-3?)
2. confident-death counts (is there anything left to prune after round 1?)
3. refill target-AUC / hit-rate (do replacements HIT their assigned hot spot? — the
   regime-correct generation score once train is interpolated)
4. pool_cv (CV-selected head) vs evolve2's honest CV 0.8435 — headline protocol, no
   best-head-on-test optimism
Decision rules: plateau+low hit-rate → stop proposing (encoder ceiling); high hit-rate+flat →
constrained-capacity selection next; CV improves → replicate seed 17. Config: 2k/-m, m=64,
patience 2, cap 6 rounds, all loop fixes active.

### 2026-07-03 — METHODOLOGY failure analysis + loop fixes (Lee's redirect)

- Lee's correction: the deliverable is the METHOD, not dataset scores; analyze failures fast and
  fix the loop. Three ~1-min tests on cached matrices:
  (A) prune-decision churn across seeds: **0.50 single-split** (half the kills were split noise),
  0.39 CV — the loop's real defect, now quantified;
  (C) refill orthogonality: median corr 0.47 to survivors, only 4% >0.8 — generation redundancy
  REFUTED; refills are novel-but-useless → the encoder's class-relevant directions saturate after
  ~2 rounds;
  (B) mRMR vs importance selection: 0.866 vs 0.862 CV — redundancy-penalized selection is not a
  lever, consistent with C.
- Loop fixes landed (22 tests): prune only CONFIDENT deaths (importance ≤0 in every fold);
  evolve stops on held-out plateau (evolve_patience); pool_cv head (CV-selected regularization)
  added to standard results. Validation run for the fixed loop: one TREC evolve w/ patience —
  pending Lee's go.

### 2026-07-03 — trec_full: RF 0.926 (+2.4 from data alone)

- **trec_full (v2 pool frozen, 4,452 train / 1,000 val): RF 0.926 / GBM 0.908** vs 0.902 at 2k —
  the variance diagnosis validated: the single biggest -m gain since the -l discovery, for $0 LM
  and one scoring pass. Curve: 63 features → 0.906 val.
- Post-hoc diagnostics: ENTY F1 0.793→0.867 (the data mostly fixed the weakest class);
  train-val gap persists (0.118) but test 0.926 > val 0.878 (val remains the harder sample).
  ABBR F1 stuck at 0.714 (rare-class floor, 9 test examples). ENTY-DESC separator AUC unchanged
  (0.836) — remaining errors concentrated exactly there.
- TREC -m ladder now: full-data pool RF **0.926** > evolve2 0.902 > evolve1/static 0.896 >
  boost 0.876 > tfidf 0.828. Obvious next cheap step when desired: same run at -l
  (projects ≥0.94-0.95, would match/beat classic supervised CNN territory).

### 2026-07-03 — bank search KILLED for good (Lee); experiment-value bar set

- Lee's call, twice affirmed: the bank search was long and low-value — nothing it could deliver
  that a faster, cheaper run can't. All copies killed (mine, the duplicate, AND a parallel
  session's nohup'd copy + its watcher). Lasting artifacts kept: the fully-scored TREC bank
  matrix in cache, and the pipeline-level lessons already merged (CV stability pruning, CV
  objectives, tuned heads, hot-spot feedback).
- **Standing principle from Lee: experiments must clear an info-per-minute bar. Prefer the
  fastest, cheapest run that answers the question; if a cheaper proxy answers 90% of it,
  run the proxy.**
- Sole survivor: trec_full (single process verified) — v2 pool frozen, 2.2x data, the variance
  test. It IS the cheap run: $0 LM, one scoring pass.

### 2026-07-03 review #39 (cron)

- trec_full scoring its enlarged matrices (running, healthy). Bank search v2 in phase 1
  (full-bank optuna, CPU, scoring replayed instantly from cache as designed). No stalls,
  nothing new to audit.

### 2026-07-03 — process untangling after session restart

- Bank search died mid-optuna during the restart (all scoring cached — the expensive part is
  banked). Its death correctly triggered the trec_full chain (now running), but a SECOND stale
  watcher would have launched a duplicate trec_full — killed before it fired.
- Bank search relaunched via nohup (session-restart-proof) on the CV-objective code; goes
  straight to CPU phases off the cache. Monitor armed on result.json.
- In flight: trec_full (data-scaling test, v2 pool frozen) + bank search CPU phases.

### 2026-07-03 review #38 (cron)

- Bank search survived the session restart as a detached process (running current CV-objective
  code); GPU scoring complete (~20 min), now in CPU search phases (119% CPU, no cache writes —
  expected silence). trec_full chained by PID watch, waiting correctly. Nothing to audit; no
  stalls.

### 2026-07-03 review #37 (cron)

- Bank search: train matrix done, val matrix at 205k/401k pairs — CPU search phases next.
  trec_full (v2 pool frozen, 2.2x data, 2x val) queued behind it, waiting correctly.
- Regularization overhaul landed this hour (Lee-driven): CV stability-selection pruning with
  "does not generalize" failure mode, CV-on-train search objectives, confusion HOT SPOTS
  (connected components of the confusion graph) with anti-single-example instructions replacing
  scattered error samples. 22 tests pass.

### 2026-07-03 review #36 (cron)

- Bank search scoring phase 41% (459k/1.12M pairs, ~500/s solo, cache live) — the 1,070-hypothesis
  bank costs more scoring than estimated (tree/boost tried-lists had little cache coverage) but
  it's a one-time asset. CPU phases (optuna + annealing) follow, ~25 min out.
- Diagnostics command landed; trec_pool_evolve2 decomposition: overfit gap 0.127 (top deficiency),
  effective rank 28/126 (redundancy #2), coverage NOT a gap (all pairs ≥0.84 AUC), zero label-noise
  evidence. The running experiments target exactly the top two.

### 2026-07-03 review #35 (cron) — idle

- No active runs; trec_pool_evolve2 (RF 0.9020) already reviewed. Pending Lee's direction:
  -l finalization of the v2 pool and/or the multi-seed sweep. Cron can be removed until then.

### 2026-07-03 — trec_pool_evolve2 (all improvements): RF 0.9020

- **v2 vs v1**: RF 0.9020/0.8710 F1 vs v1 best 0.896/0.844 — first 0.90+ on -m TREC; F1 improved
  on every head. Pools genuinely differ (2/90 replacement overlap); v2 refills visibly attack the
  ENTY/DESC boundary ("The text asks for a definition that does not involve a specific name.") —
  class definitions + confusion evidence working. Caveat: +0.6 acc is within single-seed pool
  variance (±1.5 observed); the consistent F1 gains across heads are the more credible signal.
- Failure-mode telemetry: 88 pruned as "detectable but no predictive value", 8 as
  "encoder cannot detect" — the LM mostly proposes verifiable-but-unhelpful statements, rarely
  unverifiable ones. Cost $0.007, 0 abnormal LM finishes.
- Curve: 126→0.882, 31→0.874, 15→0.848 val. Interrupted once mid-run (external process kill —
  another agent per Lee); cache made the resume free. Also: grep in our run pipelines masks exit
  codes — noted for future launch hygiene.

### 2026-07-03 — MACHINE CRASH incident (my fault)

- The throughput pass added `permutation_importance(n_jobs=-1)` — joblib process-fork inside a
  CUDA-holding process → wedged GPU driver → full machine crash → **killed Lee's futo-asr
  training (~4h progress lost)** plus the in-flight trec_pool_evolve2.
- Fixes: permutation importance back to serial (documented in code with the hazard), NLI batch
  reverted 256→128 (likely innocent; removing variables), rule recorded to memory: no
  process-parallelism in CUDA processes; flag resource-usage changes on this shared machine.
- trec_pool_evolve2 relaunched post-reboot (GPU now uncontended).

### 2026-07-03 review #34 (cron) — queue drained

- All planned v1 stages complete: baselines, pools (+ec, +evolve ×4 datasets), trees
  (+deep, +GEPA variants), boosts (binary + 2 multiclass full-length), GEPA proposer tune,
  RF/SVM/logreg_cv heads, throughput fixes. Nothing running. Next step (overnight uniform
  sweep: 3 seeds × 3 methods × 3 datasets + 20NG pool + full-train TREC + -l finalization)
  awaits Lee's go — not auto-launched.

### 2026-07-03 review #33 (cron)

- 20newsgroups_pool_evolve: final scoring pass just completed (189.7k pairs); head fits +
  distillation curve in progress, metrics imminent. Queue otherwise drained; no stalls.

### 2026-07-03 review #32 (cron)

- 20newsgroups_pool_evolve finished all 3 evolve rounds (final refill 49); now scoring final
  feature matrices (long texts → slower). No stalls.

### 2026-07-03 review #31 (cron)

- 20newsgroups_pool_evolve through evolve round 1 (kept 48 / pruned 48 / refilled 48 at m=96),
  cache writes current — alive, ~15-20 min to go. Last queue item; final sweep planning next.

### 2026-07-03 review #30 (cron) — irony hypotheses SURVIVED pruning

- Against prediction: "The text uses ironic language to express disapproval." and the
  double-negative hypothesis survived held-out importance ranking into sst2_pool_evolve's FINAL
  pool — finecat-m detects ironic phrasing well enough to rank top-half. Yet SST-2 accuracy is
  unchanged (SVM 0.9484 ties pre-refill best): the surviving irony features fire on detectable
  cases already covered; residual errors are subtler pragmatics. Selection churn is healthy
  (~half of round-1 refills pruned in round 2).
- sst2_pool_evolve complete (SVM 0.9484 / GBM,RF,logreg_cv 0.9450). 20newsgroups_pool_evolve
  running (first with failure-mode annotations). No stalls, no action.

### 2026-07-03 review #29 (cron) — confusion refill validated qualitatively

- sst2_pool_evolve evolution log: refill shown 47/27/43 held-out errors per round; its proposals
  directly target the pragmatics failure family from our error analysis — "The text uses ironic
  language to express disapproval.", "double negative to convey a favorable opinion", "pleasant
  but lacking in substance" (concessive-but). The LM independently derived the same diagnosis
  from the misclassified texts. Open question: whether finecat can VERIFY irony statements —
  held-out importance will decide; round-2 error count (43) suggests partial traction at best.
- Final scoring in progress; 20ng evolve last in queue. No stalls.

### 2026-07-03 review #28 (cron)

- ag_news_pool_evolve: GBM 0.8855 ≈ static pool 0.887 — evolution neutral on AG News, consistent
  with the label-noise ceiling diagnosis (every method lands 0.885-0.893 there). Curve:
  128→0.886, 32→0.878, 16→0.868, 8→0.832 val — 16 features costs under 2 points. (Ran with the
  pre-RF head set; final sweep covers.)
- sst2_pool_evolve mid-run — first run with confusion-driven refill + class descriptions live.
  20ng evolve queued last. No stalls.

### 2026-07-03 — failure analysis (Lee's ask) + accidental -l result

- **Accidental discovery**: analysis script defaulted to finecat-nli-l → **TREC evolved pool on
  -l features: 0.946 test** (27/500 errors) vs 0.896 on -m. +5 points from swapping the NLI
  model under the same 58 hypotheses; no fine-tuning, 2k train. Confirms the -m-search/-l-final
  strategy for the uniform sweep.
- **TREC errors (27)**: 55% are ENTY→DESC/LOC — "What is X?" questions whose ANSWER is an entity
  but whose FORM reads as definition ("What is the heaviest naturally occurring element ?"), and
  LOC pull from mentioned places ("What currency does Luxembourg use ?"). Missing hypothesis
  family: statements about the EXPECTED ANSWER ("the answer would be a substance/object name,
  not an explanation") rather than question content.
- **AG News errors (214)**: 46% are Business↔Sci/Tech, mostly dual-topic texts (tech-company
  business news: Amazon, plasma TV retail, Airbus orders); several sampled errors look like
  GOLD LABEL NOISE ("Stocks Close Higher..." labeled World). The universal ~0.89 wall (zero-shot,
  pool, boost, -m, -l all converge there) is likely partly irreducible label noise. Candidate
  hypothesis family: aspect-priority ("primarily about the business aspect rather than the
  technology itself").
- **SST-2 errors (45, balanced)**: non-compositional sentiment — sarcasm ("hilariously inept and
  ridiculous ." gold-positive), concessive but-structures ("does paint some memorable images ...
  but ..."), metaphorical digs ("valedictorian at the school for soft landings"). Literal-content
  entailment can't see pragmatics; "The text uses irony" hypotheses are possible but NLI
  verifiability is doubtful. Some gold labels here also questionable.
- Meta: the confusion-driven refill (just built) will show the LM exactly these cases
  automatically. Label-noise observation strengthens the multi-seed/rigor item.

### 2026-07-03 — improvements 1+4 implemented (confusion refill, class definitions)

- **Class-definition grounding**: every dataset spec now carries one-line class definitions
  (TREC's ENTY-vs-DESC distinction spelled out; 20NG glossed per newsgroup). All LM-facing
  prompts (tree/boost proposers, pool generate/refill, GEPA bank + judge) receive
  "name: definition" strings; internal logic keeps plain names.
- **Confusion-driven refill**: rank_by_heldout_importance now also returns held-out
  misclassifications; evolve rounds show the refill LM up to 10 "[true: X, predicted: Y] text"
  examples — boosting's residual-targeting applied to pool evolution.
- Both land in runs starting after this point (sst2/20ng evolve pick them up mid-stage; the
  final uniform sweep supersedes all v1 runs). 20 tests pass.

### 2026-07-03 — RF/SVM heads (Lee) + trec evolve result

- **Random forest is the new best pool head**: AG News 0.893 (new dataset best, beats GBM 0.887
  and zero-shot), TREC 0.896 acc / 0.879 macro-F1 (F1 +2.1 over GBM — RF handles rare classes
  better than depth-1 GBM stumps), SST-2 ec 0.9461 (close 2nd to GBM 0.9484). SVM (RBF)
  consistently behind RF. logreg_cv added to the head set for the final sweep.
- **trec_pool_evolve: GBM 0.896** (new TREC best pre-RF-sweep; evolution recovered the
  entail_contra deficit). Distillation curve: 116→0.870, 58→0.868, 29→0.864, 14→0.840, 7→0.728
  val — top-29 columns ≈ 20-25 NLI evals/pred at −0.6 points. LM cost $0.007.
- Improvement backlog agreed with Lee (ranked): confusion-driven pool refill; scale train size;
  -l final sweep on winning configs; class-definition grounding; 20NG long-text chunking;
  multi-seed rigor + bigger val for gates.

### 2026-07-03 review #27 (cron)

- trec_pool_evolve mid-run: round 0 complete — kept 32 / pruned 32 / refilled 32, the
  prune+refill mechanism working as designed. Three more rounds + distillation curve to go;
  ag_news/sst2/20ng evolve queued behind. No stalls, nothing to audit this cycle.

### 2026-07-03 — 20newsgroups_tree complete; evolve stage launched

- **20newsgroups_tree: 0.355 test / 0.368 val** (24 leaves, $0.11, 46 min, 0 abnormal LM
  finishes). Beats zero-shot (0.163), far under tfidf (0.565) — structural ceiling: 24 leaves for
  20 classes. The tree reads as a clean Usenet taxonomy (sports→hockey/baseball,
  religion→Christianity→atheism→sin/redemption); proposer + provider pin + failure bank all
  behaved. 20-class problems belong to pool/boost.
- Launched evolve stage: trec/ag_news/sst2 pool_evolve (m=64) + 20newsgroups_pool_evolve (m=96
  for 20-class coverage), all entail_contra + STS dedup + provider pinned + distillation curves.

### 2026-07-03 — trec_boost FULL RUN: 0.876 (M5 complete)

- **trec_boost (30 rounds): 0.876 test / 0.828 val** vs 0.792 truncated — +8.4 points from
  finishing the run. Now: pool 0.894 > boost 0.876 > tfidf 0.828 > tree_gepa 0.784. Val still
  falling at the cap → a continuation/uncapped run plausibly closes the pool gap. 169 stages,
  77 distinct hypotheses (≈ pool-level inference cost), audit 1/78 flagged ("famous landmark,
  work, or discovery" — mild).
- Boosting is now within noise of the pool on both multiclass datasets while being stagewise
  auditable. Next boost levers: longer runs, contradiction split axes, ensemble-aware dedup
  (STS now built), boost-proposer GEPA.

### 2026-07-03 review #26 (cron)

- trec_boost at round 29/30 (val 0.5395, still falling — the cap will bind, continuation-worthy).
  Final test scoring imminent. 20newsgroups_tree at ~34 min, $0.09 LM, still under the 1h cap but
  watch next cycle. No stalls, no flags.

### 2026-07-03 review #25 (cron)

- trec_boost rerun at round 20/30 (val logloss 0.597, down from 0.84 at the old truncation point;
  6/6 stumps accepted). Status shows stale "done 0.792" from the 8-round metrics — the live run
  overwrites at completion. ~10 min out.
- 20newsgroups_tree at depth 2-3, splits are proper newsgroup topics ("The text discusses a
  specific motor vehicle or part thereof.", "...computer hardware component or specification.",
  "...baseball games, player statistics, or trade rumors."). Tuned proposer handling 20 classes
  gracefully. No stalls, no flags, no action.

### 2026-07-03 — pool evolution + STS dedup built (Lee's designs)

- **Pool evolution** (LM-in-the-loop recursive feature elimination): each round prunes the bottom
  half of hypotheses by HELD-OUT permutation importance and refills via a RefillPool signature
  showing survivors ("don't paraphrase") and failures ("don't repeat the pattern"). Plus a
  distillation curve on every pool run: val accuracy at top-64/32/16/8/4 features — quantifies
  minimum pool size. Configs: {ag_news,trec,sst2}_pool_evolve.yaml (entail_contra, provider
  pinned, 3 rounds, keep 0.5). Queued after in-flight runs.
- **STS dedup** (dleemiller/EttinX-sts-s, 68M): paraphrase filter BEFORE NLI scoring in tree and
  boost candidate loops; boost also dedups against the accepted ensemble. Behavioral dedup stays
  as post-scoring authority. Threshold calibrated on observed pairs: paraphrases 0.77-0.80,
  distinct 0.30-0.35 → 0.75. Off by default (sts.enabled), on in evolve configs.
- Caveat logged for evolution: permutation importance splits credit between correlated
  hypotheses — audit each round's pruned list for unique-but-credit-stolen victims.

### 2026-07-03 review #24 (cron)

- **trec_boost cleared round 8** — the no-reasoning cache-free retry worked exactly as designed
  (attempt 1 replayed the cached bad response and failed; attempt 2 sampled fresh and passed).
  Now at round 9+, 6/6 stumps/round accepted, val falling. Full 30-round TREC boost incoming.
- 20newsgroups_tree fitting (root: "The text discusses a sporting event or team.", gain 0.036 —
  low absolute gains are expected at 20 classes). Both runs share the GPU; pace acceptable.

### 2026-07-03 — round-8 failure fully diagnosed and closed (Lee)

- Attribution verdict on the trec_boost round-8 failure: **finish_reason=length at 12k tokens,
  served by first-party DeepSeek** — NOT provider variance, NOT classic token-loop repetition:
  verbose-REASONING runaway truncated mid-JSON (consistent with Lee never seeing repetition).
- Retry v1 (hint line) got one fresh sample, which failed ValidationError — and Lee spotted the
  fatal flaw: that response is then cached too, so reruns replay both bad responses and pin the
  fit at round 8 permanently.
- Retry v2: second attempt now uses a **no-reasoning, cache=False clone** of the proposer LM —
  runaway can't recur without thinking, and retries always sample fresh. trec_boost relaunched
  (rounds 0-7 replay; round 8 must now pass).

### 2026-07-03 — M6 VERDICT: GEPA proposer tune is dataset-dependent

- **trec_tree_gepa: 0.784 test / 0.740 val vs 0.730 / 0.670 original (+5.4 / +7.0)** — audit
  0/8 flagged, zero lexical wh-word splits; semantic hypotheses beat the original's shortcuts
  outright. Now above trec_tree_deep (0.754), near truncated trec_boost (0.792).
- ag_news_tree_gepa: 0.8275 vs 0.8605 (−3.3). Pattern: instruction tuning pays where junk
  proposals were the bottleneck (TREC), hurts/noise where the seed proposer was already near
  ceiling (AG News). Single-seed caveat on both.
- Decision (per Lee's rule): tuned proposer kept as an available asset; **priority shifts to
  adaptive boosting**. Launched: trec_boost rerun (retry fix, cache-replay, no pin) +
  20newsgroups_tree (tuned proposer + provider pin + failure attribution).
- Judge-screening discussion: agreed NOT to give the judge veto power in the live loop
  (Goodhart risk confirmed by AG News regression + 'Who'-split effectiveness); annotate-only
  variant + verifiable-only veto available behind a flag if wanted.

### 2026-07-03 — LM failure attribution + provider pinning (Lee's suspicion)

- Lee has never seen deepseek-flash repetition failures → suspicion the "repetition loops" were
  partly our-side or provider-side. Supporting evidence: the original 4k max_tokens truncation
  produces identical AdapterParseError symptoms, and Lee's OpenRouter dashboard showed some calls
  served by Baidu Qianfan rather than DeepSeek first-party (third-party hosts = classic
  degeneration source).
- Wired: (1) every LM call now logs finish_reason + serving provider when abnormal
  (finish != "stop"), counted as lm_abnormal_finishes in costs.json — next failure is attributable,
  not guessed; (2) LMConfig.extra_body passthrough enables OpenRouter provider pinning
  ({"provider": {"order": ["deepseek"], "allow_fallbacks": false}}) — verified live (response
  provider=DeepSeek). Cache-key caveat documented: pinning is for fresh runs.

### 2026-07-03 review #23 (cron)

- trec_tree_gepa mid-fit, healthy. Notable: NO lexical wh-word splits so far — the tuned proposer
  chooses semantic phrasings ("The text asks for the name of a specific person." where the
  original picked "starts with the word 'Who'"). The judge's anti-surface training visibly
  transferred; whether that helps TREC accuracy is exactly what this run measures (the original's
  'Who' split was genuinely effective).

### 2026-07-03 review #22 — first before/after refit: NEGATIVE on AG News

- **ag_news_tree_gepa: 0.8275 test / 0.850 val vs original 0.8605 / 0.874** (-3.3 test). Audit
  0/9 flagged; splits read cleanly. A depth-2 n=503 node found no acceptable split under the
  tuned proposer and became a leaf (original split it, including via its lucky junk split).
- Interpretation candidates: (a) frozen-node metric gain (+0.01 valset) doesn't transfer to
  sequential tree accuracy; (b) judge weight (0.3) pushed instructions toward semantic cleanliness
  at the expense of empirically effective splits; (c) single-seed tree variance — one different
  branch cascades. Cannot distinguish from one pair; trec_tree_gepa (running) is the second point.
- If TREC also regresses: rerun GEPA with judge weight reduced (e.g. 0.15) or gain-weighted
  higher, and/or multi-seed refits before further conclusions.

### 2026-07-03 review #21 (cron)

- ag_news_tree_gepa fitting; tuned-proposer root split gain 0.2216 vs 0.2185 original (marginal
  edge at the root). trec_tree_gepa queued behind it. Judge hardening landed for future GEPA runs:
  per-check Field descriptions now actually reach the LM (were Python comments), independence
  instruction, near-pure-node bank filter (max_purity 0.9).

### 2026-07-03 — GEPA attempt 7 COMPLETE: proposer improved

- **Valset 0.3755 → 0.3852** (+2.6% rel on the combined gain/utility/judge metric); improvement
  landed at iterations 4-5 after frontier consolidation, as predicted. 46 bank nodes, ~110 metric
  evals, judge 0 failures throughout (boolean checks + reasoning disabled).
- Evolved instructions: ~6.6k chars, dataset-agnostic strategy rewrite (path-constraint semantics
  spelled out, exact-count discipline, verifiability rules). Saved: models/proposer_gepa.json;
  eval trail: models/proposer_gepa.evals.jsonl.
- **Launched before/after refits**: ag_news_tree_gepa + trec_tree_gepa (identical configs to
  originals except lm.proposer_path). Baselines to beat: ag_news_tree 0.874, trec_tree 0.730.
- Follow-up queued for next GEPA use: add a synthesis line to metric feedback linking the
  quantitative winner to the judge's clean list.

### 2026-07-03 review #20 (cron)

- GEPA attempt 7 at 73/100 evals, 0 judge failures. Two mutated candidates on the Pareto frontier
  (each beat the seed on its minibatch); aggregate best-on-valset still at the seed's 0.3755.
  If the run ends flat, next step is a moderate-budget continuation, not a conclusion.
- Queue still paused; nothing to audit.

### 2026-07-03 review #19 (cron)

- GEPA attempt 7: 19 evals, 0 judge failures, running at the expected faster pace with the
  non-reasoning boolean judge. Experiment queue still intentionally paused. No completions to
  audit; no action.

### 2026-07-03 — attempt 7: non-reasoning judge (Lee)

- Why the judge burned tokens: v4-pro emits ~3k+ chain-of-thought before the structured output;
  max_tokens only truncates. The judge was already dspy.Predict (no ChainOfThought wrapper) —
  the reasoning is provider-native. Per Lee, disabled it at the API:
  extra_body={"reasoning": {"enabled": false}} — verified live (completion=2 tokens,
  reasoning=0). Judge calls now ~10x cheaper/faster; max_tokens back to 4k.
- Boolean checks + no reasoning = the grounded-judge philosophy end to end: easy decidable
  questions, no vibes, deterministic aggregation.

### 2026-07-03 — attempt 6: judge max_tokens 4k→16k

Lee spotted v4-pro judge calls finishing with reason=length at exactly 4,000 output tokens on the
OpenRouter dashboard (5/14 judge failures in attempt 5): the judge LM had a hard-coded 4k cap,
too small for reasoning + 8 boolean checks + critique. Raised to 16k; attempt 6 launched (cache
replays proposals; incremental cost ≈ judge calls only).

### 2026-07-03 — judge redesigned to boolean checks (Lee); attempt 5

- Per Lee: 0-10 judge scores are un-grounded (empirically: 0.56 mush). Judge now answers four
  strict yes/no checks per statement (semantic / verifiable / non-duplicate / node-targeted,
  "when in doubt, false") + two set-level checks (minority coverage, varied specificity); score
  aggregated deterministically in code (0.85 stmt-level + 0.15 set-level); feedback names each
  failing statement with its failed checks. ScoreWithFeedback return type per dspy docs.
- Attempt 4 (holistic judge) stopped mid-run and archived to models_attempt4_holistic/; attempt 5
  launched — baseline proposals replay from LM cache, so the restart mostly costs judge calls.

### 2026-07-03 review #18 (cron)

- GEPA attempt 4 running cleanly: 22 metric evals, judge 100% available (avg 0.56), metric now
  returns the documented ScoreWithFeedback type (per Lee). Second-half score dip is GEPA
  minibatching hard examples, not regression. Pace ~1.4 min/eval → completion in roughly an hour;
  accepted overage since this is the prioritized task, checkpointed, and health-monitored.
- Comparison configs staged: ag_news_tree_gepa / trec_tree_gepa (identical to originals except
  lm.proposer_path) — launch on completion.
- Experiment queue remains intentionally paused. No new completions to audit.

### 2026-07-03 — GEPA attempt 3 crash analysis → attempt 4

Attempt 3 made real progress (15 metric evals; baseline combined score 0.32 = gain 0.21 /
utility 0.28 / judge 0.5-fallback) then crashed. Three defects fixed for attempt 4:

1. metric returned a plain dict — dspy.Evaluate's full-valset sum crashes on dicts; now returns
   dspy.Prediction(score=, feedback=) which supports arithmetic;
2. judge failed 14/15 with RuntimeError — dspy.context(lm=...) is forbidden inside GEPA's worker
   threads; judge LM now bound via predict.set_lm(). The single successful critique was
   high-quality (correctly named "The text includes a numerical figure or statistic" as
   surface-level and class-nonspecific — exactly the junk-detection Lee wants);
3. pace ~2.3 min/eval → 100 calls ≈ 4h: valset capped at 12 (was 22), per-node metric eval
   subsample 400→250.

### 2026-07-03 review #17 (cron) — GEPA attempt 3 healthy

- Attempt 2 root cause: sqlite thread-affinity — GEPA worker threads hit the main-thread cache
  connection; every metric call died pre-logging and errors consumed the budget. Fixed
  (check_same_thread=False + lock, 16-thread hammer test passes) plus a GPU lock around
  CrossEncoder.predict for threaded callers.
- Attempt 3: zero errors, cache being written seconds ago, 17 LM connections in flight — real
  optimization underway. Experiment queue intentionally paused (per Lee) until the tuned
  proposer lands; then trec_boost rerun + 20NG tree with proposer_path set.

### 2026-07-03 — M6 GEPA proposer tune, attempt 1 → bug → attempt 2 (running)

- Per Lee: paused further experiment runs (stopped trec_boost rerun + 20NG queue) to optimize
  the proposer FIRST — better instructions make all later runs faster/better. Light budget
  (100 metric calls), v4-pro reflection + judge, flash student.
- **Attempt 1 failed silently-fast:** every metric eval errored "No LM is loaded" — GEPA executes
  the student itself and needs `dspy.configure(lm=...)`; our tree/boost paths use scoped
  `dspy.context` so this never bit before. All iterations scored 0.0, no trajectories, original
  instructions returned. Fixed: optimize_proposer now configures the student LM explicitly.
- Per Lee: instructions must stay **generic across datasets** — steering line added to every
  metric feedback (encode strategies, not class names/topics); bank already spans AG News + TREC
  which structurally discourages dataset-specific instructions.
- Attempt 2 running with iteration-score monitor.

### 2026-07-03 review #16 — trec_tree_deep complete; boost rerun + 20NG launched

- **trec_tree_deep: 0.754 test / 0.728 val** (28 leaves, $0.08 LM, 40 min) vs 0.730 at 16 leaves —
  +2.4 points for 12 extra leaves; diminishing returns. Audit: 2/25 flagged, both small-node
  overfits ("The text asks for a person's age." train 0.012 → val 0.000). The tree ceiling on
  TREC with the current proposer is ~0.75 vs pool_gbm 0.894 — strongest motivation yet for the
  GEPA-optimized proposer (M6, next).
- Launched: trec_boost rerun (retry fix active) + 20newsgroups_tree.

### 2026-07-03 review #15 (cron)

- trec_tree_deep at ~32 min (within cap; ETA ~10-15 min). min_gain=0.01 visibly rejecting weak
  splits now (a depth-4 node chose "no split" — first observed refusal). Deep hypotheses still
  semantic. No completions this cycle; no action.

### 2026-07-03 review #14 (cron)

- trec_tree_deep alive and healthy (cache/progress mtimes current; deep nodes are small so node
  events are sparser). Depth-5 splits still semantic ("The text asks for a definition or
  explanation of a term.", gain 0.06; "The text asks for a person's age." on a 44-text node).
  No new completions; trec_boost rerun + 20newsgroups_tree next, then gepa-optimize (v4-pro
  reflection + judge). No action.

### 2026-07-03 — GEPA proposer metric redesigned (Lee's ideas)

Observation (Lee): many deep-node proposals are junk (vague paraphrases, gains ≤0.05). The M6
GEPA metric is now three terms, renormalized when the judge is off:

- **0.3 · best-split gain** — what greedy induction actually consumes (fraction of node impurity
  removed by the best candidate);
- **0.4 · held-out set utility** (Lee's feature-importance idea, corrected per his own
  train-set-bias objection): fit a depth-3 DT on ALL K candidate features over 70% of the node,
  score the held-out 30%, normalize vs majority baseline. Per-hypothesis **held-out permutation
  importances** go into the reflection feedback ("N statements contributed nothing") — noise-fit
  hypotheses can't game shuffling on held-out data, unlike sklearn's train-side impurity
  importances;
- **0.3 · LLM-as-a-judge** (deepseek-v4-pro default): scores the proposal SET on semantic
  task-relevance / diversity / node-specificity / verifiability, blind to measured gains; its
  critique text is appended to the GEPA feedback. Judge calls logged to *.evals.jsonl.

Metric evals subsample nodes to ≤400 texts (seeded, identical across candidates). Run when the
GPU queue drains: `nli-boost gepa-optimize runs/ag_news_tree runs/trec_tree runs/trec_tree_deep`.

### 2026-07-03 review #13 — trec_boost complete (short), retry fix

- **trec_boost: test 0.792 / val 0.786 — but only 8 of 30 rounds**: round 8's proposal hit another
  DeepSeek repetition loop; the graceful guard finalized the fit early. Even at 8 rounds it beats
  trec_tree (0.730) and -m zero-shot (0.356); still under tfidf (0.828) and pool_gbm (0.894).
  Val was still improving — clear upside left.
- **Fix:** proposal calls now retry once with a hint line appended on parse failure — the changed
  prompt bypasses the cached bad response, so one bad LM sample can no longer end a fit.
  trec_boost rerun queued after trec_tree_deep completes (rounds 0-7 replay from cache).
- Qualitative (stages.txt): top stage "The text contains the phrase 'stand for'." w=8.2 for ABBR —
  lexical but semantically dead-on ("What does X stand for?"). "The text starts with 'Who'." holds
  9.3 total weight for HUM (len_corr 0.13; audit 0/20 flagged). Reframing: for TREC answer-type
  classification, wh-word features ARE task semantics, not shortcuts — the lexical-shortcut
  concern is dataset-relative.

### 2026-07-03 — entail+contradiction pool features (Lee's idea)

NLI is ternary, so each hypothesis yields two independent features (P(entail), P(contradiction));
the cache stores raw logits, so the 128-dim variant cost zero new NLI compute. `pool.features:
entail_contra` added; *_pool_ec runs:

| dataset | GBM entail-only | GBM entail+contra |
|---|---|---|
| SST-2   | 0.9358 | **0.9484** (best SST-2 result in table; beats -l zero-shot 0.945) |
| AG News | 0.8870 | 0.8855 (unchanged) |
| TREC    | 0.8940 acc / 0.8581 F1 | 0.8800 acc / **0.8697 F1** (mixed: acc down, macro-F1 up) |

Interpretation: contradiction carries real signal where hypotheses have semantic opposites
(sentiment: contradicting "praises the acting" = criticism); topical datasets gain little
(off-topic mostly reads neutral, not contradicted). Follow-up ablation queued conceptually:
let tree/boost stumps split on the P(contradiction) axis too (each candidate hypothesis offers
two split axes) — same zero-cost trick applied to the adaptive methods.

### 2026-07-03 review #12 — ag_news_boost complete

- **ag_news_boost: test 0.8785 / val 0.8800** (114 stages, 75 distinct hypotheses, $0.08 LM,
  40 min). Ties pool_gbm (0.887) and zero-shot (0.887); beats adaptive tree (0.861), tfidf
  (0.849). Audit: **0/75 flagged**; stage table reads as ranked evidence ("The text describes a
  sporting event or athlete performance." w=+3.4 Sports; "The text is about a specific country or
  region." τ=0.98 for World — note the *high* threshold there, the τ sweep earning its keep for
  once).
- Caveats: (1) val loss was still falling at the round-30 cap (0.359) — accuracy likely left on
  the table; a warm-start/continuation mechanism or higher lr would help. (2) At 75 distinct
  hypotheses, boost inference cost exceeds the pool's 64 — the efficiency story belongs to the
  tree; boosting must argue accuracy + auditability (ranked, gated stages).
- trec_boost underway: round 0 chose "The text starts with 'Who'." as the first HUM stump —
  the lexical-shortcut pattern again, now in boosting. The val gate is active (5/6, then 6/6
  stumps accepted in early rounds). Will audit at completion.

### 2026-07-03 review #11 (cron)

- ag_news_boost round 24/30, val 0.3915. The val gate is now visibly active: rounds 19/21/22/24
  accepted 3 of 4 stumps (one class's stump rejected per round). Hypotheses remain semantic.
  ~10 min to completion; trec_boost + trec_tree_deep next. No action.

### 2026-07-03 review #10 (cron)

- ag_news_boost round 16/30 after cache replay, val 0.443 falling steadily. Qualitative: the
  residual/contrastive prompting is doing its job — late-round hypotheses target class-boundary
  confusions specifically ("The text focuses on a sports figure's personal life or
  controversies.", "The text provides details about a sports team's business operations." — both
  Sports↔Business separators; "The text involves a legal dispute or government investigation." for
  World↔Business). This is the gradient-signal-in-text mechanism working. No stalls, no flags.

### 2026-07-03 review #9 (cron)

- ag_news_boost healthy (val 1.086→0.542 by round 7, 4/4 stumps accepted per round, hypotheses
  semantic — sports/tech/regulatory splits) but pace (~2 min/round) projected 60 rounds ≈ 2h,
  over the 1h job cap. **Capped multiclass n_rounds at 30** (120 stages — ample for v1),
  restarted the remaining queue; NLI+LM caches replay finished rounds in minutes.
- No other issues; trec_boost and trec_tree_deep follow.

### 2026-07-03 review #8 — sst2_boost crash, fix, recovery (M4 complete)

- **sst2_boost crashed after round 39**: DeepSeek repetition loop ("unforgettable" ×
  thousands) → unparseable response → dspy AdapterParseError killed the run before test scoring.
  **Fix:** LM proposal failures in tree/boost/pool sources are now caught and treated as
  "no candidates" → graceful finalize with accumulated stages. Rerun replayed rounds 0-39 from
  the caches in ~4 min and finished: **test 0.9106 / val 0.9300**, 39 stages, $0.056 LM.
- Context: SST-2 is nearly saturated by NLI zero-shot (-m 0.935, pool_dt 0.939), so 0.911 is
  competitive but not a win here; TREC/AG News boosts are the real test of residual fitting.
- Audit of all 39 stages: **0 flagged**; the artifact-adjacent ones are mild
  ("The text is brief." len_corr=0.44, just under the 0.5 flag line; "uses the word 'too'"
  r=0.22). Ensemble weight concentrates on semantic hypotheses; brevity stages carry moderate
  weight. Keeping the transparent-reporting approach.
- Repetition-loop root cause is temperature 1.0 + long structured outputs on deepseek-v4-flash;
  not changing temp mid-study (would invalidate the LM cache); the catch-and-continue guard
  suffices.

### 2026-07-03 review #7 (cron) — important reward-hacking observation

- sst2_boost round 25, val 0.575→0.212, all healthy. BUT the ensemble is now accepting
  **distribution-legitimate shortcut hypotheses**: "The text is simple or brief." (r19),
  "The text is brief." (r22), "The text uses the word 'too'." (r24). These pass the val gate
  because they genuinely correlate with sentiment in SST-2 (curt dismissals; "too long/dull") —
  so they are NOT non-generalizing hacks, but they are also not sentiment *semantics*. The val
  gate bounds generalization damage; it cannot police semantic quality. Two follow-ups for Lee
  to weigh:
  1. proposer-instruction constraint banning form-level statements (length, brevity, punctuation,
     function words) — cleaner semantics, but would also ban genuinely predictive lexical cues and
     invalidates the LM cache mid-study;
  2. keep them and report transparently as "discovered features" with an artifact-labeled column
     in the final table (audit's length-corr will quantify the brevity ones).
- Also observed: near-duplicate hypotheses accepted at different rounds ("simple or brief" vs
  "brief"). On reflection this is fine — boosting legitimately revisits strong features to refine
  their weight (as sklearn stumps reuse columns); predict-time scoring dedups them anyway. Only an
  interpretability cost, visible in stages.txt.
- No stalls; ag_news_boost, trec_boost, trec_tree_deep still queued.

### 2026-07-03 review #6 (cron)

- sst2_boost mid-fit and healthy: val logloss 0.575→0.290 over 10 rounds, all stages accepted by
  the val gate, ~45s/round. Hypotheses cover distinct sentiment facets rather than paraphrasing
  ("The text implies the film is not worth watching.", "The text expresses admiration for the
  film's construction."). One artifact-adjacent stage — "The text uses a simile or metaphor to
  describe the film." — accepted because it genuinely reduced val loss; keep an eye on its audit
  length-corr. No stalls, no refinements.

### 2026-07-03 review #5 — trees complete, boosts launched

- **trec_tree: 0.730** (> -l zero-shot 0.632, ≪ pool_gbm 0.894). Qualitative: the LM discovered
  **lexical shortcut hypotheses** — "The text starts with the word 'Where'." / 'Who' / 'How' /
  'Which'. The 'Who' split generalizes (HUM leaf 0.96 pure, val gain holds); the 'Which' one is
  flagged by audit with val-gain collapse (0.022→0.003). So: NLI-verifiable surface features are
  discoverable and *sometimes* legitimate; the audit + a future val-acceptance gate for tree
  splits is the right control. ENTY/DESC remained unresolved at max_leaves=16 → launched
  trec_tree_deep (28 leaves, depth 6, min_gain 0.01, min_samples_leaf 20).
- **-m baselines:** ag_news zero-shot 0.8865 (matches -l), sst2 0.9346, but **trec zero-shot
  drops to 0.356** with -m (vs 0.632 on -l). Learned thresholds/ensembles recover what the small
  model loses zero-shot (pool_gbm 0.894 on the same -m). Insight: the method's value grows as the
  NLI model shrinks.
- **Launched:** sst2_boost, ag_news_boost, trec_boost (first runs with val_accept gate +
  screening + 12k max_tokens), then trec_tree_deep.
- ag_news_tree reuse-bank effect visible: tried=20/node (16 LM + bank reuses).

### 2026-07-03 review #4 (cron)

- **ag_news_tree completed: 0.874 test acc** ($0.026 LM, 16 min under GPU contention). Beats
  pool_dt (0.853) while using ≤4 NLI pairs/prediction vs the pool's 64 — the
  interpretability-efficiency story holds. pool_gbm (0.887) and -l zero-shot (0.886) still ahead.
- **Audit: 1 of 9 splits flagged.** "The text discusses technological innovations or scientific
  research related to sports." — train gain 0.008, val gain 0.001 (collapse). A junk depth-3 split
  that barely cleared min_gain=0.005 and shaves a noisy 25-text leaf. All major splits generalize
  cleanly (root sports 0.219→0.207 val; sci/tech 0.192→0.168; business 0.279→0.264) and length
  correlations are low (|r|≤0.37, none flagged). Tree reads like a human taxonomy: sports →
  sci/tech → business/markets → politics/world.
- **Observation:** chosen thresholds are all ≈0.00-0.03 — finecat-nli-m entailment probs are
  near-binary, so splits act as entail/not-entail gates and the τ sweep adds little on this
  dataset. Worth an ablation note (score_mode contrast may matter more for subtler hypotheses).
- **trec_tree in progress**, mostly clean semantic splits ("The text asks for a numeric value.",
  gain 0.14) but one artifact-adjacent lexical pick at depth 3: "The text begins with the word
  'Which'." (gain 0.02) — legal but surface-level; audit when done.
- **Action:** raised min_gain 0.005→0.01 in 20newsgroups_tree.yaml (deep trees + permissive
  min_gain produced the flagged junk split). Consider a val-acceptance gate for tree splits
  (mirroring the boost val_accept) as a future change.

### 2026-07-03 review #3 (cron)

- No newly completed runs; ag_news_tree healthy at depth 3 (~14 min, $0.02 LM). Splits continue to
  read semantically ("...political events, government actions...", "...company's financial
  performance..."). trec_tree + baselines_m queued behind it.
- **Fix applied:** dspy warned that LM responses hit max_tokens=4000 (DeepSeek reasoning + 8
  rationales) and were truncated — some proposals may have been dropped at 2 nodes. Raised
  LMConfig default to 12000 for all runs starting after this point (trec_tree onward). ag_news_tree
  unaffected enough to keep (16 candidates still arrived per node).

### 2026-07-03 review #2 (cron)

- **Newly completed:** ag_news_pool (GBM 0.887), sst2_pool (DT 0.939), trec_pool (**GBM 0.894** —
  beats tfidf 0.828 and zero-shot 0.632 decisively; the headline M1 result). Total LM cost for all
  three pools: <$0.01.
- **Qualitative check of generated pools:** hypotheses are diverse, specific, and contrastive —
  e.g. AG News "The text includes a numerical score or result." (clever Sports/Business splitter),
  TREC "The text asks for the full form of an abbreviation.", SST-2 proper praise/criticize pairs
  per facet (acting, plot, visuals, direction). No vacuous or artifact-bait statements observed.
- **In progress:** ag_news_tree, healthy (~2 min/node under GPU contention); root split
  "The text mentions a specific athlete or sports team." gain 0.22, then a clean sci/tech split
  gain 0.19 on the not-sports branch. trec_tree and *_baselines_m still queued; boosts launch
  when the queue drains.
- No refinements needed; no stalls.

### 2026-07-03 review (cron)

- **Killed `ag_news_pool` (-l) at ~65 min** — over the 1h job cap. Root cause found: the GPU is
  shared with a long-running `futo-asr` process (100% util, 57 GB); all throughput planning must
  assume contention. `finecat-nli-l` was doing ~40-70 pairs/s effective; the old code would also
  have discarded all uncommitted scores on kill (now fixed via 8k-pair chunk commits).
- Relaunched the full queue on `finecat-nli-m`: pools (ag_news, sst2, trec) → adaptive trees
  (ag_news, trec) → `-m` baselines for apples-to-apples tables. Expected ~10-15 min/run under
  contention.
- No new completed runs to audit this cycle (the killed run produced no artifacts).
