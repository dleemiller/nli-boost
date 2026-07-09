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

### 2026-07-04 — GLM-5.2 teacher run: WORSE than DeepSeek; + reward rebuilt (booleans)

- **GLM-5.2 as teacher underperformed DeepSeek-pro.** 400 calls, returned the BASELINE instruction
  unchanged (710 chars, best geo 0.745 ≈ baseline 0.744), explored less (3 distinct scores/dataset
  vs DeepSeek's 5). DeepSeek found candidate 1 (geo 0.761, dominating, dataset-agnostic); GLM found
  nothing above baseline. On this task DeepSeek-pro is the better reflection model.
- **BIG caveat: both runs used the OLD reward** (single float judge, no coverage terms), which we
  then replaced — so neither result is comparable to anything produced under the new reward, and
  the teacher comparison should be re-confirmed under it. The DeepSeek candidate-1 transfer gate was
  never run and is now moot (old reward).
- **Reward rebuilt for granularity (Lee's direction):** (1) judge switched from a float score to a
  16-criterion BOOLEAN rubric (10 per-hypothesis + 6 set-level), reward = fraction of booleans
  passing — LLMs can't ground continuous scores (they collapse to a few anchors; the old float
  judge gave ~5 distinct reward values across hundreds of evals). Granularity now comes from the
  QUANTITY of booleans (~280 levels for a 28-hyp pool). (2) added continuous per-class coverage
  (min/mean best single-hypothesis separation) so near-identical pools are still distinguished.
  (3) dead always-1.0 anti_hack term → penalty multiplier. See [[feedback-llm-judge-booleans]].
- **Next:** fresh GEPA run under the new reward (DeepSeek-pro teacher, the winner) before any
  adoption/transfer decision. Prior candidate instructions and all reward numbers above are
  old-reward artifacts.

### 2026-07-04 — GEPA DeepSeek-pro run complete + two fixes (judge feedback, teacher swap)

DeepSeek-v4-pro teacher run finished (700-call budget). Result judged against pre-registration:
- **Reward: PASS.** Saved instruction = candidate 1 (geo-mean 0.761 vs 0.744 baseline, +0.017,
  clears the +0.01 bar). Dominates baseline on BOTH tuning tasks (trec 0.701, sst2 0.825).
- **Qualitative: PASS.** 4379-char structured instruction (sim 0.05 to baseline — a real rewrite,
  not paraphrase), fully dataset-agnostic (no dataset/class-name leakage), not reward-hacked.
- **Transfer gate: PENDING** — the decisive test (full method on held-out ag_news at -l, McNemar
  vs baseline instruction) not yet run. +0.017 is on the tuning sets only.
- **Diversity diagnosis (Lee's Q): not temperature (already 1.0).** Only 2 candidates accepted over
  105 iters; candidate 1 found early (call 32), then ~100 non-improving near-duplicate proposals.
  Cause: reflection model in reasoning mode converges regardless of temperature + likely reward
  plateau near the encoder ceiling.

Two fixes committed as a result:
1. **Judge critique was being DISCARDED** — make_judge returned only the float score, so GEPA's
   reflection LM saw only quantitative feedback (eff rank/artifacts/CV), never the judge's semantic
   critique. Now the critique is piped into the reflection feedback and the judge signature elicits
   actionable, instruction-level critique. Likely a real contributor to the proposal plateau.
2. **Teacher is now configurable** (--teacher, --no-teacher-reasoning, --teacher-temp) with the
   judge decoupled (--judge) so a teacher swap varies only that. Next experiment: teacher =
   z-ai/glm-5.2 (Lee's pick), judge held at deepseek-pro, distinct --out. Confound to note: GLM
   run also carries the improved judge feedback, so a clean teacher ablation would re-run deepseek
   under the new feedback too; deferring that unless GLM's result is ambiguous.

### 2026-07-04 — FINAL GEPA verdict: instruction tuning delivered NO value (clean A/B)

Completed the missing cell. Full -l A/B on TREC (metrics.json headline):
| run | proposer + instruction | acc | F1 |
|---|---|---|---|
| trec_baseline_l | flash + HAND-WRITTEN | **0.952** | 0.939 |
| trec_pro_l | pro + hand-written | 0.946 | 0.925 |
| trec_tuned_l | flash + GEPA-tuned | 0.946 | 0.924 |

McNemar (compare tool, fresh refits):
- **Clean instruction A/B (same flash proposer): tuned 0.948 vs hand-written 0.958, p=0.36** — not
  significant, trending SLIGHTLY WORSE. The GEPA-tuned instruction did not beat the hand-written one.
- flash vs pro (hand-written): 0.946 vs 0.958, p=0.21 — equivalent (proposer model not a lever; cf #46).
- **Definitive close of the GEPA arc:** instruction tuning = no benefit (equivalent/slightly worse),
  proposer model = no benefit, ENCODER = the only significant lever (-m→-l +5pts, p=0.024). The simple
  hand-written prompt + cheap flash + bigger encoder is the winning config. Tuned instruction NOT
  adopted. Value delivered by the effort: a reusable, audited instruction-tuning loop + the McNemar
  `compare` tool + the covariance dedup — methodology, not scoreboard.
- Next real lever (from the 2402.12368 review): a better-GENERALIZING frozen encoder (domain+length
  robust) for the OOD/long-text weakness (20newsgroups), not more instruction/proposer work.

### 2026-07-04 — hourly: cleaning up the instruction A/B (flash+baseline+-l — the missing cell)

The GEPA verdict rests on an incomplete A/B. We have: flash+tuned+-l 0.946, pro+baseline+-l 0.946,
flash+baseline+-m 0.920. MISSING: flash+baseline+-l. Without it we can't tell if the tuned
instruction did anything, or if flash+baseline already reaches 0.946 at -l (instruction irrelevant).
Launching trec_baseline_l (flash + hand-written instruction + -l) — the clean same-proposer A/B vs
trec_tuned_l. Pre-registered decision rule:
- flash+baseline+-l ≈ 0.946  => tuned instruction added NOTHING (hand-written prompt at ceiling; GEPA
  delivered no value beyond a reusable loop). Most likely given equivalence so far.
- flash+baseline+-l < 0.946 (e.g. ~0.93) => the tuned instruction LIFTED cheap flash to pro-level =>
  GEPA delivered real value (better prompt compensates for a cheaper proposer).
Then McNemar compare trec_baseline_l vs trec_tuned_l. ~26 min at -l; one-time, disambiguates verdict.

### 2026-07-04 — VERDICT: tuned instruction EQUALS hand-written at -l (no improvement, no regression)

trec_tuned_l (GEPA-tuned instruction + flash + -l): **0.9460 / F1 0.9244**. vs trec_pro_l
(hand-written + pro + -l): 0.9460 / F1 0.9248. McNemar (compare tool): delta +0.002, discordant
21 (10/11), **p=1.0 — statistically equivalent**. It did NOT under-perform, so the lexical-
suppression contingency (tfidf variant) was not triggered; the tuned pool evolved fine (held-out to
0.885) and the wh-word suppression concern didn't bite on TREC.
- **Per the pre-registered adoption rule (significant improvement + transfer), the tuned instruction
  is NOT adopted** — it matches, doesn't beat, the hand-written one. Consistent with the whole
  session's finding: the hand-written GeneratePool prompt is already near the ceiling; the encoder is
  the real lever (bounded-upside, flagged from the start).
- Positive read: the tuned instruction + CHEAP flash matched the hand-written + expensive pro, and
  the GEPA loop is now principled and reusable (grounded-boolean incremental reward, audited terms).
- Caveats: TREC was in the tuning set (fit, not transfer) and proposer models differ (flash vs pro),
  so not a clean instruction-only A/B. A held-out transfer test (20newsgroups) remains available but
  equivalence on an in-tuning dataset already argues no adoption benefit.

### 2026-07-04 — Testing tuned instruction at -l; lexical-suppression hypothesis pre-registered

3-domain re-tune finished (sentiment leakage much reduced: v1 saturated with positive/negative/
soundtrack/movie -> now only "sentiment/acting/plot" survive; no dataset names). Wired the tuned
instruction into the proposer (lm.instruction_path, commit 6017690) and launched trec_tuned_l
(tuned instruction + flash + -l) to compare against trec_pro_l (hand-written instruction + pro + -l,
0.946/F1 0.9248). Caveat: proposer model differs (flash vs pro), so not a perfectly clean
instruction-only A/B.

**Pre-registered interpretation (Lee's point):** if trec_tuned_l UNDER-performs, a likely cause is
that the tuned instruction suppresses LEXICAL/surface hypotheses — the reward penalizes them
(semantic_not_surface judge criterion + length/vacuity hack penalty), but on TREC wh-word cues
("starts with 'Who'") are LEGITIMATE question-type signal, not hacks. Remedy: re-run with the
TF-IDF lexical channel (configs/trec_tuned_l_lex.yaml) to restore that signal externally; if that
recovers the gap, it confirms the semantic-purity reward over-penalizes task-legitimate lexical cues.

### 2026-07-04 — Reward audited + rebuilt (all terms valuable, incremental), 3-domain re-tune launched

Audited the reward on the completed run's 641-eval log — most terms were NOT valuable:
cv_skill/min_coverage/mean_coverage mutually redundant (corr 0.93-0.97), raw diversity FOUGHT
accuracy (corr -0.79), judge near-noise (corr +0.09, 9 distinct). Rebuilt (commit 3e371f1):
- **Incremental weighted sum, no binary gates** (Lee: gates too binary): score =
  (0.7*cv_skill + 0.3*judge)/norm - 0.2*hack_fraction. Every term moves the score smoothly.
- cv_skill = accuracy value (also penalizes collapse implicitly -> diversity dropped from score,
  kept in feedback). hack_fraction = incremental artifact subtraction (was a ~constant gate).
- **Judge is now a positive reward TERM** (Lee: use judge results as part of reward, not just a
  gate) AND cut to 6 SEMANTIC criteria the quantitative terms can't measure (surface-hacking,
  label-leakage, class/angle coverage, contrast, vacuity); dropped format-compliance criteria the
  LM always passes (the dead ones behind the near-noise). Per-criterion booleans now logged so each
  can be audited/pruned next run. See [[feedback-llm-judge-booleans]].
- Overfit fix: re-tune on 3 DOMAINS (trec question / sst2 sentiment / ag_news topic) so geometric-
  mean pressure scrubs domain-specific examples (v1 baked in sst2/movie-review specifics). 20ng held
  out for the transfer gate. Launched fresh: 105 train + 30 val, auto='light'. ~2h expected.

### 2026-07-04 — GEPA COMPLETE: good generic scaffolding, but sentiment-overfit examples

Run finished: 15 iters, 14 candidates accepted, 516 metric calls, 4910-char instruction.
Read the actual instruction (not just metrics):
- **GEPA learned our reward's structure** — the instruction explicitly encodes effective rank /
  avoid-paraphrases, avoid-surface-artefacts (length/punctuation/word-presence), cover-all-classes
  from multiple angles, vary specificity, contrastive splits, target-minority-classes. The
  granular boolean+coverage reward + judge critique clearly transmitted; this is a well-formed,
  mostly-generic strategy prompt. Big step up from the old-reward run (2 candidates, thin edits).
- **BUT it overfit its EXAMPLES to sst2/sentiment**: every illustration is review/sentiment —
  class defs "negative: the reviewer expresses dislike", a literal SST-2 example "[negative] a
  hokey piece", aspects "acting, pacing, soundtrack, plot", and an "emotional valence/affect"
  angle that only helps sentiment. The scaffolding is generic; the content is domain-biased. This
  is the same tuning-set overfit that shelved GEPA v1 (+5.4 TREC / -3.3 AG News), now visible in
  the text rather than only in transfer numbers.
- **Verdict vs pre-registration:** dataset-agnostic gate = QUALIFIED FAIL (generic principles, but
  sentiment-leaked examples). Decisive test remains the -l ag_news TRANSFER gate — but that needs
  the tuned instruction wired into the proposer/runner (not yet supported). If it transfers despite
  the sentiment examples, adopt; if the examples drag topic-classification generation, don't.
- **Cheap improvement idea (higher value than adopting as-is):** the sentiment leakage comes from
  tuning on only trec+sst2. Re-tune with a 3rd, different-domain context (e.g. ag_news topic) held
  IN, and keep a truly held-out dataset for the gate — the geometric-mean pressure should scrub
  domain-specific examples. Or strip/greek the illustrative examples post-hoc.

### 2026-07-04 — GEPA run PRODUCTIVE under new reward (hourly check, 2h in)

Checkpoint read (not just liveness): total_num_evals 444, iteration 13, **12 candidates accepted**
— vs only 2 under the old float-judge reward. The granular boolean+coverage reward gives GEPA a
real gradient; latest instruction is heavily evolved (6983 chars, sim 0.017 to seed). This is the
payoff of the session's rework (grounded booleans + continuous coverage + true parallelism).
- Downside: at ~4/min and iter 13, auto='light' over valset=30 will run several more HOURS. Best-so-
  far is checkpointed, so `touch models/proposer_instruction.stop` ends it early with the best kept.
  Lesson for next time: valset=30 makes auto='light' a multi-hour job — use valset ~12-16 for a
  genuinely quick light run, or medium/heavy only when you want the long search.
- Verdict (reward vs new-scale baseline, dataset-agnostic read, -l ag_news transfer gate) pending
  completion or a manual stop.

### 2026-07-04 — GEPA run >1h (hourly check): auto='light' scales with valset

Run healthy and live at 1h02m / 357 evals (~4/min sustained; wal + evals fresh). Longer than
"light" suggests because auto='light' budget scales with valset size, and I set valset=30 — so a
bigger validation pool makes "light" not-light. Tradeoff noted: if faster iteration is wanted,
shrink valset (e.g. 16) or the whole thing scales down. Not stopping it (Lee's run, progressing).
Verdict (reward vs new-scale baseline, dataset-agnostic check, -l ag_news transfer gate) still
pending on completion — the one outstanding deliverable.

### 2026-07-04 — GEPA parallelism + reward granularity fixed (hourly check)

Iterated the tuner with Lee to canonical dspy usage + fixes; run live and healthy now:
- **Simplified to canonical dspy GEPA**: `auto='light'` budget (not hand-set max_calls/stop_callbacks),
  large trainset (~60) + small valset (~30, docs say <=35). CLI is just --out/--tune/--teacher/
  --judge/--auto/--threads/--fresh. --fresh wipes the checkpoint (avoids the resume-terminate trap).
- **Judge = grounded booleans, not floats** (float scores are ungrounded, collapse to a few anchors —
  [[feedback-llm-judge-booleans]]). Kept CONCISE for speed: 18 SET-LEVEL booleans + one short fix
  line (a per-hypothesis rubric hit ~2500 tok/call, too slow). Continuous cv/coverage terms carry
  the fine granularity. Result: reward now takes 20 distinct trec / 12 sst2 values (was 5/5).
- **Parallelism fixed** (was ~2-3 req/min): the metric held a GPU lock before the judge, serializing
  the LLM calls; moved judge before scoring, then (Lee: concurrent GPU is fine here) removed the GPU
  lock entirely so scoring overlaps too. num_threads=8 now genuinely 8-wide → ~32 evals/min.
- Reward numbers are on the NEW scale — NOT comparable to old-reward runs (DeepSeek 0.761, GLM). A
  fresh run under this reward is the basis for any adoption decision; transfer gate (ag_news @ -l)
  still the acceptance test.

### 2026-07-04 — GEPA run status (in flight, hourly check)

Light-scale run live and healthy (~154/700 calls, 22 min, 149% CPU — thread cap holding,
checkpoint + WAL fresh). The 40-call probe was budget-starved (only 2 distinct instructions
explored); at 700 calls it is genuinely searching — 5 distinct rewards/dataset, best paired
geo-mean **0.77 vs 0.744 baseline (+0.026)**, clearing the pre-registered +0.01 reward bar.
NOT yet judged: pairing is adjacency-approximate, gain is on the tuning sets (trec+sst2), and
adoption still requires the -l McNemar transfer gate on held-out ag_news. Verdict when it lands.

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

### 2026-07-04 — learnings from two papers (EDEntail; synthetic-NLI 2402.12368)

Both reinforce our measured verdict: the NLI encoder's coverage/generalization is the ceiling, not
label-side cleverness (instructions/proposer are saturated — GEPA null). Prioritized takeaways:
1. [HIGH, 2402.12368] Better-generalizing FROZEN encoder for OOD + LONG premises — exactly our
   setting (OOD LM-hypotheses) and open weakness (20ng long docs; we truncate at 1200 chars). The
   encoder is the only measured lever (+5pts). Action: A/B a domain/length-robust NLI model on 20ng.
2. [CHEAP, EDEntail] Extensional/DISJUNCTIVE hypotheses for internally-diverse classes ("asks for a
   count, a date, or a percentage") — one high-recall feature; addresses the class-internal-diversity
   problem. Caveat: cuts against the tuned reward's single-claim/non-vacuous pressure -> a deliberate type.
3. [CHECK] max_text_chars=1200 is a silent length cap; 2402.12368 makes length first-class. Probe on 20ng.
4. [LOW, EDEntail] hypothesis-format ensembling — marginal (we already have many-hypothesis diversity).

In flight: trec_pro_tuned_l (missing grid cell) + trec_cov_l (from-scratch with covariance deduper).

### 2026-07-04 — CONFIRMED: tuned instruction suppresses task-legitimate lexical features

Direct pool comparison (same flash proposer + -l, only instruction differs):
- trec_baseline_l (hand-written): 13/60 lexical-wh-word hypotheses ("begins with 'Who'", "contains
  'stands for'", "How many/How much", "the name of", ...).
- trec_tuned_l (tuned): 1/49.
The GEPA reward's semantic_not_surface judge criterion + length/vacuity hack penalty trained the
instruction to avoid surface features (13->1). But on TREC wh-word cues are LEGITIMATE question-type
signal, not hacks -> tuned sits below hand-written (test 0.946 vs 0.952; evolve ~0.878 vs ~0.895).
Gap is small only because -l semantic paraphrases recover most of it.
LESSON: "surface" != "hack". semantic_not_surface over-penalizes task-legitimate lexical cues; a
better reward penalizes surface features only when they DON'T generalize (cv_skill already catches
that). Remedy queued: trec_tuned_l_lex (tuned + TF-IDF channel) — if it recovers ~0.952, diagnosis confirmed.

### 2026-07-04 — 2x2 proposer x instruction grid complete (all within noise at -l)

TREC pool_cv at -l (metrics.json):        hand-written | tuned
                                   flash:     0.952     | 0.946
                                   pro:       0.946     | 0.952
All four within the ~0.5-1pt noise floor (McNemar p>0.2 across cells). Neither proposer model nor
instruction tuning is a lever at -l. Lexical suppression (13->1 wh-word hyps in tuned pools) is real
in COMPOSITION but costs ~0 ACCURACY at -l (encoder absorbs it via semantic paraphrases; would more
likely bite at -m, untested). trec_cov_l (covariance deduper, from scratch) running next.

### 2026-07-04 — covariance deduper validated (= STS), STS dependency dropped

trec_cov_l (covariance dedup, from scratch) 0.950/F1 0.929 vs trec_baseline_l (STS dedup, identical
else) 0.952/F1 0.939. McNemar 0.948 vs 0.958, delta -0.010, p=0.27 -> not significant / equivalent.
Covariance (feature-space) dedup matches STS at no accuracy cost, in the correct space, and drops a
model dependency (ModernCE-STS). Swap validated end-to-end.
Now running: trec_tuned_l_lex (tuned instruction + TF-IDF channel) — tests if restoring lexical
signal recovers the tuned pool's small gap (tuned 0.946 vs hand-written 0.952).

### 2026-07-04 — blind-spot hunt: a NEW hypothesis STYLE (answer-imperative), + an intrinsic ceiling

Diagnosed trec_baseline_l. Only hard confusion: ENTY->DESC (7 errors, best-separator AUC 0.84; all
other pairs >0.97). Confused cases are ambiguous "What is X?/What is X made of?" (question reads DESC,
gold answer is an entity). Tested candidate hypotheses in styles we DON'T use, scored at -l on the
ENTY/DESC subset (AUC vs the 0.84 current best; distinctness = max |corr| with pool):
- paraphrase->imperative "equivalent to asking someone to EXPLAIN/DEFINE X": AUC 0.842, corr 0.71 (distinct)
- contrastive answer-type "a thing, not a description": 0.784, corr 0.48
- answer-length "answerable in a few words": 0.748, corr 0.27 (very distinct)
- content "made of/composed": 0.608 ; "nameable thing": 0.525  (encoder can't ground these)

Findings:
1. ENTY/DESC is an INTRINSIC ceiling — no style (old or new) beats ~0.84; genuine label ambiguity,
   not a coverage gap. Don't expect to fix it with better hypotheses.
2. Real blind-spot STYLE: every pool hypothesis describes the QUESTION (intent/lexical/topic); NONE
   describes the ANSWER's form or reduces the question to its imperative. "answer-imperative"
   (equivalent to 'Name X' / 'Explain X' / 'Locate X') and "answer-length/form" are new, DISTINCT,
   competitive features -> worth adding to the GeneratePool instruction as an explicit angle.
3. Encoder grounds ABSTRACT answer-type framings (explain-vs-name) >> concrete compositional ones
   (made-of) -> instruction should steer to answer-imperative, not compositional, framing.
Proposed instruction addition: an angle "what a valid ANSWER looks like — reduce the question to its
imperative (Name/Explain/Define/Locate/Count X) and to the answer's form (a short name vs a sentence)".

### 2026-07-04 — answer-oriented instruction: generated + evaluated + tweaked

Added an ANSWER-oriented angle to _RULES, then generated 30 TREC hyps and scored at -m.
- v1: 10/30 answer-oriented but flawed — vacuous forms ("answered with a single word" AUC 0.591,
  "a phrase" 0.768) and answer-FORM restatements redundant with intent ("answered with a person's
  name" 0.915 ~ "asks for a person's name" 0.941 -> covariance dedup would drop them).
- Tweaked _RULES: ban vacuous forms, forbid restating intent as answer-form, emphasize answer-
  imperative + the disambiguating short-name-vs-full-sentence contrast.
- v2: 6/30, ALL clean answer-imperatives, strong + distinct: locate-a-place 0.966 (LOC), name-a-
  person 0.961 (HUM), give-a-number 0.908 (NUM), expand-an-acronym 0.907 (ABBR), explain-in-full-
  sentences 0.750 (DESC). Zero vacuous, zero redundant.
Verdict: instruction now produces clean distinct answer-oriented hyps (genuine vocabulary
enrichment). Answer-LENGTH axis (ENTY/DESC discriminator) still light — but that's the intrinsic
ceiling, not worth pushing. Whether it lifts final accuracy needs a full run on a dataset where
answer-type ambiguity is the failure mode (TREC ceiling already hit).

### 2026-07-04 — lexical remedy result: TF-IDF closes the tuned pool's gap (diagnosis confirmed)

trec_tuned_l_lex (tuned instruction + TF-IDF channel): 0.956/F1 0.935.
- vs trec_tuned_l (tuned, no lexical) 0.946: +0.010; McNemar 0.948->0.956 p=0.50 (right direction,
  within TREC-500 noise).
- vs trec_baseline_l (hand-written) 0.952: McNemar 0.956 vs 0.958 p=1.0 — EQUAL. The TF-IDF channel
  fully recovers the ~1pt the tuned instruction lost by suppressing wh-word features (13->1).
Confirms the lexical-suppression mechanism directionally: restoring lexical signal externally brings
the semantics-only tuned pool back to parity with the hand-written pool. All configs cluster ~0.95
(TREC-500 noise floor). CLOSES the instruction-tuning arc:
- GEPA-tuned instruction: no accuracy benefit (= hand-written); it also SUPPRESSES task-legit lexical
  features, recoverable via the TF-IDF channel.
- Proposer model (flash vs pro): no benefit.
- Encoder (-m->-l): the only significant lever.
- Covariance deduper: = STS, dropped the STS dependency.
- Answer-oriented instruction style: added + tweaked; produces clean distinct hypotheses (vocabulary
  enrichment; untested on a dataset where answer-type ambiguity is the failure mode).

### 2026-07-04 — testing the updated hand-written instruction (answer-oriented) at -m
Pre-reg: trec_newinstr (new instruction + covariance dedup + -m) vs committed trec (0.920, old
instruction + STS). -m chosen because -l washed all configs to ~0.95 noise; -m is where instruction
quality can show, and TREC is a QA task the answer-oriented style fits. Expect: >0.920 if the answer-
oriented hyps add distinct signal; ~0.920 if redundant. McNemar to judge. Dedup change is ~neutral (p=0.27).

### 2026-07-04 — answer-oriented instruction: +0.014 at -m (promising); tree-structured output added

trec_newinstr (updated instruction + covariance dedup, -m) = 0.934 vs committed trec (0.920):
McNemar +0.014, discordant 31 (12 old-only / 19 new-only), p=0.28. Directionally positive, largest
instruction delta seen, at -m where instruction quality shows (-l washed to noise). Not significant
on one seed -> promising, needs more seeds/datasets to confirm. (dedup change ~neutral, p=0.27.)

Then implemented Lee's decision-tree methodology IN THE DATA STRUCTURE (commit 0248fb5): GeneratePool
returns `tree` (list[SplitNode]: depth/separates/hypotheses, root=grouping -> leaves=boundary) + a
separate flat `hypotheses` list (current approach); _flatten merges both. Not yet generation-tested.
Next: generate with tree structure, check it yields real grouping/boundary hyps + whether it helps at -m.

### 2026-07-04 — tree (grouping) structure: neutral on TREC; head already derives groups

Balanced tree (group-vs-group splits) + flat list vs flat list only:
- trec_tree_m (tree+list) 0.926 vs trec_newinstr (list only) 0.934: McNemar delta -0.008, p=0.57 —
  NOT significant (within noise, directionally slightly negative).
- ~6 group-style hyps DID survive evolution ("named entity such as a person, place, or thing";
  "a person or a place") — not pruned, just not more useful than one-vs-rest.
Why no gain: the RF/HGB head already DERIVES group boundaries by combining one-vs-rest features, so
explicit balanced-group features are largely redundant; a couple go vacuous (fire for 5-6 classes).
Caveat: TREC is a WEAK test for grouping — 6 flat, cleanly-separable classes, no hierarchy for group
features to exploit. Grouping would only earn its keep on a dataset with genuine class hierarchy that
one-vs-rest misses. Tree structure kept (interpretable, neutral); its real test is a hierarchical dataset.
Session theme reconfirmed: label-side structural cleverness (instruction tuning, grouping) is
neutral/within-noise; the ENCODER is the lever.

### 2026-07-04 — 1-cov feature weighting: no-op for tree heads (verified)

Q: weight features by 1-cov? Tested on trec cached features (weight = 1 - mean|corr| per feature):
HGB 0.8760=0.8760, RF 0.8615=0.8615 (identical — trees are scale-invariant), LogReg+scaler
0.809->0.8095 (scaler erases the weight; LogReg worse anyway). Verdict: no-op for our RF/HGB head.
Principle: redundancy belongs in feature SELECTION (drop), not WEIGHTING (trees ignore scale). Already
handled: covariance dedup (drop |corr|>0.95) + permutation importance (redundant feature -> ~0 marginal
importance -> pruned). Only matters for an unscaled linear/distance model, which we don't use.

## 2026-07-04 — trec_tree128_m (pool 128, crowding-out hypothesis) + low-N pivot

Q: does pool=128 unlock weak-but-valuable features that pool=64 crowds out? VERDICT: NO.
trec_tree128_m test acc **0.930** (macro-F1 0.9146) vs trec_tree_m@64 0.926 and trec_newinstr@64
**0.934** — 0.930 sits inside the @64 spread, within TREC-500 noise (~0.5-1pt). cv_train edged up
(0.8825 vs 0.87) but test did not -> extra capacity, no generalization gain. Doubling the pool just
carried more redundancy the CV-selected head ignores; no suppressed valuable features exist.
Pool sustained a full 128 (refill replenishes each round; NOT collapsing to ~53 like the @64 runs —
my mid-run "collapsing" read was wrong; the falling counts were post-prune, pre-refill). Evolution:
heldout 0.848->0.859->0.863->0.864->0.864->0.873 over 6 rounds.

Reward-hacking audit (read pool + pruned lists, rounds 0-5): CLEAN. Survivors dominated by meaningful
answer-type ("equivalent to asking someone to name a person"), intent ("asks for a reason"), content
hyps. Surface first-word triggers (Who/Where/When/How-many, "stand for" for ABBR) survived because
genuinely discriminative for question-type + fully interpretable — feature not bug. Pruning correctly
removed VACUOUS ("asks for a clarification", "a detailed account", "a specific named entity") and
REDUNDANT (round 3 "...calculate something" hit the "signal nearly identical to a kept hypothesis"
annotation). No val-gain collapse, no length/punctuation exploits.

PIVOT (see docs/low-n-plan.md): all data-rich null results (128, tree grouping, GEPA, style, pro
proposer) are null BECAUSE the head compensates when data is abundant. The method's value is
transfer-knowledge-for-labels, largest at low-N. Next work is the low-N learning-curve study (evolution
OFF, prior-aggregation/strong-L2 head, STS dedup, crossover lines STS-vs-cov & tree-vs-flat), NOT the
old tree/boost configs. Cron step-4 auto-launch (sst2_boost/ag_news_boost/20newsgroups_tree) is
OBSOLETE — those configs don't exist; that plan is superseded. Did not launch anything.

## 2026-07-04 — best method @ -l (trec_best_l)

Took best -m method (trec_newinstr: new answer-oriented instr + covariance dedup + flat/tree pool 64
+ flash) and ran encoder -l. Result: acc **0.954**, macroF1 0.9591, pool 64 (converged clean — round 4
pruned 0). Evolution heldout 0.908->0.916 (vs ~0.86 at -m: encoder lever in the internal metric too).

Compare vs trec_baseline_l on common refit basis (compare.py refits both heads via fit_head; note
this reads baseline_l as 0.958, not its stored 0.952 CV-head number): delta n.s., discordant 16
(7 vs 9), **McNemar p=0.80**. Accuracy TIED — instruction washes out at -l (confirms prior: -l band
~0.95 regardless of instruction/proposer).

macroF1 looked like a +0.015-0.02 win but it is ENTIRELY 2 ABBR examples (9-example class): best_l
9/9 ABBR (F1 1.000) vs baseline 7/9 (F1 0.875); ÷6 classes = the whole macro delta. On larger classes
baseline is slightly AHEAD (HUM 0.976 vs 0.952, DESC 0.965 vs 0.957), which is why accuracy tips to
baseline. Mechanistically plausible (best_l pool has ABBR answer-hyps "asks what a set of letters
stands for") but n=9 on one seed = NOT a result; needs seed-sweep to believe. Verdict: best method at
-l = tied at ~0.95; refinements only reshuffle which tiny class eats errors. Reinforces low-N thesis
(docs/low-n-plan.md): encoder-rich regime saturates method refinements.

## 2026-07-04 (hourly check) — trec_best_l_max RUNNING (lexical-aware pruning)

Status: alive/healthy (WAL fresh <30s, log advancing), round 1/10, ~5 min in. First run exercising
the new marginal-over-TF-IDF pruning (commit a595fae). Pre-registered expectation (README/prior
notes): accuracy holds in the ~0.95 -l band while the NLI pool SHRINKS (fewer forward passes/pred).
Early signature confirms more aggressive pruning vs best_l (NLI-only): round 0 pruned 13 vs 7,
round 1 pruned 10 vs 12; held-out 0.9114/0.9151 (on NLI+lexical). Verdict deferred to completion:
will judge on (1) final NLI pool size < best_l's 64, (2) test acc within noise of best_l 0.954 /
tuned_l_lex 0.956, (3) which hypothesis types got dropped (expect wh-word/keyword ones lexical covers).
No new job launched (one in flight; no CUDA parallelism).

## 2026-07-04 — trec_best_l_max = 0.964 (best TREC -l point estimate) + evolve regression diagnosis

trec_best_l_max (answer-oriented instr + covariance dedup + LEXICAL-AWARE pruning + tfidf_svd 128,
pool 64, 10 rounds, -l): **acc 0.964, macroF1 0.968**, best TREC -l number to date
(best_l 0.954, tuned_l_lex 0.956, baseline_l 0.952). BUT McNemar vs tuned_l_lex (both have tfidf, so
this isolates lexical-aware PRUNING): delta ~+4 net examples, discordant 9/5, **p=0.42 — NOT
significant**. So 0.964 is the best point estimate, statistically tied with tuned_l_lex; the
lexical-aware-pruning gain is promising but unproven on one seed. Do not report as an established win.

Evolution regression (Lee flagged): trec_best_l_max PEAKED at round 3 (heldout 0.9201) then dipped
every round to round 7 (0.9164), patience-4 stop. Old code shipped the round-7 pool (last), NOT the
round-3 peak — the checkpoint-best fix (commit 3aa0772) now ships the peak.

Root cause of regression: evolve does BLIND SWAPS — prune stability==0 (noisy 4-fold estimate kills
real features), then add UNTESTED refills; pool_{i+1}=survivors+refills, refills never compared to
what they replaced. So a good-but-unlucky feature gets swapped for a worse refill => structural
regression, not bad luck. Fix (Lee's design, next commit): grow-then-select — generate refills, MERGE
to ~2x (128), then importance+covariance PRUNE back to 64; a refill enters only if it out-ranks an
incumbent. Plus a strict accept gate (revert if the new pool regresses beyond noise) => monotonic.

## 2026-07-04 (hourly) — trec_growselect_l RUNNING: grow-then-select is MONOTONE (early)

New evolve (commit 785e051) first run, healthy (WAL fresh, 358% CPU, round 1/10). Pre-registered
expectation: (1) held-out monotone across rounds (no post-peak dip — accept gate), (2) test acc holds
or beats old-evolve best_l_max 0.964. Early evidence for (1) CONFIRMED: round0 0.9114 ->merge 0.9164
(accepted); round1 0.9164 ->merge 0.9176 (accepted). Monotone, vs old-evolve best_l_max which dipped
(0.9114->0.9151->0.9139 at round2). Consistency check holds: round1 entry heldout (0.9164) == round0
merged_acc, confirming accept-gate CV matches next-round ranking CV. Refills 64->60 (dedup trimming as
seen grows). Verdict on (2) deferred to completion (McNemar vs best_l_max). No new job launched.

## 2026-07-04 — trec_growselect_l VERDICT: grow-then-select OVERFITS held-out (significant, negative)

Pre-registered: (1) monotone held-out, (2) test holds/beats best_l_max 0.964.
(1) CONFIRMED — textbook monotone with 5 reverts: 0.9114->0.9164->0.9176->0.9226->0.9276->0.9313
(rounds 3,5,7,8,9 REVERTED by the accept gate; shipped round-7 peak checkpoint 0.9313). Mechanism
works exactly as designed.
(2) FAILED, significantly. TEST **0.948** vs best_l_max **0.964**, McNemar **p=0.0215**, discordant
9-to-1 in best_l_max's favor. Held-out went UP 1.1pt, test went DOWN 1.6pt.

Diagnosis: grow-then-select (merge 128 -> select top-64 by CV importance, accept gate, ship max-heldout
checkpoint) is a much STRONGER optimizer of the held-out estimate — which is CV on the rank_sample(800)
subsample, a NOISY PROXY. Optimizing it hard overfits it -> test generalization drops. Pool is clean
(6 legit surface hyps, no reward-hacking), so this is statistical selection-overfitting, not bad
hypotheses. Corollary: the OLD blind-swap evolve's churn/"regression" that Lee flagged was acting as
IMPLICIT REGULARIZATION — its noise prevented overfitting the CV folds (it shipped a 0.9164-heldout
pool that tested 0.964). Even checkpoint-best is implicated: shipping the MAX-heldout pool ships the
most-overfit one; best_l_max got 0.964 partly BY shipping its non-peak last pool.

DECISION FOR LEE (not auto-reverting — you designed this): grow-then-select is committed as default
(785e051) but hurts test on this seed. Options: (a) revert evolve to blind-swap (regularized, 0.964);
(b) keep grow-then-select but fix the proxy — accept gate / selection on a SEPARATE held-out not used
for ranking (nested CV), bigger rank_sample, or lighter selection pressure; (c) keep checkpoint+evolve
but drop checkpoint-best (ship last, not peak) to reduce overfit. Caveat: single seed; a 2-3 seed
confirm would harden the conclusion before reverting. No code reverted, no new run launched (cron).

## 2026-07-04 (hourly) — PRE-REGISTER: cheap overfit test via growselect checkpoints (cached, no GPU run)

Decisive test of the grow-then-select overfitting finding on ONE completed run, cached features only:
evaluate TEST accuracy of each trec_growselect_l CHECKPOINT (rounds 0-9, held-out rose 0.9114->0.9313).
EXPECTATION: if grow-then-select overfits the held-out proxy, test does NOT track the monotone held-out
climb — test is flat or declines across rounds even as held-out rises. If instead test rises with
held-out, the 0.948<0.964 gap was seed noise, not overfitting, and grow-then-select is fine. This
resolves the a/b/c decision within a single run. Head per checkpoint: fixed HGB on (pool features +
tfidf), full train, eval on held-out test set (honest, cached).

## 2026-07-04 (hourly) — RESULT: held-out is ANTI-correlated with test (decisive, within-run)

Pre-registered test ran (checkpoint trajectory of trec_growselect_l; needed light GPU to score
intermediate pools on train+test — corrects the "no GPU" claim above). Per-checkpoint TEST acc vs the
monotone held-out climb:
  round: 0     1     2     3     4     5     6     7     8     9
  hout : .9114 .9164 .9176 .9226 .9226 .9276 .9276 .9313 .9313 .9313   (monotone +0.0199)
  test : .9540 .9620 .9520 .9560 .9560 .9560 .9560 .9480 .9480 .9480   (net -0.0060)
**corr(held-out, test) = -0.585 (NEGATIVE).** Best TEST = round 1 (0.962, LOW held-out); shipped pool
= max held-out (round 7) = WORST test (0.948). EXPECTATION (test doesn't track held-out) CONFIRMED and
stronger than expected: the rank_sample(800) held-out CV is ANTI-correlated with test once optimized
hard. grow-then-select + accept gate + checkpoint-best all optimize a MISLEADING target;
checkpoint-best is actively harmful (ships max-held-out = worst-test pool). Round 0 (initial pool, ZERO
evolution) already tested 0.954 and round 1 peaked 0.962 — evolution beyond round 1 HURT test here.

Decision (Lee's call): the PROXY is the problem, not the mechanism.
- (c) drop checkpoint-best won't save it (round 9 last = 0.948).
- (b) independent held-out for selection/accept (nested CV / held-out split not used for ranking) is
  the principled fix — give the optimizer an honest signal. RECOMMENDED.
- (a) revert to blind-swap (its churn regularized -> 0.964) is the safe fallback.
First confirm on a 2nd seed that grow-select underperforms blind-swap before reverting the committed
default (785e051): within-run corr is strong (n=1 run), cross-method test gap still one seed. No code
reverted, no full run launched (cron did the cheap diagnostic only).

## 2026-07-04 (hourly) — PRE-REGISTER: is the overfit fixable by a bigger held-out? (cached, no GPU)

Follow-up to the corr(rank800-heldout, test)=-0.585 finding. Question gating option (b): would a
LARGER held-out be an honest selection signal, or is the overfit intrinsic to selection pressure?
Cheap test (checkpoint pools already cached on full train+test): per growselect checkpoint, compute a
FULL-TRAIN (2000) 4-fold CV accuracy and correlate with test, vs the rank_sample(800) held-out.
EXPECTATION: if the proxy problem is SIZE/noise, full-train CV correlates POSITIVELY with test ->
option (b) with a bigger/independent held-out is validated. If full-train CV ALSO anti-correlates with
test, the overfit is from selection pressure itself (fitting folds), not held-out size -> favors (a)
revert or true nested CV (select and validate on disjoint splits). CPU only, features cached.

## 2026-07-05 (hourly) — RESULT: overfit is a SELECTION ARTIFACT; option (b) validated but signal weak

Per growselect checkpoint (cached, CPU): rank800 held-out (the SELECTION TARGET) vs an INDEPENDENT
full-train(2000) 4-fold CV vs test:
  round:      0     1     2     3     4     5     6     7     8     9
  rank800_ho: .9114 .9164 .9176 .9226 .9226 .9276 .9276 .9313 .9313 .9313  (monotone; it's optimized)
  fulltr_cv : .9140 .9135 .9140 .9125 .9125 .9220 .9220 .9145 .9145 .9145  (FLAT ~.913-.922)
  test      : .9540 .9620 .9520 .9560 .9560 .9560 .9560 .9480 .9480 .9480
corr(rank800_ho, test) = -0.585 ; corr(fulltrain2000_cv, test) = +0.126.

Interpretation: the -0.585 anti-correlation is a SELECTION ARTIFACT — optimizing the rank800 held-out
inflates it (0.9114->0.9313) while the INDEPENDENT 2000-CV never rises (stays flat) and test drifts
down. An independent held-out is NOT anti-correlated (+0.126) => option (b) [select/validate on a
DISJOINT split, nested CV] removes the active harm — VALIDATED as the fix. BUT +0.126 is weak: the
independent CV barely predicts test, and the checkpoint test wiggle (0.948-0.962) is near the TREC-500
noise floor (~1pt). So option (b) mainly PREVENTS HARM; it won't deliver big gains on TREC because
there's little real headroom (round 0-1 already ~0.954-0.962; further optimization chases noise).

Updated recommendation for Lee:
- (b) nested/independent held-out for the accept gate + selection = the principled fix; do this if
  keeping grow-then-select. Temper expectations (weak signal on TREC).
- Given the weak signal, a lighter touch also works: cap rounds low (round 1 was best) or revert to
  blind-swap (a) whose churn regularized to 0.964. On TREC these are all within noise of each other.
- The real test of any fix is a dataset with genuine evolution headroom (20ng / ag_news), NOT TREC.
No code changed, CPU-only cached analysis (no GPU, Lee's training untouched).

## 2026-07-05 (hourly) — PRE-REGISTER: generation-time variance check (vacuous hypotheses)

Decision-INDEPENDENT of the evolve a/b/c question. Vacuous/always-true hyps (near-constant entail
across texts, e.g. "The text is a question." on TREC) carry zero discriminative signal but PASS
covariance dedup (a constant z-scores to zeros -> corr 0 with everything, never rejected). They then
occupy pool slots (= cross-encoder forward passes at inference) until evolution's _failure_reason
prunes them. Question: what fraction of a freshly-generated pool is vacuous? EXPECTATION: if a
non-trivial fraction (>~5%) have entail-std < 0.02 on train, a cheap generation-time variance filter
(drop near-constant entail vectors as candidates are scored for dedup) removes dead weight earlier —
saving inference cost with no accuracy loss (they'd be pruned anyway). Test on cached growselect
round-0 pool (initial 64, already scored on train). CPU/cache only.

## 2026-07-05 (hourly) — RESULT: generation-time variance filter NOT worth it (2% vacuous)

Checked initial generated pool (growselect round-0, 64 hyps, cached): entail-std min 0.0057 / median
0.3297 / max 0.4483. Vacuous (std<0.02): **1/64 = 2%** ("The text asks for a phone number or age.",
mean 0.001), and evolution pruned it (0 vacuous survived to shipped pool). Below the pre-registered
~5% bar -> a generation-time variance filter saves ~1 hyp, not worth the code. LM + covariance dedup
already yield discriminative hyps. Backlog item "generation-time variance check / drop always-true
statements" -> CLOSED, no deficiency to target. No code change.

STATE OF PROJECT: cheap cached analyses are now exhausted. Remaining backlog needs either (a) Lee's
evolve a/b/c decision [grow-then-select overfits on TREC via selection artifact; option (b)
independent held-out validated as the fix but signal weak on TREC], or (b) a full run on a
HEADROOM dataset (20newsgroups/ag_news) where evolution can actually matter — TREC -l is saturated
(round 0-1 already ~0.95, rest is noise). No further cheap TREC diagnostics remain. Idle; awaiting a
direction from Lee.

## 2026-07-05 (hourly) — JUDGED: cross-dataset validation (ag_news/sst2, seeds 7/17, -m)

Top backlog item, decision-independent. Committed pool method vs baselines (pool_cv, honest protocol):
  ag_news (topic, 4-cls): method 0.891 (s7) / 0.8895 (s17)  | zero-shot-NLI 0.8855 | tfidf 0.8485
  sst2 (sentiment, bin) : method 0.9404 (s7) / 0.9404 (s17) | zero-shot-NLI 0.945  | tfidf 0.7007
Pools read for reward-hacking: CLEAN — ag_news topic hyps (stock indices, sports leagues, product
launches), sst2 sentiment hyps (disappointment, enthusiasm, sarcasm, pos/neg adjectives). Meaningful,
no artifacts.

VERDICT: method >> tfidf everywhere (+4.3pt ag_news, +24pt sst2) but only TIES zero-shot NLI —
marginally above on topic (+0.005) and marginally BELOW on binary sentiment (-0.005), both within
test noise. Seed stability excellent (ag_news delta 0.0015; sst2 identical). Interpretation: the
pool+head's value concentrates on MULTI-CLASS carving (trec question-type, ag_news topic); on BINARY
sentiment a single zero-shot hypothesis already suffices, so the pool adds ~nothing over zero-shot.
The method's real, reliable win is over lexical/TF-IDF (interpretable features that beat bag-of-words),
not over zero-shot NLI at -m on these two. Backlog item "judge ag_news/sst2 seeds 7,17" -> CLOSED.
(These are -m; -l would lift both but the encoder saturates. No McNemar vs baselines — the +-0.005
gaps are visibly within noise.) No code change; decision-independent of the evolve a/b/c question.

## 2026-07-05 — STATE OF PROJECT (capstone; cheap autonomous work exhausted)

Committed method: LM-written NLI hypotheses -> frozen cross-encoder entail/contradict scores ->
covariance-deduped, evolved pool -> CV-selected classical head (+ optional TF-IDF channel). Evolve is
currently grow-then-select + accept gate + checkpoint-best (commit 785e051) — UNDER REVIEW (see below).

What this session established (all honest protocol, pool_cv):
- Encoder is the only reliable accuracy lever (-m -> -l ~+5pt, p=0.024). Instruction/proposer/tree/
  pool-size/lexical-vs-not all wash out at -l (~0.95 saturation band).
- Best -m recipe = answer-oriented instruction (0.934 vs 0.920 original, seed 7). Best -l point
  estimate 0.964 (best_l_max, tfidf + lexical-aware pruning) but NOT sig vs tuned_l_lex 0.956 (p=0.42).
- Lexical-aware pruning (marginal-over-TF-IDF) committed: minimizes redundant NLI features -> fewer
  forward passes at inference. Sound; kept.
- Cross-dataset (ag_news/sst2, -m): method >> TF-IDF (+4-24pt) but only TIES zero-shot NLI. Value =
  multi-class carving (trec/ag_news) + interpretable features beating bag-of-words; adds ~nothing over
  zero-shot on binary sentiment.

OPEN DECISION (Lee's; blocks further progress) — grow-then-select overfits on TREC:
- It's monotone on held-out (0.9114->0.9313) but held-out is ANTI-correlated with test (corr -0.585);
  a SELECTION ARTIFACT (independent 2000-CV is flat / +0.126 with test). Test DROPPED: growselect 0.948
  vs old-evolve best_l_max 0.964 (McNemar p=0.02). checkpoint-best ships the max-held-out = worst-test
  pool. Old blind-swap evolve's churn was implicit regularization.
- Options: (a) revert evolve to blind-swap (regularized, 0.964); (b) [recommended] independent held-out
  for accept gate + selection (nested CV) — validated as removing the anti-correlation, but signal is
  weak on saturated TREC; (c) drop checkpoint-best (won't save it). ANY fix must be validated where
  evolution has real headroom — 20newsgroups / ag_news — NOT TREC.

Closed this session (no action needed): generation-time variance filter (2% vacuous, already pruned);
plateau-epsilon tune (no-op on testable trajectory); cross-dataset judging.

RECOMMENDATION: pause both crons. No cheap, decision-independent, non-noise work remains. Next real
progress = Lee picks the evolve fix (then implement + validate on 20ng), or greenlights a headroom-
dataset run. Idle otherwise.

## 2026-07-05 (hourly) — PRE-REGISTER: multi-pool ensemble on cached -l pools (variance/robustness)

Targets a real measured gap: all conclusions are single-seed, no error bars. Cheap proxy on CACHED
-l pools: fit each pool's head (pool features + its lexical config) on train, average predicted
class-probabilities across the independent TREC -l pools, argmax -> ensemble test acc. Question: does
a COMMITTEE of interpretable NLI-hypothesis pools beat the best single pool / reduce variance?
EXPECTATION: ensemble beats the MEAN single-pool acc and approaches the best; if it exceeds best_l_max
0.964, ensembling is a real lever worth a proper same-method 3-seed version. Honest: fixed ensemble of
ALL pools (no test-selection). Caveat: these pools overlap in hypotheses (not seed-independent), so the
benefit is a LOWER BOUND on what true seed-diversity would give. Cached features, CPU only, no GPU.

## 2026-07-05 (hourly) — RESULT: multi-pool ensemble reaches top of -l band robustly (0.964)

Cached ensemble of 9 independent TREC -l pools (each head refit with fixed HGB for a consistent
comparison; absolute nums differ slightly from stored cv_selected_head metrics):
  individual: 0.946-0.960 | mean single 0.9524 | best single 0.9600 | ENSEMBLE(all 9) 0.9640 |
  ENSEMBLE(top-5) 0.9640.
The committee beats the MEAN (+1.2pt) and any SINGLE pool (+0.4pt), landing at the top of the -l band
(0.964) ROBUSTLY — i.e. averaging independent interpretable pools gets you the best number without
having to pick the single luckiest pool/evolve-config (all within noise 0.946-0.960). +0.4pt over best
single = ~2 examples, not significant alone, but the variance-reduction direction is the point.
Caveat: pools overlap in hypotheses (not seed-independent) -> LOWER BOUND; true seed-diverse pools
should give more, and give real error bars (the project's current single-seed gap).

IMPLICATION (decision-independent of evolve a/b/c): the robust path to the top of the band is a POOL
COMMITTEE, not optimizing one pool — which also dodges the grow-select overfit problem (no single
held-out to overfit; diverse pools average out). Next step (needs runs, Lee's call): a proper
same-method 3-seed ensemble for genuine variance reduction + error bars. Backlog "multi-pool ensemble"
-> validated on cached proxy; promote to a real 3-seed run when a direction is chosen. No code change.

## 2026-07-05 — checkpoint-best is REDUNDANT with the accept gate (verified)

Lee's question: did best-checkpoint change the result? NO. Verified on growselect_l checkpoints:
rounds 7/8/9 are the IDENTICAL pool (accept gate reverted all three), so shipped == best checkpoint
(round 7) == last-round pool. checkpoint-best rescued nothing. Reason: with the accept gate ON the
running pool never drifts below best, so last==best always. checkpoint-best only acts WITHOUT an accept
gate (old blind-swap evolve, pool drifts post-peak) — and there the overfit finding (held-out anti-
correlated with test) says it would HURT (shipping the higher-held-out peak = lower test; best_l_max
shipped its DRIFTED last pool and got 0.964).
=> Both this-session additions fail to earn their keep: checkpoint-best (redundant/harmful),
grow-then-select+accept-gate (overfits, -0.016 test, p=0.02). Strengthens option (a) REVERT to
blind-swap. Awaiting Lee's decision.

## 2026-07-05 — YAGNI cuts (1/2/3): revert grow-select, delete GEPA + diagnostics

Per Lee. (1) REVERTED evolve to blind-swap (git checkout a595fae for evolve.py/runner.py/
test_evolve.py) — keeps the lexical-aware marginal-over-TF-IDF pruning, drops the grow-then-select
loop + accept gate + checkpoint-best (they overfit: test -0.016 p=0.02; checkpoint-best was redundant
with the accept gate). Evolve returns (pool, history) again; no checkpoints.jsonl. (2) DELETED the
GEPA subsystem — gepa_tune.py, reward.py, test_gepa_tune.py, test_reward.py, cli `gepa-tune` (verdict:
tuned instruction ties hand-written, p≈1.0 — no measured benefit). (3) DELETED diagnostics.py + cli
`diagnose` (superseded by `compare`). Also dropped dead wordllama earlier. No dangling refs; 23 tests
pass; ruff clean. Inference is now the sklearn HypothesisVectorizer (dspy-free); training deps in the
`train` group. Net: smaller, one featurization path, no proven-neutral/negative machinery.

## 2026-07-05 (hourly) — PRE-REGISTER: HypothesisVectorizer correctness parity on real data

Verify the shipped inference interface reproduces the method end-to-end (tests so far use fakes only).
from_run("runs/trec_best_l") on real cached -l features: (a) vec.transform(train) must EQUAL
scorer.features(train, pool) exactly (DRY — both wrap the same encoder); (b) Pipeline(vec, HGB) test
accuracy must reproduce the run's ~0.954 ballpark. EXPECTATION: exact feature parity; test acc within
noise of 0.954 (HGB vs the run's cv_selected_head differ slightly). Confirms from_run/transform are
correct, not just the fake-scorer unit tests. Cached (-l pool scored during the run), no GPU.

## 2026-07-05 (hourly) — RESULT: HypothesisVectorizer parity CONFIRMED on real data

from_run("runs/trec_best_l"): loaded encoder finecat-nli-l, 64 hyps, entail_contradict. (a) FEATURE
PARITY EXACT — vec.transform(train) allclose scorer.features(train, pool) (both wrap the same encoder);
shape (2000,128)=(n,2m). (b) END-TO-END — Pipeline(vec, HGB) test acc 0.9520 vs run pool_cv 0.954,
within noise (HGB vs cv_selected_head). The shipped inference interface reproduces the method correctly
on real cached data, not just the fake-scorer unit tests. All cache hits, no GPU. Verdict: interface
correct; no action needed.

## 2026-07-05 (hourly) — RESULT: standard sklearn workflow works + reproduces 0.964

Built the model as a plain sklearn Pipeline(FeatureUnion([HypothesisVectorizer.from_run(best_l_max),
make_pipeline(TfidfVectorizer, TruncatedSVD(128))]), HGB(lr=.12,l2=.01)). Results: clone OK; fit/score
= **0.9640 exactly** (== best_l_max pool_cv); get_feature_names_out = 256 readable names
(nli__entail:... + tfidf__truncatedsvd*); cross_val_score(cv=3) clones per fold fine. NO sklearn-
convention issues found — nothing needed rework; the vectorizer composes idiomatically (Pipeline /
FeatureUnion / ColumnTransformer / clone / GridSearchCV-ready / feature names). Locked in as
tests/test_sklearn_workflow.py (skips unless the run + score cache are present; cached -l, no GPU).

## 2026-07-05 (hourly) — from-scratch train via STANDARD sklearn workflow = 0.956

Ran a genuine from-scratch train through plain sklearn: HypothesisVectorizer(task=..., class_definitions
=..., n_hypotheses=64, encoder=-l) with NO hypotheses -> fit(X,y) GENERATES the pool via the LM
(deepseek-flash) + fresh -l encoding, inside Pipeline(FeatureUnion([nli, tfidf->svd128]), HGB(.12,.01)).
Generated 64 meaningful answer-oriented hyps; TEST acc **0.9560** (vs evolved best_l_max 0.964; ==
tuned_l_lex 0.956, within noise). So the standard sklearn workflow trains end-to-end from scratch and
lands in the -l band; evolution (CLI, step 3) is the ~+0.008 (noise-level) squeeze on top. Confirms
fit-generation works live (LM+GPU), not just fakes. README now has training code examples.

## 2026-07-05 — full code review of the sklearn migration (fixes committed)

Combed all ~2400 lines. Found & fixed:
1. PACKAGING BUG: `pip install "nli-boost[train]"` was documented but train was a PEP-735
   dependency-GROUP — groups don't ship in wheel metadata (verified Provides-Extra: None). Now a real
   extra; dev group self-references "nli-boost[train]" so `uv sync` still gives contributors everything.
2. Version mismatch pyproject 0.1.0 vs __init__ 0.3.0 -> 0.3.0.
3. DRY/CONSISTENCY: vectorizer had its own _labeled_examples with a DIFFERENT prompt format
   ("text -> name" vs the method's "[name] text") and a non-stratified dedup ref sample — vectorizer-
   generated pools saw different prompts than CLI pools. Unified on data.labeled_examples (+
   stratified_indices); datasets import made lazy so nli_boost.data is importable without it.
   Inference path now dspy-free AND datasets-free (sklearn users bring numpy/lists — Lee's point).
4. sklearn nativeness: random_state param (stochastic fit-generation), verbose=False param gating the
   encoder's progress prints (library etiquette; CLI keeps them via EncoderConfig.verbose=True).
5. save() now requires fitted (was silently writing the constructor arg).
6. from_config: generic constructor-param passthrough (carries generation params, ignores unknowns);
   from_run passes a minimal dict + uses the run config's cache_dir (no lm-dict collision).
7. Dead code: duplicate _labeled_examples, stray numpy re-import, dead RunConfig test import.
Reviewed-and-fine: runner/compare/head/evolve/proposer/cache/dedup/cli (single-purpose, documented);
build_matrices and vectorizer both delegating to EntailmentScorer.features is the correct DRY point.
33 tests pass (incl. new: save-requires-fitted, from_config passthrough), ruff clean.

## 2026-07-05 (hourly) — trec_full RUNNING (headline full-train run)

Lee-requested headline: best recipe (-l + tfidf_svd 128 + evolve, rounds 10) trained FROM SCRATCH on
the FULL TREC train (5452 examples, vs 2000 in dev), same 500 test. Healthy: evolution done (10
rounds, rank-subsample held-out peaked ~0.90), now fitting CV head on 5452x62 hyps + test scoring
(WAL fresh, 229% CPU, ~27min). PRE-REGISTERED expectation: meet-or-beat the 2000-train best_l_max
0.964 (more train data -> cleaner head + evolution signal; but TREC-500 test noise ~0.5-1pt means
"meet" is plausible). Verdict on completion: pool_cv acc+macroF1, read shipped hypotheses for
reward-hacking, McNemar vs best_l_max. No new job launched (headline run in flight).

## 2026-07-05 — trec_full VERDICT: full-train ties 2k on accuracy, better calibrated

trec_full (best recipe -l + tfidf_svd 128 + evolve, FULL 5452 TREC train, from scratch, same 500 test):
**acc 0.964, macroF1 0.9602, logloss 0.1822, cv_train 0.9365.** vs best_l_max (2000 train): acc 0.964,
macroF1 0.968, logloss 0.2262, cv_train 0.919. Pre-registered "meet-or-beat 0.964" -> MET (exactly:
482/500 both). More train data: SAME test accuracy (TREC-500 saturated at ~0.964 for this recipe),
better calibration (logloss 0.226->0.182) + higher train-CV; macroF1 -0.008 (≈1 small-class example,
noise). Headline reportable: **0.964, stable across 2k<->5.4k train** (not data-hungry). Pool: 62 hyps,
clean (7 legit question-type surface features + answer/intent), no reward-hacking.

TOOL BUG found: `compare` refused the McNemar ("mismatched test sets: trec/seed7 vs trec/seed7") —
different train_size/val_size consume the shared RNG differently before the test draw, so both take
ALL 500 test but in different shuffle order -> np.array_equal fails though the SET is identical.
Accuracies identical so verdict unaffected. Fix option (Lee's call): draw the test split from a
seed-only RNG independent of train/val sizes — but that changes which test rows are drawn when
test_size < len(test) (e.g. ag_news), invalidating existing runs' test sets; deferred.

## 2026-07-05 — compare test-set bug FIXED + trec_full McNemar

Fixed data.load: the test split now uses its own seeded RNG (default_rng(seed)) instead of the
train/val RNG, so the test set depends only on (dataset, seed, test_size), not train_size/val_size.
Runs differing only in train size are now directly comparable. Verified: compare runs/trec_full
runs/trec_best_l_max now works -> trec_full 0.962 vs best_l_max 0.964, delta +0.002, discordant 9
(4/5), McNemar p=1.0 — NOT significant. Full-train == 2k-train on TREC test, saturation confirmed.
(Caveat: for datasets that SUBSAMPLE test, e.g. ag_news test_size 2000<7600, this changes which rows
are drawn vs old runs — intended; the new invariant is worth it. TREC test=all 500, set unchanged.)
27 affected tests pass, ruff clean.

## 2026-07-05 — low-N (5/class) run + README results table with staleness placeholders

Added exact K-shot sampling (DataConfig.shots_per_class + data.per_class_indices) — proportional
stratified starves rare classes (TREC ABBR got 1 of 30). trec_lown (5/class, -l, sts dedup, no
evolve, pool 64): **test 0.666, macroF1 0.631, cv_train 0.804**. Big cv_train>>test gap = the RF/HGB
head OVERFITS (128 features on 30 rows); barely above TREC zero-shot 0.632. Confirms the low-N plan:
the default head is wrong at low N; needs a lighter/prior-aggregation head (docs/low-n-plan.md).
README "What's measured" prose -> Results TABLE (settings + train size). Per Lee, most stored numbers
are STALE (trec -m 0.920 predates the answer-oriented instruction that hit 0.934; ag_news/sst2 predate
current instruction+dedup AND the test-split fix changed their subset) -> marked "*re-run*"
placeholders; only the current -l TREC numbers + low-N kept. TREC baselines valid (all-500 test
unchanged); ag_news/sst2 baselines placeholdered.

## 2026-07-05 (hourly) — CFPB monetary-relief run in flight (first tabular+text datapoint)

examples/cfpb.py, balanced 1k/class (2k, random stratified split — temporal would shift balance),
static pool, -l, ColumnTransformer(narrative->HypothesisVectorizer + one-hot Product/Company/State/
channel), tabular passed as baseline_features. Healthy at ~10min (WAL fresh); slower generation than
TREC — covariance dedup rejects many similar complaint hypotheses -> multiple generate attempts + LM
latency. Then ~128k-pair train+test scoring at -l ahead. EXPECTATION: this is a discrimination
datapoint (balanced/random split), NOT the temporal-natural-rate benchmark (AUC 0.78 hybrid / 0.69
tfidf) — so not directly comparable; want AUC comfortably >0.5 and beating a tfidf+tabular baseline to
show the narrative hypotheses add signal. Verdict on completion. No new job launched (run in flight).

## 2026-07-05 — CFPB run killed (user request; GPU reserved for training)
CFPB balanced run (--per-class 1000) killed at user request — Lee's futo-asr training holds the GPU
and takes priority. Not a failure: diagnosed a real perf bug first. Worker was pegged 100% on ONE CPU
core with the WAL frozen = stuck in sqlite cache LOOKUPS, not GPU scoring. Cause: score cache is now
2.9 GB and `get_logits` filters `WHERE hyp_hash=? AND model=? AND text_hash IN(...)` while the PK
leads with text_hash -> no index hit on a large DB. Compounded by CFPB's long narratives (~300 tok vs
TREC ~15). FIX (CPU/disk-only, safe anytime): add covering index (hyp_hash, model, text_hash) in
cache.py; drop max_text_chars to ~512 for CFPB. Deferred until GPU frees / Lee's go-ahead.

CRON NOTE: the 15-min review cron is STALE — references removed configs (sst2_boost, ag_news_boost,
trec_boost, 20newsgroups_tree) and a removed `audit` subcommand + tree/boost methods. Superseded by
the sklearn HypothesisVectorizer direction. Recommend deleting it.

## 2026-07-05 — plateau-epsilon analysis on cached logs (CPU-only; GPU untouched, Lee's training running)
Backlog item "plateau epsilon tune" — did the "test on cached data first" step. No GPU, no runs; read
heldout_acc round-sequences from all runs/*/log.jsonl (97 round-over-round transitions).

MEASURED round noise (|delta heldout|): median 0.0037, p75 0.0063, p90 0.0124, mean 0.0045, max 0.0237.
Confirms the ~0.003 estimate. Current plateau check (evolve.py:289) is `acc > best_acc + 1e-4` with
default patience=2, rounds cap 6.

FINDING 1 (the backlog's premise holds): eps=1e-4 is far below noise. 21% of transitions are POSITIVE
sub-noise upticks (1e-4 < d <= 0.003) that reset the patience counter -> evolution over-runs. Symptom:
trec_full (patience 4) and trec_best_l_max (patience 4) oscillated within +/-0.005 and ran to their
round caps (10/8) for a net +0.004/+0.005 that is within noise.

FINDING 2 (the catch — naive fix is HARMFUL): real gains sometimes arrive as a DELAYED JUMP after
2 flat rounds. trec_pro_l deltas [+0.0012,+0.0013,+0.0100,+0.0012,-0.0025] — the real +0.010 is at
round 3. With eps=0.003 AND the DEFAULT patience=2, best_acc+eps is not beaten at rounds 1,2 -> since_best
hits 2 -> STOP at round 2, MISSING the +0.010. So raising epsilon alone (default patience) causes
premature stopping and lost gains.

CONCLUSION: it is a JOINT (epsilon, patience) tune, not a solo epsilon bump. eps ~= 0.003 must be paired
with patience >= 3-4 so the detector is noise-robust yet still waits through a noise-plateau for a
delayed jump. (patience=4 + eps=0.003 recovers trec_pro_l's round-3 jump.)

DEFERRED (needs GPU, ~7-min validating run; gate not met this cycle):
- make plateau epsilon configurable in PoolConfig (default keep 1e-4 = no behavior change until opted in);
- validating run when GPU frees: trec at (eps=0.003, patience=4) vs (eps=1e-4, patience=4), compare
  rounds-to-stop AND pool_cv. Commit only if pool_cv holds (>= within noise) and rounds-to-stop drops.
No code changed / committed this cycle (validation gate requires a run; GPU reserved for Lee's training).

## 2026-07-05 — repo cleanup while GPU busy: train/ subpackage, isolation guard, CFPB-diagnosis CORRECTION
1. REORG: flat src -> core + train/. Core (inference, importable with base deps only): vectorizer,
   encoder, cache, dedup, config, costs. train/ (may import dspy/datasets/typer/rich): proposer,
   evolve, runner, data, head, cli, compare, lexical. Mirrors the [train] extra; public API unchanged
   (from hypothesis_vectorizer import HypothesisVectorizer). CLI entrypoint -> hypothesis_vectorizer.train.cli:app.
2. NEW tests/test_inference_isolation.py: subprocess with dspy/datasets/typer/rich/dotenv/litellm
   BLOCKED in import machinery; exercises import + fit(fixed pool) + feature names + clone + pickle +
   save/load. Enforces the "inference is dspy-free" promise structurally. Passes in <1s.
3. HONEST CORRECTION of 2026-07-05 CFPB diagnosis: the sqlite-lookup theory was WRONG. Measured on
   the real 2.9GB cache: get_logits is fast either way (20x500 lookups = 6.3ms with a (hyp_hash,
   model, text_hash) index vs 4.0ms via the PK — sqlite decomposes IN() into per-value PK seeks).
   Built the index, measured, found no benefit (+1.8GB disk), DROPPED it and reverted cache.py;
   VACUUMed the DB (4.7G -> 2.5G). What the CFPB run was ACTUALLY doing: scoring normally but
   SILENTLY — the served vectorizer defaulted verbose=False, and WAL commits land once per 8192-pair
   chunk, which at long-text throughput is minutes apart. I killed a healthy run. Prevention:
   examples/cfpb.py served vectorizer now verbose=True, and --max-text-chars default 512 (halves
   tokenize+inference for 1200-char narratives). Real lesson: long texts make TOKENIZATION+inference
   the cost, not the cache.
4. Removed stray empty CLAUDE.md.
All 40 tests pass, ruff clean. No GPU touched (index build/vacuum were disk-only).

## 2026-07-05 — maintenance: metadata/0.4.0, py3.13, profiling verdict (no Cython), length-sort batching
1. pyproject: license=Apache-2.0, authors, keywords, classifiers, project.urls -> renamed repo;
   version 0.3.0 -> 0.4.0 (the train/ reorg changed internal import paths). Wheel builds clean.
2. Cleanup: 13 root runs_*.log launcher logs -> runs/_launcher_logs/; deleted models/ (dead GEPA-era
   artifacts — gepa_tune code was removed in the sklearn rewrite).
3. Coverage (pytest --cov): 79% total. Algorithmic core 88-100% (evolve 98, vectorizer 96, runner 93,
   dedup 91). Thin: cli 0%, compare 0% (typer/CLI glue), proposer 47% (live-LM paths), encoder 73%
   (GPU paths). No alarming gap; CLI/compare tests would be low-value glue tests.
4. PROFILING (cache-hit scoring, real 2.5GB cache, TREC 2000x62=124k pairs, zero GPU): warm 0.48s
   (get_logits row loop 0.37s), cold 4.6s (sqlite page reads). Evolution does tens of such passes ->
   ~10-30s CPU per run vs hours of GPU. VERDICT: nothing in this codebase is worth Cython — hot paths
   are already C (GPU inference, sqlite, sklearn CV); a numpy rewrite of get_logits would save
   fractions of a second per run. Memory is a non-issue CPU-side (feature matrices are MBs); the
   10.6GB GPU footprint is model+activations, already halved by max_text_chars 512.
5. REAL perf lever implemented: encoder._logits now sorts pending pairs by text length before
   batching — CrossEncoder.predict pads each batch to its longest member, so random order wastes
   compute on mixed-length corpora. Semantically inert (results keyed per pair). EXPECT ~1.3-2x on
   long-text corpora (CFPB), ~none on TREC. VALIDATE at next GPU window before claiming the number.
6. Python: deps resolve AND all 40 tests pass on 3.13 and 3.14 (side venvs). Adopted 3.13 as the dev
   pin (.python-version, now tracked); floor stays >=3.11; classifiers list 3.11-3.14. 3.14 CUDA
   path untested — staying one step back. NOTE (resource-change flag): future GPU runs now execute on
   py3.13 + same torch — flagging per protocol; revert = echo 3.11 > .python-version && uv sync.

## 2026-07-05 — generation-time variance check (backlog item): measured, REFRAMED, filter implemented
MEASURED on cached pools (entail-prob std across each run's own train texts):
- The premise ("drop always-TRUE statements") is stale — no always-true hyps survive in final pools
  (evolution's zero-fold prune kills them). The real waste is always-FALSE over-specific hypotheses:
  trec_full: "equivalent to 'What is the abbreviation of X?'" std 0.013 mean 0.002;
  ag_news: sanctions/climate/espionage/ESG hyps std 0.022-0.039 mean ~0.001-0.003 (fire on nothing).
- Baseline for real signal: a detector for TREC's rarest class (ABBR, 1.6%) still has std ~0.10;
  even a 0.5%-class detector ~0.04. So std < 0.02 is unambiguous junk.
- CODE GAP found: dedup._zscore comments "vacuity is handled elsewhere" — it is handled NOWHERE.
  Flat candidates zscore to zeros -> corr 0 with everything -> always KEPT.
PRE-REGISTERED EXPECTATION: adding a min_std floor (default 0.02, conservative) to the covariance
Deduper rejects truly-dead candidates at generation with ZERO risk to rare-class detectors (5x
margin to ABBR's 0.10). Would have pruned 1/62 of trec_full's final pool; the ag_news junk
(std 0.022-0.039) sits ABOVE the conservative default — testing a stricter 0.05 floor is a GPU-window
experiment (expectation there: pool_cv holds or improves as slots reallocate; if it drops, rare
detectors matter more than measured and we revert). No accuracy claim made for the default floor;
it is a generation-efficiency fix (fewer wasted slots + LM refills), validated fully at next GPU run.

## 2026-07-05 — STATE OF PROJECT: CPU-only backlog exhausted; improvement cron standing down
All work that can happen WITHOUT the GPU is done. Lee's futo-asr training holds the GPU (untouchable),
so the hourly improvement cron has no productive move and is now firing no-ops. It should stand down
(be deleted/paused) until the GPU frees; same for the stale 15-min review cron.

Shipped this session (all CPU-only, GPU untouched): repo rename -> hypothesis-vectorizer; train/
subpackage split + dspy-free isolation guard test; corrected the bogus CFPB "sqlite stall" diagnosis
(index built/benchmarked/dropped, DB vacuumed 4.7G->2.5G; real cause = silent scoring, now verbose);
packaging metadata + v0.4.0; py3.13 pin (tests green on 3.13 & 3.14); length-sorted GPU batching;
dedup min_std variance floor. Cross-dataset judging CLOSED. Plateau-epsilon cached analysis done.

READY-TO-FIRE GPU QUEUE (each has a pre-registered expectation above; run when GPU frees):
1. length-sort validation: time one -l scoring pass on a mixed-length corpus (CFPB) vs the pre-commit
   baseline. Expect ~1.3-2x; no accuracy change (result keyed per pair).
2. CFPB re-run: examples/cfpb.py --per-class 1000 (now verbose=True, max-text-chars 512). Expect AUC
   comfortably >0.5 on the balanced/random split, ideally > tfidf+tabular. NOT the 0.78 temporal bench.
3. plateau (eps, patience) tune: make PoolConfig.plateau_epsilon configurable (default 1e-4, no-op),
   then trec run at (eps=0.003, patience=4) vs (1e-4, 4); compare rounds-to-stop AND pool_cv. Commit
   only if pool_cv holds and rounds drop. (Code change + validation in ONE GPU session — don't split.)
4. min_std=0.05 experiment: regenerate a pool at the stricter floor; expect pool_cv holds/improves as
   slots reallocate off the ag_news dead detectors (std 0.022-0.039); revert if it drops.
5. Lower-priority: -l finalization of best pools; multi-pool 3-seed ensemble (error bars); 20newsgroups
   (pool 96); two-encoder (-m + -l) union for the scaling-law story.

## 2026-07-07 — FIRST TREE-EVOLVE TEST (Lee's LLM-in-the-loop design), cache-warm, GPU shared
Lee approved sharing the GPU (ASR at 31GB/98GB; our encoder ~3GB, batch 64). Cache-warm design:
both runs seed from trec_full's pool (first 32 hyps; -l, full 5452 train — all scores cached), so
GPU work is ONLY scoring newly proposed hypotheses (~few k pairs/round).
- trec_tree32_base: the same 32-hyp pool, NO growth (tree.rounds 0) — the static baseline.
- trec_tree32: tree-evolve 16 rounds, dspy.Refine x4, reward = normalized leaf info gain.
PRE-REGISTERED EXPECTATIONS:
1. Baseline pool_cv: BELOW trec_full's 0.964 (32 is half the pool) — guess 0.94-0.955.
2. Tree run should recover a chunk of the gap vs the baseline (it targets exactly the residual
   confusion). Success = tree_32+grown > base_32 by more than noise (~0.005); stretch = approaching
   0.964 with fewer total hyps. Failure mode to watch: LLM proposals with high LEAF info gain that
   don't generalize (leaf-overfit) -> pool_cv flat while log.jsonl gains look great. Read the
   grown hypotheses for leaf-overfit wording (single-example tailoring).
3. Mechanics: expect early rounds to target the known ENTY/DESC hot spot; info_gain per accepted
   round mostly > 0.1; some no-add rounds (dedup) are fine.
Honest protocol: pool_cv only; one test eval each.

## 2026-07-07 (late) — tree-evolve v2 in flight; depth-cap frontier bug found & fixed mid-experiment
Live-watching v1 caught three design fixes (all Lee-directed, committed): (1) reward = leaf info
gain x NOVELTY (1 - max|corr| vs existing pool features on the leaf) — v1's round-0 winner was
redundant; (2) replaced dspy.Refine with our own loop feeding MEASURED LANGUAGE back ("correlates
0.95 with EXISTING '<named hypothesis>'") + related_hypotheses input (top-10 existing hyps with leaf
gain) — also halves LM latency; (3) SplitLeaf prompt pushes inference-level angles (ANTICIPATE THE
ANSWER, imperative reduction) and bans surface features after v1 proposed "begins with 'What are
the'". CRITICAL BUG (Lee spotted): 3 rounds accepted but the SAME 456-sample leaf every round — the
leaf sat AT max_depth=6, so the tree could never USE the new features. Fix: uncap depth, gate splits
on min_impurity_decrease=0.002 (measured: worst leaf n=456 H=1.78 -> n=80 H=0.99, 99 leaves, no
shredding). v2 relaunched cache-warm; round 0 now targets the n=80 DESC/ENTY leaf (frontier moves).
Bar unchanged: static-32 baseline 0.960; success > 0.965.

## 2026-07-08 — TREE-EVOLVE v3 VERDICT (vs pre-registered expectations)
Final config after Lee's live-debugging session: error-mass targeting, gate 0.002 with acceptance
bar = split gate, best_of_n x4 parallel (dspy.Parallel), 24 leaf shots, 3-class features,
inference-level prompt. Run: 8 rounds, 3 adds (32 -> 35 hyps), stopped on patience at the ABBR/DESC
leaf (needed 28% leaf-entropy removal; best attempts 11-22%).

RESULTS (all 5452 train, -l, seed 7):          acc     f1      logloss  cv_train
  static-32 baseline (2-class AND 3-class):    0.960   0.9478  0.2697   0.9255
  tree-evolved 35 (3-class):                   0.960   0.9473  0.2295   0.9290
  (reference: stability method, 62 hyps:       0.964)

VERDICT vs pre-registration:
- ACCURACY: FLAT (0.960 = baseline). Success bar (>0.965) NOT met. The residual confusion the tree
  surfaces (DESC/ENTY, ABBR/DESC leaves) is the encoder's known ceiling — 12 LLM attempts on the
  ABBR/DESC leaf maxed at 22% of the 28% bar. Honest read: on an already-strong pool at full-data
  TREC, tree-guided growth does not buy accuracy.
- LOGLOSS: -15% (0.2697 -> 0.2295) — the 3 added hypotheses improve CALIBRATION materially. cv_train
  +0.35pt. Real, unexpected win worth keeping.
- EFFICIENCY angle: 0.960 with 35 hyps vs 0.964 with 62 (stability) — comparable accuracy at 44%
  fewer hypotheses = cheaper inference. The tree method may be the better POOL-COMPRESSION tool.
- HYPOTHESIS AUDIT: all 3 adds are clean semantic properties (characteristic-of-subject,
  what/how/why description, list-answer) — zero surface features, zero leaf-overfit wording. The
  prompt + novelty reward worked qualitatively.
- 3-class features: identical baseline metrics to 2-class (0.960/0.9478/0.2697 both) — neutral adds
  nothing for a static pool head on TREC; its value (if any) is inside tree targeting/attr-hyps.
NEXT (needs decision, not compute): where tree-evolve should shine is FROM-SCRATCH growth (start
size~8, grow to ~35) and LOW-N — not polishing a mature pool. Also 20newsgroups (more classes =
more leaves = more room). Candidate experiments when Lee wants them.

## 2026-07-08 — LAUNCH: from-scratch tree growth (trec_tree_scratch)
Decision this answers: is tree-evolve a GROWTH method (its design intent) or only a polish tool?
v3 on a mature pool: accuracy flat, logloss -15%, only 3 adds before patience. From scratch (seed 8,
grow to <=35, same final settings: error-mass targeting, gate 0.002 = acceptance bar, best_of_n x4
parallel, 24 shots, 3-class) the starting tree is weak -> many big genuine leaves -> the LLM should
do real work every round. PRE-REGISTERED: (a) many more adds than v3's 3 (expect 15-25); (b) success
= within noise of static-32's 0.960 with a comparable pool size, since every hypothesis was
demand-driven; stretch = beat it; (c) failure mode = plateau far below 0.95 (seed too weak to
bootstrap targeting). Resource flag: ~3GB GPU batch 64 (Lee-approved footprint for this test
family); RAM 11GB avail (ASR DataLoader creep again — watch). Est wall ~40-60 min (justification:
this decides whether the tree-evolve line continues, per the v3 verdict's NEXT).

## 2026-07-08 — FROM-SCRATCH VERDICT + leaf-blacklist fix; rerun launched
trec_tree_scratch (seed 8): 0.948 acc / 0.9198 f1 / 0.2789 logloss @ 18 hyps (10 adds, stop round 12).
vs pre-registration: (a) 10 adds — 3x v3 but below the 15-25 band; (b) NOT within noise of 0.960
(-1.2pt) but at 56% of the pool (98.7% of the accuracy at ~half the inference cost); (c) not a
bootstrap failure. ROOT CAUSE of under-growth found in the log: rounds 10-12 all burned on ONE
stubborn leaf (n=373 DESC:351, H=0.364 -> gate demands 8%, LLM peaked 5.7%), then GLOBAL patience
killed the loop with 17 rounds unused while other targets existed.
FIX (committed): per-LEAF patience — a leaf that resists `patience` rounds is BLACKLISTED
(membership-hash key, so an add that reshapes it auto-un-blacklists) and the loop moves to the
next-worst leaf; stop only when no targetable leaf remains or rounds exhaust. PRE-REGISTERED for the
rerun (trec_tree_scratch2): expect growth well past 18 (toward the 35 cap), and acc between 0.948
and ~0.960; the pool-size-vs-accuracy CURVE is the real deliverable (efficiency story).

## 2026-07-08 — FROM-SCRATCH + LEAF-BLACKLIST VERDICT: the method's best result
trec_tree_scratch2 (seed 8, 13 grown = 21 hyps, all 27 rounds used, 2 leaves blacklisted):
                                acc     f1      logloss   hyps
  static-32 baseline            0.960   0.9478  0.2697    32
  v3 polish (mature pool)       0.960   0.9473  0.2295    35
  from-scratch v1 (global pat.) 0.948   0.9198  0.2789    18
  FROM-SCRATCH + BLACKLIST      0.962   0.9582  0.1998    21   <- beats baseline on ALL metrics
  stability method (reference)  0.964   -       -         62

vs pre-registration: STRETCH CASE HIT. Predicted 0.948-0.960; got 0.962 (+macro_f1 +1.0pt,
logloss -26% vs baseline) at 34% fewer hypotheses than the baseline and ONE-THIRD of the stability
method's pool for accuracy within noise of its 0.964 (0.002 on 2000 test ~ McNemar noise).
The blacklist did exactly its job: 2 stubborn DESC/ENTY leaves absorbed 3 rounds each then were
skipped; growth continued to the rounds cap instead of stranding (v1 stranded 17 rounds).
HYPOTHESIS AUDIT: all 13 grown are clean, demand-driven semantic properties (components-of-category,
process-vs-cause explanation, creative-work names, role/title identity, spell-out-abbreviation...)
— reads like a hand-designed taxonomy of TREC's actual confusion structure. Zero surface features.
TAKEAWAY: tree-guided growth is a POOL-COMPRESSION method — near-max accuracy with 1/3 the
inference cost, every hypothesis traceable to the exact confusion that demanded it (the
interpretability story compounds: the pool is a MAP of where the task is hard).

## 2026-07-08 — LAUNCH: seed-variance check of the from-scratch result (trec_tree_scratch_s17)
The 0.962@21 headline needs an error bar before anything builds on it. Same config, seed 17 (new
seed pool, new leaf-shot draws, new LM rollouts — full-pipeline variance, not just head variance).
PRE-REGISTERED: prior seed studies measured tiny seed deltas (ag_news 0.0015, sst2 0.000);
success = seed-17 accuracy in [0.955, 0.968] and pool size 18-26 with semantically similar (not
identical) grown hypotheses -> the result is the METHOD's, not the seed's. A miss below 0.95 means
the v1/v2 gap was luck and the claim gets retracted to "0.948-0.962 depending on seed".

## 2026-07-08 — SEED-17 VERDICT: MISSED the band; headline softened (honest protocol)
trec_tree_scratch_s17: 0.950 acc / 0.9324 f1 / 0.2997 logloss @ 19 hyps. Pre-registered success was
[0.955, 0.968]: MISS (at the 0.95 retract boundary, not below it). Pool size 19 in band; blacklist
worked (stubborn DESC/ENTY leaf skipped). CONCLUSION: from-scratch growth has FULL-PIPELINE seed
variance ~10x the old method's head-only variance (spread 0.950-0.962 = 0.012 vs ag_news 0.0015) —
the LM rollouts + seed pool draw matter. HEADLINE RETRACTED per pre-registration: not "beats
baseline"; now "PARITY with the 32-hyp baseline (2-seed mean 0.956 +/- 0.006 vs 0.960) at ~40%
fewer hypotheses". The compression story survives; the accuracy-win story does not (yet).
LAUNCHING seed 27 to disambiguate which seed is the outlier and report a 3-seed mean+/-std.
PRE-REGISTERED: expect acc in [0.945, 0.965], pool 17-26; deliverable = 3-seed mean+/-std as the
method's reported number.

## 2026-07-08 — 3-SEED FINAL VERDICT on from-scratch tree growth (line closed pending Lee)
Seeds 7/17/27: acc 0.962 / 0.950 / 0.944 -> **0.952 +/- 0.009** @ ~20 hyps (pools 21/19/20).
Baseline static-32: 0.960. Stability method: 0.964 @ 62 (near-zero seed variance).
FINAL HONEST CLAIM: from-scratch tree growth = ~0.8pt BELOW baseline on average at ~40% fewer
hypotheses, with ~10x the seed variance of the stability method. The 0.962 first run was the
best-of-3 outlier. It is a COMPRESSION/interpretability method (demand-driven pool = a map of task
confusion), not an accuracy method, on full-data TREC.
DECISION POINT (Lee): (a) tree-growth for its natural settings — low-N, 20newsgroups, or as the
CFPB narrative-feature generator (baseline_features integration is built); (b) variance reduction
(ensemble the 3 seed pools: union = 60 demand-driven hyps, or average heads — backlog #16 applied
here); (c) park the line and return to the main backlog (#11-#18). Holding for Lee — 3 overnight
runs answered the pre-registered question; more autonomous spend on this line isn't justified
before he weighs the retraction.

## 2026-07-08 — LAUNCH: 3-pool union ensemble (backlog #16 on the tree pools)
Union of scratch2+s17+s27 pools (21+19+22, text-dedup'd), head on the superset, seed-7 split.
PRE-REGISTERED: features are a superset of seed-7's pool, so expect acc >= 0.962 minus head noise;
success = >=0.960 (baseline parity from purely demand-driven hyps); strong success = ~0.964 (ties
stability at the same pool size ~60, meaning union-of-cheap-tree-pools == stability quality);
<0.956 would mean cross-seed hyps interfere (collinearity hurting the head) — would surprise me.
Cache: train fully warm (all 3 runs share the full 5452 train); test partially warm (~2-3 min GPU).

## 2026-07-08 — UNION ENSEMBLE VERDICT: strong-success band hit; backlog #16 closed
trec_tree_union (59 demand-driven hyps = union of 3 seed pools): 0.964 acc / 0.9674 macro_f1 /
0.2133 logloss. Pre-registered strong success (~0.964, ties stability) HIT exactly.
McNemar vs trec_full (stability, same seed-7 test): delta +0.002, p=1.0 — statistical TIE.
Notably macro_f1 0.9674 is the BEST f1 in the project (stability 0.960, baseline-32 0.9478).
STORY THAT NOW HOLDS (all pre-registered, seed-checked, significance-tested):
  1. single tree-growth run: cheap, interpretable, compressed (0.952+/-0.009 @ ~20 hyps) — the
     budget option; every hypothesis traceable to the confusion that demanded it.
  2. UNION of 3 tree runs: ties the stability method's accuracy with the project's best macro_f1 —
     variance averages out, demand-driven coverage compounds (rare classes gain: +0.7pt f1 at
     equal accuracy suggests better tail-class coverage than stability's survivor pool).
  3. stability method: still the single-run accuracy champion at low variance.
Backlog #16 (multi-pool ensemble) CLOSED — implemented as pool-union (simpler than head averaging,
and sufficient). Tree-evolve line now fully characterized; next moves remain Lee's call
(low-N / 20NG / CFPB integration).

## 2026-07-08 — LAUNCH: plateau-epsilon validation (backlog #13, code+run in one session)
PoolConfig.plateau_epsilon added (default 1e-4 = no behavior change). Validation per the cached
analysis (2026-07-05): trec at patience 4 / rounds 10, eps 1e-4 (control) vs 0.003 (treatment),
same seed, -m encoder, heavily cached. PRE-REGISTERED: treatment stops in FEWER rounds (analysis:
21% of transitions are sub-noise upticks that reset patience under 1e-4) with pool_cv within noise
(+/-0.005) of control. Commit the knob if both hold; revert the default-change idea if pool_cv drops.

## 2026-07-08 — FRAMING PROBE (Lee's challenge): encoder-ceiling claim RETRACTED for the DESC/ENTY leaf
Lee: "perhaps for the hypotheses we have but framed differently it might [split]". Tested directly —
hand-framed battery through finecat-l on the stuck 46-sample DESC/ENTY leaf (bar 25%, LLM peaked ~9%
in-run; best failed attempt 18.5%):
  "Answering the text requires explaining rather than naming."      36.0% gain, AUC 0.825 (entail)
  "The answer to the text is an established name, term, or title."  28.9% gain (NEUTRAL column)
Two framings CLEAR the bar -> the failure was the LLM's FRAMING, not the encoder. Principles:
(1) CONTRASTIVE/ANTITHESIS framing — state the distinction itself ("A rather than B"), not one side;
    one-sided phrasings sit at the encoder's uncertainty region.
(2) The NEUTRAL column carries real signal for answer-property presuppositions (Lee's 3-class push
    validated at the leaf level).
Also observed: the leaf contains near-duplicate structures with different gold labels ("seven deadly
sins"=DESC vs "7 articles of the constitution"=ENTY) — some residual is TREC label inconsistency.
ACTION: SplitLeaf prompt now leads with the CONTRASTIVE angle (with the measured numbers as the
instruction's evidence). Not yet re-run (GPU reserved for Lee's training per his stop order) —
rerunning a seed with the new prompt is the next cheap experiment when he frees the GPU.
Probe cost: 368 GPU pairs (~seconds).

## 2026-07-08 — REVIEW of cohort (Bradley Smith) contribution — pulled to main, VERDICT: sound, merge-quality
Scope: 6 commits, 185 files (~19k lines): paper draft, low-N/tau "paper line", 3 new datasets, cache
robustness, status daemon. Reviewed CPU-only (Lee's GPU-training hold in effect).
SHIP-CODE (the part that matters):
- data.py +299: banking77 / clinc150 / goemotions. VERIFIED hardcoded ClassLabel orders EXACT-MATCH
  the real HF datasets (all 3, incl clinc's 151=150+oos and goemotions 7-way Ekman) — the silent-
  mislabel trap is clear. multilabel goemotions handled via a pre-collapsed single-label mirror;
  oos kept as its own class — both documented honestly in-code.
- cache.py: WAL-probe on a throwaway file (ZFS/NFS accept the pragma then fail first write, poisoning
  the handle) with fallback to DELETE. Sound; closes the handle on failure. MINOR: concurrent runs
  sharing a cache dir race on the single .wal_probe.sqlite (worst case -> DELETE fallback; low sev).
- config.py/init/pyproject: dataset literal + version 0.4.0 + matplotlib dev dep. Fine.
- src ruff clean; full suite 48 pass.
PAPER/EXPERIMENTS: protocol is honest — fixed 500 test never seen in generation, head selected by
CV-on-train, 10 seeds, LM-free baselines. Leakage scan (fit-on-test / select-by-test) CLEAN. Headline
0.954 HV+RF traces exactly to committed table artifact. No cherry-pick/dropped-seed language.
Claims hedged (HV converges to 0.954 vs fine-tune 0.964, not "beats"; regime-crossover framing).
GAPS (non-blocking): (1) ZERO tests for 299 lines of new dataset code — add a (network, skipif)
test asserting ClassLabel order == hardcoded, since that's the whole risk; (2) cache probe race.
NET: high-quality, honest, correctly-orderd. The low-N prior-head result (0.594 zero-label vs
0.428 zero-shot-NLI) is the strongest new finding and aligns with the tree-evolve "natural home =
low-N" note. Recommend keep. Flagged gaps to Bradley/Lee.
