"""Offline GEPA optimization of the GeneratePool INSTRUCTION.

What and why (grounded in the experiment log):
- We optimize the instruction, not the model: the instruction is the collapse-
  prevention lever (our diversity-instructed prompt sits at effective rank 22.9/64
  vs a naive prompt's 8.3/64). Flash stays the generator; pro is the reflection LM.
- The reward is reward.pool_reward on the RAW generated pool (evolution is a separate,
  weak stage, too expensive per eval): noise-averaged CV skill + diversity + anti-hack
  + optional judge. Honest: train-CV only, test never touched.
- Contexts are FROZEN (task + class defs + a seeded example sample + a seeded train
  subsample per dataset), so the objective is stationary — GEPA's assumption.
- Cross-dataset generalization is the gate: tune on several datasets, aggregate with a
  geometric mean, and ACCEPT only after a full-method McNemar transfer test on a dataset
  held out of tuning (see cli `compare`). This is what the first GEPA attempt lacked
  (it overfit one dataset: +5.4 TREC / -3.3 AG News).

Safety (shared GPU + no-OOM rule): a single EntailmentScorer guarded by a lock so only
one thread touches the GPU; CV is serial (n_jobs=1); GEPA runs few threads.
"""

import json
from pathlib import Path

import dspy
import numpy as np
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

from .cache import ScoreCache
from .config import DataConfig, EncoderConfig, LMConfig
from .costs import CostTracker
from .data import labeled_examples, load, stratified_indices
from .encoder import EntailmentScorer
from .proposer import GeneratePool, _make_lm
from .reward import RewardConfig, geometric_mean, pool_reward

_INPUTS = ("task", "class_definitions", "labeled_examples", "n", "avoid")


def build_contexts(specs, pool_size, sub_size, seed, n_subsamples=1):
    """Frozen dspy.Examples for GEPA. Each (dataset, seed) yields `n_subsamples` contexts, each a
    DISTINCT stratified subsample + example sample. Resampling turns a handful of datasets into a
    real validation pool: a 2-example valset is a degenerate Pareto frontier / minibatch-of-2.
    Correlated within a dataset, but it cuts per-eval reward variance and gives GEPA a smoother
    selection signal. Returns (examples, bundles-by-key)."""
    bundles, examples = {}, []
    for name, sd in specs:
        bundle = load(DataConfig(name=name), sd)
        bundles[(name, sd)] = bundle
        for i in range(n_subsamples):
            rng = np.random.default_rng(sd * 10_000 + i)  # distinct subsample + examples per context
            sub = stratified_indices(bundle.y_train, min(sub_size, len(bundle.y_train)), rng)
            examples.append(
                dspy.Example(
                    task=bundle.task,
                    class_definitions=bundle.class_descriptions,
                    labeled_examples=labeled_examples(bundle, per_class=3, rng=rng),
                    n=pool_size,
                    avoid=[],
                    dataset=name,
                    seed=sd,
                    sub=sub.tolist(),
                ).with_inputs(*_INPUTS)
            )
    return examples, bundles


class PoolRewardMetric:
    """GEPA metric: generate -> score on the frozen subsample -> composite reward.

    Runs concurrently across GEPA's worker threads (num_threads): LLM generate/judge and
    GPU scoring all overlap. The score cache is internally thread-locked; GPU inference runs
    concurrently (fine on this box)."""

    def __init__(self, scorer, bundles, reward_cfg=None, judge=None, eval_log=None):
        self.scorer = scorer
        self.bundles = bundles
        self.reward_cfg = reward_cfg or RewardConfig()
        self.judge = judge
        self.eval_log = eval_log

    def __call__(self, gold, pred, trace=None, pred_name=None, pred_trace=None):
        bundle = self.bundles[(gold.dataset, gold.seed)]
        sub = np.asarray(gold.sub)
        texts = [bundle.train_texts[i] for i in sub]
        y = bundle.y_train[sub]

        pool, seen = [], set()
        for h in getattr(pred, "hypotheses", []) or []:
            s = (h.statement or "").strip()
            if s and s.casefold() not in seen:
                seen.add(s.casefold())
                pool.append(s)
        if not pool:
            return ScoreWithFeedback(score=0.0, feedback="No usable hypotheses were produced.")

        judge_score, judge_critique, judge_subs = self.judge(gold, pool) if self.judge else (None, "", {})
        x = self.scorer.features(texts, pool)  # cache-through; GPU on misses (runs concurrently)
        r = pool_reward(x, y, pool, texts, self.reward_cfg, judge_score=judge_score)
        r["judge_criteria"] = judge_subs  # per-criterion booleans, logged so we can audit each
        feedback = f"[{gold.dataset}] {r['feedback']}"
        if judge_critique:  # the judge's SEMANTIC critique is the actionable signal for reflection
            feedback += f"\n\nJudge critique of this pool (what to fix): {judge_critique}"
        feedback += (
            "\n\nKeep the instruction GENERIC and dataset-agnostic — it is reused across many "
            "tasks. Encode better strategies (angles to cover, targeting minority classes, "
            "varying specificity, avoiding paraphrase and surface tricks), never dataset-specific "
            "class names, topics, or canned statements."
        )
        if self.eval_log:
            with open(self.eval_log, "a") as f:
                f.write(json.dumps({"dataset": gold.dataset, **r, "n_pool": len(pool)}) + "\n")
        return ScoreWithFeedback(score=r["score"], feedback=feedback)


# SET-LEVEL boolean rubric. LLMs can't ground continuous 0-1 scores (they collapse to a few round
# anchors — known bad practice), so we ask many independent yes/no questions ABOUT THE WHOLE SET;
# the judge score = fraction true. Set-level (not per-hypothesis) keeps the output tiny (~18 bools +
# one short line) so a judge call is fast — the continuous cv/coverage terms already give the reward
# its fine granularity, so the judge needn't emit 28x per-hypothesis checks.
# Only SEMANTIC criteria the quantitative reward terms CANNOT measure (audit 2026-07-04: the old
# 18-criterion judge was near-noise — most were format-compliance the LM always passes, or things
# cv/coverage/diversity already measure). These 6 each catch a distinct semantic defect that
# held-out accuracy can miss (surface-hacking, label leakage, missing an angle/class, no contrast,
# vacuity). Per-criterion booleans are logged so we can prune any that turn out dead/redundant.
# Pruned to the criteria that actually DISCRIMINATE (per-criterion audit on a 49-pool run):
# dropped covers_every_class (98% true) and multiple_semantic_angles (90% true) — near-constant,
# no signal. Kept the four that vary meaningfully. Still logged per-criterion to re-audit.
_SET_CRITERIA = {
    "has_contrastive": "includes hypotheses that separate GROUPS of classes, not only one-vs-rest",
    "semantic_not_surface": "hypotheses are about MEANING, not surface tricks (word position, "
    "punctuation, casing, length, exact word presence)",
    "non_vacuous": "few hypotheses are tautological or true of almost any text",
    "no_label_leakage": "no hypothesis references the class labels, the dataset, or the task itself",
}


def make_judge(judge_lm):
    """Fast set-level boolean judge — only SEMANTIC criteria that the quantitative reward can't
    measure. 6 yes/no questions + one short fix. Returns (fraction-true, detail, per-criterion dict);
    the fraction feeds the reward's judge guard and the per-criterion booleans are logged for audit.
    Grounded booleans (no float scores); concise output keeps it fast. Caller passes no-reasoning LM."""
    fields = {
        "task": (str, dspy.InputField()),
        "class_definitions": (list[str], dspy.InputField()),
        "hypotheses": (list[str], dspy.InputField()),
    }
    for c, desc in _SET_CRITERIA.items():
        fields[c] = (bool, dspy.OutputField(desc=f"true if {desc}"))
    fields["fix"] = (
        str,
        dspy.OutputField(desc="ONE short sentence: the single most useful strategy change for the generator"),
    )
    JudgePool = dspy.Signature(
        fields,
        "Judge a set of NLI hypotheses used as features for a text classifier, on SEMANTIC quality "
        "only. Answer each yes/no question about the WHOLE set, strictly and independently (when in "
        "doubt, false). Then give ONE short sentence of the most useful fix for the GENERATOR'S "
        "INSTRUCTIONS. Be concise.",
    )
    predict = dspy.Predict(JudgePool)
    predict.set_lm(judge_lm)  # dspy.context is forbidden in GEPA worker threads

    def judge(gold, pool):
        try:
            r = predict(task=gold.task, class_definitions=gold.class_definitions, hypotheses=pool)
            subs = {c: bool(getattr(r, c, False)) for c in _SET_CRITERIA}
            score = sum(subs.values()) / len(subs)
            failed = [c for c, ok in subs.items() if not ok]
            detail = f"judge {sum(subs.values())}/{len(subs)}; failing: {', '.join(failed) or 'none'}. {(r.fix or '').strip()}"
            return score, detail, subs
        except Exception:
            return 0.5, "", {}

    return judge


# Fixed, sensible internals (kept off the CLI on purpose — dspy's auto budget handles scale).
_POOL_SIZE, _SUB_SIZE, _SUBSAMPLES = 28, 400, 45


def optimize_instruction(
    out_path: Path,
    tune_specs,
    reflection_model="openrouter/deepseek/deepseek-v4-pro",
    judge_model="openrouter/deepseek/deepseek-v4-pro",  # reasoning OFF; concise set-level rubric
    auto="light",  # dspy GEPA budget preset: light | medium | heavy (sets metric-call budget)
    threads=8,  # concurrent metric evals -> parallel OpenRouter calls (LLM I/O is the bottleneck)
    student_lm: LMConfig | None = None,
    encoder: EncoderConfig | None = None,
    fresh=False,
    seed=7,
    cache_dir=Path("cache"),
) -> dict:
    """Canonical dspy.GEPA usage (see dspy.ai GEPA tutorial): auto budget preset, a LARGE trainset
    and a SMALL valset (docs: smallest valset matching the task distribution, ~<=35). log_dir
    checkpoints each iteration so re-running resumes; fresh=True wipes it first (use when the reward
    or datasets changed and a stale checkpoint would mismatch)."""
    student_lm = student_lm or LMConfig()
    encoder = encoder or EncoderConfig()
    eval_log = out_path.with_suffix(".evals.jsonl")
    log_dir = out_path.parent / f"{out_path.stem}_gepa_logs"
    if fresh:
        import shutil

        shutil.rmtree(log_dir, ignore_errors=True)
        eval_log.unlink(missing_ok=True)
        print("--- fresh: wiped checkpoint + eval log for this run", flush=True)

    # Resample many contexts, then split into a large trainset + a small held-out valset.
    allctx, bundles = build_contexts(tune_specs, _POOL_SIZE, _SUB_SIZE, seed, n_subsamples=_SUBSAMPLES)
    order = np.random.default_rng(seed).permutation(len(allctx))
    n_val = min(30, max(4, len(allctx) // 3))  # docs: valset <= ~35
    valset = [allctx[i] for i in order[:n_val]]
    trainset = [allctx[i] for i in order[n_val:]] or valset
    print(f"--- GEPA: {len(trainset)} train + {len(valset)} val contexts, auto={auto!r}", flush=True)

    scorer = EntailmentScorer(encoder, ScoreCache(cache_dir / "nli_scores.sqlite"), CostTracker())
    # judge = grounded booleans only: NO reasoning, and a tight token cap (the structured output
    # for the rubric fits well under 4k; the cap stops any runaway).
    judge = (
        make_judge(_make_lm(LMConfig(model=judge_model, max_tokens=4000), reasoning=False))
        if judge_model
        else None
    )
    # concurrent evals share the GPU (lock-serialized) and each caps its CV to 2 threads, so with
    # `threads` workers total CPU threads stay ~2*threads (no OOM); the LLM calls run in parallel.
    metric = PoolRewardMetric(
        scorer, bundles, reward_cfg=RewardConfig(cpu_threads=2), judge=judge, eval_log=eval_log
    )
    baseline = _baseline_scores(valset, metric, student_lm)

    log_dir.mkdir(parents=True, exist_ok=True)
    dspy.configure(lm=_make_lm(student_lm))
    gepa = dspy.GEPA(
        metric=metric,
        reflection_lm=dspy.LM(reflection_model, temperature=1.0, max_tokens=32000),
        auto=auto,
        num_threads=threads,  # parallel LLM I/O; GPU is lock-serialized, CV capped -> safe
        track_stats=True,
        log_dir=str(log_dir),  # per-iteration checkpoints -> Ctrl-C then re-run resumes
    )
    compiled = gepa.compile(dspy.Predict(GeneratePool), trainset=trainset, valset=valset)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(out_path))
    return {
        "baseline_geo_mean": baseline,
        "tuned_instruction": compiled.signature.instructions,
        "saved_to": str(out_path),
        "log_dir": str(log_dir),
    }


def _baseline_scores(contexts, metric, student_lm) -> float:
    """Reward geo-mean of the CURRENT instruction on the valset — the bar GEPA must clear."""
    student = dspy.Predict(GeneratePool)
    student.set_lm(_make_lm(student_lm))
    scores = [metric(ex, student(**{k: ex[k] for k in _INPUTS})).score for ex in contexts]
    return round(geometric_mean(scores), 4)
