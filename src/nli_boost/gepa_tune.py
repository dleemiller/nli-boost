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
import threading
from pathlib import Path

import dspy
import numpy as np
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
from gepa.utils.stop_condition import (
    FileStopper,
    MaxMetricCallsStopper,
    SignalStopper,
    TimeoutStopCondition,
)

from .cache import ScoreCache
from .config import DataConfig, EncoderConfig, LMConfig
from .costs import CostTracker
from .data import labeled_examples, load, stratified_indices
from .encoder import EntailmentScorer
from .proposer import GeneratePool, _make_lm
from .reward import RewardConfig, geometric_mean, pool_reward

_INPUTS = ("task", "class_definitions", "labeled_examples", "n", "avoid")


def build_contexts(specs, pool_size, sub_size, seed):
    """One frozen dspy.Example per (dataset, seed); returns (examples, bundles-by-key)."""
    bundles, examples = {}, []
    for name, sd in specs:
        bundle = load(DataConfig(name=name), sd)
        key = (name, sd)
        bundles[key] = bundle
        rng = np.random.default_rng(sd)
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

    GPU scoring is serialized by a lock; the encoder is not thread-safe and the
    shared GPU must not be hammered by concurrent metric threads."""

    def __init__(self, scorer, bundles, reward_cfg=None, judge=None, eval_log=None):
        self.scorer = scorer
        self.bundles = bundles
        self.reward_cfg = reward_cfg or RewardConfig()
        self.judge = judge
        self.eval_log = eval_log
        self._gpu = threading.Lock()

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

        with self._gpu:
            x = self.scorer.features(texts, pool)  # cache-through; GPU only on misses

        judge_score = self.judge(gold, pool) if self.judge else None
        r = pool_reward(x, y, pool, texts, self.reward_cfg, judge_score=judge_score)
        feedback = (
            f"[{gold.dataset}] {r['feedback']}"
            + "\n\nKeep the instruction GENERIC and dataset-agnostic — it is reused across many "
            "tasks. Encode better strategies (angles to cover, targeting minority classes, "
            "varying specificity, avoiding paraphrase and surface tricks), never dataset-specific "
            "class names, topics, or canned statements."
        )
        if self.eval_log:
            with open(self.eval_log, "a") as f:
                f.write(json.dumps({"dataset": gold.dataset, **r, "n_pool": len(pool)}) + "\n")
        return ScoreWithFeedback(score=r["score"], feedback=feedback)


def make_judge(judge_lm):
    """Boolean-rubric pool judge, blind to the numeric reward; reasoning disabled."""

    class JudgePool(dspy.Signature):
        """Rate a set of NLI hypotheses written as features for a text classifier.
        Score 0-1 how well the SET would separate the classes: semantic (about content,
        not surface form), verifiable from one text alone, non-duplicated, covering all
        classes including minorities, with varied specificity. Judge strictly."""

        task: str = dspy.InputField()
        class_definitions: list[str] = dspy.InputField()
        hypotheses: list[str] = dspy.InputField()
        score: float = dspy.OutputField(desc="0.0-1.0 set quality")
        critique: str = dspy.OutputField(desc="name the weakest statements and what to change")

    predict = dspy.Predict(JudgePool)
    predict.set_lm(judge_lm)  # dspy.context is forbidden in GEPA worker threads

    def judge(gold, pool):
        try:
            r = predict(task=gold.task, class_definitions=gold.class_definitions, hypotheses=pool)
            return float(max(0.0, min(1.0, r.score)))
        except Exception:
            return 0.5

    return judge


def optimize_instruction(
    out_path: Path,
    tune_specs,
    reflection_model="openrouter/deepseek/deepseek-v4-pro",
    judge_model="openrouter/deepseek/deepseek-v4-pro",
    student_lm: LMConfig | None = None,
    encoder: EncoderConfig | None = None,
    pool_size=28,
    sub_size=400,
    max_metric_calls=40,
    timeout_min=40.0,
    seed=7,
    cache_dir=Path("cache"),
) -> dict:
    """Feasibility-scale defaults (max_metric_calls=40). Returns the tuned instruction.

    Stops on whichever comes first: max_metric_calls, timeout_min wall-clock, a
    `touch <out>.stop` sentinel, or Ctrl-C/SIGTERM — all keep the best-so-far.
    log_dir also checkpoints every iteration, so re-running the SAME command resumes.
    """
    student_lm = student_lm or LMConfig()
    encoder = encoder or EncoderConfig()
    examples, bundles = build_contexts(tune_specs, pool_size, sub_size, seed)

    scorer = EntailmentScorer(encoder, ScoreCache(cache_dir / "nli_scores.sqlite"), CostTracker())
    judge = (
        make_judge(_make_lm(LMConfig(model=judge_model), cache=True, reasoning=False))
        if judge_model
        else None
    )
    metric = PoolRewardMetric(scorer, bundles, judge=judge, eval_log=out_path.with_suffix(".evals.jsonl"))

    baseline = _baseline_scores(examples, bundles, scorer, metric, student_lm)

    # per-instruction checkpoint dir (NOT the shared models/gepa_logs, which holds the
    # stale pre-rewrite tree-GEPA state — resuming from that would load an incompatible
    # program). Same command re-runs resume from here; a fresh out_path starts clean.
    log_dir = out_path.parent / f"{out_path.stem}_gepa_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stop_file = out_path.with_suffix(".stop")
    # stop on whichever fires first; all return the best-so-far. MaxMetricCallsStopper
    # is included because passing stop_callbacks overrides the built-in call cap.
    stoppers = [
        MaxMetricCallsStopper(max_metric_calls),
        TimeoutStopCondition(timeout_min * 60.0),
        FileStopper(str(stop_file)),
        SignalStopper(),
    ]
    dspy.configure(lm=_make_lm(student_lm))
    gepa = dspy.GEPA(
        metric=metric,
        reflection_lm=dspy.LM(reflection_model, temperature=1.0, max_tokens=16000),
        max_metric_calls=max_metric_calls,
        track_stats=True,
        num_threads=2,  # GPU serialized by lock; keep CPU workers few (no-OOM rule)
        log_dir=str(log_dir),  # checkpoints every iteration -> resumable on re-run
        gepa_kwargs={"stop_callbacks": stoppers},
    )
    compiled = gepa.compile(dspy.Predict(GeneratePool), trainset=examples, valset=examples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(out_path))
    return {
        "baseline_geo_mean": baseline,
        "tuned_instruction": compiled.signature.instructions,
        "saved_to": str(out_path),
        "stop_file": str(stop_file),
        "log_dir": str(log_dir),
    }


def _baseline_scores(examples, bundles, scorer, metric, student_lm) -> float:
    """Reward geo-mean of the CURRENT instruction — the bar GEPA must clear."""
    student = dspy.Predict(GeneratePool)
    student.set_lm(_make_lm(student_lm))
    scores = []
    for ex in examples:
        pred = student(**{k: ex[k] for k in _INPUTS})
        scores.append(metric(ex, pred).score)
    return round(geometric_mean(scores), 4)
