"""The LM proposer: initial pool generation and evolution-round refills.

Failure handling is load-bearing, not decoration: providers occasionally emit
reasoning runaways that truncate mid-JSON, and a cached bad response would
otherwise pin a fit forever. Retries therefore use a NO-REASONING, CACHE-OFF
clone of the LM — the runaway cannot recur without thinking, and retry samples
are always fresh.
"""

from pathlib import Path

import dspy
from pydantic import BaseModel

from ..config import LMConfig
from ..costs import CostTracker

_RULES = (
    "Each hypothesis must be a single declarative, present-tense sentence about 'the text' "
    '(e.g. "The text describes a sporting event."). It must be verifiable from the text alone '
    "— no references to datasets, labels, or classification. Prefer affirmative phrasing over "
    "negation. Vary specificity: some broad, some narrow. Beyond describing the QUESTION (its "
    "topic, intent, or wording), also include ANSWER-oriented hypotheses that reduce the text to "
    'the imperative it is equivalent to (e.g. "The text is equivalent to asking someone to name a '
    'thing." / "...to explain or define something." / "...to locate a place." / "...to give a '
    'number."). These are most useful for separating classes whose QUESTIONS LOOK ALIKE but whose '
    "ANSWERS differ in form — contrast the answer's shape (e.g. \"The text can be answered with a "
    'short proper name." vs "The text requires a full-sentence explanation."). Do NOT restate an '
    'intent hypothesis as an answer form (e.g. "asks for a person" and "answered with a person\'s '
    'name" are redundant), and avoid vacuous forms true of almost any question (e.g. "a single '
    'word", "a phrase"). Prefer abstract answer-imperative framing over concrete compositional '
    "wording, which the encoder grounds poorly."
)


class Hypothesis(BaseModel):
    statement: str
    rationale: str  # elicited because it improves statement quality; then discarded


class SplitNode(BaseModel):
    """One node of an imagined decision tree over the classes. depth 0 = root (coarsest split);
    `separates` names what this node distinguishes; `hypotheses` implement that split."""

    depth: int
    separates: str  # a balanced group-vs-group split, e.g. "DESC/ABBR/NUM vs HUM/LOC/ENTY"
    hypotheses: list[str]


class GeneratePool(dspy.Signature):
    __doc__ = (
        "Write hypotheses for a natural-language-inference model that will check, for each input "
        "text, whether the text entails each hypothesis. The entailment scores become features "
        "for a downstream classifier. Produce TWO complementary things:\n"
        "(1) `tree` — imagine a BALANCED (as symmetric as possible) decision tree over the classes "
        "and output its splits, ROOT FIRST (depth 0). At EVERY node, split the classes under it into "
        "TWO GROUPS OF ROUGHLY EQUAL SIZE and write hypotheses TRUE for one group and FALSE for the "
        "other — GROUPING features that span several classes at once. Do NOT peel one class vs the "
        "rest: that duplicates the flat list and wastes the tree. The root splits ALL classes into "
        "two halves (e.g. 'DESC, ABBR, NUM' vs 'HUM, LOC, ENTY'); recurse on each half with even "
        "splits until leaves are single classes. Name what each node `separates` as 'group A vs "
        "group B'. Use the class definitions to decide which classes belong together.\n"
        "(2) `hypotheses` — additional standalone hypotheses covering every class from multiple "
        "angles (topic, entity, intent, style, answer-oriented), as independent features.\n"
        "Every hypothesis from BOTH becomes a feature; keep them complementary, not redundant. " + _RULES
    )

    task: str = dspy.InputField(desc="the classification task")
    class_definitions: list[str] = dspy.InputField(desc="one-line definition per class")
    labeled_examples: list[str] = dspy.InputField(desc="sample texts with their true class")
    n: int = dspy.InputField(desc="total hypotheses to write across tree + list")
    avoid: list[str] = dspy.InputField(desc="statements already written; do not repeat or paraphrase")
    tree: list[SplitNode] = dspy.OutputField(desc="BALANCED tree splits, root first; each is group-vs-group")
    hypotheses: list[Hypothesis] = dspy.OutputField(desc="additional diverse standalone hypotheses")


class RefillPool(dspy.Signature):
    __doc__ = (
        "A pool of NLI hypotheses is used as features for a text classifier. The pool is being "
        "refined by recursive elimination: hypotheses that carried no held-out signal were "
        "removed (each with the REASON it failed), and you must write replacements. Do not "
        "paraphrase the survivors; do not repeat the failure patterns. The confusion evidence "
        "lists HOT SPOTS — groups of mutually-confused classes with several example errors each: "
        "write hypotheses for what each group's errors share, that would carve the group apart. "
        "Never write a statement tailored to a single example's topic, entity, or wording. " + _RULES
    )

    task: str = dspy.InputField(desc="the classification task")
    class_definitions: list[str] = dspy.InputField(desc="one-line definition per class")
    labeled_examples: list[str] = dspy.InputField(desc="sample texts with their true class")
    survivors: list[str] = dspy.InputField(
        desc="hypotheses with real held-out signal, strongest first; do not paraphrase"
    )
    failed: list[str] = dspy.InputField(
        desc="pruned hypotheses, each annotated with WHY it failed; avoid their patterns"
    )
    confusion_evidence: list[str] = dspy.InputField(
        desc="hot spots of mutually-confused classes with grouped example errors, plus "
        "counts-only summaries of scattered errors"
    )
    n: int = dspy.InputField(desc="how many replacement hypotheses to write")
    hypotheses: list[Hypothesis] = dspy.OutputField()


class SplitLeaf(dspy.Signature):
    __doc__ = (
        "A decision tree over text classes is stuck: ONE leaf still mixes several classes it "
        "cannot tell apart (the `confused_examples`, with their true labels). Write ONE new NLI "
        "hypothesis whose ENTAILMENT SCORE would best SEPARATE those classes — high for some of the "
        "confused classes and low for the others. Study what the examples of each class share that "
        "the OTHER classes in this leaf do NOT, and name that distinguishing property. The "
        "`related_hypotheses` are what the model ALREADY has for this leaf, each with the fraction "
        "of the leaf's confusion it resolves — they are INSUFFICIENT, and a paraphrase of one of "
        "them will score zero (its signal is already measured): write a hypothesis reading a "
        "genuinely DIFFERENT property than every listed one. The tree already handles everything "
        "ABOVE this leaf, so do not restate coarse distinctions; target exactly what still mixes "
        "here. Return a single statement. " + _RULES
    )

    task: str = dspy.InputField(desc="the classification task")
    class_definitions: list[str] = dspy.InputField(desc="one-line definition per class")
    confused_examples: list[str] = dspy.InputField(desc="'[class] text' samples mixed in this leaf")
    classes_present: list[str] = dspy.InputField(desc="the classes in this leaf, with example counts")
    related_hypotheses: list[str] = dspy.InputField(
        desc="existing hypotheses most relevant to this leaf, with the (insufficient) fraction of "
        "leaf confusion each resolves; complement them, never paraphrase them"
    )
    feedback: list[str] = dspy.InputField(
        desc="MEASURED results of your previous attempts this round, each with why it scored low "
        "(weak split, or collinear with a named existing hypothesis); fix exactly those failures"
    )
    avoid: list[str] = dspy.InputField(desc="hypotheses already in the pool; do not repeat or paraphrase")
    hypothesis: str = dspy.OutputField(desc="a single declarative sentence about 'the text'")


def _attempt_feedback(i: int, hyp: str, r: dict) -> str:
    """Language feedback for the next attempt — names the covariant hypothesis when that is the
    failure, because the wording is more instructive than any scalar."""
    if r.get("covariant_with") is not None and r["novelty"] < 0.5:
        return (
            f'Attempt {i + 1}: "{hyp}" scored {r["score"]:.2f}. Its entailment scores correlate '
            f'{1 - r["novelty"]:.2f} with the EXISTING hypothesis "{r["covariant_with"]}" — the model '
            "already has that exact signal, however differently it is worded. Read a genuinely "
            "DIFFERENT property of the text."
        )
    return (
        f'Attempt {i + 1}: "{hyp}" scored {r["score"]:.2f} (info gain {r["gain"]:.2f}). Its entailment '
        "scores barely separate the confused classes — the property is too weak, too rare here, or "
        "undetectable by the NLI encoder. Try a sharper, more concrete distinguishing property."
    )


def _flatten(result) -> list[str]:
    """Collect statements from a decision-tree output (`tree` of SplitNodes) and/or a flat
    `hypotheses` list — GeneratePool returns both, RefillPool only the flat list."""
    out = []
    for node in getattr(result, "tree", None) or []:
        for s in node.hypotheses or []:
            if s and s.strip():
                out.append(s.strip())
    for h in getattr(result, "hypotheses", None) or []:
        if h.statement and h.statement.strip():
            out.append(h.statement.strip())
    return out


def _make_lm(cfg: LMConfig, cache: bool = True, reasoning: bool = True) -> dspy.LM:
    kwargs: dict = {}
    extra = dict(cfg.extra_body or {})
    if not reasoning:
        extra["reasoning"] = {"enabled": False}
    if extra:
        kwargs["extra_body"] = extra
    return dspy.LM(
        model=cfg.model, max_tokens=cfg.max_tokens, temperature=cfg.temperature, cache=cache, **kwargs
    )


class Proposer:
    def __init__(self, cfg: LMConfig, costs: CostTracker):
        self.cfg = cfg
        self.costs = costs
        self._lm = _make_lm(cfg)
        self._retry_lm = None  # built on first failure
        self._generate = dspy.Predict(GeneratePool)
        self._refill = dspy.Predict(RefillPool)
        self._split = dspy.Predict(SplitLeaf)
        if cfg.instruction_path:  # swap in a GEPA-tuned GeneratePool instruction
            import json

            tuned = json.loads(Path(cfg.instruction_path).read_text())["signature"]["instructions"]
            self._generate.signature = self._generate.signature.with_instructions(tuned)
            print(f"    proposer: using tuned instruction from {cfg.instruction_path}", flush=True)

    def generate(
        self, task: str, class_definitions: list[str], examples: list[str], n: int, avoid: list[str]
    ) -> list[str]:
        return self._call(
            self._generate,
            dict(task=task, class_definitions=class_definitions, labeled_examples=examples, n=n, avoid=avoid),
        )

    def refill(
        self,
        task: str,
        class_definitions: list[str],
        examples: list[str],
        survivors: list[str],
        failed: list[str],
        confusion_evidence: list[str],
        n: int,
    ) -> list[str]:
        return self._call(
            self._refill,
            dict(
                task=task,
                class_definitions=class_definitions,
                labeled_examples=examples,
                survivors=survivors,
                failed=failed,
                confusion_evidence=confusion_evidence,
                n=n,
            ),
        )

    def split_leaf(
        self,
        task: str,
        class_definitions: list[str],
        confused_examples: list[str],
        classes_present: list[str],
        related_hypotheses: list[str],
        avoid: list[str],
        evaluate_fn,
        attempts: int = 4,
        strategy: str = "refine",
        threshold: float = 1.0,
    ) -> tuple[str | None, float]:
        """Propose ONE hypothesis that splits a confused tree leaf: up to `attempts` LM samples,
        each scored by `evaluate_fn(hyp) -> {score, gain, novelty, covariant_with}` (leaf info gain
        x novelty, from tree_evolve); best scorer wins, early exit at `threshold`.

        Our OWN refine loop, not dspy.Refine: its feedback module only sees the scalar reward (and
        burns a second LM call guessing advice), while we KNOW the failure — so 'refine' feeds the
        next attempt measured language ("collinear with <named hypothesis>", "info gain 0.04").
        'best_of_n' = same loop, no feedback (independent samples). Never crashes a fit."""
        inputs = dict(
            task=task,
            class_definitions=class_definitions,
            confused_examples=confused_examples,
            classes_present=classes_present,
            related_hypotheses=related_hypotheses,
            avoid=avoid,
        )
        best_hyp, best_score = None, -1.0
        feedback: list[str] = []
        tried: set[str] = set()
        for k in range(attempts):
            n_before = len(self._lm.history)
            try:
                with dspy.context(lm=self._lm):
                    # rollout_id busts the LM cache so identical inputs still get fresh samples
                    pred = self._split(**inputs, feedback=list(feedback), config={"rollout_id": k})
                hyp = (getattr(pred, "hypothesis", "") or "").strip()
            except Exception as e:  # LM/parse pathologies must not kill a fit
                print(f"      attempt {k + 1} failed ({type(e).__name__})", flush=True)
                continue
            finally:
                self._track(self._lm, n_before)
            if not hyp:
                continue
            if hyp in tried:  # resampling the same statement carries no information
                continue
            tried.add(hyp)
            r = evaluate_fn(hyp)
            if r["score"] > best_score:
                best_hyp, best_score = hyp, r["score"]
            if r["score"] >= threshold:
                break
            if strategy == "refine":
                feedback.append(_attempt_feedback(k, hyp, r))
        return best_hyp, max(best_score, 0.0)

    # -- internals -----------------------------------------------------------

    def _call(self, predictor, inputs: dict) -> list[str]:
        """One primary attempt (cached LM), one fresh no-reasoning retry, never a crash."""
        for attempt in range(2):
            lm = self._lm if attempt == 0 else self._get_retry_lm()
            n_before = len(lm.history)
            try:
                with dspy.context(lm=lm):
                    result = predictor(**inputs)
                return _flatten(result)
            except Exception as e:  # LM pathologies must not kill a fit
                print(f"    proposal failed ({type(e).__name__}), attempt {attempt + 1}/2", flush=True)
            finally:
                self._track(lm, n_before)
        return []

    def _get_retry_lm(self) -> dspy.LM:
        if self._retry_lm is None:
            self._retry_lm = _make_lm(self.cfg, cache=False, reasoning=False)
        return self._retry_lm

    def _track(self, lm: dspy.LM, n_before: int) -> None:
        """Cost + abnormal-finish attribution (finish_reason, serving provider)."""
        for entry in lm.history[n_before:]:
            self.costs.lm_calls += 1
            usage = entry.get("usage") or {}
            self.costs.lm_input_tokens += usage.get("prompt_tokens") or 0
            self.costs.lm_output_tokens += usage.get("completion_tokens") or 0
            self.costs.lm_usd += entry.get("cost") or 0.0
            try:
                response = entry.get("response")
                finish = getattr(response.choices[0], "finish_reason", None)
                provider = getattr(response, "provider", None)
            except Exception:
                finish, provider = None, None
            if finish and finish != "stop":
                self.costs.lm_abnormal_finishes += 1
                print(f"    lm finish_reason={finish!r} provider={provider or 'unknown'}", flush=True)


def generate_pool(
    proposer, deduper, task, class_definitions, examples, size: int, fixed: list[str] = ()
) -> list[str]:
    """Generate a deduped pool of up to `size` NEW hypotheses (excluding `fixed`): a few generate
    passes, keeping only novel (deduped) statements. `fixed` are user-written hypotheses already in
    the model — the LM is told to avoid them and candidates are deduped against them, but they do
    not count toward `size`. Shared by the training runner and HypothesisVectorizer.fit."""
    from ..dedup import norm_statement

    pool: list[str] = []
    seen: set[str] = {norm_statement(f) for f in fixed}
    for _ in range(5):  # a few attempts in case the LM under-delivers or dedup trims
        if len(pool) >= size:
            break
        proposed = proposer.generate(
            task, class_definitions, examples, n=size - len(pool), avoid=list(fixed) + pool
        )
        kept, _ = deduper.filter(proposed, against=list(fixed) + pool, seen=seen)
        pool += kept
    return pool[:size]
