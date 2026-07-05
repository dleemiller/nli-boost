"""HypothesisVectorizer: the inference interface for the method.

Turns a column of text into features by asking a frozen NLI cross-encoder, for each
input text, how strongly it entails (and contradicts) each of a fixed set of
natural-language hypotheses. Those scores are the columns. The "model" is just the
hypothesis list + the encoder name, so INFERENCE needs no LM and no dspy — only the
encoder. Hypothesis *generation/evolution* (the training side) lives elsewhere and
merely produces the list this consumes.

It is a plain scikit-learn transformer, so it composes the usual ways:

    Pipeline([("hyp", HypothesisVectorizer(hypotheses)), ("clf", HistGradientBoostingClassifier())])

    # optional TF-IDF channel — standard sklearn, not baked in:
    FeatureUnion([("nli", HypothesisVectorizer(hypotheses)),
                  ("tfidf", make_pipeline(TfidfVectorizer(), TruncatedSVD(128)))])

    # one text column alongside other tabular features:
    ColumnTransformer([("nli", HypothesisVectorizer(hypotheses), "text"),
                       ("num", StandardScaler(), ["price", "age"])])
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .cache import ScoreCache
from .config import EncoderConfig
from .costs import CostTracker
from .encoder import EntailmentScorer

_SCORE_MODES = ("entail_contradict", "entail", "contrast")
_DEFAULT_ENCODER = "dleemiller/finecat-nli-l"


class HypothesisVectorizer(BaseEstimator, TransformerMixin):
    """NLI-entailment features over a fixed hypothesis set.

    Parameters
    ----------
    hypotheses : list[str] | None
        The natural-language hypotheses — the feature vocabulary. Required before
        ``transform``; produced by the (separate) generation/evolution step.
    encoder : str
        HuggingFace cross-encoder id (entailment=0, neutral=1, contradiction=2).
    score_mode : {"entail_contradict", "entail", "contrast"}
        Columns per hypothesis: both P(entail) and P(contradict) (2 cols), just
        P(entail) (1 col), or their difference P(entail)-P(contradict) (1 col).
    device, batch_size, max_text_chars : encoder inference knobs.
    cache_path : str | Path | None
        sqlite score cache. A path persists scores across processes; ``None`` uses
        an in-process cache (repeat transforms are free within the instance).
    task, class_definitions, class_names, n_hypotheses, lm, dedup_corr, evolve
        Generation knobs, used ONLY by ``fit(X, y)`` when ``hypotheses`` is None — the
        pool is then generated from the data via the LM proposer, which requires the
        ``train`` extras (dspy). Ignored when ``hypotheses`` is supplied. ``evolve=True``
        additionally refines the generated pool with the CV-prune/refill loop (stronger
        pool, more LM calls); ``evolve=False`` (default) stops at a static pool.

    Notes
    -----
    All parameters follow the sklearn convention: stored verbatim in ``__init__``,
    with fitted state on ``hypotheses_``. The live encoder/cache is lazy and dropped
    on pickling, so fitted pipelines serialize cleanly.
    """

    def __init__(
        self,
        hypotheses=None,
        *,
        encoder=_DEFAULT_ENCODER,
        score_mode="entail_contradict",
        device="cuda",
        batch_size=128,
        max_text_chars=1200,
        cache_path=None,
        task=None,
        class_definitions=None,
        class_names=None,
        n_hypotheses=64,
        lm="openrouter/deepseek/deepseek-v4-flash",
        dedup_corr=0.95,
        evolve=False,
    ):
        self.hypotheses = hypotheses
        self.encoder = encoder
        self.score_mode = score_mode
        self.device = device
        self.batch_size = batch_size
        self.max_text_chars = max_text_chars
        self.cache_path = cache_path
        # generation params (only used by fit when `hypotheses` is None; require the `train` extras)
        self.task = task
        self.class_definitions = class_definitions
        self.class_names = class_names
        self.n_hypotheses = n_hypotheses
        self.lm = lm
        self.dedup_corr = dedup_corr
        self.evolve = evolve

    # -- sklearn API ---------------------------------------------------------

    def fit(self, X=None, y=None):
        """Fix the hypothesis set. If `hypotheses` was given it is used as-is (pure transformer,
        no LM — X/y ignored). If not, the hypotheses are GENERATED from (X, y) via the LM proposer,
        which requires the `train` extras (dspy); `task` and `class_definitions` must be set."""
        if self.score_mode not in _SCORE_MODES:
            raise ValueError(f"score_mode must be one of {_SCORE_MODES}, got {self.score_mode!r}")
        if self.hypotheses:
            self.hypotheses_ = list(self.hypotheses)
            self._scorer = None  # lazy; built on first transform
        else:
            self.hypotheses_ = self._generate(X, y)  # builds self._scorer (reused by dedup)
            if not self.hypotheses_:
                raise ValueError("hypothesis generation produced an empty pool")
        return self

    def transform(self, X):
        check_is_fitted(self, "hypotheses_")
        texts = self._coerce_texts(X)
        feats = self._get_scorer().features(texts, self.hypotheses_)  # (n, 2m) [entail | contradict]
        m = len(self.hypotheses_)
        if self.score_mode == "entail_contradict":
            return feats
        if self.score_mode == "entail":
            return feats[:, :m]
        return feats[:, :m] - feats[:, m:]  # contrast

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "hypotheses_")
        if self.score_mode == "entail_contradict":
            names = [f"entail: {h}" for h in self.hypotheses_] + [
                f"contradict: {h}" for h in self.hypotheses_
            ]
        elif self.score_mode == "entail":
            names = [f"entail: {h}" for h in self.hypotheses_]
        else:
            names = [f"contrast: {h}" for h in self.hypotheses_]
        return np.asarray(names, dtype=object)

    # -- text-column input handling -----------------------------------------

    @staticmethod
    def _coerce_texts(X) -> list[str]:
        """Accept a 1-D sequence/Series of strings or a single-column 2-D array/frame
        (as ColumnTransformer hands over), returning a list[str]."""
        arr = np.asarray(X, dtype=object)
        if arr.ndim == 2:
            if arr.shape[1] != 1:
                raise ValueError(
                    f"HypothesisVectorizer scores ONE text column; got shape {arr.shape}. "
                    "Select a single text column (e.g. via ColumnTransformer)."
                )
            arr = arr[:, 0]
        return [("" if t is None else str(t)) for t in arr.ravel()]

    # -- lazy encoder (never pickled) ---------------------------------------

    def _get_scorer(self) -> EntailmentScorer:
        if getattr(self, "_scorer", None) is None:
            cfg = EncoderConfig(
                model=self.encoder,
                device=self.device,
                batch_size=self.batch_size,
                max_text_chars=self.max_text_chars,
            )
            cache = ScoreCache(self.cache_path if self.cache_path is not None else ":memory:")
            self._scorer = EntailmentScorer(cfg, cache, CostTracker())
        return self._scorer

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "_scorer"}

    # -- generation (training side; needs the `train` extras) ----------------

    def _generate(self, X, y) -> list[str]:
        if X is None or y is None:
            raise ValueError(
                "No `hypotheses` given: call fit(X, y) with texts+labels to generate them "
                "(needs the `train` extras), or pass hypotheses=... / use from_run()."
            )
        if not self.task or not self.class_definitions:
            raise ValueError("Generating hypotheses requires `task` and `class_definitions`.")
        try:
            from .proposer import Proposer, generate_pool  # train extra: pulls dspy
        except ImportError as e:  # pragma: no cover - depends on install
            raise ImportError(
                "Generating hypotheses needs the training dependencies (dspy). Install nli-boost "
                "with the `train` dependency group."
            ) from e
        from .config import LMConfig
        from .dedup import Deduper

        texts = self._coerce_texts(X)
        y = np.asarray(y)
        examples = _labeled_examples(texts, y, self.class_names)
        rng = np.random.default_rng(0)
        ref_idx = rng.choice(len(texts), size=min(400, len(texts)), replace=False)
        scorer = self._get_scorer()
        deduper = Deduper(scorer, [texts[i] for i in ref_idx], self.dedup_corr)
        proposer = Proposer(LMConfig(model=self.lm), CostTracker())
        pool = generate_pool(
            proposer, deduper, self.task, self.class_definitions, examples, self.n_hypotheses
        )
        if self.evolve:  # opt-in: refine the static pool (CV-prune weak, refill hot-spots)
            from types import SimpleNamespace

            from .config import PoolConfig
            from .evolve import evolve as evolve_pool

            names = self.class_names or [f"class {c}" for c in range(int(y.max()) + 1)]
            bundle = SimpleNamespace(
                task=self.task,
                class_names=names,
                class_descriptions=self.class_definitions,
                train_texts=texts,
                y_train=y,
                n_classes=len(names),
            )
            pool, _history = evolve_pool(
                bundle, pool, scorer, proposer, deduper, PoolConfig(size=self.n_hypotheses), seed=0
            )
        return pool

    # -- persistence & config -----------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write the inference artifact (hypotheses + encoder config) as JSON."""
        Path(path).write_text(
            json.dumps(
                {
                    "hypotheses": self.hypotheses_ if hasattr(self, "hypotheses_") else self.hypotheses,
                    "encoder": self.encoder,
                    "score_mode": self.score_mode,
                    "device": self.device,
                    "batch_size": self.batch_size,
                    "max_text_chars": self.max_text_chars,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "HypothesisVectorizer":
        """Load a saved artifact into a fitted, transform-ready vectorizer."""
        return cls.from_config(json.loads(Path(path).read_text()))

    @classmethod
    def from_config(cls, config: dict | str | Path) -> "HypothesisVectorizer":
        """Build from a dict or YAML file. `encoder` may be a model-id string or a
        mapping ({model, device, batch_size, max_text_chars}) — so run config.yaml
        files work directly. If `hypotheses` is present the result is fitted."""
        if isinstance(config, (str, Path)):
            config = yaml.safe_load(Path(config).read_text())
        enc = config.get("encoder", {})
        enc = {"model": enc} if isinstance(enc, str) else dict(enc or {})
        vec = cls(
            hypotheses=config.get("hypotheses"),
            encoder=enc.get("model", _DEFAULT_ENCODER),
            score_mode=config.get("score_mode", "entail_contradict"),
            device=enc.get("device", "cuda"),
            batch_size=enc.get("batch_size", 128),
            max_text_chars=enc.get("max_text_chars", 1200),
            cache_path=config.get("cache_path"),
        )
        return vec.fit() if vec.hypotheses else vec

    @classmethod
    def from_run(cls, run_dir: str | Path) -> "HypothesisVectorizer":
        """Load a trained run (its config.yaml encoder + model.json hypotheses) into a
        fitted vectorizer ready for inference."""
        run_dir = Path(run_dir)
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        model = json.loads((run_dir / "model.json").read_text())
        cfg["hypotheses"] = model["hypotheses"]
        cfg.setdefault("cache_path", str(run_dir.parent.parent / "cache" / "nli_scores.sqlite"))
        return cls.from_config(cfg)


def _labeled_examples(texts, y, class_names, per_class: int = 3) -> list[str]:
    """A few 'text -> class' lines per class, to ground the proposer's generation prompt."""
    import numpy as np

    y = np.asarray(y)
    names = class_names or [f"class {c}" for c in range(int(y.max()) + 1)]
    rng = np.random.default_rng(0)
    out: list[str] = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        for i in rng.choice(idx, size=min(per_class, len(idx)), replace=False):
            out.append(f"{texts[int(i)]} -> {names[int(c)]}")
    return out
