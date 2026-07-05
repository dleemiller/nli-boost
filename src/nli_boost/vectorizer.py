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
    verbose : bool
        Print progress lines during long encoder scoring passes (default False).
    task, class_definitions, class_names, n_hypotheses, lm, evolve, random_state
        Generation knobs, used ONLY by ``fit(X, y)`` when ``hypotheses`` is None — the
        pool is then generated from the data via the LM proposer, which requires the
        ``train`` extra (``pip install "nli-boost[train]"``). Ignored when ``hypotheses``
        is supplied. ``evolve=True`` additionally refines the generated pool with the
        CV-prune/refill loop (stronger pool, more LM calls); ``evolve=False`` (default)
        stops at a static pool. ``random_state`` seeds example sampling and evolution.
    dedup : {"covariance", "sts"} or object
        Candidate dedup during generation. ``"covariance"`` (default) rejects candidates
        whose entail-score vectors are ~collinear with a kept one — behaviorally exact but
        needs enough data to estimate. ``"sts"`` compares hypothesis TEXTS with a bi-encoder
        (cosine > threshold = duplicate) — data-free, the right choice at a few examples
        per class. Any object with ``.filter(candidates, against, seen)`` also works.
    dedup_threshold : float
        Rejection threshold for the chosen backend (|Pearson| for covariance, cosine for
        sts; ~0.9 is a sensible sts value).

    Attributes
    ----------
    hypotheses_ : list[str]
        The fitted hypothesis set (the feature vocabulary).
    evolution_history_ : list[dict]
        Present after ``fit`` with ``evolve=True``: one dict per round with the exact
        ``pool`` scored that round and its ``heldout_acc`` — every round is recoverable.

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
        verbose=False,
        task=None,
        class_definitions=None,
        class_names=None,
        n_hypotheses=64,
        lm="openrouter/deepseek/deepseek-v4-flash",
        dedup="covariance",
        dedup_threshold=0.95,
        evolve=False,
        random_state=0,
    ):
        self.hypotheses = hypotheses
        self.encoder = encoder
        self.score_mode = score_mode
        self.device = device
        self.batch_size = batch_size
        self.max_text_chars = max_text_chars
        self.cache_path = cache_path
        self.verbose = verbose
        # generation params (only used by fit when `hypotheses` is None; require the `train` extra)
        self.task = task
        self.class_definitions = class_definitions
        self.class_names = class_names
        self.n_hypotheses = n_hypotheses
        self.lm = lm
        self.dedup = dedup
        self.dedup_threshold = dedup_threshold
        self.evolve = evolve
        self.random_state = random_state

    # -- sklearn API ---------------------------------------------------------

    def __sklearn_tags__(self):
        # a text transformer: consumes a 1-D column of strings, not 2-D numeric arrays,
        # and needs no y at transform time (same shape as TfidfVectorizer's tags)
        tags = super().__sklearn_tags__()
        tags.input_tags.string = True
        tags.input_tags.one_d_array = True
        tags.input_tags.two_d_array = False
        tags.target_tags.required = False
        return tags

    def fit(self, X=None, y=None, baseline_features=None):
        """Fix the hypothesis set. If `hypotheses` was given it is used as-is (pure transformer,
        no LM — X/y ignored). If not, the hypotheses are GENERATED from (X, y) via the LM proposer,
        which requires the `train` extra; `task` and `class_definitions` must be set.

        `baseline_features` (n_samples, d), optional: any extra feature block the downstream head
        will ALSO see — other tabular columns, TF-IDF, embeddings. With ``evolve=True`` the
        hypotheses are then pruned by their MARGINAL value over these fixed columns, so the pool
        keeps only what the baseline can't carry. (In a Pipeline, route it as
        ``pipe.fit(X, y, hyp__baseline_features=Z)``.)"""
        if self.score_mode not in _SCORE_MODES:
            raise ValueError(f"score_mode must be one of {_SCORE_MODES}, got {self.score_mode!r}")
        if self.hypotheses:
            self.hypotheses_ = list(self.hypotheses)
            self._scorer = None  # lazy; built on first transform
        else:
            self.hypotheses_ = self._generate(X, y, baseline_features)  # builds self._scorer
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
                verbose=self.verbose,
            )
            cache = ScoreCache(self.cache_path if self.cache_path is not None else ":memory:")
            self._scorer = EntailmentScorer(cfg, cache, CostTracker())
        return self._scorer

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "_scorer"}

    # -- generation (training side; needs the `train` extras) ----------------

    def _generate(self, X, y, baseline_features=None) -> list[str]:
        if X is None or y is None:
            raise ValueError(
                "No `hypotheses` given: call fit(X, y) with texts+labels to generate them "
                "(needs the `train` extra), or pass hypotheses=... / use from_run()."
            )
        if not self.task or not self.class_definitions:
            raise ValueError("Generating hypotheses requires `task` and `class_definitions`.")
        try:
            from .proposer import Proposer, generate_pool  # train extra: pulls dspy
        except ImportError as e:  # pragma: no cover - depends on install
            raise ImportError(
                'Generating hypotheses needs the training dependencies: pip install "nli-boost[train]".'
            ) from e
        from .config import LMConfig
        from .data import labeled_examples

        texts = self._coerce_texts(X)
        y = np.asarray(y)
        if baseline_features is not None:
            baseline_features = np.asarray(baseline_features, dtype=np.float64)
            if baseline_features.ndim != 2 or baseline_features.shape[0] != len(texts):
                raise ValueError(
                    f"baseline_features must be (n_samples, d) aligned with X; "
                    f"got {baseline_features.shape} for {len(texts)} texts"
                )
        names = self.class_names or [f"class {c}" for c in range(int(y.max()) + 1)]
        rng = np.random.default_rng(self.random_state)
        examples = labeled_examples(texts, y, names, per_class=3, rng=rng)
        scorer = self._get_scorer()
        deduper = self._make_deduper(texts, y, rng)
        proposer = Proposer(LMConfig(model=self.lm), CostTracker())
        pool = generate_pool(
            proposer, deduper, self.task, self.class_definitions, examples, self.n_hypotheses
        )
        if self.evolve:  # opt-in: refine the static pool (CV-prune weak, refill hot-spots)
            from types import SimpleNamespace

            from .config import PoolConfig
            from .evolve import evolve as evolve_pool

            bundle = SimpleNamespace(
                task=self.task,
                class_names=names,
                class_descriptions=self.class_definitions,
                train_texts=texts,
                y_train=y,
                n_classes=len(names),
            )
            pool, self.evolution_history_ = evolve_pool(
                bundle,
                pool,
                scorer,
                proposer,
                deduper,
                PoolConfig(size=self.n_hypotheses),
                seed=self.random_state,
                baseline_train=baseline_features,
            )
        return pool

    def _make_deduper(self, texts, y, rng):
        """Dedup backend: 'covariance' (behavioral, needs data), 'sts' (text-similarity,
        data-free — the low-data choice), or any object with a .filter(candidates, against,
        seen) method."""
        if hasattr(self.dedup, "filter"):
            return self.dedup
        if self.dedup == "sts":
            from .dedup import STSDeduper

            return STSDeduper(threshold=self.dedup_threshold, device=self.device)
        if self.dedup == "covariance":
            from .data import stratified_indices
            from .dedup import Deduper

            ref_idx = stratified_indices(y, min(400, len(texts)), rng)  # correlate on this sample
            return Deduper(self._get_scorer(), [texts[int(i)] for i in ref_idx], self.dedup_threshold)
        raise ValueError(f"dedup must be 'covariance', 'sts', or a deduper object; got {self.dedup!r}")

    # -- persistence & config -----------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write the fitted inference artifact (hypotheses + encoder config) as JSON."""
        check_is_fitted(self, "hypotheses_")
        Path(path).write_text(
            json.dumps(
                {
                    "hypotheses": self.hypotheses_,
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
        """Build from a dict or YAML file of constructor params. `encoder` may be a
        model-id string or a mapping ({model, device, batch_size, max_text_chars}) — so
        run config.yaml files work directly. Unknown keys are ignored. If `hypotheses`
        is present the result is fitted."""
        if isinstance(config, (str, Path)):
            config = yaml.safe_load(Path(config).read_text())
        config = dict(config)
        enc = config.get("encoder")
        if isinstance(enc, dict):  # nested run-config form -> flatten onto constructor params
            config["encoder"] = enc.get("model", _DEFAULT_ENCODER)
            for k in ("device", "batch_size", "max_text_chars"):
                if k in enc:
                    config.setdefault(k, enc[k])
        params = cls().get_params()
        vec = cls(**{k: v for k, v in config.items() if k in params})
        return vec.fit() if vec.hypotheses else vec

    @classmethod
    def from_run(cls, run_dir: str | Path) -> "HypothesisVectorizer":
        """Load a trained run (its config.yaml encoder + model.json hypotheses) into a
        fitted vectorizer ready for inference, sharing the run's score cache."""
        run_dir = Path(run_dir)
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        model = json.loads((run_dir / "model.json").read_text())
        return cls.from_config(
            {
                "hypotheses": model["hypotheses"],
                "encoder": cfg.get("encoder", {}),
                "score_mode": cfg.get("score_mode", "entail_contradict"),
                "cache_path": str(Path(cfg.get("cache_dir", "cache")) / "nli_scores.sqlite"),
            }
        )
