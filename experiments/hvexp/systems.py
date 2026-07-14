"""Pluggable classification systems for the learning-curve / baseline study.

Every system implements the same call:

    proba = system.run(train_texts, y_train, test_texts)   # -> (n_test, n_classes)

so the harness can treat baselines and HV variants uniformly. NLI-based systems share one
cache-backed `NLIFeaturizer`, so scoring the same corpus twice is free.

Systems implemented here (all LM-free):
  * TfidfLogReg           — TF-IDF (word or char n-grams) + logistic regression
  * TfidfUnionLogReg      — word ∪ char TF-IDF + logistic regression
  * EmbeddingLogReg       — frozen sentence-embeddings + logistic regression
  * ZeroShotNLI           — score class templates, argmax entailment (no training; N=0 anchor)
  * HVHead                — HV features + a classifier head ('auto' RF/HGB grid | 'logreg' L2)
  * PriorAggregation      — the low-N prior head: per-class mean entailment of class-tagged
                            hypotheses; 'fixed' (N=0 zero-shot ensemble) or 'reweight' (strong-L2).

LLM-generated pools plug straight into HVHead / PriorAggregation once a pool is generated —
only the `pool` argument changes.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import NLIFeaturizer


# --------------------------------------------------------------------------- helpers
def _safe_folds(y: np.ndarray, max_folds: int = 4) -> int:
    """Largest CV fold count that every class can populate (>=2 to be a CV at all)."""
    _, counts = np.unique(y, return_counts=True)
    return int(max(2, min(max_folds, counts.min())))


def _cv_head_light(x, y, seed, folds):
    """A faster stand-in for the library's cv_selected_head for the learning-curve sweep.

    Same idea — CV-select (family, regularization) on train, refit — but a trimmed grid at 120
    trees instead of 300 (the sweep fits this 90+ times; 300-tree RFs on <60 rows are ~5x slower
    for no low-N benefit). The full 300-tree library head is available as head='auto_full' for
    any single headline number that must match the CLI exactly.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

    grid = (
        [dict(kind="rf", min_samples_leaf=msl, max_features=mf) for msl in (2, 5) for mf in (0.6, 1.0)]
        + [dict(kind="hgb", learning_rate=lr, l2_regularization=l2) for lr in (0.06, 0.12) for l2 in (0.01, 0.3)]
    )

    def build(p):
        p = dict(p)
        if p.pop("kind") == "rf":
            return RandomForestClassifier(n_estimators=120, random_state=seed, n_jobs=4, **p)
        return HistGradientBoostingClassifier(max_iter=200, random_state=seed, **p)

    scored = [(float(cross_val_score(build(p), x, y, cv=folds).mean()), i) for i, p in enumerate(grid)]
    _cv, best_i = max(scored)
    head = build(grid[best_i])
    head.fit(x, y)
    return head


def _proba_to_full(clf, X, n_classes: int) -> np.ndarray:
    """predict_proba aligned to 0..n_classes-1 even if some classes are absent from train."""
    p = clf.predict_proba(X)
    if p.shape[1] == n_classes and list(clf.classes_) == list(range(n_classes)):
        return p
    full = np.zeros((X.shape[0], n_classes), dtype=float)
    for j, c in enumerate(clf.classes_):
        full[:, int(c)] = p[:, j]
    return full


def _cols_to_full(p: np.ndarray, classes: np.ndarray, n_classes: int) -> np.ndarray:
    """Align a raw proba matrix (columns ordered by `classes`) to 0..n_classes-1."""
    if p.shape[1] == n_classes and list(classes) == list(range(n_classes)):
        return p
    full = np.zeros((p.shape[0], n_classes), dtype=float)
    for j, c in enumerate(classes):
        full[:, int(c)] = p[:, j]
    return full


# --------------------------------------------------------------------------- lexical baselines
class TfidfLogReg:
    def __init__(self, n_classes: int, analyzer: str = "word",
                 ngram_range: tuple[int, int] = (1, 2), name: str | None = None):
        self.n_classes = n_classes
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.name = name or f"tfidf_{analyzer}+logreg"

    def run(self, train_texts, y_train, test_texts) -> np.ndarray:
        vec = TfidfVectorizer(analyzer=self.analyzer, ngram_range=self.ngram_range,
                              min_df=1, sublinear_tf=True)
        Xtr = vec.fit_transform(train_texts)
        Xte = vec.transform(test_texts)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, y_train)
        return _proba_to_full(clf, Xte, self.n_classes)


class TfidfUnionLogReg:
    def __init__(self, n_classes: int, name: str = "tfidf_word+char+logreg"):
        self.n_classes = n_classes
        self.name = name

    def run(self, train_texts, y_train, test_texts) -> np.ndarray:
        from sklearn.pipeline import FeatureUnion

        union = FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True)),
        ])
        Xtr = union.fit_transform(train_texts)
        Xte = union.transform(test_texts)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, y_train)
        return _proba_to_full(clf, Xte, self.n_classes)


# --------------------------------------------------------------------------- embedding baseline
@lru_cache(maxsize=4)
def _sentence_model(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


class EmbeddingLogReg:
    def __init__(self, n_classes: int, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cuda", name: str | None = None):
        self.n_classes = n_classes
        self.model_name = model_name
        self.device = device
        self.name = name or f"emb:{model_name.split('/')[-1]}+logreg"
        self._cache: dict[str, np.ndarray] = {}

    def _embed(self, texts) -> np.ndarray:
        model = _sentence_model(self.model_name, self.device)
        return np.asarray(model.encode(list(texts), batch_size=256, show_progress_bar=False,
                                       normalize_embeddings=True), dtype=np.float32)

    def run(self, train_texts, y_train, test_texts) -> np.ndarray:
        Xtr, Xte = self._embed(train_texts), self._embed(test_texts)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(Xtr, y_train)
        return _proba_to_full(clf[-1], clf[:-1].transform(Xte), self.n_classes)


# --------------------------------------------------------------------------- fine-tuned encoder
class FineTunedEncoder:
    """Fine-tune a small pretrained encoder (DistilBERT) end-to-end on the training subsample.

    The strong *data-rich* supervised reference: weak at low N (few gradient steps, no semantic
    prior) and strong at high N. Unlike HV it is opaque and needs a GPU fine-tune per fit.
    A fixed recipe (AdamW, lr 2e-5) trained for `epochs` passes capped at `max_steps` total —
    so low-N runs get many epochs over few rows and full-data runs stop at a few epochs.
    """

    def __init__(self, n_classes: int, model_name: str = "distilbert-base-uncased",
                 device: str = "cuda", max_len: int = 128, lr: float = 2e-5, batch_size: int = 16,
                 epochs: int = 20, max_steps: int = 1000, seed: int = 0, name: str | None = None):
        self.n_classes = n_classes
        self.model_name = model_name
        self.device = device
        self.max_len = max_len
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_steps = max_steps
        self.seed = seed
        self.name = name or f"finetune:{model_name.split('/')[-1]}"

    def run(self, train_texts, y_train, test_texts) -> np.ndarray:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.manual_seed(self.seed)
        tok = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.n_classes).to(self.device)

        def encode(texts):
            enc = tok(list(texts), truncation=True, padding=True, max_length=self.max_len,
                      return_tensors="pt")
            return enc["input_ids"], enc["attention_mask"]

        ids, mask = encode(train_texts)
        y = torch.tensor(np.asarray(y_train), dtype=torch.long)
        loader = DataLoader(TensorDataset(ids, mask, y), batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0.01)

        model.train()
        step = 0
        for _ in range(self.epochs):
            for bi, bm, by in loader:
                opt.zero_grad()
                out = model(input_ids=bi.to(self.device), attention_mask=bm.to(self.device),
                            labels=by.to(self.device))
                out.loss.backward()
                opt.step()
                step += 1
                if step >= self.max_steps:
                    break
            if step >= self.max_steps:
                break

        model.eval()
        tids, tmask = encode(test_texts)
        probs = []
        with torch.no_grad():
            for s in range(0, len(test_texts), 64):
                logits = model(input_ids=tids[s:s + 64].to(self.device),
                               attention_mask=tmask[s:s + 64].to(self.device)).logits
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        del model
        torch.cuda.empty_cache()
        return np.concatenate(probs, axis=0)


# --------------------------------------------------------------------------- zero-shot NLI
class ZeroShotNLI:
    """Score each class template's entailment; softmax over classes. No labels used."""

    def __init__(self, n_classes: int, templates: list[str], featurizer: NLIFeaturizer,
                 temperature: float = 1.0, name: str = "zeroshot_nli"):
        assert len(templates) == n_classes, "one template per class, in class-index order"
        self.n_classes = n_classes
        self.templates = templates
        self.fz = featurizer
        self.temperature = temperature
        self.name = name

    def run(self, train_texts, y_train, test_texts) -> np.ndarray:
        e = self.fz.features(test_texts, self.templates, score_mode="entail")  # (n, n_classes)
        z = e / self.temperature
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- HV + head
class HVHead:
    """HV features (P(entail)|P(contradict)) + a classifier head."""

    def __init__(self, n_classes: int, pool: list[str], featurizer: NLIFeaturizer,
                 head: str = "auto", score_mode: str = "entail_contradict",
                 seed: int = 0, name: str | None = None):
        self.n_classes = n_classes
        self.pool = pool
        self.fz = featurizer
        self.head = head
        self.score_mode = score_mode
        self.seed = seed
        self.name = name or f"hv_{head}"

    def _make_head(self, y):
        if self.head == "logreg":
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"),
            )
        if self.head == "logreg_cv":
            _, counts = np.unique(y, return_counts=True)
            if counts.min() < 2:  # cannot CV with <2/class — fall back to a mild default
                return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
            folds = _safe_folds(y)
            best_c, best = 1.0, -1.0
            for c in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
                pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=c))
                try:
                    s = cross_val_score(pipe, self._Xtr, y, cv=StratifiedKFold(folds)).mean()
                except Exception:
                    s = -1.0
                if s > best:
                    best, best_c = s, c
            return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=best_c))
        raise ValueError(self.head)

    def run(self, train_texts, y_train, test_texts) -> np.ndarray:
        Xtr = self.fz.features(train_texts, self.pool, self.score_mode)
        Xte = self.fz.features(test_texts, self.pool, self.score_mode)
        if self.head in ("rf", "hgb"):  # single flexible tree-ensemble head — fast, no CV grid
            from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

            if self.head == "hgb":  # note: single HGB degenerates to a constant class on <10 rows;
                head = HistGradientBoostingClassifier(  # RF degrades gracefully and is the default
                    max_iter=200, learning_rate=0.1, l2_regularization=0.1, random_state=self.seed)
            else:
                head = RandomForestClassifier(n_estimators=200, random_state=self.seed, n_jobs=4)
            head.fit(Xtr, y_train)
            return _proba_to_full(head, Xte, self.n_classes)
        if self.head in ("auto", "auto_full"):
            _, counts = np.unique(y_train, return_counts=True)
            if counts.min() < 2:  # too few rows to CV-select — mild RF default
                from sklearn.ensemble import RandomForestClassifier

                head = RandomForestClassifier(n_estimators=120, random_state=self.seed,
                                              n_jobs=4, min_samples_leaf=1)
                head.fit(Xtr, y_train)
                return _proba_to_full(head, Xte, self.n_classes)
            if self.head == "auto_full":  # exact library head (300 trees) for headline single points
                from hypothesis_vectorizer.train.head import cv_selected_head

                head, _p, _cv = cv_selected_head(Xtr, y_train, self.seed, folds=_safe_folds(y_train))
                return _proba_to_full(head, Xte, self.n_classes)
            head = _cv_head_light(Xtr, y_train, self.seed, _safe_folds(y_train))
            return _proba_to_full(head, Xte, self.n_classes)
        self._Xtr = Xtr
        clf = self._make_head(y_train)
        clf.fit(Xtr, y_train)
        return _proba_to_full(clf[-1], clf[:-1].transform(Xte), self.n_classes)


# --------------------------------------------------------------------------- prior-aggregation head
class PriorAggregation:
    """Low-N head: per-class score = mean entailment of that class's tagged hypotheses.

    mode='fixed'    : argmax of the per-class prior. Uses NO labels — at any N this is a
                      zero-shot NLI *ensemble* over multiple hypotheses per class.
    mode='reweight' : strong-L2 multinomial logreg over the (n, n_classes) aggregated scores,
                      i.e. let a little data reweight the class votes without touching DOF per
                      hypothesis. The intended low-N crossover head (docs/low-n-plan.md).
    """

    def __init__(self, n_classes: int, pool: list[str], tags: list[str], class_names: list[str],
                 featurizer: NLIFeaturizer, mode: str = "fixed", temperature: float = 0.1,
                 name: str | None = None):
        self.n_classes = n_classes
        self.pool = pool
        self.fz = featurizer
        self.class_names = class_names
        self.mode = mode
        self.temperature = temperature
        self.name = name or f"prior_{mode}"
        # boolean (n_classes, n_hyp) membership: hypothesis j supports class i
        idx = {c: i for i, c in enumerate(class_names)}
        self.member = np.zeros((n_classes, len(pool)), dtype=float)
        for j, t in enumerate(tags):
            self.member[idx[t], j] = 1.0
        self.member /= np.clip(self.member.sum(axis=1, keepdims=True), 1.0, None)

    def _class_scores(self, texts) -> np.ndarray:
        e = self.fz.features(texts, self.pool, score_mode="entail")  # (n, n_hyp) P(entail)
        return e @ self.member.T  # (n, n_classes): mean entailment of each class's hypotheses

    def _softmax(self, s):
        z = s / self.temperature
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def run(self, train_texts, y_train, test_texts) -> np.ndarray:
        Ste = self._class_scores(test_texts)
        if self.mode == "fixed":
            return self._softmax(Ste)
        # reweight: strong-L2 multinomial logreg on aggregated class scores
        Str = self._class_scores(train_texts)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.25))
        clf.fit(Str, y_train)
        return _proba_to_full(clf[-1], clf[:-1].transform(Ste), self.n_classes)


# --------------------------------------------------------------- shrinkage heads (the low-N gap)
# Both interpolate the label-free prior (owns 1-3/class) and a data-driven head (owns high N),
# so one head traces a continuous curve across the whole sweep instead of hand-swapping heads by
# N (docs/low-n-plan.md, "elegant unifying version"). Shrinkage is chosen on TRAIN ONLY.
class PriorShrinkageBlend:
    """Design A — convex blend of the label-free prior and a data-driven head.

        p = lam * prior_proba + (1 - lam) * learned_proba

    `lam` in [0,1] is CV-selected on train (grid includes 0 and 1). With <2 examples/class a CV
    is impossible, so lam=1 -> the pure prior (the correct low-N default). The prior head is
    label-free, so its train proba carries no leakage; the learned head is scored out-of-fold so
    lam is picked fairly. learned='rf' matches the hv_expert_rf ceiling; 'logreg' is lighter.
    """

    def __init__(self, n_classes: int, pool: list[str], tags: list[str], class_names: list[str],
                 featurizer: NLIFeaturizer, learned: str = "rf", temperature: float = 0.1,
                 seed: int = 0, name: str | None = None):
        self.prior = PriorAggregation(n_classes, pool, tags, class_names, featurizer,
                                      mode="fixed", temperature=temperature)
        self.n_classes = n_classes
        self.pool = pool
        self.fz = featurizer
        self.learned = learned
        self.seed = seed
        self.name = name or f"hv_prior_shrink_blend_{learned}"

    def _learned_est(self):
        if self.learned == "logreg":
            return make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=2000, C=0.5))
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=200, random_state=self.seed, n_jobs=4)

    def run(self, train_texts, y_train, test_texts) -> np.ndarray:
        prior_te = self.prior._softmax(self.prior._class_scores(test_texts))
        _, counts = np.unique(y_train, return_counts=True)
        if counts.min() < 2:  # cannot CV -> pure prior (lam=1)
            return prior_te
        from sklearn.model_selection import cross_val_predict

        Xtr = self.fz.features(train_texts, self.pool, "entail_contradict")
        Xte = self.fz.features(test_texts, self.pool, "entail_contradict")
        prior_tr = self.prior._softmax(self.prior._class_scores(train_texts))  # label-free: no leak
        folds = _safe_folds(y_train)
        est = self._learned_est()
        oof = cross_val_predict(est, Xtr, y_train, cv=StratifiedKFold(folds),
                                method="predict_proba")
        oof = _cols_to_full(oof, np.unique(y_train), self.n_classes)
        best_lam, best = 1.0, -1.0
        for lam in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            blend = lam * prior_tr + (1 - lam) * oof
            acc = float((blend.argmax(axis=1) == y_train).mean())
            if acc > best:
                best, best_lam = acc, lam
        est.fit(Xtr, y_train)
        learned_te = _proba_to_full(est, Xte, self.n_classes)
        return best_lam * prior_te + (1 - best_lam) * learned_te


class PriorScheduleBlend:
    """Design A v2 — deterministic shrinkage *schedule* (no fragile CV-lambda).

        anchor  = prior_fixed        if k < 2      (label-free; the N=1 default)
                = prior_reweight      if k >= 2     (best low-variance prior head)
        lam(k)  = tau / (tau + k),    k = min examples/class     (monotone 1 -> 0 as k grows)
        p       = lam * anchor_proba + (1 - lam) * learned_proba

    `tau` is a SINGLE global constant (the crossover scale ~ examples/class where a flexible head
    overtakes the prior), fixed a priori and held identical across datasets — if it needed
    per-dataset tuning the schedule would be overfit. Removes the v1 CV-lambda dip at 2-3/class:
    lambda is now a smooth function of N instead of an accuracy argmax over ~12-18 noisy rows.
    Being a convex blend it matches the better endpoint at the extremes and interpolates between.
    """

    def __init__(self, n_classes: int, pool: list[str], tags: list[str], class_names: list[str],
                 featurizer: NLIFeaturizer, tau: float = 4.0, learned: str = "rf",
                 temperature: float = 0.1, seed: int = 0, name: str | None = None):
        self.prior_fixed = PriorAggregation(n_classes, pool, tags, class_names, featurizer,
                                            mode="fixed", temperature=temperature)
        self.prior_rw = PriorAggregation(n_classes, pool, tags, class_names, featurizer,
                                         mode="reweight", temperature=temperature)
        self.n_classes = n_classes
        self.pool = pool
        self.fz = featurizer
        self.tau = tau
        self.learned = learned
        self.seed = seed
        self.name = name or f"hv_prior_sched_blend_{learned}"

    def _learned_proba(self, train_texts, y_train, test_texts) -> np.ndarray:
        Xtr = self.fz.features(train_texts, self.pool, "entail_contradict")
        Xte = self.fz.features(test_texts, self.pool, "entail_contradict")
        if self.learned == "logreg":
            est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
        else:
            from sklearn.ensemble import RandomForestClassifier

            est = RandomForestClassifier(n_estimators=200, random_state=self.seed, n_jobs=4)
        est.fit(Xtr, y_train)
        final = est[-1] if hasattr(est, "steps") else est
        Xte_t = est[:-1].transform(Xte) if hasattr(est, "steps") else Xte
        return _proba_to_full(final, Xte_t, self.n_classes)

    def run(self, train_texts, y_train, test_texts) -> np.ndarray:
        _, counts = np.unique(y_train, return_counts=True)
        k = int(counts.min())
        if k < 2:  # lambda = 1 -> pure label-free prior (cannot reweight or fit a head)
            return self.prior_fixed._softmax(self.prior_fixed._class_scores(test_texts))
        lam = self.tau / (self.tau + k)
        anchor = self.prior_rw.run(train_texts, y_train, test_texts)
        learned = self._learned_proba(train_texts, y_train, test_texts)
        return lam * anchor + (1 - lam) * learned


class PriorAnchoredLogReg:
    """Design B — one multinomial logreg over the full hypothesis features with the label-free
    prior wired in as (effectively unpenalized) offset features.

    The design matrix is [alpha * prior_logits | scaled hyp-features]. The prior columns are
    scaled up by `alpha`, so the tiny weights (~1/alpha) that use them are barely touched by L2 —
    they act as an offset. Strong L2 at low N crushes the hyp-feature weights to ~0, leaving the
    prior; as C grows (CV-selected on train) the data correction activates. Empirical-Bayes
    shrinkage toward the prior, in a single model (contrast the blend, which mixes two fits).
    """

    def __init__(self, n_classes: int, pool: list[str], tags: list[str], class_names: list[str],
                 featurizer: NLIFeaturizer, temperature: float = 0.1, alpha: float = 8.0,
                 seed: int = 0, name: str | None = None):
        self.prior = PriorAggregation(n_classes, pool, tags, class_names, featurizer,
                                      mode="fixed", temperature=temperature)
        self.n_classes = n_classes
        self.pool = pool
        self.fz = featurizer
        self.alpha = alpha
        self.name = name or "hv_prior_anchored_logreg"
        self._scaler: StandardScaler | None = None

    def _design(self, texts, fit: bool = False) -> np.ndarray:
        plog = self.prior._class_scores(texts) / self.prior.temperature  # (n, n_classes)
        X = self.fz.features(texts, self.pool, "entail_contradict")
        if fit:
            self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        return np.hstack([self.alpha * plog, Xs])

    def run(self, train_texts, y_train, test_texts) -> np.ndarray:
        Dtr = self._design(train_texts, fit=True)
        Dte = self._design(test_texts)
        _, counts = np.unique(y_train, return_counts=True)
        if counts.min() < 2:  # cannot CV -> strong reg (leans on the prior offset)
            best_c = 0.02
        else:
            folds = _safe_folds(y_train)
            best_c, best = 0.02, -1.0
            for c in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0):
                clf = LogisticRegression(max_iter=3000, C=c)
                try:
                    s = cross_val_score(clf, Dtr, y_train, cv=StratifiedKFold(folds)).mean()
                except Exception:
                    s = -1.0
                if s > best:
                    best, best_c = s, c
        clf = LogisticRegression(max_iter=3000, C=best_c).fit(Dtr, y_train)
        return _proba_to_full(clf, Dte, self.n_classes)


# --------------------------------------------------------------- llm-trees induction (label-free)
class LLMForestInduction:
    """Route text through an LLM-written decision-tree *forest* via NLI — the induction half of
    llm-trees (arXiv 2409.18594), label-free.

    Each internal node holds an NLI hypothesis; `P(entail)` is the probability the text takes the
    `yes` branch. routing='soft' propagates that probability (reach(yes)=p, reach(no)=1-p) and each
    leaf contributes its class one-hot weighted by the mass reaching it — a probabilistic tree, the
    natural fit for a soft encoder. routing='hard' thresholds p at 0.5 (the paper's hard traversal).
    Per-tree leaf distributions are renormalized (mass that reaches a *labeled* leaf) then averaged
    over the trees that produced any mass for that row. Uses NO labels: `run` ignores y_train.
    """

    def __init__(self, n_classes: int, forest, class_names: list[str], featurizer: NLIFeaturizer,
                 routing: str = "soft", name: str | None = None):
        from hypothesis_vectorizer.dedup import norm_statement

        from .forest import flatten_conditions

        assert routing in ("soft", "hard")
        self.n_classes = n_classes
        self.forest = forest
        self.fz = featurizer
        self.routing = routing
        self.name = name or f"llm_forest_induction_{routing}"
        self._classidx = {c: i for i, c in enumerate(class_names)}
        self._conditions = flatten_conditions(forest)  # unique, stable order
        self._colmap = {norm_statement(c): j for j, c in enumerate(self._conditions)}

    def _route(self, node, E, reach, out) -> None:
        from hypothesis_vectorizer.dedup import norm_statement

        if node is None:
            return
        internal = node.condition is not None and (node.yes is not None or node.no is not None)
        if not internal:  # leaf (or dead node) — deposit mass on its class if it names a known one
            ci = self._classidx.get((node.leaf_class or "").strip())
            if ci is not None:
                out[:, ci] += reach
            return
        col = self._colmap.get(norm_statement(node.condition))
        if col is None:  # condition somehow unscored — cannot branch; treat as labeled leaf if any
            ci = self._classidx.get((node.leaf_class or "").strip())
            if ci is not None:
                out[:, ci] += reach
            return
        p = E[:, col]
        if self.routing == "hard":
            p = (p >= 0.5).astype(float)
        self._route(node.yes, E, reach * p, out)
        self._route(node.no, E, reach * (1.0 - p), out)

    def run(self, train_texts, y_train, test_texts) -> np.ndarray:
        n = len(test_texts)
        E = (self.fz.features(test_texts, self._conditions, "entail")
             if self._conditions else np.zeros((n, 0)))
        acc = np.zeros((n, self.n_classes), dtype=float)
        contributing = np.zeros(n, dtype=float)  # trees that routed any mass, per row
        for tree in self.forest:
            out = np.zeros((n, self.n_classes), dtype=float)
            self._route(tree, E, np.ones(n), out)
            mass = out.sum(axis=1)  # mass that reached a labeled leaf
            hit = mass > 0
            out[hit] /= mass[hit, None]  # per-tree class distribution
            acc[hit] += out[hit]
            contributing[hit] += 1.0
        ok = contributing > 0
        acc[ok] /= contributing[ok, None]
        acc[~ok] = 1.0 / self.n_classes  # no tree could route this row -> uniform
        return acc
