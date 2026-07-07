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
