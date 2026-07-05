"""Candidate deduplication in FEATURE space (covariance), not text space.

A hypothesis IS its entailment score vector across the data — that vector is the feature the head
sees. Two hypotheses with ~collinear score vectors are a redundant feature (multicollinearity),
however differently they are worded; two differently-behaving hypotheses are distinct even if the
wording is similar. So we dedup on the |correlation| (normalized covariance) of the score vectors,
which is both behaviorally exact and cheaper than a separate STS model (it reuses cached scores).

An exact-text pass runs first (free); the covariance pass then drops candidates whose entailment
vector correlates above `corr_threshold` with a kept feature.
"""

import re

import numpy as np

_WS = re.compile(r"\s+")


def norm_statement(s: str) -> str:
    return _WS.sub(" ", s).strip().rstrip(".").casefold()


def _zscore(col: np.ndarray) -> np.ndarray:
    """Standardize a column so a dot product / n is the Pearson correlation. A ~constant
    column becomes zeros -> correlation 0 with everything (vacuity is caught by the
    min_std floor in Deduper.filter, BEFORE the correlation pass)."""
    c = col - col.mean()
    s = c.std()
    return c / s if s > 1e-9 else np.zeros_like(c)


def _exact_text_pass(candidates: list[str], seen: set[str]) -> tuple[list[str], list[str]]:
    """First pass, free: drop empty/already-seen/intra-batch textual duplicates.
    Returns (unique candidates, rejected)."""
    rejected, uniq, batch = [], [], set()
    for c in candidates:
        key = norm_statement(c)
        if not key or key in seen or key in batch:
            rejected.append(c)
        else:
            batch.add(key)
            uniq.append(c)
    return uniq, rejected


class Deduper:
    def __init__(self, scorer, ref_texts: list[str], corr_threshold: float = 0.95, min_std: float = 0.02):
        self.scorer = scorer
        self.ref_texts = ref_texts
        self.thr = corr_threshold
        # variance floor: a candidate whose entailment is ~constant on the ref texts is a dead
        # feature (measured: always-false over-specific statements, entail mean ~0.002). It would
        # zscore to zeros and pass the correlation check unchallenged. 0.02 is conservative — a
        # detector for even a 1.6%-prevalence class measures std ~0.10 (see NOTES 2026-07-05).
        self.min_std = min_std

    def _entail(self, hyps: list[str]) -> np.ndarray:
        """(n_ref, len(hyps)) entailment score vectors — the features to correlate."""
        if not hyps:
            return np.empty((len(self.ref_texts), 0))
        x = self.scorer.features(self.ref_texts, hyps)  # (n_ref, 2*len) [entail | contradict]
        return x[:, : len(hyps)]

    def filter(
        self, candidates: list[str], against: list[str], seen: set[str]
    ) -> tuple[list[str], list[str]]:
        """Returns (kept, rejected). Mutates `seen` with kept normalized forms. A candidate is
        rejected if its entailment vector is ~collinear with a kept feature (in `against` or an
        earlier keep this batch)."""
        uniq, rejected = _exact_text_pass(candidates, seen)
        if not uniq:
            return [], rejected

        cols = [_zscore(v) for v in self._entail(list(against)).T]  # kept feature vectors
        cand = self._entail(uniq).T
        n = max(1, len(self.ref_texts))
        kept = []
        for c, raw in zip(uniq, cand):
            spread = float(raw.std())
            if spread < self.min_std:  # flat = vacuous feature, junk regardless of wording
                rejected.append(f"{c} (flat: entail std {spread:.3f} on ref texts)")
                continue
            v = _zscore(raw)
            corr = max((abs(float(v @ k) / n) for k in cols), default=0.0)  # max |Pearson| vs kept
            if corr > self.thr:
                rejected.append(f"{c} (|corr| {corr:.2f} with a kept feature)")
            else:
                seen.add(norm_statement(c))  # only KEPT enter persistent seen
                cols.append(v)
                kept.append(c)
        return kept, rejected


class STSDeduper:
    """Text-similarity dedup for LOW-DATA settings. With only a few examples per class the
    covariance estimate over score vectors is noise (it would keep/reject at random), so
    near-duplicates are caught in TEXT space instead: embed the hypotheses with a bi-encoder
    and reject a candidate whose cosine similarity to a kept one exceeds `threshold`.
    Data-free — needs no reference texts. Same .filter contract as Deduper."""

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.9,
        device: str | None = None,
    ):
        self.model_name = model
        self.thr = threshold
        self.device = device
        self._model = None  # lazy

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return np.asarray(self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False))

    def filter(
        self, candidates: list[str], against: list[str], seen: set[str]
    ) -> tuple[list[str], list[str]]:
        uniq, rejected = _exact_text_pass(candidates, seen)
        if not uniq:
            return [], rejected

        kept_vecs = [v for v in self._embed(list(against))] if against else []
        kept = []
        for c, v in zip(uniq, self._embed(uniq)):
            sim = max((float(v @ k) for k in kept_vecs), default=0.0)  # cosine (normalized)
            if sim > self.thr:
                rejected.append(f"{c} (sts {sim:.2f} with a kept hypothesis)")
            else:
                seen.add(norm_statement(c))
                kept_vecs.append(v)
                kept.append(c)
        return kept, rejected
