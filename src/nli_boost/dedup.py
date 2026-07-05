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
    column becomes zeros -> correlation 0 with everything (vacuity is handled elsewhere)."""
    c = col - col.mean()
    s = c.std()
    return c / s if s > 1e-9 else np.zeros_like(c)


class Deduper:
    def __init__(self, scorer, ref_texts: list[str], corr_threshold: float = 0.95):
        self.scorer = scorer
        self.ref_texts = ref_texts
        self.thr = corr_threshold

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
        rejected, uniq, batch = [], [], set()
        for c in candidates:  # exact-text pass first (free); catches cross-batch and intra-batch
            key = norm_statement(c)
            if not key or key in seen or key in batch:
                rejected.append(c)
            else:
                batch.add(key)
                uniq.append(c)
        if not uniq:
            return [], rejected

        cols = [_zscore(v) for v in self._entail(list(against)).T]  # kept feature vectors
        cand = self._entail(uniq).T
        n = max(1, len(self.ref_texts))
        kept = []
        for c, raw in zip(uniq, cand):
            v = _zscore(raw)
            corr = max((abs(float(v @ k) / n) for k in cols), default=0.0)  # max |Pearson| vs kept
            if corr > self.thr:
                rejected.append(f"{c} (|corr| {corr:.2f} with a kept feature)")
            else:
                seen.add(norm_statement(c))  # only KEPT enter persistent seen
                cols.append(v)
                kept.append(c)
        return kept, rejected
