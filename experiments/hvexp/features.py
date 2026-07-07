"""NLI feature computation for the experiment harness, reusing the library's cached scorer.

For a learning curve we score the *entire* train pool + the fixed test set against a
hypothesis pool exactly once (per encoder), persist the raw logits in the shared sqlite
cache, and then every (k-shot, seed) subsample just slices rows out of the cached matrix —
so the whole seed/size sweep is free CPU after one GPU pass.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from hypothesis_vectorizer.cache import ScoreCache
from hypothesis_vectorizer.config import EncoderConfig
from hypothesis_vectorizer.costs import CostTracker
from hypothesis_vectorizer.encoder import EntailmentScorer

# The workspace is a 9P-mounted ZFS share where SQLite's per-lookup RPCs dominate a large-pool
# scoring pass. Point HV_CACHE_DIR at a LOCAL filesystem (e.g. /tmp) to keep cache I/O off 9P;
# defaults to the repo `cache/` for portability.
_CACHE_DIR = Path(os.environ.get("HV_CACHE_DIR", Path(__file__).resolve().parents[2] / "cache"))
DEFAULT_CACHE = _CACHE_DIR / "nli_scores.sqlite"


class NLIFeaturizer:
    """Thin wrapper: (texts, pool) -> probability tensor / feature matrix, cached on disk."""

    def __init__(self, encoder: str = "dleemiller/finecat-nli-l", *, device: str = "cuda",
                 batch_size: int = 128, max_text_chars: int = 1200,
                 cache_path: str | Path = DEFAULT_CACHE, verbose: bool = True):
        self.cfg = EncoderConfig(
            model=encoder, device=device, batch_size=batch_size,
            max_text_chars=max_text_chars, verbose=verbose,
        )
        self.cache = ScoreCache(str(cache_path))
        self.costs = CostTracker()
        self.scorer = EntailmentScorer(self.cfg, self.cache, self.costs)
        self._probs_memo: dict = {}  # in-process memo of full prob tensors, keyed by content

    def probs(self, texts: list[str], pool: list[str]) -> np.ndarray:
        """(n_texts, n_hyp, 3) probabilities [entail, neutral, contradict].

        Memoized on the (texts, pool) content so re-scoring the fixed test set across a whole
        learning-curve sweep is a single dict hit rather than a fresh pass of SQLite lookups.
        """
        key = (hash(tuple(texts)), hash(tuple(pool)))
        cached = self._probs_memo.get(key)
        if cached is not None:
            return cached
        out = self.scorer.probs(list(texts), list(pool))
        self._probs_memo[key] = out
        return out

    def features(self, texts: list[str], pool: list[str],
                 score_mode: str = "entail_contradict") -> np.ndarray:
        """Feature matrix per score_mode.

        entail_contradict -> (n, 2m) = [P(entail) | P(contradict)]
        entail            -> (n, m)  = P(entail)
        contrast          -> (n, m)  = P(entail) - P(contradict)
        """
        p = self.probs(texts, pool)
        e, c = p[:, :, 0], p[:, :, 2]
        if score_mode == "entail":
            return e
        if score_mode == "contrast":
            return e - c
        if score_mode == "entail_contradict":
            return np.concatenate([e, c], axis=1)
        raise ValueError(f"unknown score_mode {score_mode!r}")

    def cost_summary(self) -> dict:
        return self.costs.to_dict()
