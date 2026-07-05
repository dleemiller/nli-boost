"""The frozen NLI cross-encoder, cache-through and chunk-committed.

One representation everywhere (METHOD.md): each hypothesis contributes TWO
features per text — P(entailment) and P(contradiction) — from a single forward
pass. Contradiction is independent signal wherever hypotheses have semantic
opposites (measured +1.3 on sentiment); it is never an extra inference cost.
"""

import hashlib
import re

import numpy as np

from .cache import ScoreCache
from .config import EncoderConfig
from .costs import CostTracker

_WS = re.compile(r"\s+")
_GPU_CHUNK = 8192  # commit cadence: an interrupted run loses at most one chunk


def normalize(text: str, max_chars: int) -> str:
    return _WS.sub(" ", text).strip()[:max_chars]


def digest(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:32]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class EntailmentScorer:
    """Scores (text, hypothesis) pairs; GPU only for cache misses.

    Label order of the finecat family: entailment=0, neutral=1, contradiction=2.
    """

    def __init__(self, cfg: EncoderConfig, cache: ScoreCache, costs: CostTracker):
        self.cfg = cfg
        self.cache = cache
        self.costs = costs
        self._model = None  # lazy: cache-only paths never touch the GPU

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.cfg.model, device=self.cfg.device)
        return self._model

    def probs(self, texts: list[str], pool: list[str]) -> np.ndarray:
        """(n_texts, n_hypotheses, 3) probabilities."""
        return _softmax(self._logits(texts, pool))

    def features(self, texts: list[str], pool: list[str]) -> np.ndarray:
        """The method's feature representation: (n, 2m) = [P(entail) | P(contradict)].

        Column j is hypothesis j's entailment; column m+j its contradiction.
        """
        p = self.probs(texts, pool)
        return np.concatenate([p[:, :, 0], p[:, :, 2]], axis=1)

    def _logits(self, texts: list[str], pool: list[str]) -> np.ndarray:
        norm_t = [normalize(t, self.cfg.max_text_chars) for t in texts]
        t_hashes = [digest(t) for t in norm_t]
        norm_h = [_WS.sub(" ", h).strip() for h in pool]
        h_hashes = [digest(h) for h in norm_h]

        out = np.empty((len(texts), len(pool), 3), dtype=np.float32)
        pending: list[tuple[int, int]] = []
        self.costs.encoder_pairs_requested += len(texts) * len(pool)
        for j, hh in enumerate(h_hashes):
            hit = self.cache.get_logits(self.cfg.model, hh, t_hashes)
            self.costs.encoder_cache_hits += len(hit)
            for i, th in enumerate(t_hashes):
                if th in hit:
                    out[i, j] = hit[th]
                else:
                    pending.append((i, j))

        for start in range(0, len(pending), _GPU_CHUNK):
            batch = pending[start : start + _GPU_CHUNK]
            pairs = [(norm_t[i], norm_h[j]) for i, j in batch]
            logits = np.asarray(
                self.model.predict(pairs, batch_size=self.cfg.batch_size, show_progress_bar=False),
                dtype=np.float32,
            )
            self.costs.encoder_gpu_pairs += len(pairs)
            by_hyp: dict[int, list[tuple[str, str, np.ndarray]]] = {}
            for (i, j), z in zip(batch, logits):
                out[i, j] = z
                by_hyp.setdefault(j, []).append((t_hashes[i], norm_t[i], z))
            for j, rows in by_hyp.items():
                self.cache.put_logits(self.cfg.model, h_hashes[j], norm_h[j], rows)
            if self.cfg.verbose and len(pending) > 2 * _GPU_CHUNK:
                print(
                    f"    encoder: {min(start + _GPU_CHUNK, len(pending))}/{len(pending)} pairs scored",
                    flush=True,
                )
        return out
