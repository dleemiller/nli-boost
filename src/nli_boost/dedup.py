"""Candidate deduplication: exact-text and STS-paraphrase, BEFORE encoder scoring.

Paraphrases waste both a candidate slot and a scoring pass; the STS filter
kills them for the cost of a few cross-encoder pairs. Behavioral duplicates
that STS cannot see (different wording, identical encoder signal) surface
later as "redundant" in evolution's failure diagnosis.
"""

import re

import numpy as np

from .config import StsConfig

_WS = re.compile(r"\s+")


def norm_statement(s: str) -> str:
    return _WS.sub(" ", s).strip().rstrip(".").casefold()


class Deduper:
    def __init__(self, cfg: StsConfig):
        self.cfg = cfg
        self._model = None
        self._failed = False

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.cfg.model, device=self.cfg.device)
        return self._model

    def _paraphrase_of(self, statement: str, against: list[str]) -> str | None:
        if self._failed or not against:
            return None
        try:
            sims = np.asarray(
                self.model.predict([(statement, other) for other in against], show_progress_bar=False)
            )
        except Exception as e:  # dedup must never kill a fit
            print(f"    sts dedup unavailable ({type(e).__name__}); textual-only for this run", flush=True)
            self._failed = True
            return None
        j = int(np.argmax(sims))
        return against[j] if sims[j] >= self.cfg.threshold else None

    def filter(
        self, candidates: list[str], against: list[str], seen: set[str]
    ) -> tuple[list[str], list[str]]:
        """Returns (kept, rejected). Mutates `seen` with kept normalized forms.

        `against` are existing statements (pool + earlier keeps) that a
        candidate may not paraphrase.
        """
        kept, rejected = [], []
        against = list(against)
        for c in candidates:
            key = norm_statement(c)
            if not key or key in seen:
                rejected.append(c)
                continue
            dup = self._paraphrase_of(c, against)
            if dup is not None:
                rejected.append(f"{c} (paraphrase of: {dup})")
                continue
            seen.add(key)
            against.append(c)
            kept.append(c)
        return kept, rejected
