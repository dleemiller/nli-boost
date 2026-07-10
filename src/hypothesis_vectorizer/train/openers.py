"""Opener frames to diversify hypothesis phrasing (NOTES 2026-07-10).

Our generated corpus starts "The text …" 99% of the time and "The text asks …" 32% — one
grammatical groove that likely caps the diversity of behavioral directions. Injecting varied opening
frames (mined from dleemiller/CrossingGuard-NLI, a prompt-classification NLI set, subject remapped
to "the text") into the generation prompt nudges the LLM into different SEMANTIC stances. Frames are
stochastically sampled per call (seeded rng) so each round explores a different subset.
"""

import json
from pathlib import Path

import numpy as np

_PATH = Path("configs/openers.json")  # mined, checked-in; regenerate with mine_openers()


def load_openers(path: str | Path | None = None) -> list[str]:
    p = Path(path) if path else _PATH
    return json.loads(p.read_text()) if p.exists() else []


def sample_openers(openers: list[str], rng: np.random.Generator, k: int = 6) -> list[str]:
    """A seeded random subset of `k` opener frames (empty if none / k<=0)."""
    if not openers or k <= 0:
        return []
    idx = rng.choice(len(openers), size=min(k, len(openers)), replace=False)
    return [openers[int(i)] for i in idx]


def mine_openers(n: int = 30000, top: int = 24, config: str = "v0") -> list[str]:
    """(Re)generate the opener table from CrossingGuard-NLI: the most common opening verb-frames,
    subject remapped to 'The text', minus modals/aux and our own 'asks' groove. Network; writes
    configs/openers.json."""
    import re
    from collections import Counter

    from datasets import load_dataset

    subj = re.compile(r"^the (?:prompt|request|message|text|user|input)\s+(\w+)", re.I)
    drop = {
        "is",
        "may",
        "might",
        "can",
        "could",
        "would",
        "will",
        "does",
        "has",
        "asks",
        "that",
        "which",
        "seems",
        "appears",
    }
    verbs: Counter = Counter()
    for i, r in enumerate(
        load_dataset("dleemiller/CrossingGuard-NLI", config, split="train", streaming=True)
    ):
        m = subj.match(str(r.get("hypothesis") or "").strip())
        if m:
            verbs[m.group(1).lower()] += 1
        if i >= n:
            break
    frames = [f"The text {v}" for v, _ in verbs.most_common(top * 2) if v not in drop][:top]
    _PATH.write_text(json.dumps(frames, indent=2))
    return frames
