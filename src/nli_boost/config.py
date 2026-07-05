"""Run configuration. Only the knobs of the converged method (METHOD.md) exist."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class DataConfig(BaseModel):
    name: Literal["ag_news", "sst2", "trec", "20newsgroups"]
    train_size: int = 2000
    val_size: int = 500
    test_size: int = 2000


class EncoderConfig(BaseModel):
    """The frozen NLI cross-encoder — the method's capacity knob (-m -> -l = +5 pts)."""

    model: str = "dleemiller/finecat-nli-m"
    batch_size: int = 128
    max_text_chars: int = 1200  # normalize+truncate BEFORE hashing so cache keys are stable
    device: str = "cuda"


class DedupConfig(BaseModel):
    """Feature-space (covariance) dedup: drop a candidate whose entailment score vector is
    ~collinear with a kept one. Behavioral, not textual; no separate STS model."""

    corr_threshold: float = 0.95  # |Pearson| above this = redundant feature (multicollinearity)
    ref_size: int = 400  # stratified train subsample the score vectors are correlated on


class LMConfig(BaseModel):
    """The hypothesis proposer. Cheap by design; a full fit costs ~$0.01."""

    model: str = "openrouter/deepseek/deepseek-v4-flash"
    max_tokens: int = 12000  # reasoning + hypotheses can exceed 4k and truncate mid-JSON
    temperature: float = 1.0
    # provider passthrough, e.g. {"provider": {"order": ["deepseek"], "allow_fallbacks": false}}
    # NOTE: part of the LM cache key — changing it invalidates cached proposals.
    extra_body: dict | None = None
    # optional GEPA-tuned GeneratePool instruction (a saved dspy program json); overrides the
    # hand-written GeneratePool docstring when set.
    instruction_path: str | None = None


class PoolConfig(BaseModel):
    size: int = 64  # ~30 useful directions exist per task (measured); 64 gives slack
    rounds: int = 6  # hard cap; patience exits ~round 2-3 in practice
    patience: int = 2  # stop when held-out CV accuracy stops improving
    min_keep_frac: float = 0.5  # never prune below this fraction in one round
    rank_sample: int = 800  # stability ranking needs no full-matrix precision
    # reuse the pool from a previous run instead of generating: this is how a
    # pool is finalized with a bigger encoder (hypotheses transfer; only re-score)
    from_run: str | None = None


class LexicalConfig(BaseModel):
    """Optional static lexical channel concatenated with hypothesis features."""

    kind: Literal["none", "tfidf_svd", "wordllama"] = "none"
    dims: int = 128


class RunConfig(BaseModel):
    run_name: str
    seed: int = 7
    data: DataConfig
    encoder: EncoderConfig = EncoderConfig()
    dedup: DedupConfig = DedupConfig()
    lm: LMConfig = LMConfig()
    pool: PoolConfig = PoolConfig()
    lexical: LexicalConfig = LexicalConfig()
    cache_dir: Path = Path("cache")
    runs_dir: Path = Path("runs")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)
