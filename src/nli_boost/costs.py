"""Run-level cost accounting: LM spend, encoder pairs, abnormal LM responses."""

import time
from dataclasses import dataclass, field


@dataclass
class CostTracker:
    lm_calls: int = 0
    lm_input_tokens: int = 0
    lm_output_tokens: int = 0
    lm_usd: float = 0.0
    lm_abnormal_finishes: int = 0  # finish_reason != "stop": truncation/degeneration, attributed
    encoder_pairs_requested: int = 0
    encoder_cache_hits: int = 0
    encoder_gpu_pairs: int = 0
    _start: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict:
        return {
            "lm_calls": self.lm_calls,
            "lm_input_tokens": self.lm_input_tokens,
            "lm_output_tokens": self.lm_output_tokens,
            "lm_usd": round(self.lm_usd, 4),
            "lm_abnormal_finishes": self.lm_abnormal_finishes,
            "encoder_pairs_requested": self.encoder_pairs_requested,
            "encoder_cache_hits": self.encoder_cache_hits,
            "encoder_gpu_pairs": self.encoder_gpu_pairs,
            "wall_seconds": round(time.monotonic() - self._start, 1),
        }
