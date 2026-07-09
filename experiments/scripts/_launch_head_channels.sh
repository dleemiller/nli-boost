#!/usr/bin/env bash
# Resilient, low-priority launcher for the per-head channel sweep. The shared box is memory-tight
# and runs other GPU/CPU jobs (trading + Lee's encoder_size_sweep), so we nice ourselves, cap
# threads, and retry on transient OOM/import failures with backoff instead of dying on the first
# memory spike. Idempotent per-cell: results.jsonl is rewritten each attempt, cheap to re-run.
set -u
cd "$(dirname "$0")/../.." || exit 1
export HV_CACHE_DIR="${HV_CACHE_DIR:-/tmp/hv_head_channels_cache}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
LOG=experiments/results/logs/probe_head_channels.log
for attempt in 1 2 3 4 5 6; do
  echo "=== attempt $attempt $(date -u +%H:%M:%S) (mem-free $(free -m | awk '/Mem:/{print $7}')MB, load $(cut -d' ' -f1 /proc/loadavg)) ===" >> "$LOG"
  nice -n 19 uv run python experiments/scripts/probe_head_channels.py \
    --datasets trec sst2 goemotions ag_news --sizes xxs xs s m l \
    --device cuda --run-id probe_head_channels >> "$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then echo "=== SUCCESS attempt $attempt ===" >> "$LOG"; exit 0; fi
  echo "=== attempt $attempt FAILED rc=$rc, backing off ===" >> "$LOG"
  sleep $((attempt * 120))
done
echo "=== GAVE UP after retries ===" >> "$LOG"
exit 1
