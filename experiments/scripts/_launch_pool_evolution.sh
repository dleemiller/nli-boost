#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../.." || exit 1
export HV_CACHE_DIR="${HV_CACHE_DIR:-/tmp/hv_head_channels_cache}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
LOG=experiments/results/logs/probe_pool_evolution.log
for attempt in 1 2 3 4 5; do
  echo "=== attempt $attempt $(date -u +%H:%M:%S) (free $(free -m|awk '/Mem:/{print $7}')MB load $(cut -d' ' -f1 /proc/loadavg)) ===" >> "$LOG"
  nice -n 19 uv run python experiments/scripts/probe_pool_evolution.py --dataset trec --device cuda >> "$LOG" 2>&1
  [ $? -eq 0 ] && { echo "=== SUCCESS ===" >> "$LOG"; exit 0; }
  echo "=== attempt $attempt FAILED, backing off ===" >> "$LOG"; sleep $((attempt*120))
done
echo "=== GAVE UP ===" >> "$LOG"; exit 1
