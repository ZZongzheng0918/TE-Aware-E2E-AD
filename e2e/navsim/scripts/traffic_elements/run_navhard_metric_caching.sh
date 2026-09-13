#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAVSIM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${NAVSIM_ROOT}"

METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-exp/metric_cache_navhard}"

python navsim/planning/script/run_metric_caching.py \
  train_test_split=navhard \
  metric_cache_path="${METRIC_CACHE_PATH}" \
  "$@"
