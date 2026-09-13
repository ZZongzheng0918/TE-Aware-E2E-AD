#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAVSIM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${NAVSIM_ROOT}"

ACTION="${1:-}"
if [[ -z "${ACTION}" ]]; then
  echo "Usage: $0 {cache|train|eval} [Hydra overrides...]"
  exit 2
fi
shift

CACHE_PATH="${CACHE_PATH:-exp/cache/navtrain_cache}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-exp/diffusiondrive_te_agent/diffusiondrive_te.ckpt}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-exp/metric_cache_navhard}"
WORKER_THREADS="${WORKER_THREADS:-32}"

case "${ACTION}" in
  cache)
    CACHE_PATH="${CACHE_PATH}" WORKER_THREADS="${WORKER_THREADS}" \
      bash "${SCRIPT_DIR}/dataset_cache.sh" diffusiondrive_te_agent navtrain "$@"
    ;;
  train)
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      python navsim/planning/script/run_training_diffusiondrive.py \
      agent=diffusiondrive_te_agent \
      experiment_name=diffusiondrive_te_agent_training \
      train_test_split=navtrain \
      split=trainval \
      use_cache_without_dataset=True \
      force_cache_computation=False \
      cache_path="${CACHE_PATH}" \
      "$@"
    ;;
  eval)
    python navsim/planning/script/run_pdm_score.py \
      agent=diffusiondrive_te_agent \
      experiment_name=diffusiondrive_te_agent \
      experiment_uid=navhard \
      train_test_split=navhard_two_stage \
      agent.checkpoint_path="${CHECKPOINT_PATH}" \
      metric_cache_path="${METRIC_CACHE_PATH}" \
      worker.threads_per_node="${WORKER_THREADS}" \
      "$@"
    ;;
  *)
    echo "Unknown action: ${ACTION}"
    echo "Usage: $0 {cache|train|eval} [Hydra overrides...]"
    exit 2
    ;;
esac
