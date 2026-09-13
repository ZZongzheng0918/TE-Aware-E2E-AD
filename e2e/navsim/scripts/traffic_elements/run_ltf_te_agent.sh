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

CACHE_PATH="${CACHE_PATH:-exp/training_cache_ltf_te_agent}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-exp/ltf_te_agent/ltf_te.ckpt}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-exp/metric_cache_navhard}"
WORKER_THREADS="${WORKER_THREADS:-32}"

case "${ACTION}" in
  cache)
    python navsim/planning/script/run_dataset_caching.py \
      cache_path="${CACHE_PATH}" \
      agent=ltf_te_agent \
      experiment_name=ltf_te_agent_caching \
      train_test_split=navtrain \
      worker.threads_per_node="${WORKER_THREADS}" \
      "$@"
    ;;
  train)
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      python navsim/planning/script/run_training.py \
      agent=ltf_te_agent \
      experiment_name=ltf_te_agent_training \
      train_test_split=navtrain \
      split=trainval \
      use_cache_without_dataset=True \
      force_cache_computation=False \
      cache_path="${CACHE_PATH}" \
      "$@"
    ;;
  eval)
    python navsim/planning/script/run_pdm_score.py \
      agent=ltf_te_agent \
      experiment_name=ltf_te_agent \
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
