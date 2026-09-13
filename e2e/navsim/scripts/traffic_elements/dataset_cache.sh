#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAVSIM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${NAVSIM_ROOT}"
export PYTHONPATH="${NAVSIM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_ROOT}"
export NAVSIM_EXP_ROOT="${E2E_NAVSIM_EXP_ROOT:-${NAVSIM_ROOT}/exp}"

AGENT_NAME="${1:-}"
TRAIN_TEST_SPLIT="${2:-}"
if [[ -z "${AGENT_NAME}" || -z "${TRAIN_TEST_SPLIT}" ]]; then
  echo "Usage: $0 <agent_name> <train_test_split> [Hydra overrides...]"
  exit 2
fi
shift 2

CACHE_PATH="${CACHE_PATH:-${NAVSIM_EXP_ROOT}/cache/${TRAIN_TEST_SPLIT}_cache}"
WORKER_THREADS="${WORKER_THREADS:-32}"
if [[ "${TRAIN_TEST_SPLIT}" == "navtrain" ]]; then
  TRAFFIC_ELEMENT_SPLIT="${SPLIT:-trainval}"
else
  TRAFFIC_ELEMENT_SPLIT="${SPLIT:-${TRAIN_TEST_SPLIT}}"
fi

SPLIT="${TRAFFIC_ELEMENT_SPLIT}" \
  python navsim/planning/script/run_dataset_caching.py \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  agent="${AGENT_NAME}" \
  experiment_name="cache_dataset_${TRAIN_TEST_SPLIT}" \
  cache_path="${CACHE_PATH}" \
  worker.threads_per_node="${WORKER_THREADS}" \
  "$@"
