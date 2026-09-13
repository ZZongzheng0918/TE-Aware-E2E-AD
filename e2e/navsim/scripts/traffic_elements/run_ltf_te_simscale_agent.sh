#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAVSIM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${NAVSIM_ROOT}"
export PYTHONPATH="${NAVSIM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_ROOT}"
export NAVSIM_EXP_ROOT="${E2E_NAVSIM_EXP_ROOT:-${NAVSIM_ROOT}/exp}"

ACTION="${1:-}"
if [[ "${ACTION}" != "cache" && "${ACTION}" != "pdm" && "${ACTION}" != "recovery" && "${ACTION}" != "eval" ]]; then
  echo "Usage: $0 {cache|pdm|recovery|eval} [Hydra overrides...]"
  exit 2
fi
shift

if [[ "${ACTION}" == "cache" ]]; then
  for split in \
    navtrain \
    navtrain_reaction_pdm_v1.0-1 \
    navtrain_reaction_pdm_v1.0-2 \
    navtrain_reaction_recovery_v1.0-1 \
    navtrain_reaction_recovery_v1.0-2; do
    bash "${SCRIPT_DIR}/dataset_cache.sh" ltf_te_simscale_agent "${split}" "$@"
  done
  exit 0
fi

if [[ "${ACTION}" == "eval" ]]; then
  CHECKPOINT_PATH="${CHECKPOINT_PATH:-${NAVSIM_EXP_ROOT}/ltf_te_simscale_agent/ltf_te_simscale.ckpt}"
  METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/metric_cache_navhard}"
  WORKER_THREADS="${WORKER_THREADS:-32}"

  python navsim/planning/script/run_pdm_score.py \
    agent=ltf_te_simscale_agent \
    experiment_name=ltf_te_simscale_agent \
    experiment_uid=navhard \
    train_test_split=navhard_two_stage \
    agent.checkpoint_path="${CHECKPOINT_PATH}" \
    metric_cache_path="${METRIC_CACHE_PATH}" \
    worker.threads_per_node="${WORKER_THREADS}" \
    "$@"
  exit 0
fi

CACHE_ROOT="${SIMSCALE_CACHE_ROOT:-${NAVSIM_EXP_ROOT}/cache}"
MANIFEST_ROOT="${SIMSCALE_MANIFEST_ROOT:-navsim/planning/script/config/training/simscale}"
BASE_CACHE_PATH="${BASE_CACHE_PATH:-${CACHE_ROOT}/navtrain_cache}"
SYNTHETIC_CACHE_PATHS="[${CACHE_ROOT}/navtrain_reaction_${ACTION}_v1.0-1_cache,${CACHE_ROOT}/navtrain_reaction_${ACTION}_v1.0-2_cache]"
SYNTHETIC_MANIFEST_PATHS="[${MANIFEST_ROOT}/default_log_split_synthetic_reaction_${ACTION}_v1.0-1.yaml,${MANIFEST_ROOT}/default_log_split_synthetic_reaction_${ACTION}_v1.0-2.yaml]"

EXPERIMENT_NAME="train_ltf_te_simscale_${ACTION}_v1.0-2"
if [[ "${ACTION}" == "pdm" ]]; then
  INPUT_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-${NAVSIM_EXP_ROOT}/ltf_simscale/ltf_sim_navhard.ckpt}"
else
  INPUT_CHECKPOINT_PATH="${PDM_CHECKPOINT_PATH:-${NAVSIM_EXP_ROOT}/train_ltf_te_simscale_pdm_v1.0-2/epoch=19.ckpt}"
fi

if [[ ! -f "${INPUT_CHECKPOINT_PATH}" ]]; then
  echo "Input checkpoint does not exist: ${INPUT_CHECKPOINT_PATH}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python navsim/planning/script/run_training_simscale.py \
  agent=ltf_te_simscale_agent \
  experiment_name="${EXPERIMENT_NAME}" \
  train_test_split=navtrain \
  cache_path="${BASE_CACHE_PATH}" \
  simscale.cache_paths="${SYNTHETIC_CACHE_PATHS}" \
  simscale.log_manifest_paths="${SYNTHETIC_MANIFEST_PATHS}" \
  dataloader.params.batch_size=32 \
  trainer.params.max_epochs=20 \
  agent.lr=5e-5 \
  "agent.checkpoint_path='${INPUT_CHECKPOINT_PATH}'" \
  "$@"
