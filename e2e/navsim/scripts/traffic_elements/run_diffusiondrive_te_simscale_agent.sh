#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAVSIM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${NAVSIM_ROOT}"
export PYTHONPATH="${NAVSIM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_ROOT}"
export NAVSIM_EXP_ROOT="${E2E_NAVSIM_EXP_ROOT:-${NAVSIM_ROOT}/exp}"

STAGE="${1:-}"
if [[ "${STAGE}" != "cache" && "${STAGE}" != "pdm" && "${STAGE}" != "recovery" && "${STAGE}" != "all" && "${STAGE}" != "eval" ]]; then
  echo "Usage: $0 {cache|pdm|recovery|all|eval} [Hydra overrides...]"
  exit 2
fi
shift

if [[ "${STAGE}" == "cache" ]]; then
  for split in \
    navtrain \
    navtrain_reaction_pdm_v1.0-1 \
    navtrain_reaction_pdm_v1.0-2 \
    navtrain_reaction_recovery_v1.0-1 \
    navtrain_reaction_recovery_v1.0-2; do
    bash "${SCRIPT_DIR}/dataset_cache.sh" diffusiondrive_te_simscale_agent "${split}" "$@"
  done
  exit 0
fi

if [[ "${STAGE}" == "eval" ]]; then
  CHECKPOINT_PATH="${CHECKPOINT_PATH:-${NAVSIM_EXP_ROOT}/diffusiondrive_te_simscale_agent/diffusiondrive_te_simscale.ckpt}"
  METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/metric_cache_navhard}"
  WORKER_THREADS="${WORKER_THREADS:-32}"

  python navsim/planning/script/run_pdm_score.py \
    agent=diffusiondrive_te_simscale_agent \
    experiment_name=diffusiondrive_te_simscale_agent \
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
NUM_GPUS="${NUM_GPUS:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

if [[ "${NUM_GPUS}" -eq 1 ]]; then
  TRAIN_COMMAND=(python)
else
  TRAIN_COMMAND=(torchrun --nproc_per_node="${NUM_GPUS}")
fi

run_stage() {
  local stage="$1"
  shift
  local experiment_name="train_diffusiondrive_te_simscale_${stage}_v1.0-2"
  local synthetic_cache_paths="[${CACHE_ROOT}/navtrain_reaction_${stage}_v1.0-1_cache,${CACHE_ROOT}/navtrain_reaction_${stage}_v1.0-2_cache]"
  local synthetic_manifest_paths="[${MANIFEST_ROOT}/default_log_split_synthetic_reaction_${stage}_v1.0-1.yaml,${MANIFEST_ROOT}/default_log_split_synthetic_reaction_${stage}_v1.0-2.yaml]"
  local input_checkpoint_path

  if [[ "${stage}" == "pdm" ]]; then
    input_checkpoint_path="${BASE_CHECKPOINT_PATH:-${NAVSIM_EXP_ROOT}/ltf_simscale_diffusiondrive/diffusiondrive_sim_navhard.ckpt}"
  else
    input_checkpoint_path="${PDM_CHECKPOINT_PATH:-${NAVSIM_EXP_ROOT}/train_diffusiondrive_te_simscale_pdm_v1.0-2/epoch=19.ckpt}"
  fi

  if [[ ! -f "${input_checkpoint_path}" ]]; then
    echo "Input checkpoint does not exist: ${input_checkpoint_path}" >&2
    exit 1
  fi

  "${TRAIN_COMMAND[@]}" navsim/planning/script/run_training_diffusiondrive_simscale.py \
    agent=diffusiondrive_te_simscale_agent \
    experiment_name="${experiment_name}" \
    train_test_split=navtrain \
    cache_path="${BASE_CACHE_PATH}" \
    simscale.cache_paths="${synthetic_cache_paths}" \
    simscale.log_manifest_paths="${synthetic_manifest_paths}" \
    dataloader.params.batch_size=32 \
    trainer.params.max_epochs=20 \
    agent.lr=5e-5 \
    "agent.checkpoint_path='${input_checkpoint_path}'" \
    "$@"
}

if [[ "${STAGE}" == "all" ]]; then
  run_stage pdm "$@"
  run_stage recovery "$@"
else
  run_stage "${STAGE}" "$@"
fi
