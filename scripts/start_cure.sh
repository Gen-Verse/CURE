#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONDA_ACTIVATE_SCRIPT="${CONDA_ACTIVATE_SCRIPT:-/mnt/shared-storage-user/zhupengyu1/anaconda3/bin/activate}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-CURE}"
CUDA_LAUNCH_BLOCKING_VALUE="${CUDA_LAUNCH_BLOCKING_VALUE:-1}"

cd "${PROJECT_DIR}"

if [[ ! -f "${CONDA_ACTIVATE_SCRIPT}" ]]; then
  echo "Conda activate script not found: ${CONDA_ACTIVATE_SCRIPT}"
  echo "Please set CONDA_ACTIVATE_SCRIPT before running this script."
  exit 1
fi

source "${CONDA_ACTIVATE_SCRIPT}" "${CONDA_ENV_NAME}"

if [[ -z "${VLLM_CUDART_SO_PATH:-}" ]]; then
  for candidate in \
    "${CONDA_PREFIX:-}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12" \
    "${CONDA_PREFIX:-}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12" \
    "/mnt/shared-storage-user/zhupengyu1/anaconda3/envs/CURE/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
  do
    if [[ -n "${candidate}" && -f "${candidate}" ]]; then
      export VLLM_CUDART_SO_PATH="${candidate}"
      break
    fi
  done
fi

mkdir -p logs
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING_VALUE}"

LOG_FILE="${LOG_FILE:-logs/train_$(date +%Y%m%d_%H%M%S).log}"
export CURE_LOG_FILE="${LOG_FILE}"

python run.py 2>&1 | tee "${LOG_FILE}"
