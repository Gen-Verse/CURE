#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/rebuild_bootstrap_report.sh experiments/<exp_name>"
  exit 1
fi

EXP_DIR="$1"
CONDA_ACTIVATE_SCRIPT="${CONDA_ACTIVATE_SCRIPT:-/mnt/shared-storage-user/zhupengyu1/anaconda3/bin/activate}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-CURE}"

if [[ ! -f "${CONDA_ACTIVATE_SCRIPT}" ]]; then
  echo "Conda activate script not found: ${CONDA_ACTIVATE_SCRIPT}"
  echo "Please set CONDA_ACTIVATE_SCRIPT before running this script."
  exit 1
fi

source "${CONDA_ACTIVATE_SCRIPT}" "${CONDA_ENV_NAME}"
python analysis/generate_report.py --exp_dir "$EXP_DIR" --output_dir "$EXP_DIR/report"
