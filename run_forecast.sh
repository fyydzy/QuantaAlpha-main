#!/bin/bash
# 燃气旬度预测启动脚本
#
# Usage:
#   ./run_forecast.sh
#   ./run_forecast.sh --model lasso --province 河北
#   CONFIG=configs/forecast.yaml ./run_forecast.sh --model auto

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [ -f "${SCRIPT_DIR}/.env" ]; then
    set -a
    source "${SCRIPT_DIR}/.env"
    set +a
else
    echo "Warning: .env not found; using defaults"
fi

export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_QUANTAALPHA="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_QUANTAALPHA:-0.1.0}"

eval "$(conda shell.bash hook)" 2>/dev/null
conda activate "${CONDA_ENV_NAME:-quantaalpha}" 2>/dev/null

CONFIG_PATH="${CONFIG:-configs/forecast.yaml}"
EXTRA_ARGS=("$@")

if [ -x "$(command -v quantaalpha)" ]; then
    quantaalpha forecast --config "${CONFIG_PATH}" "${EXTRA_ARGS[@]}"
else
    python -m quantaalpha.cli forecast --config "${CONFIG_PATH}" "${EXTRA_ARGS[@]}"
fi
