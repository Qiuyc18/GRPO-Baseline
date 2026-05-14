#!/usr/bin/env bash
set -euo pipefail

# ============ 加载项目级 .env（与 train 脚本一致）============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -f "${PROJECT_ROOT}/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/.env"
  set +a
fi

# ============ 基础环境 ============
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES:-1}"

export HOST_CHECKPOINT_PATH="${HOST_CHECKPOINT_PATH:-/etc/moreh/checkpoint}"
export MODEL_PATH="${MODEL_PATH:-${HOST_CHECKPOINT_PATH}/Qwen/Qwen3-4B-Base}"

export MATH_DATA_DIR="${MATH_DATA_DIR:-${HOST_CHECKPOINT_PATH}/data/math}"
export DATA_PATH="${DATA_PATH:-${MATH_DATA_DIR}/train.parquet}"

# ============ I/O ============
export OUTPUT_JSON="${OUTPUT_JSON:-${PROJECT_ROOT}/logs/pure_rollout_4k_stats.json}"
export OUTPUT_JSONL="${OUTPUT_JSONL:-${PROJECT_ROOT}/logs/pure_rollout_4k_samples.jsonl}"

# ============ Data / sampling ============
export NUM_PROMPTS="${NUM_PROMPTS:-512}"
export SHUFFLE="${SHUFFLE:-1}"
export SEED="${SEED:-42}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export N_ROLLOUTS="${N_ROLLOUTS:-5}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-1.0}"
export TOP_K="${TOP_K:--1}"
export PRESENCE_PENALTY="${PRESENCE_PENALTY:-0.0}"
export FREQUENCY_PENALTY="${FREQUENCY_PENALTY:-0.0}"
export REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"

# ============ vLLM ============
export TP_SIZE="${TP_SIZE:-4}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
export DTYPE="${DTYPE:-auto}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
export DISABLE_LOG_STATS="${DISABLE_LOG_STATS:-0}"

# ============ W&B ============
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-deng-lab}"
export WANDB_PROJECT="${WANDB_PROJECT:-GRPO-Baseline}"
export WANDB_NAME="${WANDB_NAME:-pure_rollout_eval_4k}"
export WANDB_GROUP="${WANDB_GROUP:-}"
export WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-rollout_eval}"
export WANDB_TAGS="${WANDB_TAGS:-4k,qwen3-4b,math}"
export WANDB_MODE="${WANDB_MODE:-online}"

mkdir -p "$(dirname "${OUTPUT_JSON}")" "$(dirname "${OUTPUT_JSONL}")" 2>/dev/null || true

exec python3 "${PROJECT_ROOT}/tests/test_rollout_time.py" "$@"