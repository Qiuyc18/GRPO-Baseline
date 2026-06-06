#!/usr/bin/env bash
set -euo pipefail

# Aggressive 8-GPU long-tail timing profile for mi250-007.
# This wraps test_long_tail_gen.sh with higher rollout pressure and a shorter
# actor-update token budget. It does not enable replay or spec verification.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/test_long_tail_gen.sh"

if [ ! -f "${BASE_SCRIPT}" ]; then
  echo "Base script not found: ${BASE_SCRIPT}" >&2
  exit 1
fi

export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-test_long_tail_gen_12k_2}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-deepscaler_8gpu_long_tail_gen_aggressive}"

# More aggressive than the 4-GPU/8k profile:
# - longer cap exposes more tail completions;
# - rollout.n=6 increases fresh generation pressure;
# - batch=96 keeps per-DP prompt load moderate on 8 GPUs with TP=2.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-96}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-12288}"
export ROLLOUT_N="${ROLLOUT_N:-6}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-49152}"

# Keep actor/ref passes conservative for memory, and push the phase ratio toward
# rollout generation by truncating actor update tokens more aggressively.
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-24}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"

export NAT_TOKEN_SAMPLING="${NAT_TOKEN_SAMPLING:-True}"
export NAT_MODE="${NAT_MODE:-rpc_urs}"
export NAT_KEEP_RATIO="${NAT_KEEP_RATIO:-0.15}"
export NAT_MIN_TOKENS="${NAT_MIN_TOKENS:-1}"
export NAT_TRUNCATE_RATIO="${NAT_TRUNCATE_RATIO:-0.15}"
export NAT_TRUNCATE_MIN_TOKENS="${NAT_TRUNCATE_MIN_TOKENS:-384}"
export NAT_SAMPLE_MIN_TOKENS="${NAT_SAMPLE_MIN_TOKENS:-128}"
export NAT_TRUNCATE_RPC="${NAT_TRUNCATE_RPC:-True}"

export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TEST_FREQ="${TEST_FREQ:-5}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export DUMP_VALIDATION_GENERATIONS="${DUMP_VALIDATION_GENERATIONS:-0}"
export FOLLOW_LOG="${FOLLOW_LOG:-1}"

echo ">>> Aggressive long-tail overrides"
echo "    GPUS_PER_NODE=${GPUS_PER_NODE}, TP=${TENSOR_MODEL_PARALLEL_SIZE}"
echo "    TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}, ROLLOUT_N=${ROLLOUT_N}, MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}"
echo "    PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE}, PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU}, LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
echo "    NAT_MODE=${NAT_MODE}, NAT_KEEP_RATIO=${NAT_KEEP_RATIO}, NAT_TRUNCATE_RATIO=${NAT_TRUNCATE_RATIO}"
echo "    If this OOMs, first try MAX_RESPONSE_LENGTH=8192 or TRAIN_BATCH_SIZE=64."

exec bash "${BASE_SCRIPT}" "$@"
