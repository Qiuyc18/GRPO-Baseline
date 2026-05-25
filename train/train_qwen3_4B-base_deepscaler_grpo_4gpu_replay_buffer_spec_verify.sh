#!/usr/bin/env bash
set -euo pipefail

# Replay-buffer experiment with speculative-style verification enabled.
# Cached trajectories are treated as draft outputs and accepted only when the
# current actor still assigns sufficiently similar log probability.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-train_qwen3_4B-base_deepscaler_grpo_4gpu_replay_buffer_spec_verify}"
export WANDB_NAME="${WANDB_NAME:-${EXPERIMENT_NAME}}"
export TEST_FREQ="${TEST_FREQ:-5}"
export DUMP_VALIDATION_GENERATIONS="${DUMP_VALIDATION_GENERATIONS:-1}"
export REPLAY_BUFFER_SPEC_VERIFY="${REPLAY_BUFFER_SPEC_VERIFY:-True}"
export REPLAY_BUFFER_SPEC_VERIFY_MIN_MEAN_LOGPROB_DELTA="${REPLAY_BUFFER_SPEC_VERIFY_MIN_MEAN_LOGPROB_DELTA:--1.0}"
export REPLAY_BUFFER_SPEC_VERIFY_MIN_SEQ_LOGPROB_DELTA="${REPLAY_BUFFER_SPEC_VERIFY_MIN_SEQ_LOGPROB_DELTA:--5.0}"

exec bash "${SCRIPT_DIR}/train_qwen3_4B-base_deepscaler_grpo_4gpu_replay_buffer.sh"
