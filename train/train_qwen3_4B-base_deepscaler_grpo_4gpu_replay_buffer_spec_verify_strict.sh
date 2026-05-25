#!/usr/bin/env bash
set -euo pipefail

# Stricter speculative-style verification experiment.
# Compared with the default spec-verify run, this accepts cached trajectories
# only when the current actor remains closer to the cached draft policy.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-train_qwen3_4B-base_deepscaler_grpo_4gpu_replay_buffer_spec_verify_strict}"
export WANDB_NAME="${WANDB_NAME:-${EXPERIMENT_NAME}}"

# Keep dense validation and dump full validation generations for trajectory
# comparison across checkpoints.
export TEST_FREQ="${TEST_FREQ:-5}"
export DUMP_VALIDATION_GENERATIONS="${DUMP_VALIDATION_GENERATIONS:-1}"

# Same replay rate as the first spec-verify run, but stricter acceptance.
export REPLAY_BUFFER_SPEC_VERIFY="${REPLAY_BUFFER_SPEC_VERIFY:-True}"
export REPLAY_BUFFER_SPEC_VERIFY_MIN_MEAN_LOGPROB_DELTA="${REPLAY_BUFFER_SPEC_VERIFY_MIN_MEAN_LOGPROB_DELTA:--0.2}"
export REPLAY_BUFFER_SPEC_VERIFY_MIN_SEQ_LOGPROB_DELTA="${REPLAY_BUFFER_SPEC_VERIFY_MIN_SEQ_LOGPROB_DELTA:--2.0}"

exec bash "${SCRIPT_DIR}/train_qwen3_4B-base_deepscaler_grpo_4gpu_replay_buffer.sh"
