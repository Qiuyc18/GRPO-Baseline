#!/usr/bin/env bash
set -euo pipefail

# DeepScaleR GRPO run on Qwen3-4B-Base, 4 GPUs, with replay buffer plus
# NAT-style token-efficient actor updates.
# Defaults to RPC with 50% target token keep ratio.

# ============ Load project .env ============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -f "${PROJECT_ROOT}/.env" ]; then
  set -a
  source "${PROJECT_ROOT}/.env"
  set +a
fi

# Keep datasets cache out of a possibly root-owned ~/.cache/huggingface tree.
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

# ============ Base env ============
export HOST_CHECKPOINT_PATH="${HOST_CHECKPOINT_PATH:-/etc/moreh/checkpoint}"
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-train_qwen3_4B-base_deepscaler_grpo_4gpu_replay_buffer_nat_rpc_loose}"

# Optional: pin this run to specific devices, for example GPU_DEVICES=4,5,6,7.
if [ -n "${GPU_DEVICES:-}" ]; then
  export HIP_VISIBLE_DEVICES="${GPU_DEVICES}"
  unset ROCR_VISIBLE_DEVICES
  export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
fi

# ============ Model, data, cache ============
export MODEL_PATH="${MODEL_PATH:-${HOST_CHECKPOINT_PATH}/Qwen/Qwen3-4B-Base}"
export DATA_PATH="${DATA_PATH:-${HOST_CHECKPOINT_PATH}/data/deepscaler}"
DEFAULT_CKPT_PARENT="${CKPT_PARENT:-${HOST_CHECKPOINT_PATH}/GRPO-Baseline}"
if [ -z "${CKPT_ROOT:-}" ] && { [ ! -d "${DEFAULT_CKPT_PARENT}" ] || [ ! -w "${DEFAULT_CKPT_PARENT}" ]; }; then
  DEFAULT_CKPT_PARENT="${PROJECT_ROOT}/checkpoints"
fi
export CKPT_ROOT="${CKPT_ROOT:-${DEFAULT_CKPT_PARENT}/${EXPERIMENT_NAME}}"
export REPLAY_BUFFER_DIR="${REPLAY_BUFFER_DIR:-${CKPT_ROOT}/replay_buffer}"

# ============ W&B ============
export WANDB_ENTITY="${WANDB_ENTITY:-deng-lab}"
export WANDB_PROJECT="${WANDB_PROJECT:-GRPO-Baseline}"
export WANDB_NAME="${WANDB_NAME:-${EXPERIMENT_NAME}}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-deepscaler_4gpu_replay_nat_rpc_loose}"
export TRAIN_LOGGER="${TRAIN_LOGGER:-[\"console\",\"wandb\"]}"

# ============ Training knobs ============
# Keep the main training shape aligned with the no-replay baseline.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
export ROLLOUT_N="${ROLLOUT_N:-5}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.72}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-12288}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
export SAVE_FREQ="${SAVE_FREQ:-20}"
export TEST_FREQ="${TEST_FREQ:-5}"
export CLEAN_OLD_CKPT="${CLEAN_OLD_CKPT:-1}"
export CLEAN_REPLAY_BUFFER="${CLEAN_REPLAY_BUFFER:-1}"
export SKIP_MODEL_LOAD_TEST="${SKIP_MODEL_LOAD_TEST:-0}"
export FOLLOW_LOG="${FOLLOW_LOG:-1}"
export AUTO_SELECT_GPUS="${AUTO_SELECT_GPUS:-1}"
export DUMP_VALIDATION_GENERATIONS="${DUMP_VALIDATION_GENERATIONS:-1}"

if [ -z "${GPU_DEVICES:-}" ] && [ "${AUTO_SELECT_GPUS}" = "1" ]; then
  echo ">>> Auto-select free GPUs with rocm-smi"
  GPU_DEVICES="$(python3 - <<PY
import os
import re
import subprocess
import sys

need = int(os.environ.get("GPUS_PER_NODE", "4"))
util = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.72"))
try:
    out = subprocess.check_output(
        ["rocm-smi", "--showmeminfo", "vram"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
except Exception as exc:
    print(f"Failed to run rocm-smi: {exc}", file=sys.stderr)
    sys.exit(1)

total_by_gpu = {}
used_by_gpu = {}
for line in out.splitlines():
    m = re.search(r"GPU\\[(\\d+)\\].*VRAM Total Memory \\(B\\):\\s*(\\d+)", line)
    if m:
        total_by_gpu[int(m.group(1))] = int(m.group(2))
    m = re.search(r"GPU\\[(\\d+)\\].*VRAM Total Used Memory \\(B\\):\\s*(\\d+)", line)
    if m:
        used_by_gpu[int(m.group(1))] = int(m.group(2))

candidates = []
for gpu in sorted(total_by_gpu):
    total = total_by_gpu[gpu]
    used = used_by_gpu.get(gpu, 0)
    free = total - used
    required = total * util
    if free >= required:
        candidates.append(gpu)

if len(candidates) < need:
    print(
        f"Need {need} GPUs with free VRAM >= {util:.2%} of total, "
        f"but only found {len(candidates)}: {candidates}",
        file=sys.stderr,
    )
    for gpu in sorted(total_by_gpu):
        total = total_by_gpu[gpu]
        used = used_by_gpu.get(gpu, 0)
        free_gib = (total - used) / 1024**3
        total_gib = total / 1024**3
        print(f"GPU {gpu}: free={free_gib:.2f}/{total_gib:.2f} GiB", file=sys.stderr)
    sys.exit(2)

print(",".join(str(gpu) for gpu in candidates[:need]))
PY
)"
fi

if [ -n "${GPU_DEVICES:-}" ]; then
  export HIP_VISIBLE_DEVICES="${GPU_DEVICES}"
  unset ROCR_VISIBLE_DEVICES
  export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
fi

# Replay knobs. p_fresh=0.5 keeps half the steps fresh by default, which is a
# safer full-training setting than the smoke test's forced replay mode.
export REPLAY_BUFFER_MAX_SIZE="${REPLAY_BUFFER_MAX_SIZE:-64}"
export REPLAY_BUFFER_P_FRESH="${REPLAY_BUFFER_P_FRESH:-0.5}"
export REPLAY_BUFFER_HOT_CACHE_SIZE="${REPLAY_BUFFER_HOT_CACHE_SIZE:-4}"
export REPLAY_BUFFER_SPEC_VERIFY="${REPLAY_BUFFER_SPEC_VERIFY:-True}"
export REPLAY_BUFFER_SPEC_VERIFY_MIN_MEAN_LOGPROB_DELTA="${REPLAY_BUFFER_SPEC_VERIFY_MIN_MEAN_LOGPROB_DELTA:--1.0}"
export REPLAY_BUFFER_SPEC_VERIFY_MIN_SEQ_LOGPROB_DELTA="${REPLAY_BUFFER_SPEC_VERIFY_MIN_SEQ_LOGPROB_DELTA:--5.0}"

# NAT knobs. RPC can shorten the actor update micro-batch to the sampled max
# prefix while keeping rollout and verifier reward on the full response.
export NAT_TOKEN_SAMPLING="${NAT_TOKEN_SAMPLING:-True}"
export NAT_MODE="${NAT_MODE:-rpc_urs}"
export NAT_KEEP_RATIO="${NAT_KEEP_RATIO:-0.5}"
export NAT_MIN_TOKENS="${NAT_MIN_TOKENS:-1}"
export NAT_TRUNCATE_RATIO="${NAT_TRUNCATE_RATIO:-0.5}"
export NAT_TRUNCATE_MIN_TOKENS="${NAT_TRUNCATE_MIN_TOKENS:-256}"
export NAT_SAMPLE_MIN_TOKENS="${NAT_SAMPLE_MIN_TOKENS:-64}"
export NAT_TRUNCATE_RPC="${NAT_TRUNCATE_RPC:-True}"
export NAT_EPS="${NAT_EPS:-1e-6}"

echo ">>> Check local data path"
if [ ! -d "${DATA_PATH}" ]; then
  echo "Data directory does not exist: ${DATA_PATH}"
  echo "Please run: python3 data/format_deepscaler.py"
  exit 1
fi

if [ ! -f "${DATA_PATH}/train.parquet" ] || [ ! -f "${DATA_PATH}/test.parquet" ]; then
  echo "Data files missing under ${DATA_PATH}: train.parquet or test.parquet"
  exit 1
fi

echo ">>> Check local model path"
if [ ! -d "${MODEL_PATH}" ]; then
  echo "Model directory does not exist: ${MODEL_PATH}"
  exit 1
fi

echo ">>> Prepare checkpoint/cache dir"
mkdir -p "${CKPT_ROOT}"
if [ "${CLEAN_OLD_CKPT}" = "1" ] && ls "${CKPT_ROOT}"/global_step_* 1>/dev/null 2>&1; then
  echo "  Remove old checkpoints: ${CKPT_ROOT}/global_step_*"
  rm -rf "${CKPT_ROOT}"/global_step_*
fi
if [ "${CLEAN_REPLAY_BUFFER}" = "1" ]; then
  echo "  Reset replay buffer: ${REPLAY_BUFFER_DIR}"
  rm -rf "${REPLAY_BUFFER_DIR}"
fi
mkdir -p "${REPLAY_BUFFER_DIR}"

if [ "${SKIP_MODEL_LOAD_TEST}" != "1" ]; then
  echo ">>> Test local model load"
  python3 - <<PY
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "${MODEL_PATH}"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
print("Local model load OK:", model_path)
PY
fi

echo ">>> Check disk for checkpoint dir"
df -h "${CKPT_ROOT}" || true

# ============ GPU monitor ============
export GPU_PLATFORM=amd
export PYTHONPATH="${PROJECT_ROOT}/verl-src:${PROJECT_ROOT}/monitor:${PYTHONPATH:-}"

# ============ Log file ============
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_${TIMESTAMP}.log"
HYDRA_OUTPUT_DIR="${HYDRA_OUTPUT_DIR:-${LOG_DIR}/hydra_outputs/${EXPERIMENT_NAME}_${TIMESTAMP}}"
export GPU_MONITOR_OUTPUT="${GPU_MONITOR_OUTPUT:-${LOG_DIR}/monitor/${EXPERIMENT_NAME}_${TIMESTAMP}}"
if [ "${DUMP_VALIDATION_GENERATIONS:-0}" = "1" ]; then
  export VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${LOG_DIR}/validation_generations/${EXPERIMENT_NAME}_${TIMESTAMP}}"
else
  export VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-}"
fi
mkdir -p "${GPU_MONITOR_OUTPUT}"
if [ -n "${VALIDATION_DATA_DIR}" ]; then
  mkdir -p "${VALIDATION_DATA_DIR}"
fi
PID_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_${TIMESTAMP}.pid"
LATEST_PID_FILE="${LOG_DIR}/${EXPERIMENT_NAME}.pid"

echo ">>> Start DeepScaleR GRPO training, replay buffer + NAT token sampling enabled"
echo "    Log file: ${LOG_FILE}"
echo "    Stop: kill \$(cat ${PID_FILE})"
echo "    Replay cache: ${REPLAY_BUFFER_DIR}"
echo "    NAT: enabled=${NAT_TOKEN_SAMPLING}, mode=${NAT_MODE}, keep_ratio=${NAT_KEEP_RATIO}, truncate_ratio=${NAT_TRUNCATE_RATIO}, truncate_min=${NAT_TRUNCATE_MIN_TOKENS}, sample_min=${NAT_SAMPLE_MIN_TOKENS}, truncate_rpc=${NAT_TRUNCATE_RPC}"
echo "    GPUS_PER_NODE=${GPUS_PER_NODE}, GPU_DEVICES=${GPU_DEVICES:-all visible}"
echo "    GPU_MONITOR_OUTPUT=${GPU_MONITOR_OUTPUT}"
echo "    VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-disabled}"

nohup env PYTHONUNBUFFERED=1 python3 "${PROJECT_ROOT}/monitor/launch_verl.py" \
  hydra.run.dir="${HYDRA_OUTPUT_DIR}" \
  data.train_files=${DATA_PATH}/train.parquet \
  data.val_files=${DATA_PATH}/test.parquet \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  reward_model.strategy=fsdp2 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.entropy_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.token_sampling.enabled="${NAT_TOKEN_SAMPLING}" \
  actor_rollout_ref.actor.token_sampling.mode="${NAT_MODE}" \
  actor_rollout_ref.actor.token_sampling.keep_ratio="${NAT_KEEP_RATIO}" \
  actor_rollout_ref.actor.token_sampling.min_tokens="${NAT_MIN_TOKENS}" \
  actor_rollout_ref.actor.token_sampling.truncate_ratio="${NAT_TRUNCATE_RATIO}" \
  actor_rollout_ref.actor.token_sampling.truncate_min_tokens="${NAT_TRUNCATE_MIN_TOKENS}" \
  actor_rollout_ref.actor.token_sampling.sample_min_tokens="${NAT_SAMPLE_MIN_TOKENS}" \
  actor_rollout_ref.actor.token_sampling.truncate_rpc="${NAT_TRUNCATE_RPC}" \
  actor_rollout_ref.actor.token_sampling.eps="${NAT_EPS}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.001 \
  +algorithm.replay_buffer.enabled=True \
  +algorithm.replay_buffer.cache_dir="${REPLAY_BUFFER_DIR}" \
  +algorithm.replay_buffer.max_size="${REPLAY_BUFFER_MAX_SIZE}" \
  +algorithm.replay_buffer.p_fresh="${REPLAY_BUFFER_P_FRESH}" \
  +algorithm.replay_buffer.hot_cache_size="${REPLAY_BUFFER_HOT_CACHE_SIZE}" \
  +algorithm.replay_buffer.spec_verify="${REPLAY_BUFFER_SPEC_VERIFY}" \
  +algorithm.replay_buffer.spec_verify_min_mean_logprob_delta="${REPLAY_BUFFER_SPEC_VERIFY_MIN_MEAN_LOGPROB_DELTA}" \
  +algorithm.replay_buffer.spec_verify_min_seq_logprob_delta="${REPLAY_BUFFER_SPEC_VERIFY_MIN_SEQ_LOGPROB_DELTA}" \
  trainer.critic_warmup=0 \
  trainer.logger="${TRAIN_LOGGER}" \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${WANDB_NAME}" \
  trainer.default_local_dir="${CKPT_ROOT}" \
  trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.max_actor_ckpt_to_keep=3 \
  > "${LOG_FILE}" 2>&1 &

TRAIN_PID=$!
echo "${TRAIN_PID}" > "${PID_FILE}"
if [ -e "${LATEST_PID_FILE}" ] && [ ! -w "${LATEST_PID_FILE}" ]; then
  echo "Warning: cannot update latest pid file: ${LATEST_PID_FILE}"
elif ! echo "${TRAIN_PID}" > "${LATEST_PID_FILE}"; then
  echo "Warning: cannot update latest pid file: ${LATEST_PID_FILE}"
fi
echo "    Training PID: ${TRAIN_PID}"
echo ""
echo "    Reopen: tail -f ${LOG_FILE}"
if [ "${FOLLOW_LOG}" = "1" ]; then
  echo ">>> Following log (Ctrl+C stops tail only, training continues)"
  echo "=========================================="
  tail -f "${LOG_FILE}"
else
  echo ">>> FOLLOW_LOG=0, not tailing log."
fi
