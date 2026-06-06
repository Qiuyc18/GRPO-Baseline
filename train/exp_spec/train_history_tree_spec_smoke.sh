#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "${PROJECT_ROOT}/.env" ]; then
  set -a
  source "${PROJECT_ROOT}/.env"
  set +a
fi

export HOST_CHECKPOINT_PATH="${HOST_CHECKPOINT_PATH:-/etc/moreh/checkpoint}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export GPU_PLATFORM=amd
export PYTHONPATH="${PROJECT_ROOT}/verl-src:${PROJECT_ROOT}/monitor:${PYTHONPATH:-}"
export WANDB_MODE=disabled

mkdir -p "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-history_tree_spec_smoke}"
export MODEL_PATH="${MODEL_PATH:-${HOST_CHECKPOINT_PATH}/Qwen/Qwen3-4B-Base}"
export DATA_PATH="${DATA_PATH:-${HOST_CHECKPOINT_PATH}/data/deepscaler}"
export CKPT_ROOT="${CKPT_ROOT:-${HOST_CHECKPOINT_PATH}/GRPO-Baseline/${EXPERIMENT_NAME}}"

export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
export ROLLOUT_N="${ROLLOUT_N:-2}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"

export SMOKE_UNIQUE_PROMPTS="${SMOKE_UNIQUE_PROMPTS:-4}"
export SMOKE_REPEAT_TIMES="${SMOKE_REPEAT_TIMES:-4}"
export TRAIN_LOGGER="${TRAIN_LOGGER:-[\"console\"]}"
export SKIP_MODEL_LOAD_TEST="${SKIP_MODEL_LOAD_TEST:-0}"
export FOLLOW_LOG="${FOLLOW_LOG:-1}"

if [ -n "${GPU_DEVICES:-}" ]; then
  export HIP_VISIBLE_DEVICES="${GPU_DEVICES}"
  unset ROCR_VISIBLE_DEVICES
  export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
fi

for path in "${MODEL_PATH}" "${DATA_PATH}"; do
  if [ ! -d "${path}" ]; then
    echo "Missing path: ${path}" >&2
    exit 1
  fi
done
if [ ! -f "${DATA_PATH}/train.parquet" ] || [ ! -f "${DATA_PATH}/test.parquet" ]; then
  echo "Missing train.parquet or test.parquet under ${DATA_PATH}" >&2
  exit 1
fi

if [ "${SKIP_MODEL_LOAD_TEST}" != "1" ]; then
  echo ">>> Test local model load"
  python3 - <<PY
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "${MODEL_PATH}"
AutoTokenizer.from_pretrained(model_path, local_files_only=True)
AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
print("Local model load OK:", model_path)
PY
fi

LOG_DIR="${PROJECT_ROOT}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SMOKE_DATA_DIR="${LOG_DIR}/smoke_data/${EXPERIMENT_NAME}_${TIMESTAMP}"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_${TIMESTAMP}.pid"
LATEST_PID_FILE="${LOG_DIR}/${EXPERIMENT_NAME}.pid"
HYDRA_OUTPUT_DIR="${LOG_DIR}/hydra_outputs/${EXPERIMENT_NAME}_${TIMESTAMP}"
export GPU_MONITOR_OUTPUT="${GPU_MONITOR_OUTPUT:-${LOG_DIR}/monitor/${EXPERIMENT_NAME}_${TIMESTAMP}}"

mkdir -p "${CKPT_ROOT}" "${HYDRA_OUTPUT_DIR}" "${GPU_MONITOR_OUTPUT}" "${SMOKE_DATA_DIR}"

echo ">>> Build repeated smoke parquet"
python3 - <<PY
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

src_dir = Path("${DATA_PATH}")
out_dir = Path("${SMOKE_DATA_DIR}")
unique = int(os.environ["SMOKE_UNIQUE_PROMPTS"])
repeat = int(os.environ["SMOKE_REPEAT_TIMES"])

train = pq.read_table(src_dir / "train.parquet")
val = pq.read_table(src_dir / "test.parquet")
if train.num_rows < unique:
    raise RuntimeError(f"train.parquet has only {train.num_rows} rows, need {unique}")

base = train.take(pa.array(list(range(unique)), type=pa.int64()))
indices = [i for _ in range(repeat) for i in range(unique)]
pq.write_table(base.take(pa.array(indices, type=pa.int64())), out_dir / "train.parquet")
pq.write_table(val.take(pa.array(list(range(min(unique, val.num_rows))), type=pa.int64())), out_dir / "test.parquet")
print(f"smoke train rows={len(indices)}, unique_prompts={unique}, repeat_times={repeat}")
PY

echo ">>> History-tree speculative smoke run"
echo "    log=${LOG_FILE}"
echo "    pid_file=${PID_FILE}"
echo "    gpu_monitor=${GPU_MONITOR_OUTPUT}"
echo "    smoke_data=${SMOKE_DATA_DIR}"
echo "    logger=${TRAIN_LOGGER}, wandb=${WANDB_MODE}"
echo "    response_length=${MAX_RESPONSE_LENGTH}, batch=${TRAIN_BATCH_SIZE}, rollout.n=${ROLLOUT_N}"
echo "    unique_prompts=${SMOKE_UNIQUE_PROMPTS}, repeat_times=${SMOKE_REPEAT_TIMES}"

nohup env PYTHONUNBUFFERED=1 WANDB_MODE="${WANDB_MODE}" python3 "${PROJECT_ROOT}/monitor/launch_verl.py" \
  hydra.run.dir="${HYDRA_OUTPUT_DIR}" \
  data.train_files="${SMOKE_DATA_DIR}/train.parquet" \
  data.val_files="${SMOKE_DATA_DIR}/test.parquet" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  reward_model.strategy=fsdp2 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.entropy_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.token_sampling.enabled=True \
  actor_rollout_ref.actor.token_sampling.mode=rpc_urs \
  actor_rollout_ref.actor.token_sampling.keep_ratio=0.15 \
  actor_rollout_ref.actor.token_sampling.min_tokens=1 \
  actor_rollout_ref.actor.token_sampling.truncate_ratio=0.15 \
  actor_rollout_ref.actor.token_sampling.truncate_min_tokens=128 \
  actor_rollout_ref.actor.token_sampling.sample_min_tokens=64 \
  actor_rollout_ref.actor.token_sampling.truncate_rpc=True \
  actor_rollout_ref.actor.token_sampling.eps=1e-6 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.history_tree_speculation.enabled=True \
  actor_rollout_ref.rollout.history_tree_speculation.max_depth="${SPEC_MAX_DEPTH:-1}" \
  actor_rollout_ref.rollout.history_tree_speculation.max_branch_width="${SPEC_MAX_BRANCH_WIDTH:-8}" \
  actor_rollout_ref.rollout.history_tree_speculation.max_tokens_to_store="${SPEC_MAX_TOKENS_TO_STORE:-1024}" \
  actor_rollout_ref.rollout.history_tree_speculation.min_visits="${SPEC_MIN_VISITS:-1}" \
  actor_rollout_ref.rollout.history_tree_speculation.use_reward_prior=False \
  actor_rollout_ref.rollout.history_tree_speculation.use_nll_prior=True \
  actor_rollout_ref.rollout.history_tree_speculation.candidate_prompt_logprobs="${SPEC_CANDIDATE_PROMPT_LOGPROBS:-20}" \
  actor_rollout_ref.rollout.history_tree_speculation.exact_residual=True \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.critic_warmup=0 \
  trainer.logger="${TRAIN_LOGGER}" \
  trainer.project_name=history_tree_spec_smoke \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${CKPT_ROOT}" \
  trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.total_epochs=1 \
  trainer.max_actor_ckpt_to_keep=1 \
  > "${LOG_FILE}" 2>&1 &

TRAIN_PID=$!
echo "${TRAIN_PID}" > "${PID_FILE}"
echo "${TRAIN_PID}" > "${LATEST_PID_FILE}" || true

echo "    training_pid=${TRAIN_PID}"
echo "    stop: kill \$(cat ${PID_FILE})"
echo "    reopen: tail -f ${LOG_FILE}"
echo "    metrics: grep -E 'history_tree/(tree_hit_rate|tree_hit_count|root_miss_count|proposal_miss_count|candidate_score_miss_count|draft_tokens_proposed|draft_tokens_accepted)' ${LOG_FILE}"

if [ "${FOLLOW_LOG}" = "1" ]; then
  echo ">>> Following log (Ctrl+C stops tail only, training continues)"
  tail -f "${LOG_FILE}"
fi
