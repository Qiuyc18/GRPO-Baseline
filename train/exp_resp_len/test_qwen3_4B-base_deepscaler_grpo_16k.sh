#!/usr/bin/env bash
set -euo pipefail

# ============ 加载项目级 .env ============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [ -f "${PROJECT_ROOT}/.env" ]; then
  set -a  # 自动 export
  source "${PROJECT_ROOT}/.env"
  set +a
fi

# ============ 基础环境 ============
export HOST_CHECKPOINT_PATH="${HOST_CHECKPOINT_PATH:-/etc/moreh/checkpoint}"  # checkpoint 根目录
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1  # AMD GPU 需要
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"  # 每个节点 GPU 数量
export EXPERIMENT_NAME="test_qwen3_4B-base_deepscaler_grpo_4k"

# ============ 模型与数据 ============
export MODEL_PATH="${MODEL_PATH:-${HOST_CHECKPOINT_PATH}/Qwen/Qwen3-4B-Base}"
export DATA_PATH="${DATA_PATH:-${HOST_CHECKPOINT_PATH}/data/deepscaler}"
export CKPT_ROOT="${CKPT_ROOT:-${HOST_CHECKPOINT_PATH}/GRPO-Baseline/${EXPERIMENT_NAME}}"

# ============ Wandb ============
export WANDB_ENTITY="${WANDB_ENTITY:-deng-lab}"
export WANDB_PROJECT="${WANDB_PROJECT:-GRPO-Baseline}"
export WANDB_NAME="${WANDB_NAME:-${EXPERIMENT_NAME}}"

echo ">>> Check local data path"
if [ ! -d "${DATA_PATH}" ]; then
  echo "❌ 数据目录不存在: ${DATA_PATH}"
  echo "请先运行:"
  echo "  python3 download.py download agentica-org/DeepScaleR-Preview-Dataset --type dataset"
  echo "  python3 data/format_deepscaler.py"
  exit 1
else
  echo "🆗 数据目录存在: ${DATA_PATH}"
fi

echo ">>> Check data files"
if [ ! -f "${DATA_PATH}/train.parquet" ] || [ ! -f "${DATA_PATH}/test.parquet" ]; then
  echo "❌ 数据文件不存在: train.parquet 或 test.parquet"
  echo "请先运行: python3 data/format_deepscaler.py"
  exit 1
else
  echo "🆗 数据文件存在"
fi

echo ">>> Check local model path"
if [ ! -d "${MODEL_PATH}" ]; then
  echo "❌ 模型目录不存在: ${MODEL_PATH}"
  exit 1
else
  echo "🆗 模型目录存在: ${MODEL_PATH}"
fi

echo ">>> Prepare checkpoint dir (clean old checkpoints)"
if ls "${CKPT_ROOT}"/global_step_* 1>/dev/null 2>&1; then
  echo "  清理旧 checkpoint: ${CKPT_ROOT}/global_step_*"
  rm -rf "${CKPT_ROOT}"/global_step_*
fi
mkdir -p "${CKPT_ROOT}"

echo ">>> Test local model load"
python3 - <<PY
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "${MODEL_PATH}"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
print("Local model load OK:", model_path)
PY

echo ">>> Check disk for checkpoint dir"
df -h "${CKPT_ROOT}" || true

# ============ GPU 监控 ============
export GPU_PLATFORM=amd
export GPU_MONITOR_OUTPUT=logs/${EXPERIMENT_NAME}
export PYTHONPATH="${PROJECT_ROOT}/monitor:${PYTHONPATH:-}"

# ============ 日志文件 ============
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_${TIMESTAMP}.log"

echo ">>> Start GRPO training (Math) with W&B (nohup)"
echo "    日志文件: ${LOG_FILE}"
echo "    停止训练: kill \$(cat ${LOG_DIR}/${EXPERIMENT_NAME}.pid)"

# ============ 训练参数说明 ============
# Qwen3-4B-Base, 8x AMD GPU, tensor_parallel=2 -> data_parallel=4

nohup env PYTHONUNBUFFERED=1 python3 "${PROJECT_ROOT}/monitor/launch_verl.py" \
  algorithm.adv_estimator=grpo \
  data.train_files=${DATA_PATH}/train.parquet \
  data.val_files=${DATA_PATH}/test.parquet \
  data.train_batch_size=896 \
  data.max_prompt_length=384 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  reward_model.strategy=fsdp2 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.entropy_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=224 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=28 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=28 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=28 \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${WANDB_NAME}" \
  trainer.default_local_dir="${CKPT_ROOT}" \
  trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.save_freq=999 \
  trainer.test_freq=5 \
  trainer.total_epochs=10 \
  > "${LOG_FILE}" 2>&1 &

TRAIN_PID=$!
echo "${TRAIN_PID}" > "${LOG_DIR}/${EXPERIMENT_NAME}.pid"
echo "    训练进程 PID: ${TRAIN_PID}"
echo ""
echo ">>> 正在跟踪日志 (Ctrl+C 退出跟踪，训练不会停止)"
echo "    重新查看: tail -f ${LOG_FILE}"
echo "=========================================="
tail -f "${LOG_FILE}"