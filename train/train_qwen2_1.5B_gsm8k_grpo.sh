#!/usr/bin/env bash
set -euo pipefail

# ============ 加载项目级 .env（每人各自的 key）============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -f "${PROJECT_ROOT}/.env" ]; then
  set -a  # 自动 export
  source "${PROJECT_ROOT}/.env"
  set +a
fi

# ============ 基础环境 ============
export HOST_CHECKPOINT_PATH="${HOST_CHECKPOINT_PATH:-/etc/moreh/checkpoint}"  # checkpoint 根目录
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1  # AMD GPU 需要
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"  # 每个节点 GPU 数量，显卡不足时改小

# ============ 模型与数据 ============
export MODEL_PATH="${MODEL_PATH:-${HOST_CHECKPOINT_PATH}/Qwen2-1.5B}"  # 基座模型路径
export DATA_PATH="${DATA_PATH:-${HOST_CHECKPOINT_PATH}/data/gsm8k}"    # 数据集路径
export CKPT_ROOT="${CKPT_ROOT:-${HOST_CHECKPOINT_PATH}/GRPO-Baseline}" # checkpoint 保存路径

# ============ Wandb ============
export WANDB_ENTITY="${WANDB_ENTITY:-qiuyc24-tsinghua-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-GRPO-Baseline}"
export WANDB_NAME="${WANDB_NAME:-verl-grpo-gsm8k-demo}"  # 每次实验改一下名字

echo ">>> Check local data path"
if [ ! -d "${DATA_PATH}" ]; then
  echo "❌ 数据目录不存在: ${DATA_PATH}"
  exit 1
else
  echo "🆗 数据目录存在: ${DATA_PATH}"
fi

echo ">>> Check data files"
if [ ! -f "${DATA_PATH}/train.parquet" ] || [ ! -f "${DATA_PATH}/test.parquet" ]; then
  echo "❌ 数据文件不存在: train.parquet 或 test.parquet"
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

echo ">>> Start GRPO training with W&B"
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  data.train_files=${DATA_PATH}/train.parquet \
  data.val_files=${DATA_PATH}/test.parquet \
  data.train_batch_size=256 \
  data.max_prompt_length=256 \
  data.max_response_length=512 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  algorithm.adv_estimator=grpo \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${WANDB_NAME}" \
  trainer.default_local_dir="${CKPT_ROOT}" \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.total_epochs=10