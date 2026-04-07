#!/usr/bin/env bash
set -euo pipefail

# 这里设置 checkpoint 的根目录，以及 GPU 数量
export HOST_CHECKPOINT_PATH="${HOST_CHECKPOINT_PATH:-/etc/moreh/checkpoint}"  # 把根目录设置在 /etc/moreh/checkpoint 下，因为当前路径挂载的硬盘分区太小
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1  # 设置 GPU 可见设备为 1
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"  # 设置每个节点使用的 GPU 数量，当显卡数量不足的时候需要改小一点
export MODEL_PATH="${MODEL_PATH:-${HOST_CHECKPOINT_PATH}/Qwen2-1.5B}"  # 设置模型，默认路径是 /etc/moreh/checkpoint/，这里选择 Qwen2-1.5B
export DATA_PATH="${DATA_PATH:-${HOST_CHECKPOINT_PATH}/data/gsm8k}"  # 设置数据，保存路径是 /etc/moreh/checkpoint/data/，这里选择 gsm8k
export CKPT_ROOT="${CKPT_ROOT:-${HOST_CHECKPOINT_PATH}/GRPO-Baseline}"  # 设置 checkpoint 保存路径，默认路径是 /etc/moreh/checkpoint/

# 这里分别设置 wandb 的入口、项目名称、实验名称（实验名称最好每次改一下，避免不记得）
export WANDB_ENTITY="${WANDB_ENTITY:-qiuyc24-tsinghua-university}"  # 设置 wandb 的入口
export WANDB_PROJECT="${WANDB_PROJECT:-GRPO-Baseline}"  # 设置 wandb 的项目名称
export WANDB_NAME="${WANDB_NAME:-verl-ppo-gsm8k-demo}"  # 设置 wandb 的实验名称

echo ">>> Check local data path"
if [ ! -d "${DATA_PATH}" ]; then
  echo "❌ 数据目录不存在: ${DATA_PATH}"
  exit 1
else
  echo "🆗 数据目录存在: ${DATA_PATH}"
fi

echo ">>> Check local model path"
if [ ! -d "${MODEL_PATH}" ]; then
  echo "❌ 模型目录不存在: ${MODEL_PATH}"
  exit 1
else
  echo "🆗 模型目录存在: ${MODEL_PATH}"
fi

echo ">>> Prepare checkpoint dir"
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

echo ">>> Start PPO training with W&B"
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  data.train_files=${DATA_PATH}/train.parquet \
  data.val_files=${DATA_PATH}/test.parquet \
  data.train_batch_size=256 \
  data.val_batch_size=1312 \
  data.max_prompt_length=512 \
  data.max_response_length=256 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  critic.optim.lr=1e-5 \
  critic.model.path="${MODEL_PATH}" \
  critic.ppo_micro_batch_size_per_gpu=4 \
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
  trainer.total_epochs=1