#!/usr/bin/env bash
set -euo pipefail

# ============ 加载项目级 .env ============
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
export EXPERIMENT_NAME="train_qwen2_1.5B_math_grpo"

# ============ 模型与数据 ============
export MODEL_PATH="${MODEL_PATH:-${HOST_CHECKPOINT_PATH}/Qwen2-1.5B}"  # 基座模型路径
export DATA_PATH="${DATA_PATH:-${HOST_CHECKPOINT_PATH}/data/math}"     # MATH 数据集路径
export CKPT_ROOT="${CKPT_ROOT:-${HOST_CHECKPOINT_PATH}/GRPO-Baseline/${EXPERIMENT_NAME}}" # checkpoint 保存路径

# ============ Wandb ============
export WANDB_ENTITY="${WANDB_ENTITY:-deng-lab}"
export WANDB_PROJECT="${WANDB_PROJECT:-GRPO-Baseline}"
export WANDB_NAME="${WANDB_NAME:-${EXPERIMENT_NAME}}"  # 每次实验改一下名字

echo ">>> Check local data path"
if [ ! -d "${DATA_PATH}" ]; then
  echo "❌ 数据目录不存在: ${DATA_PATH}"
  echo "请先运行: python3 data/preprocess_math.py"
  exit 1
else
  echo "🆗 数据目录存在: ${DATA_PATH}"
fi

echo ">>> Check data files"
if [ ! -f "${DATA_PATH}/train.parquet" ] || [ ! -f "${DATA_PATH}/test.parquet" ]; then
  echo "❌ 数据文件不存在: train.parquet 或 test.parquet"
  echo "请先运行: python3 data/preprocess_math.py"
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
# 如需启用 GPU 监控，参考 monitor/ 子模块（RL_GRPO_Monitor）的文档
# export GPU_PLATFORM=amd
# python3 -m monitor &

export GPU_PLATFORM=amd
export GPU_MONITOR_OUTPUT=logs/${EXPERIMENT_NAME}
export PYTHONPATH="${PROJECT_ROOT}/monitor:${PYTHONPATH:-}"

# ============ 日志文件 ============
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_${TIMESTAMP}.log"

echo ">>> Start GRPO training (MATH dataset) with W&B (nohup)"
echo "    日志文件: ${LOG_FILE}"
echo "    停止训练: kill \$(cat ${LOG_DIR}/${EXPERIMENT_NAME}.pid)"

nohup env PYTHONUNBUFFERED=1 python3 "${PROJECT_ROOT}/monitor/launch_verl.py" \
  data.train_files=${DATA_PATH}/train.parquet \
  data.val_files=${DATA_PATH}/test.parquet \
  data.train_batch_size=256 \
  data.max_prompt_length=3072 \
  data.max_response_length=2048 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
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
  trainer.save_freq=20 \
  trainer.test_freq=10 \
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
