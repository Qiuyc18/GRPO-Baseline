#!/usr/bin/env bash
set -euo pipefail

# Small end-to-end replay-buffer smoke test for Qwen3-4B-Base on DeepScaleR.
# It builds a tiny parquet subset, runs a few GRPO steps, and forces replay
# after the first fresh rollout so logs should contain rollout/from_cache=1.

# ============ Load project .env ============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -f "${PROJECT_ROOT}/.env" ]; then
  set -a
  source "${PROJECT_ROOT}/.env"
  set +a
fi

# ============ Base env ============
export HOST_CHECKPOINT_PATH="${HOST_CHECKPOINT_PATH:-/etc/moreh/checkpoint}"
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export GPUS_PER_NODE="${GPUS_PER_NODE:-6}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-test_qwen3_4B-base_deepscaler_replay_buffer_smoke}"

# ============ Model, data, cache ============
export MODEL_PATH="${MODEL_PATH:-${HOST_CHECKPOINT_PATH}/Qwen/Qwen3-4B-Base}"
export DATA_PATH="${DATA_PATH:-${HOST_CHECKPOINT_PATH}/data/deepscaler}"
export CKPT_ROOT="${CKPT_ROOT:-${HOST_CHECKPOINT_PATH}/GRPO-Baseline/${EXPERIMENT_NAME}}"
export SMOKE_DATA_DIR="${SMOKE_DATA_DIR:-${CKPT_ROOT}/smoke_data}"
export REPLAY_BUFFER_DIR="${REPLAY_BUFFER_DIR:-${CKPT_ROOT}/replay_buffer}"

# Keep the subset and run short. Override these from the shell if needed.
export SMOKE_TRAIN_SAMPLES="${SMOKE_TRAIN_SAMPLES:-96}"
export SMOKE_TEST_SAMPLES="${SMOKE_TEST_SAMPLES:-16}"
export SMOKE_MAX_PROMPT_CHARS="${SMOKE_MAX_PROMPT_CHARS:-3000}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-6}"

# Training knobs chosen for a quick functional test, not for final quality.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-48}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-24}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"

# p_fresh=0.0 means: first step must be fresh because the buffer is empty,
# then later steps replay cached batches. This is best for a branch smoke test.
export REPLAY_BUFFER_MAX_SIZE="${REPLAY_BUFFER_MAX_SIZE:-8}"
export REPLAY_BUFFER_P_FRESH="${REPLAY_BUFFER_P_FRESH:-0.0}"
export REPLAY_BUFFER_HOT_CACHE_SIZE="${REPLAY_BUFFER_HOT_CACHE_SIZE:-2}"

# Console-only by default to keep smoke tests lightweight.
export WANDB_ENTITY="${WANDB_ENTITY:-deng-lab}"
export WANDB_PROJECT="${WANDB_PROJECT:-GRPO-Baseline}"
export WANDB_NAME="${WANDB_NAME:-${EXPERIMENT_NAME}}"
export TRAIN_LOGGER="${TRAIN_LOGGER:-[\"console\"]}"

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

mkdir -p "${CKPT_ROOT}" "${SMOKE_DATA_DIR}" "${REPLAY_BUFFER_DIR}"

echo ">>> Build small DeepScaleR smoke subset"
python3 - <<PY
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

data_path = Path("${DATA_PATH}")
out_dir = Path("${SMOKE_DATA_DIR}")
train_n = int("${SMOKE_TRAIN_SAMPLES}")
test_n = int("${SMOKE_TEST_SAMPLES}")
max_prompt_chars = int("${SMOKE_MAX_PROMPT_CHARS}")


def prompt_len(value):
    try:
        prompt = value[0]
        if isinstance(prompt, dict):
            return len(str(prompt.get("content", "")))
        return len(str(prompt))
    except Exception:
        return len(str(value))


def take_rows(src: Path, n: int) -> pa.Table:
    pf = pq.ParquetFile(src)
    chunks = []
    count = 0
    for batch in pf.iter_batches(batch_size=1024):
        table = pa.Table.from_batches([batch])
        data = table.to_pylist()
        kept = [row for row in data if prompt_len(row.get("prompt")) <= max_prompt_chars]
        if kept:
            chunks.append(pa.Table.from_pylist(kept[: max(0, n - count)]))
            count += len(kept[: max(0, n - count)])
        if count >= n:
            break
    if count < n:
        raise RuntimeError(f"Only collected {count}/{n} rows from {src}")
    return pa.concat_tables(chunks)


out_dir.mkdir(parents=True, exist_ok=True)
train_table = take_rows(data_path / "train.parquet", train_n)
test_table = take_rows(data_path / "test.parquet", test_n)
pq.write_table(train_table, out_dir / "train.parquet")
pq.write_table(test_table, out_dir / "test.parquet")
print(f"Wrote smoke train rows: {train_table.num_rows} -> {out_dir / 'train.parquet'}")
print(f"Wrote smoke test rows:  {test_table.num_rows} -> {out_dir / 'test.parquet'}")
PY

echo ">>> Prepare checkpoint/cache dir"
if ls "${CKPT_ROOT}"/global_step_* 1>/dev/null 2>&1; then
  echo "  Remove old checkpoints: ${CKPT_ROOT}/global_step_*"
  rm -rf "${CKPT_ROOT}"/global_step_*
fi
rm -rf "${REPLAY_BUFFER_DIR}"
mkdir -p "${REPLAY_BUFFER_DIR}"

echo ">>> Check disk for checkpoint dir"
df -h "${CKPT_ROOT}" || true

# ============ GPU monitor ============
export GPU_PLATFORM=amd
export GPU_MONITOR_OUTPUT=logs/${EXPERIMENT_NAME}
export PYTHONPATH="${PROJECT_ROOT}/verl-src:${PROJECT_ROOT}/monitor:${PYTHONPATH:-}"

# ============ Log file ============
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_${TIMESTAMP}.log"

echo ">>> Start Qwen3-4B-Base DeepScaleR replay-buffer smoke test"
echo "    Log file: ${LOG_FILE}"
echo "    Stop: kill \$(cat ${LOG_DIR}/${EXPERIMENT_NAME}.pid)"
echo "    Replay cache: ${REPLAY_BUFFER_DIR}"

nohup env PYTHONUNBUFFERED=1 python3 "${PROJECT_ROOT}/monitor/launch_verl.py" \
  hydra.run.dir="${CKPT_ROOT}/hydra_outputs/${TIMESTAMP}" \
  data.train_files=${SMOKE_DATA_DIR}/train.parquet \
  data.val_files=${SMOKE_DATA_DIR}/test.parquet \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.ref.strategy=fsdp2 \
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
  trainer.logger="${TRAIN_LOGGER}" \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${WANDB_NAME}" \
  trainer.default_local_dir="${CKPT_ROOT}" \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=10 \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  > "${LOG_FILE}" 2>&1 &

TRAIN_PID=$!
echo "${TRAIN_PID}" > "${LOG_DIR}/${EXPERIMENT_NAME}.pid"
echo "    Training PID: ${TRAIN_PID}"
echo ""
echo ">>> Following log (Ctrl+C stops tail only, training continues)"
echo "    Reopen: tail -f ${LOG_FILE}"
echo "=========================================="
tail -f "${LOG_FILE}"
