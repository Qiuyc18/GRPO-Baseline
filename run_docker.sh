#!/usr/bin/env bash
set -euo pipefail

IMAGE="rocm/verl:verl-0.6.0.amd0_rocm7.0_vllm0.11.0.dev"
CONTAINER_NAME="rocm_verl"

HOST_WORKDIR="${HOST_WORKDIR:-$PWD}"
HOST_CHECKPOINT_PATH="${HOST_CHECKPOINT_PATH:-/etc/moreh/checkpoint}"

CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-$PWD}"
CONTAINER_CHECKPOINT_PATH="${CONTAINER_CHECKPOINT_PATH:-/etc/moreh/checkpoint}"

# echo ">>> Pull image: ${IMAGE}"
# docker pull "${IMAGE}"

# 检查容器是否已存在
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  # 容器存在，检查是否正在运行
  if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo ">>> 容器正在运行，将连接到已有容器: ${CONTAINER_NAME}"
    echo ">>>>> PS: 如需重建容器，先运行: docker rm -f ${CONTAINER_NAME})"
    exec docker exec -it "${CONTAINER_NAME}" /bin/bash
  else
    echo ">>> 容器已停止，将重新启动容器并连接: ${CONTAINER_NAME}"
    docker start "${CONTAINER_NAME}"
    exec docker exec -it "${CONTAINER_NAME}" /bin/bash
  fi
fi

echo ">>> 创建新容器: ${CONTAINER_NAME}"
docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  --device /dev/dri \
  --device /dev/kfd \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  --ipc=host \
  --shm-size 128G \
  -p 8265:8265 \
  -v "$HOME/.ssh:/root/.ssh:ro" \
  -v "$HOME:$HOME" \
  -v "${HOST_WORKDIR}:${HOST_WORKDIR}" \
  -v "${HOST_CHECKPOINT_PATH}:${CONTAINER_CHECKPOINT_PATH}" \
  -w "${CONTAINER_WORKDIR}" \
  -e HOME="$HOME" \
  -e USER="$(id -un)" \
  -e LANG="${LANG:-en_US.UTF-8}" \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -e RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 \
  "${IMAGE}" sleep infinity

echo ">>> 正在连接到容器: ${CONTAINER_NAME}"
exec docker exec -it "${CONTAINER_NAME}" /bin/bash