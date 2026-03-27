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

echo ">>> Remove old container if exists"
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

echo ">>> Start container: ${CONTAINER_NAME}"
docker run --rm -it \
  --name "${CONTAINER_NAME}" \
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
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -e RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 \
  "${IMAGE}" /bin/bash