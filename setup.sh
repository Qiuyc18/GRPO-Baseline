#!/bin/bash
# setup_env.sh - 为当前目录创建独立的 uv 虚拟环境并安装依赖

echo "开始初始化环境..."

if [ ! -d ".venv" ]; then
    uv venv
    echo "✅ 已创建 .venv 目录"
else
    echo "⚡ .venv 已存在，跳过创建"
fi

source .venv/bin/activate

echo "📦 正在安装 vllm (ROCm)..."
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/

echo "📦 正在安装其他依赖..."
uv pip install verl ray deepspeed wandb pandas pyarrow

echo "🎉 环境初始化完成！"