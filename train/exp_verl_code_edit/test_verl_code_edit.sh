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

# ============ 加载自定义源码 =============
export PYTHONPATH="${PROJECT_ROOT}/verl-src:${PYTHONPATH:-}"

# ============ 测试自定义代码 =============
python3 "${PROJECT_ROOT}/verl-src/hello_verl.py"