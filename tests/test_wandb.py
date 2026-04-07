#!/usr/bin/env python3
"""
测试 .env 中的 Wandb 配置是否正确。

用法:
    python3 tests/test_wandb.py
"""
import os
import random
import sys
from pathlib import Path

import dotenv

# 加载项目根目录的 .env
project_root = Path(__file__).resolve().parent.parent
dotenv.load_dotenv(project_root / ".env")

# ============ 检查必要的环境变量 ============
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "GRPO-Baseline")

print("=== Wandb 配置检查 ===")
print(f"  .env 路径:      {project_root / '.env'} ({'存在' if (project_root / '.env').exists() else '不存在'})")
print(f"  WANDB_API_KEY:  {'已设置 (' + WANDB_API_KEY[:8] + '...)' if WANDB_API_KEY else '❌ 未设置'}")
print(f"  WANDB_ENTITY:   {WANDB_ENTITY or '❌ 未设置'}")
print(f"  WANDB_PROJECT:  {WANDB_PROJECT}")
print()

if not WANDB_API_KEY:
    print("❌ WANDB_API_KEY 未在 .env 中设置")
    print("   请在 .env 中添加: WANDB_API_KEY=your_key_here")
    print("   获取 key: https://wandb.ai/authorize")
    sys.exit(1)

if not WANDB_ENTITY:
    print("⚠️  WANDB_ENTITY 未设置，将使用 wandb 默认 entity")

# ============ 测试 wandb 连接 ============
import wandb

print(">>> 测试 wandb 登录...")
try:
    wandb.login(key=WANDB_API_KEY, relogin=True)
    print("  ✅ 登录成功")
except Exception as e:
    print(f"  ❌ 登录失败: {e}")
    sys.exit(1)

print(">>> 创建测试 run...")
run = wandb.init(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    name="test-env-config",
    config={
        "test": True,
        "purpose": "验证 .env 中的 wandb 配置",
    },
    tags=["test"],
)

print(f"  ✅ Run 创建成功: {run.url}")

# 模拟几步训练数据
print(">>> 模拟训练数据上传...")
offset = random.random() / 5
for epoch in range(1, 6):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    run.log({"acc": acc, "loss": loss, "epoch": epoch})

run.finish()
print()
print("🆗 Wandb 测试完成！请检查上面的 URL 确认数据已上传。")
