#!/usr/bin/env python3
"""
将 agentica-org/DeepScaleR-Preview-Dataset 转换为 veRL 训练所需的 parquet 格式。

用法:
    # 1. 先下载原始数据集
    python3 download.py download agentica-org/DeepScaleR-Preview-Dataset --type dataset

    # 2. 运行预处理
    python3 data/format_deepscaler.py

    # 3. 指定自定义路径
    python3 data/format_deepscaler.py \
        --raw-dir /path/to/DeepScaleR-Preview-Dataset \
        --out-dir /path/to/output
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def extract_boxed_answer(text: str) -> str | None:
    """从文本中提取 \\boxed{...} 里的答案。"""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i]
            depth -= 1
    return None


def convert_row(row: pd.Series, index: int) -> dict:
    """将 DeepScaleR 的一行转为 veRL 格式。"""
    problem = row["problem"]

    # DeepScaleR 的 answer 字段可能是纯文本或 \boxed{} 格式
    raw_answer = str(row.get("answer", ""))
    boxed = extract_boxed_answer(raw_answer)
    answer = boxed if boxed is not None else raw_answer.strip()

    prompt_text = (
        f"{problem} "
        f"Let's think step by step and output the final answer in \\boxed{{}}."
    )

    return {
        # veRL 内置 "lighteval/MATH" reward function，复用其评测逻辑
        "data_source": "lighteval/MATH",
        "prompt": np.array(
            [{"role": "user", "content": prompt_text}], dtype=object
        ),
        "ability": "math",
        "reward_model": {"ground_truth": answer, "style": "rule"},
        "extra_info": {
            "index": index,
            "problem": problem,
            "source": row.get("source", "deepscaler"),
        },
    }


def load_dataset(raw_dir: Path) -> pd.DataFrame:
    """加载 DeepScaleR 数据集，支持多种文件布局。"""
    # 尝试多种可能的文件格式
    candidates = [
        list(raw_dir.rglob("train*.parquet")),
        list(raw_dir.rglob("*.parquet")),
        list(raw_dir.rglob("*.jsonl")),
        list(raw_dir.rglob("*.json")),
    ]
    for found in candidates:
        found = [f for f in found if ".cache" not in str(f)]
        if found:
            break
    else:
        raise FileNotFoundError(
            f"在 {raw_dir} 下找不到数据文件（parquet/jsonl/json）。请检查数据是否已下载。"
        )

    dfs = []
    for path in sorted(found):
        print(f"  读取: {path.name} ", end="")
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".jsonl":
            df = pd.read_json(path, lines=True)
        else:
            df = pd.read_json(path)
        print(f"({len(df)} 行)")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    print(f"  合计: {len(merged)} 行")
    print(f"  列:   {merged.columns.tolist()}")
    return merged


def main():
    default_raw = "/etc/moreh/checkpoint/data/agentica-org/DeepScaleR-Preview-Dataset"
    default_out = "/etc/moreh/checkpoint/data/deepscaler"

    parser = argparse.ArgumentParser(description="预处理 DeepScaleR 数据集为 veRL 格式")
    parser.add_argument("--raw-dir", default=default_raw, help="原始数据集路径")
    parser.add_argument("--out-dir", default=default_out, help="输出路径")
    parser.add_argument(
        "--test-ratio", type=float, default=0.05,
        help="测试集比例（DeepScaleR 无官方 test split，需手动切分）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not raw_dir.exists():
        print(f"❌ 原始数据目录不存在: {raw_dir}")
        print(
            "请先下载:\n"
            "  python3 download.py download "
            "agentica-org/DeepScaleR-Preview-Dataset --type dataset"
        )
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n>>> 加载 DeepScaleR 数据集")
    df_raw = load_dataset(raw_dir)

    # 检查必需列
    if "problem" not in df_raw.columns:
        print(f"❌ 数据集缺少 'problem' 列。现有列: {df_raw.columns.tolist()}")
        return 1

    print("\n>>> 转换为 veRL 格式")
    rows = [convert_row(row, i) for i, (_, row) in enumerate(df_raw.iterrows())]
    df = pd.DataFrame(rows)

    # 统计答案提取情况
    n_empty = sum(1 for r in rows if r["reward_model"]["ground_truth"] == "")
    if n_empty > 0:
        print(f"  ⚠️  {n_empty}/{len(rows)} 行未能提取到答案")

    # 切分 train/test
    print(f"\n>>> 切分 train/test (test_ratio={args.test_ratio}, seed={args.seed})")
    rng = np.random.default_rng(args.seed)
    n_test = max(1, int(len(df) * args.test_ratio))
    test_idx = rng.choice(len(df), size=n_test, replace=False)
    mask = np.zeros(len(df), dtype=bool)
    mask[test_idx] = True

    df_train = df[~mask].reset_index(drop=True)
    df_test = df[mask].reset_index(drop=True)

    for split, split_df in [("train", df_train), ("test", df_test)]:
        out_path = out_dir / f"{split}.parquet"
        split_df.to_parquet(out_path, index=False)
        print(f"  ✅ {split}: {len(split_df)} 行 → {out_path}")

    # prompt 长度统计
    prompt_lens = df["prompt"].apply(lambda x: len(x[0]["content"]))
    print(f"\n>>> Prompt 长度统计:")
    print(f"    min={prompt_lens.min()}, median={int(prompt_lens.median())}, "
          f"max={prompt_lens.max()}, mean={prompt_lens.mean():.0f}")

    print(f"\n🆗 预处理完成！输出目录: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
