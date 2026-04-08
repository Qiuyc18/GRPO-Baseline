#!/usr/bin/env python3
"""
将 EleutherAI/hendrycks_math 数据集转换为 veRL 训练所需的 parquet 格式。

该数据集按子目录存放（algebra, geometry, ...），每个子目录下有 train/test parquet 文件，
本脚本将它们合并为单个 train.parquet 和 test.parquet。

用法:
    # 1. 先下载原始数据集
    python3 download.py download EleutherAI/hendrycks_math --type dataset

    # 2. 运行预处理
    python3 data/preprocess_math.py

    # 3. 指定自定义路径
    python3 data/preprocess_math.py --raw-dir /path/to/hendrycks_math --out-dir /path/to/output
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def extract_boxed_answer(solution: str) -> str | None:
    """从 MATH 数据集的 solution 中提取 \\boxed{...} 里的答案。"""
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        return None
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(solution)):
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            if depth == 0:
                return solution[start:i]
            depth -= 1
    return None


def convert_row(row: pd.Series, index: int, split: str) -> dict:
    """将原始 MATH 数据集的一行转为 veRL 格式。"""
    problem = row["problem"]
    solution = row["solution"]
    answer = extract_boxed_answer(solution) or ""

    prompt_text = (
        f"{problem} "
        f'Let\'s think step by step and output the final answer in \\boxed{{}}.'
    )

    return {
        "data_source": "lighteval/MATH",  # verl 内置支持此 data_source 的 reward function
        "prompt": np.array(
            [{"role": "user", "content": prompt_text}], dtype=object
        ),
        "ability": "math",
        "reward_model": {"ground_truth": answer, "style": "rule"},
        "extra_info": {
            "solution": solution,
            "level": row.get("level", ""),
            "type": row.get("type", ""),
            "index": index,
            "problem": problem,
            "split": split,
        },
    }


def collect_parquets(raw_dir: Path, split: str) -> pd.DataFrame:
    """从多个子目录中收集并合并同一 split 的 parquet 文件。"""
    found = sorted(raw_dir.rglob(f"{split}*.parquet"))
    # 排除 .cache 目录
    found = [f for f in found if ".cache" not in str(f)]

    if not found:
        raise FileNotFoundError(
            f"在 {raw_dir} 下找不到 {split} parquet 文件。请检查数据是否已下载。"
        )

    dfs = []
    for path in found:
        df = pd.read_parquet(path)
        subdir = path.parent.name
        print(f"  读取: {subdir}/{path.name} ({len(df)} 行)")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    print(f"  合并后: {len(merged)} 行")
    return merged


def process_split(raw_dir: Path, split: str) -> pd.DataFrame:
    """处理单个 split（train / test）。"""
    df_raw = collect_parquets(raw_dir, split)
    print(f"  列: {df_raw.columns.tolist()}")

    rows = [convert_row(row, i, split) for i, (_, row) in enumerate(df_raw.iterrows())]
    df = pd.DataFrame(rows)
    df["prompt_len"] = df["prompt"].apply(lambda x: len(x[0]["content"]))

    # 统计答案提取成功率
    n_empty = sum(1 for r in rows if r["reward_model"]["ground_truth"] == "")
    if n_empty > 0:
        print(f"  ⚠️  {n_empty}/{len(rows)} 行未能提取到 \\boxed{{}} 答案")

    return df


def main():
    default_raw = "/etc/moreh/checkpoint/data/EleutherAI/hendrycks_math"
    default_out = "/etc/moreh/checkpoint/data/math"

    parser = argparse.ArgumentParser(description="预处理 MATH 数据集为 veRL 格式")
    parser.add_argument("--raw-dir", default=default_raw, help="原始数据集路径")
    parser.add_argument("--out-dir", default=default_out, help="输出路径")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not raw_dir.exists():
        print(f"❌ 原始数据目录不存在: {raw_dir}")
        print("请先下载: python3 download.py download EleutherAI/hendrycks_math --type dataset")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        print(f"\n>>> 处理 {split} split")
        try:
            df = process_split(raw_dir, split)
            print(f"最大prompt长度: {df['prompt_len'].max()}")
            df.drop(columns=["prompt_len"], inplace=True)
            df.reset_index(drop=True, inplace=True)
            out_path = out_dir / f"{split}.parquet"
            df.to_parquet(out_path, index=False)
            print(f"  ✅ 已保存: {out_path} ({len(df)} 行)")
        except FileNotFoundError as e:
            print(f"  ⚠️  跳过 {split}: {e}")

    print("\n🆗 预处理完成！")
    print(f"输出目录: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
