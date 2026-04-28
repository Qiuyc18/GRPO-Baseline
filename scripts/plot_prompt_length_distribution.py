#!/usr/bin/env python3
"""统计数据集 prompt 长度并绘制直方图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DATASET_NAME = "math"
DEFAULT_DATA_PATH = f"/etc/moreh/checkpoint/data/"
DEFAULT_OUTPUT_PATH = f"data/{DEFAULT_DATASET_NAME}_prompt_length_hist.png"
DEFAULT_TOKENIZER_PATH = "/etc/moreh/checkpoint/Qwen/Qwen3-4B-Base/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 prompt 长度并绘图")
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help=f"数据集存放路径，默认: {DEFAULT_DATA_PATH}",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_NAME,
        help=f"数据集名称，默认: {DEFAULT_DATASET_NAME}",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test", "all"],
        help=f"当 data-path 为目录时读取哪个 split，默认: train",
    )
    parser.add_argument(
        "--length-unit",
        default="token",
        choices=["char", "token"],
        help=f"长度单位：字符数或 token 数，默认: token",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER_PATH,
        help=f"当 length-unit=token 时使用的 tokenizer 名称或本地路径，默认: {DEFAULT_TOKENIZER_PATH}",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help=f"直方图分桶数，默认: 50",
    )
    parser.add_argument(
        "--output",
        default="",
        help=f"输出图片路径，默认: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--stats-json",
        default=None,
        help=f"可选：将统计信息保存到 json 文件，默认: None",
    )
    return parser.parse_args()


def resolve_input_files(data_path: Path, split: str) -> list[Path]:
    if data_path.is_file():
        return [data_path]
    if not data_path.is_dir():
        raise FileNotFoundError(f"数据路径不存在: {data_path}")

    if split == "all":
        files = sorted(data_path.glob("*.parquet"))
    else:
        files = sorted(data_path.glob(f"{split}*.parquet"))

    if not files:
        raise FileNotFoundError(f"在 {data_path} 下未找到匹配的 parquet 文件（split={split}）")
    return files


def prompt_to_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            return "\n".join(str(item.get("content", "")) for item in value)
        return " ".join(str(item) for item in value)
    if isinstance(value, dict):
        if "content" in value:
            return str(value["content"])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def iter_prompt_texts(files: Iterable[Path]) -> Iterable[str]:
    for file in files:
        df = pd.read_parquet(file)
        if "prompt" in df.columns:
            series = df["prompt"]
        elif "problem" in df.columns:
            series = df["problem"]
        elif "question" in df.columns:
            series = df["question"]
        else:
            raise ValueError(f"{file} 不包含 prompt/problem/question 列，现有列: {df.columns.tolist()}")
        for value in series:
            yield prompt_to_text(value)


def build_length_fn(length_unit: str, tokenizer_name_or_path: str | None):
    if length_unit == "char":
        return len

    if not tokenizer_name_or_path:
        raise ValueError(
            "length-unit=token 时必须提供 --tokenizer，例如 "
            "'--tokenizer /etc/moreh/checkpoint/Qwen/Qwen3-4B-Base/'"
        )

    from transformers import AutoTokenizer

    tokenizer_path = Path(tokenizer_name_or_path).expanduser()
    tokenizer_source = str(tokenizer_path) if tokenizer_path.exists() else tokenizer_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def summarize(lengths: np.ndarray) -> dict[str, float]:
    percentiles = [50, 75, 90, 95, 99]
    stats = {
        "count": int(lengths.size),
        "min": float(np.min(lengths)),
        "max": float(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "std": float(np.std(lengths)),
    }
    for p in percentiles:
        stats[f"p{p}"] = float(np.percentile(lengths, p))
    return stats


def plot_hist(lengths: np.ndarray, bins: int, output_path: Path, length_unit: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean_value = float(np.mean(lengths))
    plt.figure(figsize=(5, 3))
    _, _, patches = plt.hist(lengths, bins=bins, edgecolor="black", alpha=0.75)
    y_top = max((patch.get_height() for patch in patches), default=1.0)

    plt.axvline(
        mean_value,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_value:.2f}",
    )
    plt.text(
        mean_value,
        y_top * 0.92,
        f"mean={mean_value:.2f}",
        color="red",
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "red", "alpha": 0.8, "boxstyle": "round,pad=0.2"},
    )
    plt.title(f"{DEFAULT_DATASET_NAME} Prompt Length Distribution ({length_unit})")
    plt.xlabel(f"Prompt Length ({length_unit})")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> int:
    args = parse_args()
    data_path = Path(args.data_path)
    data_path = data_path / args.dataset
    files = resolve_input_files(data_path, args.split)
    length_fn = build_length_fn(args.length_unit, args.tokenizer)

    prompt_texts = list(iter_prompt_texts(files))
    if not prompt_texts:
        raise ValueError("没有读取到任何 prompt 文本")

    lengths = np.array([length_fn(text) for text in prompt_texts], dtype=np.int64)
    stats = summarize(lengths)

    if not args.output:
        output_path = DEFAULT_OUTPUT_PATH
    else:
        output_path = f"data/{args.dataset}_prompt_length_hist.png"

    output_path = Path(args.output)
    plot_hist(lengths, args.bins, output_path, args.length_unit)

    print("已处理文件:")
    for path in files:
        print(f"  - {path}")
    print("长度统计:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"直方图已保存: {output_path}")

    if args.stats_json:
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"统计信息已保存: {stats_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())