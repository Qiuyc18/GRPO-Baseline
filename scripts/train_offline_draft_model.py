#!/usr/bin/env python3
"""Train a small offline draft head from policy rollouts.

Example:
    PYTHONPATH=verl-src python3 scripts/train_offline_draft_model.py \
        --model-path /etc/moreh/checkpoint/Qwen/Qwen3-4B-Base \
        --train-file /etc/moreh/checkpoint/data/deepscaler/train.parquet \
        --output-dir checkpoints/offline_draft/qwen3_4b_deepscaler_smoke \
        --num-prompts 8 --max-new-tokens 16 --num-epochs 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.trainer.ppo.draft_model import OfflineDraftTrainingConfig, train_offline_draft_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Local HuggingFace policy model path.")
    parser.add_argument("--train-file", required=True, help="DeepScaleR train.parquet path.")
    parser.add_argument("--output-dir", required=True, help="Directory to save draft_model.pt and metadata.")
    parser.add_argument("--num-prompts", type=int, default=8, help="Number of training prompts to sample.")
    parser.add_argument("--prompt-offset", type=int, default=0, help="Start row in the parquet file.")
    parser.add_argument("--max-prompt-length", type=int, default=1024, help="Skip prompts longer than this many tokens.")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Rollout response length for draft data.")
    parser.add_argument("--num-rollouts-per-prompt", type=int, default=1)
    parser.add_argument("--prompt-batch-size", type=int, default=1)
    parser.add_argument("--draft-batch-size", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-multiplier", type=int, default=2)
    parser.add_argument("--do-sample", action="store_true", help="Sample policy rollouts instead of greedy decoding.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float32", "float16", "bfloat16"])
    return parser.parse_args()


def torch_dtype_from_name(name: str):
    if name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def normalize_messages(value: Any) -> list[dict[str, str]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [{"role": str(item["role"]), "content": str(item["content"])} for item in value]
    raise TypeError(f"unsupported prompt value: {type(value)!r}")


def build_prompt_dataset(
    train_file: str,
    tokenizer,
    num_prompts: int,
    prompt_offset: int,
    max_prompt_length: int,
) -> list[dict[str, torch.Tensor]]:
    dataframe = pd.read_parquet(train_file)
    prompts: list[dict[str, torch.Tensor]] = []
    skipped_long = 0

    for _, row in dataframe.iloc[prompt_offset:].iterrows():
        messages = normalize_messages(row["prompt"])
        token_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        if len(token_ids) > max_prompt_length:
            skipped_long += 1
            continue

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        prompts.append(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        )
        if len(prompts) >= num_prompts:
            break

    if not prompts:
        raise RuntimeError(
            f"no prompts selected from {train_file}; skipped_long={skipped_long}, max_prompt_length={max_prompt_length}"
        )

    print(
        f"Selected {len(prompts)} prompts from {train_file}; "
        f"skipped_long={skipped_long}, max_prompt_length={max_prompt_length}"
    )
    return prompts


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device requested but torch.cuda.is_available() is False")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = build_prompt_dataset(
        train_file=args.train_file,
        tokenizer=tokenizer,
        num_prompts=args.num_prompts,
        prompt_offset=args.prompt_offset,
        max_prompt_length=args.max_prompt_length,
    )

    print(f"Loading policy model from {args.model_path} on {device} with dtype={args.torch_dtype}")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch_dtype_from_name(args.torch_dtype),
    ).to(device)

    config = OfflineDraftTrainingConfig(
        max_new_tokens=args.max_new_tokens,
        num_rollouts_per_prompt=args.num_rollouts_per_prompt,
        prompt_batch_size=args.prompt_batch_size,
        draft_batch_size=args.draft_batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        hidden_multiplier=args.hidden_multiplier,
    )

    draft_model = train_offline_draft_model(
        policy_model=policy_model,
        dataset_of_prompts=prompts,
        tokenizer=None,
        config=config,
        device=device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": draft_model.cpu().state_dict(),
        "hidden_size": draft_model.hidden_size,
        "vocab_size": draft_model.vocab_size,
        "config": vars(config),
        "metrics": draft_model.offline_init_metrics,
        "model_path": args.model_path,
        "train_file": args.train_file,
        "num_prompts": len(prompts),
    }
    torch.save(checkpoint, output_dir / "draft_model.pt")
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(
            {k: v for k, v in checkpoint.items() if k not in {"state_dict"}},
            f,
            indent=2,
            default=str,
        )

    print(f"Saved draft checkpoint: {output_dir / 'draft_model.pt'}")
    print(f"Saved metadata: {output_dir / 'metadata.json'}")
    print(f"Metrics: {draft_model.offline_init_metrics}")


if __name__ == "__main__":
    main()
