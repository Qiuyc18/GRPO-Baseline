#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import math
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

try:
    import wandb
except Exception:
    wandb = None


@dataclass
class SampleStat:
    index: int
    prompt_len: int
    rollout_lens: List[int]
    mean_rollout_len: float
    max_rollout_len: int
    min_rollout_len: int


def parse_args():
    parser = argparse.ArgumentParser(description="Pure rollout evaluation with optional W&B logging.")

    # I/O
    parser.add_argument("--model-path", type=str, default=os.environ.get("MODEL_PATH", ""))
    parser.add_argument("--data-path", type=str, default=os.environ.get("DATA_PATH", ""))
    parser.add_argument("--output-json", type=str, default=os.environ.get("OUTPUT_JSON", "pure_rollout_stats.json"))
    parser.add_argument("--output-jsonl", type=str, default=os.environ.get("OUTPUT_JSONL", "pure_rollout_samples.jsonl"))

    # Data / sampling
    parser.add_argument("--num-prompts", type=int, default=int(os.environ.get("NUM_PROMPTS", 512)))
    parser.add_argument("--shuffle", action="store_true", default=os.environ.get("SHUFFLE", "1") == "1")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 42)))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", 128)))
    parser.add_argument("--n-rollouts", type=int, default=int(os.environ.get("N_ROLLOUTS", 4)))
    parser.add_argument("--max-response-length", type=int, default=int(os.environ.get("MAX_RESPONSE_LENGTH", 8192)))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", 1.0)))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", 1.0)))
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("TOP_K", -1)))
    parser.add_argument("--presence-penalty", type=float, default=float(os.environ.get("PRESENCE_PENALTY", 0.0)))
    parser.add_argument("--frequency-penalty", type=float, default=float(os.environ.get("FREQUENCY_PENALTY", 0.0)))
    parser.add_argument("--repetition-penalty", type=float, default=float(os.environ.get("REPETITION_PENALTY", 1.0)))
    parser.add_argument("--stop", type=str, nargs="*", default=None)

    # vLLM
    parser.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("TP_SIZE", 4)))
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(os.environ.get("GPU_MEMORY_UTILIZATION", 0.5)))
    parser.add_argument("--max-num-seqs", type=int, default=int(os.environ.get("MAX_NUM_SEQS", 256)))
    parser.add_argument("--trust-remote-code", action="store_true", default=os.environ.get("TRUST_REMOTE_CODE", "1") == "1")
    parser.add_argument("--dtype", type=str, default=os.environ.get("DTYPE", "auto"))
    parser.add_argument("--enforce-eager", action="store_true", default=os.environ.get("ENFORCE_EAGER", "0") == "1")
    parser.add_argument("--disable-log-stats", action="store_true", default=os.environ.get("DISABLE_LOG_STATS", "0") == "1")

    # W&B
    parser.add_argument("--use-wandb", action="store_true", default=os.environ.get("USE_WANDB", "1") == "1")
    parser.add_argument("--wandb-project", type=str, default=os.environ.get("WANDB_PROJECT", ""))
    parser.add_argument("--wandb-entity", type=str, default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-name", type=str, default=os.environ.get("WANDB_NAME", "pure_rollout_eval"))
    parser.add_argument("--wandb-group", type=str, default=os.environ.get("WANDB_GROUP", ""))
    parser.add_argument("--wandb-job-type", type=str, default=os.environ.get("WANDB_JOB_TYPE", "rollout_eval"))
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=os.environ.get("WANDB_TAGS", "").split(",") if os.environ.get("WANDB_TAGS") else [])
    parser.add_argument("--wandb-mode", type=str, default=os.environ.get("WANDB_MODE", "online"))

    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def percentile(values: List[float], p: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(values, p))


def ensure_parent_dir(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def init_wandb(args, run_config: Dict[str, Any]):
    if not args.use_wandb:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed, but --use-wandb is enabled.")

    wandb_kwargs = {
        "project": args.wandb_project or None,
        "entity": args.wandb_entity or None,
        "name": args.wandb_name or None,
        "group": args.wandb_group or None,
        "job_type": args.wandb_job_type or None,
        "tags": [x for x in args.wandb_tags if x],
        "mode": args.wandb_mode,
        "config": run_config,
    }
    wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}
    return wandb.init(**wandb_kwargs)


def load_dataframe(data_path: str, num_prompts: int, shuffle: bool, seed: int) -> pd.DataFrame:
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    elif data_path.endswith(".jsonl"):
        df = pd.read_json(data_path, lines=True)
    elif data_path.endswith(".json"):
        df = pd.read_json(data_path)
    elif data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if num_prompts > 0:
        df = df.iloc[:num_prompts].reset_index(drop=True)

    return df


def extract_raw_prompt(row: pd.Series) -> str:
    """
    TODO:
    1. 在这里读取你的数据字段
    2. 返回原始题面 / prompt 文本

    示例:
        return str(row["problem"])
        return str(row["question"])
        return str(row["prompt"])
    """
    raise NotImplementedError("Please implement extract_raw_prompt(row) for your dataset.")


def build_prompt(raw_prompt: str, tokenizer: AutoTokenizer) -> str:
    """
    TODO:
    1. 在这里填你的 prompt 模板
    2. 需要的话，用 tokenizer.apply_chat_template(...)
    3. 返回最终送给模型的完整 prompt

    示例:
        messages = [{"role": "user", "content": raw_prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    或者:
        return f\"Solve the following problem carefully:\\n\\n{raw_prompt}\\n\"
    """
    raise NotImplementedError("Please implement build_prompt(raw_prompt, tokenizer).")


def prepare_prompts(df: pd.DataFrame, tokenizer: AutoTokenizer):
    prompts: List[str] = []
    prompt_lens: List[int] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building prompts"):
        raw_prompt = extract_raw_prompt(row)
        prompt = build_prompt(raw_prompt, tokenizer)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        prompts.append(prompt)
        prompt_lens.append(prompt_len)

    return prompts, prompt_lens


def make_llm(args):
    llm_kwargs = {
        "model": args.model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "dtype": args.dtype,
        "enforce_eager": args.enforce_eager,
        "disable_log_stats": args.disable_log_stats,
        "max_num_seqs": args.max_num_seqs,
    }
    return LLM(**llm_kwargs)


def make_sampling_params(args):
    return SamplingParams(
        n=args.n_rollouts,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_response_length,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        repetition_penalty=args.repetition_penalty,
        stop=args.stop,
    )


def summarize_lengths(lengths: List[int]) -> Dict[str, float]:
    if len(lengths) == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": 0,
            "min": 0,
        }
    return {
        "count": int(len(lengths)),
        "mean": float(np.mean(lengths)),
        "p50": percentile(lengths, 50),
        "p75": percentile(lengths, 75),
        "p90": percentile(lengths, 90),
        "p95": percentile(lengths, 95),
        "p99": percentile(lengths, 99),
        "max": int(max(lengths)),
        "min": int(min(lengths)),
    }


def simulate_sync_tail(per_instance_mean_lens: List[float], batch_size: int) -> Dict[str, float]:
    batch_idle_ratios = []
    batch_max_lens = []
    batch_mean_lens = []

    for start in range(0, len(per_instance_mean_lens), batch_size):
        chunk = per_instance_mean_lens[start:start + batch_size]
        if not chunk:
            continue

        mx = max(chunk)
        avg = float(np.mean(chunk))
        idle = 0.0 if mx <= 0 else 1.0 - (sum(chunk) / (len(chunk) * mx))

        batch_idle_ratios.append(idle)
        batch_max_lens.append(mx)
        batch_mean_lens.append(avg)

    return {
        "num_sync_batches": int(len(batch_idle_ratios)),
        "batch_idle_ratio_mean": float(np.mean(batch_idle_ratios)) if batch_idle_ratios else 0.0,
        "batch_idle_ratio_p50": percentile(batch_idle_ratios, 50),
        "batch_idle_ratio_p95": percentile(batch_idle_ratios, 95),
        "batch_max_len_mean": float(np.mean(batch_max_lens)) if batch_max_lens else 0.0,
        "batch_mean_len_mean": float(np.mean(batch_mean_lens)) if batch_mean_lens else 0.0,
    }


def log_batch_to_wandb(
    wb_run,
    batch_idx: int,
    batch_prompt_lens: List[int],
    batch_rollout_lens_flat: List[int],
    batch_per_instance_mean: List[float],
    batch_elapsed_sec: float,
):
    if wb_run is None:
        return

    payload = {
        "batch/index": batch_idx,
        "batch/elapsed_sec": batch_elapsed_sec,
        "batch/num_prompts": len(batch_prompt_lens),
        "batch/num_rollouts": len(batch_rollout_lens_flat),
        "batch/prompt_len_mean": float(np.mean(batch_prompt_lens)) if batch_prompt_lens else 0.0,
        "batch/prompt_len_p95": percentile(batch_prompt_lens, 95),
        "batch/response_len_mean": float(np.mean(batch_rollout_lens_flat)) if batch_rollout_lens_flat else 0.0,
        "batch/response_len_p50": percentile(batch_rollout_lens_flat, 50),
        "batch/response_len_p95": percentile(batch_rollout_lens_flat, 95),
        "batch/response_len_p99": percentile(batch_rollout_lens_flat, 99),
        "batch/response_len_max": int(max(batch_rollout_lens_flat)) if batch_rollout_lens_flat else 0,
        "batch/per_instance_mean_len_mean": float(np.mean(batch_per_instance_mean)) if batch_per_instance_mean else 0.0,
        "batch/per_instance_mean_len_p95": percentile(batch_per_instance_mean, 95),
    }

    try:
        payload["batch/response_len_hist"] = wandb.Histogram(batch_rollout_lens_flat)
        payload["batch/prompt_len_hist"] = wandb.Histogram(batch_prompt_lens)
        payload["batch/per_instance_mean_len_hist"] = wandb.Histogram(batch_per_instance_mean)
    except Exception:
        pass

    wb_run.log(payload)


def main():
    args = parse_args()
    seed_everything(args.seed)

    if not args.model_path:
        raise ValueError("Missing --model-path")
    if not args.data_path:
        raise ValueError("Missing --data-path")

    run_config = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "num_prompts": args.num_prompts,
        "shuffle": args.shuffle,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "n_rollouts": args.n_rollouts,
        "max_response_length": args.max_response_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "presence_penalty": args.presence_penalty,
        "frequency_penalty": args.frequency_penalty,
        "repetition_penalty": args.repetition_penalty,
        "stop": args.stop,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "trust_remote_code": args.trust_remote_code,
        "dtype": args.dtype,
        "enforce_eager": args.enforce_eager,
        "disable_log_stats": args.disable_log_stats,
        "output_json": args.output_json,
        "output_jsonl": args.output_jsonl,
    }

    wb_run = init_wandb(args, run_config)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )

    df = load_dataframe(
        data_path=args.data_path,
        num_prompts=args.num_prompts,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    prompts, prompt_lens = prepare_prompts(df, tokenizer)

    llm = make_llm(args)
    sampling_params = make_sampling_params(args)

    all_sample_stats: List[SampleStat] = []
    all_rollout_lens: List[int] = []
    all_per_instance_mean_lens: List[float] = []
    batch_elapsed_all: List[float] = []

    ensure_parent_dir(args.output_json)
    ensure_parent_dir(args.output_jsonl)

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for batch_idx, start in enumerate(range(0, len(prompts), args.batch_size)):
            end = min(start + args.batch_size, len(prompts))
            batch_prompts = prompts[start:end]
            batch_prompt_lens = prompt_lens[start:end]

            t0 = time.time()
            batch_outputs = llm.generate(batch_prompts, sampling_params)
            elapsed = time.time() - t0
            batch_elapsed_all.append(elapsed)

            batch_rollout_lens_flat: List[int] = []
            batch_per_instance_mean: List[float] = []

            for local_i, output in enumerate(batch_outputs):
                rollout_lens = [len(o.token_ids) for o in output.outputs]
                batch_rollout_lens_flat.extend(rollout_lens)
                all_rollout_lens.extend(rollout_lens)

                mean_len = float(np.mean(rollout_lens)) if rollout_lens else 0.0
                batch_per_instance_mean.append(mean_len)
                all_per_instance_mean_lens.append(mean_len)

                stat = SampleStat(
                    index=start + local_i,
                    prompt_len=batch_prompt_lens[local_i],
                    rollout_lens=rollout_lens,
                    mean_rollout_len=mean_len,
                    max_rollout_len=max(rollout_lens) if rollout_lens else 0,
                    min_rollout_len=min(rollout_lens) if rollout_lens else 0,
                )
                all_sample_stats.append(stat)

                fout.write(json.dumps(asdict(stat), ensure_ascii=False) + "\n")

            log_batch_to_wandb(
                wb_run=wb_run,
                batch_idx=batch_idx,
                batch_prompt_lens=batch_prompt_lens,
                batch_rollout_lens_flat=batch_rollout_lens_flat,
                batch_per_instance_mean=batch_per_instance_mean,
                batch_elapsed_sec=elapsed,
            )

    response_len_stats = summarize_lengths(all_rollout_lens)
    per_instance_mean_stats = summarize_lengths([int(x) for x in all_per_instance_mean_lens])
    prompt_len_stats = summarize_lengths(prompt_lens)
    sync_tail_stats = simulate_sync_tail(all_per_instance_mean_lens, args.batch_size)

    summary = {
        "config": run_config,
        "num_rows": int(len(df)),
        "num_prompts_built": int(len(prompts)),
        "prompt_len_stats": prompt_len_stats,
        "response_len_stats": response_len_stats,
        "per_instance_mean_len_stats": per_instance_mean_stats,
        "sync_tail_stats": sync_tail_stats,
        "timing": {
            "num_batches": int(len(batch_elapsed_all)),
            "batch_elapsed_sec_mean": float(np.mean(batch_elapsed_all)) if batch_elapsed_all else 0.0,
            "batch_elapsed_sec_p50": percentile(batch_elapsed_all, 50),
            "batch_elapsed_sec_p95": percentile(batch_elapsed_all, 95),
            "batch_elapsed_sec_max": float(max(batch_elapsed_all)) if batch_elapsed_all else 0.0,
            "total_elapsed_sec": float(sum(batch_elapsed_all)),
        },
        "top_longest_samples": [
            asdict(x)
            for x in sorted(all_sample_stats, key=lambda s: s.mean_rollout_len, reverse=True)[:20]
        ],
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if wb_run is not None:
        final_log = {
            "final/num_rows": summary["num_rows"],
            "final/num_prompts_built": summary["num_prompts_built"],
            "final/prompt_len_mean": summary["prompt_len_stats"]["mean"],
            "final/prompt_len_p95": summary["prompt_len_stats"]["p95"],
            "final/response_len_mean": summary["response_len_stats"]["mean"],
            "final/response_len_p50": summary["response_len_stats"]["p50"],
            "final/response_len_p95": summary["response_len_stats"]["p95"],
            "final/response_len_p99": summary["response_len_stats"]["p99"],
            "final/response_len_max": summary["response_len_stats"]["max"],
            "final/per_instance_mean_len_mean": summary["per_instance_mean_len_stats"]["mean"],
            "final/per_instance_mean_len_p95": summary["per_instance_mean_len_stats"]["p95"],
            "final/sync_batch_idle_ratio_mean": summary["sync_tail_stats"]["batch_idle_ratio_mean"],
            "final/sync_batch_idle_ratio_p95": summary["sync_tail_stats"]["batch_idle_ratio_p95"],
            "final/batch_elapsed_sec_mean": summary["timing"]["batch_elapsed_sec_mean"],
            "final/batch_elapsed_sec_p95": summary["timing"]["batch_elapsed_sec_p95"],
            "final/batch_elapsed_sec_max": summary["timing"]["batch_elapsed_sec_max"],
            "final/total_elapsed_sec": summary["timing"]["total_elapsed_sec"],
        }

        try:
            final_log["final/response_len_hist"] = wandb.Histogram(all_rollout_lens)
            final_log["final/prompt_len_hist"] = wandb.Histogram(prompt_lens)
            final_log["final/per_instance_mean_len_hist"] = wandb.Histogram(all_per_instance_mean_lens)
        except Exception:
            pass

        wb_run.log(final_log)
        wb_run.summary.update({
            "output_json": os.path.abspath(args.output_json),
            "output_jsonl": os.path.abspath(args.output_jsonl),
        })
        wb_run.finish()

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()