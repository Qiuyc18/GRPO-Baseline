#!/usr/bin/env python3
"""Summarize GRPO replay/spec-verify experiment logs.

The script parses veRL console logs and reports timing, cache, spec-verify,
validation, and NAT token-sampling metrics. It is intentionally log-based so it
can be run after experiments without querying W&B.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as stats
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


DEFAULT_LOGS = {
    "no_replay": "logs/train_qwen3_4B-base_deepscaler_grpo_4gpu_no_replay_20260524_074312.log",
    "naive_replay_pfresh05": "logs/train_qwen3_4B-base_deepscaler_grpo_4gpu_replay_buffer_20260523_051105.log",
    "spec_verify_loose_pfresh05": "logs/train_qwen3_4B-base_deepscaler_grpo_4gpu_replay_buffer_spec_verify_20260525_034821.log",
    "spec_verify_strict_pfresh05": "logs/train_qwen3_4B-base_deepscaler_grpo_4gpu_replay_buffer_spec_verify_strict_20260525_105548.log",
    "spec_verify_strict_pfresh0": "logs/train_qwen3_4B-base_deepscaler_grpo_4gpu_replay_buffer_spec_verify_strict_pfresh0_20260526_015944.log",
    "nat_rpc_partial": "logs/train_qwen3_4B-base_deepscaler_grpo_4gpu_replay_buffer_nat_rpc_20260526_083556.log",
}


def as_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def mean(values: list[float]) -> float | None:
    return stats.mean(values) if values else None


def median(values: list[float]) -> float | None:
    return stats.median(values) if values else None


def first_group(match: re.Match[str] | None) -> str | None:
    if match is None:
        return None
    return next((g for g in match.groups() if g is not None), None)


def extract_config(clean_log: str) -> dict[str, str]:
    patterns = {
        "p_fresh": r"'p_fresh':\s*([0-9.]+)|replay_buffer\.p_fresh=([0-9.]+)",
        "spec_verify": r"'spec_verify':\s*(True|False)|replay_buffer\.spec_verify=(True|False)",
        "mean_delta": r"'spec_verify_min_mean_logprob_delta':\s*(-?[0-9.]+)|spec_verify_min_mean_logprob_delta=(-?[0-9.]+)",
        "min_seq_delta": r"'spec_verify_min_seq_logprob_delta':\s*(-?[0-9.]+)|spec_verify_min_seq_logprob_delta=(-?[0-9.]+)",
        "test_freq": r"trainer\.test_freq=([0-9]+)",
        "nat_enabled": r"token_sampling\.enabled=(True|False)|'enabled':\s*(True|False)",
        "nat_mode": r"token_sampling\.mode=([A-Za-z0-9_]+)|'mode':\s*'([^']+)'",
        "nat_keep_ratio": r"token_sampling\.keep_ratio=([0-9.]+)|'keep_ratio':\s*([0-9.]+)",
    }
    config: dict[str, str] = {}
    for key, pattern in patterns.items():
        value = first_group(re.search(pattern, clean_log, re.S))
        if value is not None:
            config[key] = value
    return config


def parse_step_line(line: str) -> dict[str, Any] | None:
    if "step:" not in line or "timing_s/step:" not in line:
        return None
    line = line[line.find("step:") :]
    parsed: dict[str, Any] = {}
    for part in [item.strip() for item in line.split(" - ")]:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        if key == "step":
            key = "step_id"
        value = value.strip()
        parsed[key] = as_float(value) if as_float(value) is not None else value
    return parsed if "step_id" in parsed else None


def parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="ignore")
    clean = ANSI_RE.sub("", text)
    steps: list[dict[str, Any]] = []
    vals: list[tuple[int, float]] = []

    for raw_line in clean.splitlines():
        parsed = parse_step_line(raw_line)
        if parsed is None:
            continue
        steps.append(parsed)
        val = parsed.get("val-core/lighteval/MATH/reward/mean@1")
        if isinstance(val, float):
            vals.append((int(parsed["step_id"]), val))

    raw = [s["timing_s/step"] for s in steps if isinstance(s.get("timing_s/step"), float)]
    testing = [s.get("timing_s/testing", 0.0) for s in steps if isinstance(s.get("timing_s/step"), float)]
    adjusted = [s["timing_s/step"] - s.get("timing_s/testing", 0.0) for s in steps if isinstance(s.get("timing_s/step"), float)]
    from_cache = [s["rollout/from_cache"] for s in steps if isinstance(s.get("rollout/from_cache"), float)]
    cache_adj = [
        s["timing_s/step"] - s.get("timing_s/testing", 0.0)
        for s in steps
        if s.get("rollout/from_cache") == 1 and isinstance(s.get("timing_s/step"), float)
    ]
    fresh_adj = [
        s["timing_s/step"] - s.get("timing_s/testing", 0.0)
        for s in steps
        if s.get("rollout/from_cache") == 0 and isinstance(s.get("timing_s/step"), float)
    ]

    attempts = sum(s.get("replay/spec_verify_attempt", 0.0) for s in steps)
    accepts = sum(s.get("replay/spec_verify_accept", 0.0) for s in steps)
    rejects = sum(s.get("replay/spec_verify_reject", 0.0) for s in steps)
    nat_keep = [s["actor/nat_keep_ratio"] for s in steps if isinstance(s.get("actor/nat_keep_ratio"), float)]
    nat_resp = [s["actor/nat_response_len_ratio"] for s in steps if isinstance(s.get("actor/nat_response_len_ratio"), float)]
    update_actor = [s["timing_s/update_actor"] for s in steps if isinstance(s.get("timing_s/update_actor"), float)]
    gen = [s["timing_s/gen"] for s in steps if isinstance(s.get("timing_s/gen"), float)]
    ref = [s["timing_s/ref"] for s in steps if isinstance(s.get("timing_s/ref"), float)]
    clip = [s["response_length/clip_ratio"] for s in steps if isinstance(s.get("response_length/clip_ratio"), float)]
    score = [s["critic/score/mean"] for s in steps if isinstance(s.get("critic/score/mean"), float)]

    return {
        "path": str(path),
        "config": extract_config(clean),
        "complete": "Final validation metrics" in clean or "Training Progress: 100%" in clean or "global_step_118" in clean,
        "failed": "Traceback" in clean or "Error executing job" in clean or "RuntimeError" in clean,
        "steps": len(steps),
        "raw_total_h": sum(raw) / 3600 if raw else None,
        "raw_mean_s": mean(raw),
        "raw_median_s": median(raw),
        "test_total_h": sum(testing) / 3600 if testing else 0,
        "test_steps": sum(1 for value in testing if value),
        "adjusted_total_h": sum(adjusted) / 3600 if adjusted else None,
        "adjusted_mean_s": mean(adjusted),
        "adjusted_median_s": median(adjusted),
        "cache_rate": sum(from_cache) / len(from_cache) if from_cache else None,
        "cache_steps": sum(1 for value in from_cache if value == 1),
        "fresh_steps": sum(1 for value in from_cache if value == 0),
        "cache_adjusted_mean_s": mean(cache_adj),
        "fresh_adjusted_mean_s": mean(fresh_adj),
        "spec_attempts": attempts,
        "spec_accepts": accepts,
        "spec_rejects": rejects,
        "spec_accept_rate": accepts / attempts if attempts else None,
        "val_count": len(vals),
        "val_best": max([value for _, value in vals]) if vals else None,
        "val_final": vals[-1][1] if vals else None,
        "val_points": vals,
        "score_mean": mean(score),
        "score_final": score[-1] if score else None,
        "clip_mean": mean(clip),
        "clip_final": clip[-1] if clip else None,
        "update_actor_mean_s": mean(update_actor),
        "gen_mean_s": mean(gen),
        "ref_mean_s": mean(ref),
        "nat_keep_mean": mean(nat_keep),
        "nat_response_len_ratio_mean": mean(nat_resp),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a compact table.")
    parser.add_argument("logs", nargs="*", help="Optional name=path entries. Uses the default experiment set when omitted.")
    args = parser.parse_args()

    if args.logs:
        log_map = dict(item.split("=", 1) for item in args.logs)
    else:
        log_map = DEFAULT_LOGS

    results = {name: parse_log(Path(path)) for name, path in log_map.items() if Path(path).exists()}
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    for name, result in results.items():
        print(f"\n=== {name} ===")
        for key in [
            "path",
            "complete",
            "failed",
            "steps",
            "raw_mean_s",
            "adjusted_mean_s",
            "test_total_h",
            "cache_rate",
            "spec_accept_rate",
            "val_best",
            "val_final",
            "update_actor_mean_s",
            "nat_keep_mean",
        ]:
            print(f"{key}: {result.get(key)}")


if __name__ == "__main__":
    main()
