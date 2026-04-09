#!/usr/bin/env python3
"""Compute GPU Bubble Ratio across the entire training process.

BubbleRatio = ∑_k (Q - r_k) * Δt_k / (T * Q)
  Q: number of GPUs
  r_k: effective busy GPUs at time k = sum(gpu_util_i / 100)
  Δt_k: time slice duration
  T: total training time
GPU_Utilization = 1 - BubbleRatio

Also breaks down bubble ratio by phase (gen, reward, adv, update_actor, gap, idle).
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PHASE_DEFS = [
    ("PhaseEvent.ROLLOUT_PHASE_START", "PhaseEvent.ROLLOUT_PHASE_END", "gen", "gen"),
    ("PhaseEvent.REWARD_CALC_START", "PhaseEvent.REWARD_CALC_END", "reward", "reward"),
    ("PhaseEvent.BATCH_PREP_START", "PhaseEvent.BATCH_PREP_END", "adv", "adv"),
    ("PhaseEvent.FORWARD_START", "PhaseEvent.BACKWARD_END", "update_actor", "update_actor"),
    ("PhaseEvent.REWARD_CALC_START", "PhaseEvent.REWARD_CALC_END", "testing", "testing"),
]

PHASE_LABELS = {
    "gen": "Rollout (Gen)",
    "reward": "Reward Calc",
    "adv": "Advantage Calc",
    "update_actor": "Actor Update",
    "testing": "Testing",
    "idle": "Idle / Overhead",
}

PHASE_COLORS = {
    "gen": "#4C72B0",
    "reward": "#DD8452",
    "adv": "#55A868",
    "update_actor": "#C44E52",
    "testing": "#8172B3",
    "idle": "#CCCCCC",
}


def build_intervals(events: pd.DataFrame) -> list[dict]:
    """Build sorted list of (start, end, phase) intervals."""
    intervals = []
    for start_evt, end_evt, role, phase_name in PHASE_DEFS:
        starts = events[(events["event_type"] == start_evt) & (events["role"] == role)]["timestamp"].values
        ends = events[(events["event_type"] == end_evt) & (events["role"] == role)]["timestamp"].values
        n = min(len(starts), len(ends))
        for i in range(n):
            intervals.append({"start": starts[i], "end": ends[i], "phase": phase_name})
    intervals.sort(key=lambda x: x["start"])
    return intervals


def assign_phase(ts: float, intervals: list[dict]) -> str:
    """Binary search for phase assignment."""
    lo, hi = 0, len(intervals) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        iv = intervals[mid]
        if ts < iv["start"]:
            hi = mid - 1
        elif ts > iv["end"]:
            lo = mid + 1
        else:
            return iv["phase"]
    return "idle"


def compute_bubble_by_phase(metrics: pd.DataFrame, intervals: list[dict], Q: int) -> dict:
    """Compute bubble ratio overall and per phase."""
    # Group by timestamp
    grouped = metrics.groupby("timestamp").agg(
        r_k=("gpu_utilization", lambda x: x.sum() / 100.0),
    ).sort_index()

    timestamps = grouped.index.values
    r_k = grouped["r_k"].values

    # Assign phase to each timestamp
    phases = np.array([assign_phase(ts, intervals) for ts in timestamps])

    # Compute dt
    dt = np.zeros(len(timestamps))
    if len(timestamps) > 1:
        gaps = np.diff(timestamps)
        dt[0] = gaps[0]
        dt[-1] = gaps[-1]
        dt[1:-1] = (gaps[:-1] + gaps[1:]) / 2.0

    T_total = dt.sum()
    bubble_total = np.sum((Q - r_k) * dt)

    results = {
        "overall": {
            "T": T_total,
            "bubble_ratio": bubble_total / (T_total * Q),
            "utilization": 1.0 - bubble_total / (T_total * Q),
            "mean_r_k": np.average(r_k, weights=dt),
        }
    }

    # Per phase
    all_phases = ["gen", "reward", "adv", "update_actor", "testing", "idle"]
    for phase in all_phases:
        mask = phases == phase
        if not mask.any():
            continue
        T_phase = dt[mask].sum()
        bubble_phase = np.sum((Q - r_k[mask]) * dt[mask])
        results[phase] = {
            "T": T_phase,
            "time_pct": T_phase / T_total * 100,
            "bubble_ratio": bubble_phase / (T_phase * Q) if T_phase > 0 else 0,
            "utilization": 1.0 - bubble_phase / (T_phase * Q) if T_phase > 0 else 0,
            "mean_r_k": np.average(r_k[mask], weights=dt[mask]) if T_phase > 0 else 0,
            "bubble_contribution": bubble_phase / (T_total * Q) * 100,  # contribution to overall bubble
        }

    return results


def print_summary(results: dict, Q: int):
    overall = results["overall"]
    print(f"\n{'='*80}")
    print(f"GPU Bubble Analysis — Full Training")
    print(f"{'='*80}")
    print(f"  GPUs (Q):            {Q}")
    print(f"  Total time:          {overall['T']:.1f}s ({overall['T']/60:.1f}min)")
    print(f"  Mean r_k:            {overall['mean_r_k']:.2f} / {Q} GPUs")
    print(f"  Overall Bubble:      {overall['bubble_ratio']*100:.2f}%")
    print(f"  Overall Utilization: {overall['utilization']*100:.2f}%")

    print(f"\n{'Phase':<18} {'Time%':>7} {'Bubble%':>9} {'Util%':>8} {'r_k':>6} {'Contrib':>9}")
    print("-" * 62)
    phase_order = ["gen", "update_actor", "reward", "adv", "testing", "idle"]
    for phase in phase_order:
        if phase not in results:
            continue
        r = results[phase]
        print(f"{PHASE_LABELS[phase]:<18} {r['time_pct']:>6.1f}% "
              f"{r['bubble_ratio']*100:>8.2f}% {r['utilization']*100:>7.2f}% "
              f"{r['mean_r_k']:>5.2f} {r['bubble_contribution']:>8.2f}%")
    print(f"\n  Contrib = contribution to overall bubble ratio (sums to {overall['bubble_ratio']*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="GPU Bubble Ratio analysis for full training")
    parser.add_argument("log_dir", type=str, help="Path to log directory")
    parser.add_argument("-o", "--output-dir", type=str, default=None)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.output_dir) if args.output_dir else log_dir
    experiment_name = log_dir.name
    events = pd.read_csv(log_dir / "gpu_events.csv")
    metrics = pd.read_csv(log_dir / "gpu_metrics.csv")
    Q = metrics["gpu_id"].nunique()

    print(f"Loaded {len(events)} events, {len(metrics)} metric rows ({Q} GPUs)")

    intervals = build_intervals(events)
    results = compute_bubble_by_phase(metrics, intervals, Q)

    print_summary(results, Q)
    import json
    with open(out_dir / f"{experiment_name}_bubble_ratio_by_phase.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
