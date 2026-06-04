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


def build_training_intervals(events: pd.DataFrame, cells: pd.DataFrame) -> list[tuple[float, float]]:
    """Use explicit training start/end events when present."""
    starts = events[
        (events["event_type"] == "PhaseEvent.STEP_START") & (events["role"] == "training_start")
    ]["timestamp"].values
    ends = events[
        (events["event_type"] == "PhaseEvent.STEP_END") & (events["role"] == "training_end")
    ]["timestamp"].values
    if len(starts) > 0 and len(ends) > 0:
        return [(float(starts[0]), float(ends[-1]))]
    if cells.empty:
        return []
    return [(float(cells["start"].min()), float(cells["end"].max()))]


def build_metric_cells(metrics: pd.DataFrame) -> pd.DataFrame:
    """Convert point samples into midpoint-bounded time cells."""
    grouped = metrics.groupby("timestamp").agg(
        r_k=("gpu_utilization", lambda x: x.sum() / 100.0),
    ).sort_index()

    timestamps = grouped.index.values
    if len(timestamps) == 0:
        return pd.DataFrame(columns=["start", "end", "timestamp", "r_k"])

    if len(timestamps) > 1:
        midpoints = (timestamps[:-1] + timestamps[1:]) / 2.0
        starts = np.empty(len(timestamps))
        ends = np.empty(len(timestamps))
        starts[0] = timestamps[0] - (timestamps[1] - timestamps[0]) / 2.0
        starts[1:] = midpoints
        ends[:-1] = midpoints
        ends[-1] = timestamps[-1] + (timestamps[-1] - timestamps[-2]) / 2.0
    else:
        starts = np.array([timestamps[0] - 0.5])
        ends = np.array([timestamps[0] + 0.5])

    return pd.DataFrame({
        "start": starts,
        "end": ends,
        "timestamp": timestamps,
        "r_k": grouped["r_k"].values,
    })


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping intervals."""
    sorted_intervals = sorted((float(s), float(e)) for s, e in intervals if e > s)
    merged = []
    for s, e in sorted_intervals:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(s, e) for s, e in merged]


def subtract_intervals(
    base_intervals: list[tuple[float, float]],
    remove_intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return base intervals after removing all overlapping phase intervals."""
    remove = merge_intervals(remove_intervals)
    result = []
    for base_start, base_end in merge_intervals(base_intervals):
        cursor = base_start
        for rem_start, rem_end in remove:
            if rem_end <= cursor:
                continue
            if rem_start >= base_end:
                break
            if rem_start > cursor:
                result.append((cursor, min(rem_start, base_end)))
            cursor = max(cursor, rem_end)
            if cursor >= base_end:
                break
        if cursor < base_end:
            result.append((cursor, base_end))
    return result


def integrate_bubble(cells: pd.DataFrame, intervals: list[tuple[float, float]], Q: int) -> dict:
    """Compute bubble by intersecting metric cells with intervals."""
    if cells.empty or not intervals:
        return {"T": 0.0, "bubble_sum": 0.0, "bubble_ratio": 0.0, "utilization": 0.0, "mean_r_k": 0.0}

    intervals = merge_intervals(intervals)
    starts = cells["start"].values
    ends = cells["end"].values
    r_k = cells["r_k"].values

    T = 0.0
    busy_sum = 0.0
    bubble_sum = 0.0
    j = 0
    for s, e in intervals:
        while j < len(cells) and ends[j] <= s:
            j += 1
        k = j
        while k < len(cells) and starts[k] < e:
            overlap = max(0.0, min(ends[k], e) - max(starts[k], s))
            if overlap > 0:
                T += overlap
                busy_sum += r_k[k] * overlap
                bubble_sum += (Q - r_k[k]) * overlap
            k += 1

    bubble_ratio = bubble_sum / (T * Q) if T > 0 else 0.0
    return {
        "T": float(T),
        "bubble_sum": float(bubble_sum),
        "bubble_ratio": float(bubble_ratio),
        "utilization": float(1.0 - bubble_ratio),
        "mean_r_k": float(busy_sum / T) if T > 0 else 0.0,
    }


def compute_bubble_by_phase(metrics: pd.DataFrame, events: pd.DataFrame, intervals: list[dict], Q: int) -> dict:
    """Compute bubble ratio overall and per phase."""
    cells = build_metric_cells(metrics)
    training_intervals = build_training_intervals(events, cells)
    overall = integrate_bubble(cells, training_intervals, Q)
    T_total = overall["T"]

    results = {"overall": {
        "T": overall["T"],
        "bubble_ratio": overall["bubble_ratio"],
        "utilization": overall["utilization"],
        "mean_r_k": overall["mean_r_k"],
    }}

    phase_intervals = {phase: [] for phase in ["gen", "reward", "adv", "update_actor", "testing"]}
    for interval in intervals:
        phase_intervals.setdefault(interval["phase"], []).append((interval["start"], interval["end"]))

    explicit_intervals = []
    for phase in ["gen", "reward", "adv", "update_actor", "testing"]:
        explicit_intervals.extend(phase_intervals.get(phase, []))
    phase_intervals["idle"] = subtract_intervals(training_intervals, explicit_intervals)

    for phase in ["gen", "reward", "adv", "update_actor", "testing", "idle"]:
        result = integrate_bubble(cells, phase_intervals.get(phase, []), Q)
        if result["T"] <= 0:
            continue
        results[phase] = {
            "T": result["T"],
            "time_pct": result["T"] / T_total * 100 if T_total > 0 else 0.0,
            "bubble_ratio": result["bubble_ratio"],
            "utilization": result["utilization"],
            "mean_r_k": result["mean_r_k"],
            "bubble_contribution": result["bubble_sum"] / (T_total * Q) * 100 if T_total > 0 else 0.0,
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
    results = compute_bubble_by_phase(metrics, events, intervals, Q)

    print_summary(results, Q)
    import json
    try:
        with open(out_dir / f"{experiment_name}_bubble_ratio_by_phase.json", "w") as f:
            json.dump(results, f)
    except PermissionError as error:
        print(f"\nWarning: could not write JSON output: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
