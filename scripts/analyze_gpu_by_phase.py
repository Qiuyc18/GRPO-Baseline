#!/usr/bin/env python3
"""Analyze GPU utilization by training phase from gpu_events.csv and gpu_metrics.csv."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Phase definitions: (start_event, end_event, role) -> phase_name
PHASE_DEFS = [
    ("PhaseEvent.ROLLOUT_PHASE_START", "PhaseEvent.ROLLOUT_PHASE_END", "gen", "gen"),
    ("PhaseEvent.REWARD_CALC_START", "PhaseEvent.REWARD_CALC_END", "reward", "reward"),
    ("PhaseEvent.BATCH_PREP_START", "PhaseEvent.BATCH_PREP_END", "adv", "adv"),
    ("PhaseEvent.FORWARD_START", "PhaseEvent.BACKWARD_END", "update_actor", "update_actor"),
    ("PhaseEvent.REWARD_CALC_START", "PhaseEvent.REWARD_CALC_END", "testing", "testing"),
]

PHASE_COLORS = {
    "gen": "#4C72B0",
    "reward": "#DD8452",
    "adv": "#55A868",
    "update_actor": "#C44E52",
    "testing": "#8172B3",
    "idle": "#CCCCCC",
}

PHASE_LABELS = {
    "gen": "Rollout (Gen)",
    "reward": "Reward Calc",
    "adv": "Advantage Calc",
    "update_actor": "Actor Update",
    "testing": "Testing",
    "idle": "Idle",
}


def build_intervals(events: pd.DataFrame) -> list[dict]:
    """Pair START/END events into (start, end, phase) intervals."""
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
    """Binary search to find which phase a timestamp belongs to."""
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


def compute_stats(metrics: pd.DataFrame, intervals: list[dict], total_duration: float):
    """Compute per-phase time and GPU utilization statistics."""
    # Phase durations
    phase_durations = {}
    for iv in intervals:
        p = iv["phase"]
        phase_durations[p] = phase_durations.get(p, 0.0) + (iv["end"] - iv["start"])
    idle_time = total_duration - sum(phase_durations.values())
    phase_durations["idle"] = max(idle_time, 0.0)

    # Assign each metric row to a phase
    timestamps = metrics["timestamp"].values
    phases = np.array([assign_phase(ts, intervals) for ts in timestamps])
    metrics = metrics.copy()
    metrics["phase"] = phases

    # Per-phase GPU utilization stats
    phase_stats = {}
    for phase in list(PHASE_LABELS.keys()):
        mask = metrics["phase"] == phase
        subset = metrics.loc[mask, "gpu_utilization"]
        if len(subset) == 0:
            phase_stats[phase] = {"mean": 0, "p50": 0, "p95": 0, "mem_mean": 0, "count": 0}
        else:
            mem_subset = metrics.loc[mask, "memory_utilization"]
            phase_stats[phase] = {
                "mean": subset.mean(),
                "p50": subset.median(),
                "p95": subset.quantile(0.95),
                "mem_mean": mem_subset.mean(),
                "count": len(subset),
            }

    return phase_durations, phase_stats


def print_summary(phase_durations: dict, phase_stats: dict, total_duration: float):
    """Print summary table to stdout."""
    print(f"\n{'='*80}")
    print(f"Total training duration: {total_duration:.1f}s ({total_duration/60:.1f}min)")
    print(f"{'='*80}\n")

    header = f"{'Phase':<16} {'Duration':>10} {'Time%':>7} {'GPU Mean':>9} {'GPU P50':>8} {'GPU P95':>8} {'Mem%':>7}"
    print(header)
    print("-" * len(header))

    ordered = ["gen", "update_actor", "reward", "adv", "testing", "idle"]
    for phase in ordered:
        dur = phase_durations.get(phase, 0)
        pct = dur / total_duration * 100 if total_duration > 0 else 0
        s = phase_stats.get(phase, {})
        print(
            f"{PHASE_LABELS[phase]:<16} {dur:>8.1f}s {pct:>6.1f}% "
            f"{s.get('mean', 0):>8.1f}% {s.get('p50', 0):>7.1f}% "
            f"{s.get('p95', 0):>7.1f}% {s.get('mem_mean', 0):>6.1f}%"
        )

    # Overall
    active_phases = [p for p in ordered if p != "idle"]
    active_dur = sum(phase_durations.get(p, 0) for p in active_phases)
    print(f"\nActive GPU time ratio: {active_dur / total_duration * 100:.1f}%")


def plot_charts(phase_durations: dict, phase_stats: dict, total_duration: float, output_dir: Path):
    """Generate bar charts."""
    ordered = ["gen", "update_actor", "reward", "adv", "testing", "idle"]
    # Filter out phases with 0 duration
    ordered = [p for p in ordered if phase_durations.get(p, 0) > 0]
    labels = [PHASE_LABELS[p] for p in ordered]
    colors = [PHASE_COLORS[p] for p in ordered]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Time distribution
    ax = axes[0]
    durations = [phase_durations.get(p, 0) for p in ordered]
    pcts = [d / total_duration * 100 for d in durations]
    bars = ax.bar(labels, pcts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Time (%)")
    ax.set_title("Time Distribution by Phase")
    ax.set_ylim(0, max(pcts) * 1.15)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", rotation=30)

    # 2. GPU utilization (mean / P50 / P95)
    ax = axes[1]
    x = np.arange(len(ordered))
    w = 0.25
    means = [phase_stats.get(p, {}).get("mean", 0) for p in ordered]
    p50s = [phase_stats.get(p, {}).get("p50", 0) for p in ordered]
    p95s = [phase_stats.get(p, {}).get("p95", 0) for p in ordered]
    ax.bar(x - w, means, w, label="Mean", color=colors, alpha=0.8, edgecolor="white")
    ax.bar(x, p50s, w, label="P50", color=colors, alpha=1.0, edgecolor="white")
    ax.bar(x + w, p95s, w, label="P95", color=colors, alpha=0.5, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.set_ylabel("GPU Utilization (%)")
    ax.set_title("GPU Utilization by Phase")
    ax.legend()
    ax.set_ylim(0, 105)

    # 3. Memory utilization
    ax = axes[2]
    mems = [phase_stats.get(p, {}).get("mem_mean", 0) for p in ordered]
    bars = ax.bar(labels, mems, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Memory Utilization (%)")
    ax.set_title("Memory Utilization by Phase")
    ax.set_ylim(0, max(mems) * 1.15 if max(mems) > 0 else 100)
    for bar, m in zip(bars, mems):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{m:.1f}%",
                ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out_path = output_dir / "gpu_analysis.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nChart saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze GPU utilization by training phase")
    parser.add_argument("log_dir", type=str, help="Path to log directory containing gpu_events.csv and gpu_metrics.csv")
    parser.add_argument("--no-plot", action="store_true", help="Skip chart generation")
    parser.add_argument("-o", "--output-dir", type=str, default=None,
                        help="Directory to save the chart (default: same as log_dir)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    events_path = log_dir / "gpu_events.csv"
    metrics_path = log_dir / "gpu_metrics.csv"

    for p in [events_path, metrics_path]:
        if not p.exists():
            print(f"Error: {p} not found", file=sys.stderr)
            sys.exit(1)

    print("Loading data...")
    events = pd.read_csv(events_path)
    metrics = pd.read_csv(metrics_path)

    print(f"  Events: {len(events)} rows, Metrics: {len(metrics)} rows ({metrics['gpu_id'].nunique()} GPUs)")

    intervals = build_intervals(events)
    total_duration = events["timestamp"].max() - events["timestamp"].min()

    print("Computing statistics...")
    phase_durations, phase_stats = compute_stats(metrics, intervals, total_duration)

    print_summary(phase_durations, phase_stats, total_duration)

    if not args.no_plot:
        out_dir = Path(args.output_dir) if args.output_dir else log_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_charts(phase_durations, phase_stats, total_duration, out_dir)


if __name__ == "__main__":
    main()
