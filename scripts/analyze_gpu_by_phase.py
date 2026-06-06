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
    "unlabeled": "#CCCCCC",
}

PHASE_LABELS = {
    "gen": "Rollout (Gen)",
    "reward": "Reward Calc",
    "adv": "Advantage Calc",
    "update_actor": "Actor Update",
    "testing": "Testing",
    "unlabeled": "Unlabeled / Overhead",
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
    """Return base intervals after removing all labeled phase intervals."""
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


def build_metric_cells(metrics: pd.DataFrame) -> pd.DataFrame:
    """Convert point samples into midpoint-bounded time cells."""
    timestamps = np.sort(metrics["timestamp"].unique())
    if len(timestamps) == 0:
        return pd.DataFrame(columns=["timestamp", "start", "end"])

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

    return pd.DataFrame({"timestamp": timestamps, "start": starts, "end": ends})


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


def interval_duration(intervals: list[tuple[float, float]]) -> float:
    """Total duration after merging intervals."""
    return sum(e - s for s, e in merge_intervals(intervals))


def overlap_weights(cells: pd.DataFrame, intervals: list[tuple[float, float]]) -> pd.DataFrame:
    """Return timestamp weights equal to overlap duration with intervals."""
    if cells.empty or not intervals:
        return pd.DataFrame(columns=["timestamp", "weight"])

    intervals = merge_intervals(intervals)
    starts = cells["start"].values
    ends = cells["end"].values
    timestamps = cells["timestamp"].values
    weights = np.zeros(len(cells), dtype=float)

    j = 0
    for s, e in intervals:
        while j < len(cells) and ends[j] <= s:
            j += 1
        k = j
        while k < len(cells) and starts[k] < e:
            weights[k] += max(0.0, min(ends[k], e) - max(starts[k], s))
            k += 1

    mask = weights > 0
    return pd.DataFrame({"timestamp": timestamps[mask], "weight": weights[mask]})


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Compute a weighted quantile for non-empty arrays."""
    if len(values) == 0 or weights.sum() <= 0:
        return 0.0
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cdf = np.cumsum(sorted_weights)
    cutoff = q * sorted_weights.sum()
    return float(sorted_values[np.searchsorted(cdf, cutoff, side="left")])


def compute_stats(
    events: pd.DataFrame,
    metrics: pd.DataFrame,
    intervals: list[dict],
):
    """Compute per-phase time and GPU utilization statistics."""
    cells = build_metric_cells(metrics)
    training_intervals = build_training_intervals(events, cells)
    total_duration = interval_duration(training_intervals)

    phase_intervals = {phase: [] for phase in ["gen", "reward", "adv", "update_actor", "testing"]}
    for iv in intervals:
        phase_intervals.setdefault(iv["phase"], []).append((iv["start"], iv["end"]))

    labeled_intervals = []
    for phase in ["gen", "reward", "adv", "update_actor", "testing"]:
        labeled_intervals.extend(phase_intervals.get(phase, []))
    phase_intervals["unlabeled"] = subtract_intervals(training_intervals, labeled_intervals)

    phase_durations = {
        phase: interval_duration(phase_intervals.get(phase, []))
        for phase in PHASE_LABELS
    }

    # Per-phase GPU utilization stats
    phase_stats = {}
    for phase in PHASE_LABELS:
        weights = overlap_weights(cells, phase_intervals.get(phase, []))
        if weights.empty:
            phase_stats[phase] = {"mean": 0, "p50": 0, "p95": 0, "mem_mean": 0, "count": 0}
            continue

        weighted = metrics.merge(weights, on="timestamp", how="inner")
        util = weighted["gpu_utilization"].to_numpy(dtype=float)
        mem = weighted["memory_utilization"].to_numpy(dtype=float)
        sample_weights = weighted["weight"].to_numpy(dtype=float)

        phase_stats[phase] = {
            "mean": float(np.average(util, weights=sample_weights)),
            "p50": weighted_quantile(util, sample_weights, 0.50),
            "p95": weighted_quantile(util, sample_weights, 0.95),
            "mem_mean": float(np.average(mem, weights=sample_weights)),
            "count": len(weighted),
        }

    return phase_durations, phase_stats, total_duration


def print_summary(phase_durations: dict, phase_stats: dict, total_duration: float):
    """Print summary table to stdout."""
    print(f"\n{'='*80}")
    print(f"Total training duration: {total_duration:.1f}s ({total_duration/60:.1f}min)")
    print(f"{'='*80}\n")

    header = f"{'Phase':<16} {'Duration':>10} {'Time%':>7} {'GPU Mean':>9} {'GPU P50':>8} {'GPU P95':>8} {'Mem%':>7}"
    print(header)
    print("-" * len(header))

    ordered = ["gen", "update_actor", "reward", "adv", "testing", "unlabeled"]
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
    active_phases = [p for p in ordered if p != "unlabeled"]
    active_dur = sum(phase_durations.get(p, 0) for p in active_phases)
    print(f"\nLabeled phase time ratio: {active_dur / total_duration * 100:.1f}%")
    print("Unlabeled / Overhead is time not covered by known phase events; it is not GPU idle.")


def plot_charts(phase_durations: dict, phase_stats: dict, total_duration: float, output_dir: Path):
    """Generate bar charts."""
    ordered = ["gen", "update_actor", "reward", "adv", "testing", "unlabeled"]
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
    print("Computing statistics...")
    phase_durations, phase_stats, total_duration = compute_stats(events, metrics, intervals)

    print_summary(phase_durations, phase_stats, total_duration)

    if not args.no_plot:
        out_dir = Path(args.output_dir) if args.output_dir else log_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_charts(phase_durations, phase_stats, total_duration, out_dir)


if __name__ == "__main__":
    main()
