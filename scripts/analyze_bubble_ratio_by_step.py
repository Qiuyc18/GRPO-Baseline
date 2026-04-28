#!/usr/bin/env python3
"""Compute GPU Bubble Ratio for Rollout (Gen) phases only.

BubbleRatio = ∑_k (Q - r_k) * Δt_k / (T * Q)
  Q: number of GPUs
  r_k: effective busy GPUs at time k = sum(gpu_util_i / 100)
  Δt_k: time slice duration
  T: total rollout time
GPU_Utilization = 1 - BubbleRatio
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_rollout_intervals(events: pd.DataFrame) -> list[tuple[float, float]]:
    """Extract (start, end) intervals for rollout phases."""
    starts = events[
        (events["event_type"] == "PhaseEvent.ROLLOUT_PHASE_START") & (events["role"] == "gen")
    ]["timestamp"].values
    ends = events[
        (events["event_type"] == "PhaseEvent.ROLLOUT_PHASE_END") & (events["role"] == "gen")
    ]["timestamp"].values
    n = min(len(starts), len(ends))
    return list(zip(starts[:n], ends[:n]))


def filter_metrics_by_intervals(metrics: pd.DataFrame, intervals: list[tuple[float, float]]) -> pd.DataFrame:
    """Keep only metrics rows whose timestamp falls within any interval."""
    mask = np.zeros(len(metrics), dtype=bool)
    ts = metrics["timestamp"].values
    for s, e in intervals:
        mask |= (ts >= s) & (ts <= e)
    return metrics[mask].copy()


def compute_bubble(metrics: pd.DataFrame, Q: int) -> dict:
    """Compute bubble ratio from filtered metrics.

    Returns dict with per-step and overall statistics.
    """
    # Group by timestamp, compute r_k = sum(gpu_util / 100) per timestamp
    grouped = metrics.groupby("timestamp").agg(
        r_k=("gpu_utilization", lambda x: x.sum() / 100.0),
        n_gpus=("gpu_id", "count"),
    ).sort_index()

    timestamps = grouped.index.values
    r_k = grouped["r_k"].values

    # Compute Δt_k (use midpoint rule: half the gap to next + half the gap to prev)
    dt = np.zeros(len(timestamps))
    if len(timestamps) > 1:
        gaps = np.diff(timestamps)
        # Cap gaps to avoid counting inter-interval dead time as a single huge slice
        median_gap = np.median(gaps)
        cap = median_gap * 3
        gaps_capped = np.minimum(gaps, cap)
        dt[0] = gaps_capped[0]
        dt[-1] = gaps_capped[-1]
        dt[1:-1] = (gaps_capped[:-1] + gaps_capped[1:]) / 2.0
    else:
        dt[0] = 1.0

    T = dt.sum()
    bubble_sum = np.sum((Q - r_k) * dt)
    bubble_ratio = bubble_sum / (T * Q)
    utilization = 1.0 - bubble_ratio

    return {
        "T": T,
        "Q": Q,
        "n_samples": len(timestamps),
        "bubble_ratio": bubble_ratio,
        "utilization": utilization,
        "mean_r_k": np.mean(r_k),
        "timestamps": timestamps,
        "r_k": r_k,
        "dt": dt,
    }


def compute_per_step_bubble(metrics: pd.DataFrame, intervals: list[tuple[float, float]], Q: int) -> pd.DataFrame:
    """Compute bubble ratio per rollout step."""
    rows = []
    for i, (s, e) in enumerate(intervals):
        ts = metrics["timestamp"].values
        mask = (ts >= s) & (ts <= e)
        step_metrics = metrics[mask]
        if len(step_metrics) == 0:
            continue
        result = compute_bubble(step_metrics, Q)
        rows.append({
            "step": i + 1,
            "duration": e - s,
            "bubble_ratio": result["bubble_ratio"],
            "utilization": result["utilization"],
            "mean_r_k": result["mean_r_k"],
        })
    return pd.DataFrame(rows)


def print_summary(overall: dict, per_step: pd.DataFrame):
    print(f"\n{'='*70}")
    print(f"Rollout Phase GPU Bubble Analysis")
    print(f"{'='*70}")
    print(f"  GPUs (Q):            {overall['Q']}")
    print(f"  Total rollout time:  {overall['T']:.1f}s ({overall['T']/60:.1f}min)")
    print(f"  Rollout steps:       {len(per_step)}")
    print(f"  Mean r_k:            {overall['mean_r_k']:.2f} / {overall['Q']} GPUs")
    print(f"  Bubble Ratio:        {overall['bubble_ratio']*100:.2f}%")
    print(f"  GPU Utilization:     {overall['utilization']*100:.2f}%")
    print(f"\nPer-step stats:")
    print(f"  Bubble Ratio:  mean={per_step['bubble_ratio'].mean()*100:.2f}%  "
          f"std={per_step['bubble_ratio'].std()*100:.2f}%  "
          f"min={per_step['bubble_ratio'].min()*100:.2f}%  "
          f"max={per_step['bubble_ratio'].max()*100:.2f}%")
    print(f"  Utilization:   mean={per_step['utilization'].mean()*100:.2f}%  "
          f"std={per_step['utilization'].std()*100:.2f}%  "
          f"min={per_step['utilization'].min()*100:.2f}%  "
          f"max={per_step['utilization'].max()*100:.2f}%")


def plot_per_step(per_step_by_run: list[tuple[str, pd.DataFrame]], output_path: Path):
    """Plot per-step bubble ratio for one or more runs."""
    fig, axes = plt.subplots(1, 1, figsize=(6, 3))

    yticks = np.arange(0, 101, 20)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#4C72B0"])

    # Bubble ratio per step
    ax = axes
    for idx, (run_name, per_step) in enumerate(per_step_by_run):
        if per_step.empty:
            continue
        x = per_step["step"].values
        y = per_step["bubble_ratio"].values * 100
        color = colors[idx % len(colors)]
        ax.plot(x, y, color=color, linewidth=2.0, alpha=0.9, label=run_name)
        mean_y = per_step["bubble_ratio"].mean() * 100
        ax.axhline(mean_y, color=color, linestyle="--", linewidth=1, alpha=0.9)
        x_text = x.max() if len(x) > 0 else 1
        ax.text(
            x_text + 0.2,
            mean_y,
            f"{mean_y:.1f}%",
            color=color,
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_xlabel("Step")
    ax.set_ylabel("Bubble Ratio (%)")
    ax.set_title("Rollout Bubble Ratio per Step")
    ax.set_ylim(0, 105)
    ax.set_yticks(yticks)
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nChart saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="GPU Bubble Ratio analysis for Rollout phase")
    parser.add_argument(
        "--log-dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more log directories",
    )
    parser.add_argument("-o", "--output-dir", type=str, default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    log_dirs = [Path(p) for p in args.log_dirs]
    if len(log_dirs) > 2 and args.output_dir is None:
        parser.error("When specifying more than two --log-dirs, --output-dir is required.")

    per_step_by_run = []
    for log_dir in log_dirs:
        events = pd.read_csv(log_dir / "gpu_events.csv")
        metrics = pd.read_csv(log_dir / "gpu_metrics.csv")
        Q = metrics["gpu_id"].nunique()

        intervals = build_rollout_intervals(events)
        rollout_metrics = filter_metrics_by_intervals(metrics, intervals)
        print(
            f"[{log_dir.name}] Loaded {len(metrics)} metric rows, "
            f"{len(rollout_metrics)} in rollout phases"
        )

        overall = compute_bubble(rollout_metrics, Q)
        per_step = compute_per_step_bubble(metrics, intervals, Q)
        print_summary(overall, per_step)
        per_step_by_run.append((log_dir.name, per_step))

    if not args.no_plot:
        out_dir = Path(args.output_dir) if args.output_dir else log_dirs[0]
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_per_step(per_step_by_run, out_dir / "bubble_rollout.png")


if __name__ == "__main__":
    main()
