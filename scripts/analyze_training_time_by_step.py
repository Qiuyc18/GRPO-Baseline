#!/usr/bin/env python3
"""Stacked bar chart of per-step training time breakdown by phase."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The repeating cycle within each step (order matters)
PHASES = ["gen", "reward", "adv", "update_actor"]

PHASE_EVENTS = {
    "gen": ("PhaseEvent.ROLLOUT_PHASE_START", "PhaseEvent.ROLLOUT_PHASE_END"),
    "reward": ("PhaseEvent.REWARD_CALC_START", "PhaseEvent.REWARD_CALC_END"),
    "adv": ("PhaseEvent.BATCH_PREP_START", "PhaseEvent.BATCH_PREP_END"),
    "update_actor": ("PhaseEvent.FORWARD_START", "PhaseEvent.BACKWARD_END"),
}

PHASE_LABELS = {
    "gen": "Rollout (Gen)",
    "reward": "Reward Calc",
    "adv": "Advantage Calc",
    "update_actor": "Actor Update",
    "gap": "Gap (overhead)",
}

PHASE_COLORS = {
    "gen": "#4C72B0",
    "reward": "#DD8452",
    "adv": "#55A868",
    "update_actor": "#C44E52",
    "gap": "#CCCCCC",
}


def parse_steps(events: pd.DataFrame) -> pd.DataFrame:
    """Parse events into per-step phase durations.

    Each step cycle: gen -> reward -> adv -> update_actor
    We identify steps by grouping consecutive cycles of these 4 phases (role != 'testing').
    """
    # Filter out testing events and sort by timestamp
    train_events = events[events["role"] != "testing"].sort_values("timestamp").reset_index(drop=True)

    # Extract per-phase start/end timestamp lists
    phase_times = {}
    for phase, (start_evt, end_evt) in PHASE_EVENTS.items():
        role = phase
        starts = train_events[
            (train_events["event_type"] == start_evt) & (train_events["role"] == role)
        ]["timestamp"].values
        ends = train_events[
            (train_events["event_type"] == end_evt) & (train_events["role"] == role)
        ]["timestamp"].values
        n = min(len(starts), len(ends))
        phase_times[phase] = (starts[:n], ends[:n])

    n_steps = min(len(v[0]) for v in phase_times.values())

    rows = []
    for i in range(n_steps):
        row = {"step": i + 1}
        step_start = phase_times["gen"][0][i]  # gen start = step start
        step_end = phase_times["update_actor"][1][i]  # backward end = step end
        total_phase_time = 0.0
        for phase in PHASES:
            dur = phase_times[phase][1][i] - phase_times[phase][0][i]
            row[phase] = dur
            total_phase_time += dur
        row["total"] = step_end - step_start
        row["gap"] = row["total"] - total_phase_time
        rows.append(row)

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame):
    """Print per-step summary."""
    print(f"\n{'='*90}")
    print(f"Total steps: {len(df)}, Total training time: {df['total'].sum():.1f}s ({df['total'].sum()/60:.1f}min)")
    print(f"{'='*90}\n")

    # Phase averages
    cols = PHASES + ["gap", "total"]
    header = f"{'Phase':<16}" + "".join(f"{'Mean':>9}{'Std':>9}{'%':>7}")
    print(f"{'Phase':<16} {'Mean (s)':>10} {'Std (s)':>10} {'Time%':>7}")
    print("-" * 50)
    total_mean = df["total"].mean()
    for col in PHASES + ["gap"]:
        m = df[col].mean()
        s = df[col].std()
        pct = m / total_mean * 100
        print(f"{PHASE_LABELS[col]:<16} {m:>10.2f} {s:>10.2f} {pct:>6.1f}%")
    print(f"{'Total':<16} {total_mean:>10.2f} {df['total'].std():>10.2f} {'100.0%':>7}")


def plot_stacked_bar(df: pd.DataFrame, output_path: Path):
    """Plot stacked bar chart."""
    n = len(df)
    x = np.arange(n)
    fig_width = max(12, n * 0.15)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    bottom = np.zeros(n)
    draw_order = PHASES + ["gap"]
    for phase in draw_order:
        vals = df[phase].values
        ax.bar(x, vals, bottom=bottom, label=PHASE_LABELS[phase],
               color=PHASE_COLORS[phase], width=1.0, edgecolor="white", linewidth=0.3)
        bottom += vals

    ax.set_xlabel("Step")
    ax.set_ylabel("Time (s)")
    ax.set_title("Per-Step Training Time Breakdown")
    ax.legend(loc="upper right")

    # Tick every 10 steps if many steps
    if n > 50:
        tick_step = max(n // 20, 10)
        ticks = np.arange(0, n, tick_step)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks + 1)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(df["step"].values)

    ax.set_xlim(-0.5, n - 0.5)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nChart saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Per-step training time breakdown (stacked bar)")
    parser.add_argument("log_dir", type=str, help="Path to log directory containing gpu_events.csv")
    parser.add_argument("-o", "--output-dir", type=str, default=None,
                        help="Directory to save the chart (default: same as log_dir)")
    parser.add_argument("--no-plot", action="store_true", help="Skip chart generation")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    events_path = log_dir / "gpu_events.csv"
    if not events_path.exists():
        print(f"Error: {events_path} not found", file=sys.stderr)
        sys.exit(1)

    events = pd.read_csv(events_path)
    df = parse_steps(events)
    print_summary(df)

    if not args.no_plot:
        out_dir = Path(args.output_dir) if args.output_dir else log_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_stacked_bar(df, out_dir / "training_time_by_step.png")


if __name__ == "__main__":
    main()
