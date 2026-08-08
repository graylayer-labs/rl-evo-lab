#!/usr/bin/env python3
"""Monitor Phase 3 experiment progress in real-time.

Shows which runs are complete, in progress, or pending.
Updates automatically as new results are generated.

Usage:
    python scripts/monitor_phase3_progress.py              # single snapshot
    python scripts/monitor_phase3_progress.py --watch 30   # update every 30s
    python scripts/monitor_phase3_progress.py --full       # detailed per-seed info
"""

import time
from pathlib import Path
from datetime import datetime


def check_run_status(run_dir: Path) -> dict:
    """Check status of a single run directory."""
    if not run_dir.exists():
        return {"status": "not_started", "seeds": 0, "episodes": 0}

    seed_dirs = list(run_dir.glob("*__seed*"))
    if not seed_dirs:
        return {"status": "pending", "seeds": 0, "episodes": 0}

    total_episodes = 0
    completed_seeds = 0

    for seed_dir in seed_dirs:
        status_file = seed_dir / "status.json"
        metrics_file = seed_dir / "metrics.csv"

        if status_file.exists():
            completed_seeds += 1

        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].strip().split(',')
                        if len(last_line) > 0:
                            total_episodes += int(last_line[0]) + 1
            except (ValueError, IndexError):
                pass

    return {
        "status": "complete" if completed_seeds == 3 else "in_progress",
        "seeds": completed_seeds,
        "episodes": total_episodes // max(1, completed_seeds),
    }


def format_status(status: dict) -> str:
    """Format status for display."""
    if status["status"] == "not_started":
        return "⏳ Not started"
    elif status["status"] == "pending":
        return "⏳ Pending"
    elif status["status"] == "in_progress":
        return f"🔄 In progress ({status['seeds']}/3 seeds, avg {status['episodes']} eps)"
    elif status["status"] == "complete":
        return f"✓ Complete ({status['seeds']}/3 seeds)"
    return "?"


def show_progress(full: bool = False):
    """Display Phase 3 progress."""
    environments = ["cartpole", "lunarlander", "cartpole_sparse", "acrobot"]
    methods = ["baseline_dqn", "evolutionary_rl", "novelty_guided_rl"]

    print("\n" + "="*100)
    print(f"PHASE 3 PROGRESS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)

    total_runs = len(environments) * len(methods)
    complete_runs = 0

    for env_name in environments:
        print(f"\n{env_name.upper()}")
        print("-" * 100)

        for method in methods:
            run_dir = Path(f"runs/{env_name}_{method}")
            status = check_run_status(run_dir)

            status_str = format_status(status)

            if status["status"] == "complete":
                complete_runs += 1

            print(f"  {method:25s} {status_str}")

    print("\n" + "="*100)
    print(f"OVERALL: {complete_runs}/{total_runs} runs complete")
    print("="*100 + "\n")

    # Show estimated time remaining
    if complete_runs < total_runs:
        remaining = total_runs - complete_runs
        avg_time_per_run = 0.5  # hours (rough estimate)
        est_hours = remaining * avg_time_per_run
        print(f"Estimated time remaining: ~{est_hours:.1f} hours")
        print(f"Expected completion: ~{datetime.fromtimestamp(time.time() + est_hours*3600).strftime('%H:%M')}")

    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Phase 3 experiment progress.")
    parser.add_argument(
        "--watch",
        type=int,
        metavar="SECONDS",
        help="Update every N seconds (runs until interrupted)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show detailed per-seed information",
    )

    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                show_progress(full=args.full)
                print(f"Next update in {args.watch} seconds (Ctrl+C to stop)...\n")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        show_progress(full=args.full)


if __name__ == "__main__":
    main()
