#!/usr/bin/env python3
"""Diagnostic: Check if IDN is actually learning from existing runs."""

import csv
from collections import defaultdict
from pathlib import Path


def check_idn_convergence(run_dir: str) -> dict:
    """Check IDN loss decay across an experiment."""
    results = defaultdict(list)

    runs_path = Path(f"runs/{run_dir}")
    if not runs_path.exists():
        print(f"Run directory not found: {run_dir}")
        return {}

    for run_folder in runs_path.iterdir():
        if not run_folder.is_dir():
            continue

        metrics_file = run_folder / "metrics.csv"
        if not metrics_file.exists():
            continue

        run_name = run_folder.name
        condition = run_name.split("__")[0]

        with open(metrics_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            continue

        # Track IDN loss across episodes
        idn_losses = []
        for row in rows:
            try:
                loss = float(row.get("idn_loss", "nan"))
                if loss == loss:  # not NaN
                    idn_losses.append(loss)
            except (ValueError, TypeError):
                pass

        if idn_losses:
            start = idn_losses[0]
            end = idn_losses[-1]
            min_loss = min(idn_losses)
            max(idn_losses)
            decay = (start - end) / start if start > 0 else 0

            results[condition].append(
                {
                    "run": run_name,
                    "start_loss": start,
                    "end_loss": end,
                    "min_loss": min_loss,
                    "decay_pct": decay * 100,
                    "n_episodes": len(idn_losses),
                }
            )

    return results


print("╔════════════════════════════════════════════════════════════════╗")
print("║              IDN LEARNING DIAGNOSTIC                          ║")
print("╚════════════════════════════════════════════════════════════════╝\n")

for exp in ["cartpole_normal", "lunarlander_normal"]:
    print(f"\n{exp.upper()}")
    print("=" * 70)

    results = check_idn_convergence(exp)

    if not results:
        print(f"No EDER/ES+DQN runs found in runs/{exp}")
        continue

    for condition in ["EDER", "ES+DQN"]:
        if condition not in results:
            continue

        runs = results[condition]
        print(f"\n{condition}:")

        for run in runs:
            print(f"  {run['run'][:35]:35s}", end=" ")
            print(f"start={run['start_loss']:7.4f} end={run['end_loss']:7.4f} ", end="")
            print(f"decay={run['decay_pct']:5.1f}% ", end="")

            if run["decay_pct"] > 50:
                print("✓ LEARNING")
            elif run["decay_pct"] > 20:
                print("⚠ SLOW")
            else:
                print("✗ NOT LEARNING")

        # Summary
        avg_start = sum(r["start_loss"] for r in runs) / len(runs)
        avg_end = sum(r["end_loss"] for r in runs) / len(runs)
        avg_decay = sum(r["decay_pct"] for r in runs) / len(runs)

        print(f"  Average: start={avg_start:.4f} → end={avg_end:.4f} (decay {avg_decay:.1f}%)")

        if avg_decay > 50:
            print(f"  ✓ IDN is learning well for {condition}")
        else:
            print(f"  ✗ IDN NOT learning enough for {condition} — novelty signal is weak/noisy")

print("\n" + "=" * 70)
print("\nINTERPRETATION:")
print("  - Decay > 50%: IDN is learning, embeddings should be useful")
print("  - Decay 20-50%: IDN learning slowly, might need more updates/higher LR")
print("  - Decay < 20%: IDN not learning, novelty signal is garbage")
