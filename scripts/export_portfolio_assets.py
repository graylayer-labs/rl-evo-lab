#!/usr/bin/env python3
"""Export experiment results to portfolio image directory.

Copies comparison plots from runs/ to docs/img/ for GitHub rendering.
Documents which run each image came from via manifest.

Usage:
    python scripts/export_portfolio_assets.py --experiment cartpole_normal
    python scripts/export_portfolio_assets.py --all
"""

import argparse
import json
import shutil
from pathlib import Path


def export_comparison_plot(exp_name: str, target_name: str | None = None) -> bool:
    """Export comparison.png from a completed experiment.

    Args:
        exp_name: experiment directory name (e.g., 'cartpole_normal')
        target_name: target filename in docs/img/ (default: {exp_name}_comparison.png)

    Returns:
        True if successful, False otherwise.
    """
    source_plot = Path("runs") / exp_name / "comparison.png"
    if not source_plot.exists():
        print(f"❌ No comparison.png found in runs/{exp_name}/")
        return False

    target_dir = Path("docs/img")
    target_dir.mkdir(parents=True, exist_ok=True)

    target_name = target_name or f"{exp_name}_comparison.png"
    target_plot = target_dir / target_name

    shutil.copy2(source_plot, target_plot)
    print(f"✅ Exported {exp_name} → docs/img/{target_name}")

    # Record in manifest
    manifest_path = target_dir / "portfolio_manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    manifest[target_name] = {
        "source_run": exp_name,
        "description": f"Comparison plot for {exp_name} experiment",
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return True


def main():
    parser = argparse.ArgumentParser(description="Export experiment results to portfolio directory")
    parser.add_argument(
        "--experiment",
        help="Export a single experiment (e.g., cartpole_normal)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all completed experiments",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check which experiments have comparison plots ready",
    )

    args = parser.parse_args()

    if args.check:
        runs_dir = Path("runs")
        ready = []
        for exp_dir in sorted(runs_dir.iterdir()):
            if exp_dir.is_dir() and (exp_dir / "comparison.png").exists():
                ready.append(exp_dir.name)
        if ready:
            print("Ready for export:")
            for name in ready:
                print(f"  - {name}")
        else:
            print("No experiments with comparison.png found.")
        return

    if args.experiment:
        success = export_comparison_plot(args.experiment)
        return 0 if success else 1

    if args.all:
        runs_dir = Path("runs")
        exported = 0
        for exp_dir in sorted(runs_dir.iterdir()):
            if exp_dir.is_dir() and (exp_dir / "comparison.png").exists():
                if export_comparison_plot(exp_dir.name):
                    exported += 1
        print(f"\n✅ Exported {exported} comparisons to docs/img/")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
