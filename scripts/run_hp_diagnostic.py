#!/usr/bin/env python3
"""Quick HP diagnostic before full training.

Runs a short experiment (50-100 episodes) with default category HPs to:
1. Verify HPs work for the environment
2. Detect if early adaptation is needed
3. Warn if anything looks wrong before expensive full run

Usage:
    python scripts/run_hp_diagnostic.py --env cartpole
    python scripts/run_hp_diagnostic.py --env lunarlander --episodes 100
"""

import argparse
import sys
from pathlib import Path

from rl_evo_lab.utils.config import make_config, ENV_CATEGORIES, ENV_PRESETS
from rl_evo_lab.train import train


def run_diagnostic(env: str, episodes: int = 50, force: bool = False) -> dict:
    """Run quick diagnostic and report findings."""

    # Get preset
    preset = ENV_PRESETS.get(env)
    if not preset:
        print(f"❌ Unknown environment: {env}")
        print(f"   Available: {list(ENV_PRESETS.keys())}")
        return {}

    # Get category
    category = preset.get("category", "unknown")
    category_info = ENV_CATEGORIES.get(category, {})

    print(f"\n╔════════════════════════════════════════════════════════════╗")
    print(f"║          HP DIAGNOSTIC: {env:25s}         ║")
    print(f"╚════════════════════════════════════════════════════════════╝\n")

    print(f"Category:  {category}")
    print(f"  {category_info.get('description', 'N/A')}\n")

    print(f"HPs from category defaults:")
    print(f"  es_sigma:              {preset.get('es_sigma', 0.06)}")
    print(f"  beta (novelty):        {preset.get('beta', 0.02)}")
    print(f"  novelty_ramp_episodes: {preset.get('novelty_ramp_episodes', 100)}\n")

    print(f"Running {episodes}-episode diagnostic...\n")

    # Create config with diagnostic episodes limit
    cfg = make_config(env, total_episodes=episodes)

    # Run training
    run_dir = Path(f"runs/{env}_diagnostic")
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        train(cfg, verbose=False, log_dir="runs", run_dir=run_dir)
        print(f"\n✅ Diagnostic completed successfully")
        print(f"   Results saved to: {run_dir}")

        # Read final metrics
        import csv
        metrics_file = list(run_dir.glob("*/metrics.csv"))
        if metrics_file:
            with open(metrics_file[0]) as f:
                rows = list(csv.DictReader(f))
                if rows:
                    final = rows[-1]
                    learner_reward = float(final.get('learner_eval_reward', 0))
                    idn_loss = float(final.get('idn_loss', 0))
                    print(f"\n   Final learner reward: {learner_reward:.1f}")
                    print(f"   Final IDN loss:      {idn_loss:.4f}")

                    # Quick sanity checks
                    if learner_reward < 0:
                        print(f"\n⚠️  WARNING: Negative final reward suggests instability")
                    if idn_loss > 0.5:
                        print(f"\n⚠️  WARNING: High IDN loss suggests poor embeddings")

        return {"status": "ok", "reward": learner_reward}

    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}")
        return {"status": "failed", "error": str(e)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run quick HP diagnostic before full training"
    )
    parser.add_argument(
        "--env",
        required=True,
        help="Environment preset name (cartpole, lunarlander, etc.)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of episodes for diagnostic (default: 50)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if diagnostic results exist",
    )

    args = parser.parse_args()
    result = run_diagnostic(args.env, args.episodes, args.force)

    sys.exit(0 if result.get("status") == "ok" else 1)
