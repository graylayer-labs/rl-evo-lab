"""Compare all three methods (DQN, Evolutionary RL, Novelty-Guided RL) side-by-side.

This script orchestrates comparison plots across the 4-environment spectrum,
showing how each technique performs relative to the DQN baseline.

Usage:
    # Compare all three methods on CartPole
    python experiments/compare_all_methods.py --env cartpole --show

    # Compare all four environments (generate all comparison plots)
    python experiments/compare_all_methods.py --all-envs --show

    # Compare specific environments
    python experiments/compare_all_methods.py --envs cartpole lunarlander --show

    # Generate summary table of all results
    python experiments/compare_all_methods.py --summary
"""

from pathlib import Path
from rl_evo_lab.experiment import Condition, Experiment


def run_comparison(env_name: str, show: bool = False, force: bool = False):
    """Run comparison of all three methods on a single environment."""

    # Define conditions (DQN, ES+DQN, Novelty-Guided)
    conditions = [
        Condition("DQN", use_es=False, use_novelty=False),
        Condition("Evolutionary RL", use_es=True, use_novelty=False),
        Condition("Novelty-Guided RL", use_es=True, use_novelty=True),
    ]

    exp = Experiment(
        name=f"{env_name}_all_methods_comparison",
        env=env_name,
        seeds=[42, 7, 123],
        conditions=conditions,
    )

    print(f"\n{'='*70}")
    print(f"Comparing all three methods on {env_name}")
    print(f"{'='*70}\n")

    exp.run(force=force, show=show, x_axis="env_steps")

    # Print path to comparison plot
    plot_path = Path(f"runs/{env_name}_all_methods_comparison/comparison.png")
    if plot_path.exists():
        print(f"\n✓ Comparison plot saved: {plot_path}")

    return exp


def generate_summary():
    """Generate a summary table of all results."""
    import json

    print("\n" + "="*80)
    print("PHASE 3 RESULTS SUMMARY")
    print("="*80 + "\n")

    environments = ["cartpole", "lunarlander", "cartpole_sparse", "acrobot"]

    results = {}
    for env in environments:
        print(f"\n{env.upper()}")
        print("-" * 60)

        for method, run_prefix in [
            ("DQN", "baseline_dqn"),
            ("Evolutionary RL", "evolutionary_rl"),
            ("Novelty-Guided RL", "novelty_guided_rl"),
        ]:
            run_dir = Path(f"runs/{env}_{run_prefix}")
            if not run_dir.exists():
                print(f"  {method:20s} - NOT RUN YET")
                continue

            # Try to extract results from individual runs
            seed_dirs = list(run_dir.glob(f"*__seed*"))
            if seed_dirs:
                final_rewards = []
                for seed_dir in seed_dirs:
                    metrics_file = seed_dir / "metrics.csv"
                    if metrics_file.exists():
                        # Get last line (final eval reward)
                        with open(metrics_file) as f:
                            lines = f.readlines()
                            if len(lines) > 1:
                                parts = lines[-1].strip().split(',')
                                if len(parts) > 5:  # learner_eval_reward is column 5
                                    try:
                                        reward = float(parts[5])
                                        final_rewards.append(reward)
                                    except ValueError:
                                        pass

                if final_rewards:
                    mean = sum(final_rewards) / len(final_rewards)
                    std = (sum((r - mean)**2 for r in final_rewards) / len(final_rewards))**0.5
                    print(f"  {method:20s} {mean:8.1f} ± {std:6.1f}")
                else:
                    print(f"  {method:20s} - INCOMPLETE")
            else:
                print(f"  {method:20s} - NOT STARTED")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare all three methods (DQN, ES+DQN, Novelty-Guided) across environments."
    )
    parser.add_argument(
        "--env",
        help="Compare on a single environment.",
        choices=["cartpole", "lunarlander", "cartpole_sparse", "acrobot"],
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        help="Compare on specific environments.",
        choices=["cartpole", "lunarlander", "cartpole_sparse", "acrobot"],
    )
    parser.add_argument(
        "--all-envs",
        action="store_true",
        help="Compare on all four environments.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots after running.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if results exist.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary table of all results (no experiments run).",
    )

    args = parser.parse_args()

    if args.summary:
        generate_summary()
        return

    # Determine which environments to compare
    envs_to_compare = []
    if args.all_envs:
        envs_to_compare = ["cartpole", "lunarlander", "cartpole_sparse", "acrobot"]
    elif args.envs:
        envs_to_compare = args.envs
    elif args.env:
        envs_to_compare = [args.env]
    else:
        parser.print_help()
        return

    # Run comparisons
    for env_name in envs_to_compare:
        run_comparison(env_name, show=args.show, force=args.force)

    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    print("\nComparison plots saved to:")
    for env_name in envs_to_compare:
        print(f"  runs/{env_name}_all_methods_comparison/comparison.png")

    print("\nTo view results summary:")
    print("  python experiments/compare_all_methods.py --summary")


if __name__ == "__main__":
    main()
