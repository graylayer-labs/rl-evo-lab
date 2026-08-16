"""Compare all four methods (DDQN, DDQN+ES, DDQN+Novelty, DDQN+ES+Novelty) side-by-side.

This script orchestrates comparison plots across the 5-environment spectrum,
showing how each technique performs relative to the DDQN baseline.

Usage:
    # Compare all four methods on CartPole
    python experiments/ddqn/compare_all.py --env cartpole --show

    # Compare all five environments (generate all comparison plots)
    python experiments/ddqn/compare_all.py --all-envs --show

    # Compare specific environments
    python experiments/ddqn/compare_all.py --envs cartpole lunarlander --show

For a results summary, use `python scripts/build_results.py` — it reads the
authoritative post-training final_eval, not a training-time checkpoint.
"""

from pathlib import Path

from runner.experiment import Condition, Experiment

_ENVS = ["cartpole", "lunarlander", "cartpole_sparse", "acrobot", "montezuma"]


def run_comparison(env_name: str, show: bool = False, force: bool = False):
    """Run comparison of all four methods on a single environment."""

    # Define conditions (DDQN, DDQN+ES, DDQN+Novelty, DDQN+ES+Novelty)
    conditions = [
        Condition("DDQN", use_es=False, use_novelty=False),
        Condition("DDQN+ES", use_es=True, use_novelty=False),
        Condition("DDQN+Novelty", use_es=False, use_novelty=True),
        Condition("DDQN+ES+Novelty", use_es=True, use_novelty=True),
    ]

    exp = Experiment(
        name=f"{env_name}_all_methods_comparison",
        env=env_name,
        seeds=[42, 7, 123],
        conditions=conditions,
    )

    print(f"\n{'='*70}")
    print(f"Comparing all four methods on {env_name}")
    print(f"{'='*70}\n")

    exp.run(force=force, show=show, x_axis="env_steps")

    # Print path to comparison plot
    plot_path = Path(f"runs/{env_name}_all_methods_comparison/comparison.png")
    if plot_path.exists():
        print(f"\n✓ Comparison plot saved: {plot_path}")

    return exp


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Compare all four methods (DDQN, DDQN+ES, DDQN+Novelty, "
            "DDQN+ES+Novelty) across environments."
        )
    )
    parser.add_argument(
        "--env",
        help="Compare on a single environment.",
        choices=_ENVS,
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        help="Compare on specific environments.",
        choices=_ENVS,
    )
    parser.add_argument(
        "--all-envs",
        action="store_true",
        help="Compare on all five environments.",
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
    args = parser.parse_args()

    # Determine which environments to compare
    envs_to_compare = []
    if args.all_envs:
        envs_to_compare = _ENVS
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
    print("  python scripts/build_results.py")


if __name__ == "__main__":
    main()
