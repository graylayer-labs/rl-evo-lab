"""Novelty-Only RL: DDQN learner + KNN novelty signal, no ES population.

Tests whether intrinsic novelty motivation improves exploration on its own,
without the population diversity that ES provides. Isolates the novelty
contribution from the ES contribution (see evolutionary_rl.py for the
inverse isolation).

Environments (3 seeds each):
  1. CartPole-v1          — dense, baseline overhead test
  2. LunarLander-v3       — dense, precision test (should novelty hurt?)
  3. CartPole-sparse      — sparse, discovery with intrinsic signal
  4. Acrobot-v1           — sparse, rare behavior with novelty guidance
  5. Montezuma's Revenge  — hard exploration, canonical novelty-search benchmark

Run all environments:
    python experiments/ddqn/novelty.py --all

Run individual environments:
    python experiments/ddqn/novelty.py --env cartpole
    python experiments/ddqn/novelty.py --env lunarlander
    python experiments/ddqn/novelty.py --env cartpole_sparse
    python experiments/ddqn/novelty.py --env acrobot
    python experiments/ddqn/novelty.py --env montezuma

Plot results:
    python experiments/ddqn/novelty.py --env cartpole --plot-only --show
"""

from runner.experiment import Condition, Experiment

# Novelty-Only condition: use_es=False, use_novelty=True
_novelty_only = Condition("DDQN+Novelty", use_es=False, use_novelty=True)

# Standard seeds for main environments (3 seeds each)
_standard_seeds = [42, 7, 123]

# Experiment registry: env -> experiment object
_experiments = {
    "cartpole": Experiment(
        name="cartpole_novelty_only_rl",
        env="cartpole",
        seeds=_standard_seeds,
        conditions=[_novelty_only],
    ),
    "lunarlander": Experiment(
        name="lunarlander_novelty_only_rl",
        env="lunarlander",
        seeds=_standard_seeds,
        conditions=[_novelty_only],
    ),
    "cartpole_sparse": Experiment(
        name="cartpole_sparse_novelty_only_rl",
        env="cartpole_sparse",
        seeds=_standard_seeds,
        conditions=[_novelty_only],
    ),
    "acrobot": Experiment(
        name="acrobot_novelty_only_rl",
        env="acrobot",
        seeds=_standard_seeds,
        conditions=[_novelty_only],
    ),
    "montezuma": Experiment(
        name="montezuma_novelty_only_rl",
        env="montezuma",
        seeds=_standard_seeds,
        conditions=[_novelty_only],
    ),
}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Novelty-Only RL (DDQN+Novelty) across the 5-environment spectrum."
    )
    parser.add_argument(
        "--env",
        choices=list(_experiments.keys()),
        default=None,
        help="Run specific environment. Default: None (use --all to run all).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all environments sequentially.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Plot existing runs without re-running.",
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
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers.",
    )
    args = parser.parse_args()

    if args.all:
        for env_name in _experiments:
            print(f"\n{'='*60}")
            print(f"Running Novelty-Only RL on {env_name}")
            print(f"{'='*60}\n")
            exp = _experiments[env_name]
            if args.plot_only:
                exp.plot(show=args.show, x_axis="env_steps")
            else:
                exp.run(force=args.force, show=args.show, workers=args.workers, x_axis="env_steps")
    elif args.env:
        exp = _experiments[args.env]
        if args.plot_only:
            exp.plot(show=args.show, x_axis="env_steps")
        else:
            exp.run(force=args.force, show=args.show, workers=args.workers, x_axis="env_steps")
    else:
        parser.print_help()
        print("\nNo environment selected. Use --env <env> or --all")


if __name__ == "__main__":
    main()
