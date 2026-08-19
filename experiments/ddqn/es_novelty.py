"""Novelty-Guided RL: ES actor + DQN learner + KNN novelty signal.

Phase 3: Tests whether adding intrinsic novelty motivation improves upon
evolutionary strategy alone, particularly on sparse/discovery tasks.

This combines both exploration techniques to test if they synergize.

Environments (3 seeds each):
  1. CartPole-v1          — dense, baseline overhead test
  2. LunarLander-v3       — dense, precision test (should novelty hurt?)
  3. CartPole-sparse      — sparse, discovery with intrinsic signal
  4. Acrobot-v1           — sparse, rare behavior with novelty guidance
  5. Montezuma's Revenge  — hard exploration, canonical novelty-search benchmark

Run all environments:
    python experiments/ddqn/es_novelty.py --all

Run individual environments:
    python experiments/ddqn/es_novelty.py --env cartpole
    python experiments/ddqn/es_novelty.py --env lunarlander
    python experiments/ddqn/es_novelty.py --env cartpole_sparse
    python experiments/ddqn/es_novelty.py --env acrobot
    python experiments/ddqn/es_novelty.py --env montezuma

Plot results:
    python experiments/ddqn/es_novelty.py --env cartpole --plot-only --show
"""

from runner.experiment import Condition, Experiment

# DDQN + ES + Novelty condition: use_es=True, use_novelty=True
_novelty_guided_rl = Condition("DDQN+ES+Novelty", use_es=True, use_novelty=True)

# Standard seeds for main environments (3 seeds each)
_standard_seeds = [42, 7, 123]

# Experiment registry: env -> experiment object
_experiments = {
    "cartpole": Experiment(
        name="cartpole_novelty_guided_rl",
        env="cartpole",
        seeds=_standard_seeds,
        conditions=[_novelty_guided_rl],
    ),
    "lunarlander": Experiment(
        name="lunarlander_novelty_guided_rl",
        env="lunarlander",
        seeds=_standard_seeds,
        conditions=[_novelty_guided_rl],
    ),
    "cartpole_sparse": Experiment(
        name="cartpole_sparse_novelty_guided_rl",
        env="cartpole_sparse",
        seeds=_standard_seeds,
        conditions=[_novelty_guided_rl],
    ),
    "acrobot": Experiment(
        name="acrobot_novelty_guided_rl",
        env="acrobot",
        seeds=_standard_seeds,
        conditions=[_novelty_guided_rl],
    ),
    "montezuma": Experiment(
        name="montezuma_novelty_guided_rl",
        env="montezuma",
        seeds=_standard_seeds,
        conditions=[_novelty_guided_rl],
    ),
}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Novelty-Guided RL (ES+novelty) across the 4-environment spectrum."
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
            print(f"Running Novelty-Guided RL on {env_name}")
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
