"""Evolutionary RL baseline: ES actor + DQN learner, no intrinsic novelty.

Phase 3: Tests whether evolutionary strategy (population diversity) alone
can improve RL on exploration-stuck environments, compared to DQN baseline.

This isolates the ES contribution without novelty confounds.

Environments (3 seeds each):
  1. CartPole-v1          — dense, baseline cost
  2. LunarLander-v3       — dense, precision test
  3. CartPole-sparse      — sparse, zero-gradient discovery
  4. Acrobot-v1           — sparse, rare behavior discovery

Run all environments:
    python experiments/evolutionary_rl.py --all

Run individual environments:
    python experiments/evolutionary_rl.py --env cartpole
    python experiments/evolutionary_rl.py --env lunarlander
    python experiments/evolutionary_rl.py --env cartpole_sparse
    python experiments/evolutionary_rl.py --env acrobot

Plot results:
    python experiments/evolutionary_rl.py --env cartpole --plot-only --show
"""

from rl_evo_lab.experiment import Condition, Experiment

# Evolutionary RL condition: use_es=True, use_novelty=False
_evolutionary_rl = Condition("Evolutionary RL", use_es=True, use_novelty=False)

# Standard seeds for main environments (3 seeds each)
_standard_seeds = [42, 7, 123]

# Experiment registry: env -> experiment object
_experiments = {
    "cartpole": Experiment(
        name="cartpole_evolutionary_rl",
        env="cartpole",
        seeds=_standard_seeds,
        conditions=[_evolutionary_rl],
    ),
    "lunarlander": Experiment(
        name="lunarlander_evolutionary_rl",
        env="lunarlander",
        seeds=_standard_seeds,
        conditions=[_evolutionary_rl],
    ),
    "cartpole_sparse": Experiment(
        name="cartpole_sparse_evolutionary_rl",
        env="cartpole_sparse",
        seeds=_standard_seeds,
        conditions=[_evolutionary_rl],
    ),
    "acrobot": Experiment(
        name="acrobot_evolutionary_rl",
        env="acrobot",
        seeds=_standard_seeds,
        conditions=[_evolutionary_rl],
    ),
}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Evolutionary RL (ES+DQN) across the 4-environment spectrum."
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
            print(f"Running Evolutionary RL on {env_name}")
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
