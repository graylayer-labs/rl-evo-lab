"""DQN Baseline across 5-environment spectrum.

Thesis-style baseline: validate that DQN alone can solve dense-reward tasks
before testing ES+DQN and EDER on the full environment spectrum.

This script establishes ground truth for "does exploration matter here" —
a prerequisite for understanding whether novelty-driven ES adds value.

Environments (3 seeds each):
  1. CartPole-v1          — dense, trivial baseline
  2. LunarLander-v3       — dense, continuous control
  3. CartPole-sparse      — sparse (0/step, +500 at episode end)
  4. Acrobot-v1           — sparse, discovery task

Montezuma's Revenge (1 seed, capped):
  5. MontezumaRevenge-ram-v5 — hard exploration, exploratory run only

Run all environments:
    python experiments/baseline_dqn.py --all

Run individual environments:
    python experiments/baseline_dqn.py --env cartpole
    python experiments/baseline_dqn.py --env lunarlander
    python experiments/baseline_dqn.py --env cartpole_sparse
    python experiments/baseline_dqn.py --env acrobot
    python experiments/baseline_dqn.py --env montezuma

Plot results:
    python experiments/baseline_dqn.py --env cartpole --plot-only --show
"""

from rl_evo_lab.experiment import Condition, Experiment

# DQN baseline condition: use_es=False, use_novelty=False
_dqn = Condition("DQN", use_es=False, use_novelty=False)

# Standard seeds for main environments (3 seeds each)
_standard_seeds = [42, 7, 123]

# Montezuma's gets 1 seed with capped budget (exploratory baseline)
_montezuma_seeds = [42]

# Experiment registry: env -> experiment object
_experiments = {
    "cartpole": Experiment(
        name="cartpole_baseline_dqn",
        env="cartpole",
        seeds=_standard_seeds,
        conditions=[_dqn],
    ),
    "lunarlander": Experiment(
        name="lunarlander_baseline_dqn",
        env="lunarlander",
        seeds=_standard_seeds,
        conditions=[_dqn],
    ),
    "cartpole_sparse": Experiment(
        name="cartpole_sparse_baseline_dqn",
        env="cartpole_sparse",
        seeds=_standard_seeds,
        conditions=[_dqn],
    ),
    "acrobot": Experiment(
        name="acrobot_baseline_dqn",
        env="acrobot",
        seeds=_standard_seeds,
        conditions=[_dqn],
    ),
    "montezuma": Experiment(
        name="montezuma_baseline_dqn",
        env="montezuma",
        seeds=_montezuma_seeds,
        conditions=[_dqn],
    ),
}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run DQN baseline across the 5-environment spectrum."
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
            print(f"Running baseline on {env_name}")
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
