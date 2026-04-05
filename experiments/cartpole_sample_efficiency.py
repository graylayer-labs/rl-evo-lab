"""Fair sample efficiency comparison: equal env-step budget across conditions.

ES uses N workers per episode → N× more env steps than DQN per training
iteration. Comparing on episodes alone makes EDER look better than it is.

This experiment gives DQN 10,000 episodes (vs 2,000 for ES variants) so all
conditions have roughly the same total env-step budget. Plot on env_steps
x-axis to make the fair comparison visible.

Run:
    python experiments/cartpole_sample_efficiency.py               # episodes x-axis
    python experiments/cartpole_sample_efficiency.py --show        # open plot
    python experiments/cartpole_sample_efficiency.py --force       # re-run all
"""

from rl_evo_lab.experiment import Condition, Experiment

experiment = Experiment(
    name="cartpole_sample_efficiency",
    env="cartpole",
    seeds=[7, 42, 123],
    conditions=[
        Condition("EDER", use_es=True, use_novelty=True),
        Condition("ES+DQN", use_es=True, use_novelty=False),
        # DQN gets more episodes to match the ES env-step budget
        Condition("DQN", use_es=False, use_novelty=False, total_episodes=10_000),
    ],
)

if __name__ == "__main__":
    import sys

    # Default to env_steps x-axis for this experiment
    if "--x-axis" not in sys.argv:
        sys.argv += ["--x-axis", "env_steps"]
    experiment.main()
