"""CartPole-Tough: Real robustness test for learned control.

Modified CartPole with harder requirements to test genuine learning vs. luck:
- Starting: random pole angle ∈ [-0.2, +0.2] rad, random cart pos ∈ [-0.5, +0.5]
- Termination: pole > 12° (stricter than default 24°)
- Episode limit: 1000 steps (vs default 500) — forces sustained control
- Success: % episodes completing full 1000 steps with pole upright

This distinguishes methods that learn robust control from those that game the metric.
A policy that drifts and gets lucky won't handle random starts or strict angle limits.

Compares EDER, ES+DQN, DQN on CartPole-Tough.

Run:
    python experiments/cartpole_tough.py
    python experiments/cartpole_tough.py --force --show
"""

from rl_evo_lab.experiment import Condition, Experiment

experiment = Experiment(
    name="cartpole_tough",
    env="cartpole_tough",
    seeds=[7, 42, 123],
    conditions=[
        Condition("EDER", use_es=True, use_novelty=True),
        Condition("ES+DQN", use_es=True, use_novelty=False),
        Condition("DQN", use_es=False, use_novelty=False),
    ],
)

if __name__ == "__main__":
    experiment.main()
