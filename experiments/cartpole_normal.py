"""CartPole-v1 Normal: Standard benchmark to validate baseline performance.

Standard Gymnasium CartPole-v1 with default settings:
- Starting: pole angle = 0, cart pos = 0
- Termination: pole > 24° OR cart > 2.4
- Episode limit: 500 steps
- Success: achieve 475+ reward (moving average)

Compares EDER, ES+DQN, DQN on standard CartPole.

Run:
    python experiments/cartpole_normal.py
    python experiments/cartpole_normal.py --force --show
"""

from rl_evo_lab.experiment import Condition, Experiment

experiment = Experiment(
    name="cartpole_normal",
    env="cartpole",
    seeds=[7, 42, 123],
    conditions=[
        Condition("EDER", use_es=True, use_novelty=True),
        Condition("ES+DQN", use_es=True, use_novelty=False),
        Condition("DQN", use_es=False, use_novelty=False),
    ],
)

if __name__ == "__main__":
    experiment.main()
