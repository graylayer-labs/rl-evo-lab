"""LunarLander-v3 Normal: Standard benchmark on longer-horizon task.

Standard Gymnasium LunarLander-v3 with default settings:
- Starting: default spawn (position/velocity = 0)
- Termination: crash or time limit (1000 steps)
- Reward: -1/step landing penalty, +100 for upright landing, etc.
- Success: episode reward > 200 (smooth landing)

Compares EDER, ES+DQN, DQN on LunarLander with natural exploration challenge.
This is where EDER showed catastrophic forgetting in prior experiments.

Run:
    python experiments/lunarlander_normal.py
    python experiments/lunarlander_normal.py --force --show
"""

from rl_evo_lab.experiment import Condition, Experiment

experiment = Experiment(
    name="lunarlander_normal",
    env="lunarlander",
    seeds=[7, 42, 123],
    conditions=[
        Condition("EDER", use_es=True, use_novelty=True),
        Condition("ES+DQN", use_es=True, use_novelty=False),
        Condition("DQN", use_es=False, use_novelty=False),
    ],
)

if __name__ == "__main__":
    experiment.main()
