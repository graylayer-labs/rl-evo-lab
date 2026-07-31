"""Does EDER shine on hard exploration? Acrobot-v1 trial.

Acrobot is a classic hard exploration problem: swing a 2-link pendulum up.
Reward is -1 per step (sparse until solved). DQN epsilon-greedy gets stuck in
local swing patterns. ES population explores diverse swing sequences.
Novelty reward (IDN + KNN) should drive further exploration variety.

Expected: EDER > ES+DQN > DQN (in final reward, higher is better / less negative).

Run:
    python experiments/acrobot_exploration.py
    python experiments/acrobot_exploration.py --force --show
    python experiments/acrobot_exploration.py --plot-only --show
"""

from rl_evo_lab.experiment import Condition, Experiment

experiment = Experiment(
    name="acrobot_exploration",
    env="acrobot",
    seeds=[7, 42, 123],
    conditions=[
        Condition("EDER", use_es=True, use_novelty=True),
        Condition("ES+DQN", use_es=True, use_novelty=False),
        Condition("DQN", use_es=False, use_novelty=False),
    ],
)

if __name__ == "__main__":
    experiment.main()
