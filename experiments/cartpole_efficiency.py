"""Does the ES actor improve sample efficiency vs pure DQN on CartPole?

Three conditions on CartPole-v1 (solved at 475):
  EDER    — ES actor + DQN learner + IDN novelty
  ES+DQN  — ES actor + DQN learner, no novelty (novelty ablation)
  DQN     — pure DQN with ε-greedy, no ES (baseline)

Key finding: EDER reaches peak reward ~75 episodes vs ~3,500 for DQN.
Without novelty (ES+DQN), the ES population converges and diversity collapses —
intrinsic reward is necessary to maintain buffer diversity.

Run:
    python experiments/cartpole_efficiency.py
    python experiments/cartpole_efficiency.py --force --show
    python experiments/cartpole_efficiency.py --plot-only --show
"""

from rl_evo_lab.experiment import Condition, Experiment

experiment = Experiment(
    name="cartpole_efficiency",
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
