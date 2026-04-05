"""Does ES diversity compensate for a smaller network?

Tests whether the diverse experience injected by the ES population allows
a small-network EDER agent to match or exceed a larger pure-DQN network.

Conditions (CartPole-v1):
  EDER-64  — ES + novelty, hidden_dim=64
  EDER-128 — ES + novelty, hidden_dim=128
  DQN-64   — pure DQN, hidden_dim=64
  DQN-128  — pure DQN, hidden_dim=128  (largest baseline)

Hypothesis: EDER-64 ≈ DQN-128, i.e. buffer diversity substitutes for
model capacity on simple envs.

Run:
    python experiments/cartpole_model_size.py
    python experiments/cartpole_model_size.py --force --show
"""

from rl_evo_lab.experiment import Condition, Experiment

experiment = Experiment(
    name="cartpole_model_size",
    env="cartpole",
    seeds=[7, 42, 123],
    conditions=[
        Condition("EDER-64", use_es=True, use_novelty=True, hidden_dim=64),
        Condition("EDER-128", use_es=True, use_novelty=True, hidden_dim=128),
        Condition("DQN-64", use_es=False, use_novelty=False, hidden_dim=64),
        Condition("DQN-128", use_es=False, use_novelty=False, hidden_dim=128),
    ],
)

if __name__ == "__main__":
    experiment.main()
