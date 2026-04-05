"""Isolated novelty ablation: does IDN novelty help the ES actor?

Both conditions use ES + DQN. The only difference is whether the IDN
novelty signal is added to the actor's augmented reward:
  EDER     — ES + DQN + IDN novelty (rₐ = rₑ + β·rᵢ)
  Baseline — ES + DQN only          (rₐ = rₑ)

Use this to confirm that novelty is the active ingredient, not just the ES
population structure. Expected: EDER maintains higher buffer diversity and
avoids the ES convergence plateau seen in the Baseline.

Run:
    python experiments/cartpole_eder_vs_baseline.py
    python experiments/cartpole_eder_vs_baseline.py --force --show
"""

from rl_evo_lab.experiment import Condition, Experiment

experiment = Experiment(
    name="cartpole_eder_vs_baseline",
    env="cartpole",
    seeds=[7, 42, 123],
    conditions=[
        Condition("EDER", use_es=True, use_novelty=True),
        Condition("Baseline", use_es=True, use_novelty=False),
    ],
)

if __name__ == "__main__":
    experiment.main()
