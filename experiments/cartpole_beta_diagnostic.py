"""CartPole-v1 Beta Diagnostic: Find optimal novelty weight.

Quick sweep of 3 beta values on CartPole Normal:
- β=0.05 (baseline 0.02 × 2.5)
- β=0.1  (baseline × 5)
- β=0.2  (baseline × 10)

SHORT runs (100 episodes, seed=42 only) to diagnose if increasing β
makes EDER > ES+DQN.

Key metric: does EDER learner_eval increase with higher β?
"""

from rl_evo_lab.experiment import Condition, Experiment

experiment = Experiment(
    name="cartpole_beta_diagnostic",
    env="cartpole",
    seeds=[42],
    conditions=[
        Condition("EDER_beta0.05", use_es=True, use_novelty=True, beta=0.05),
        Condition("EDER_beta0.1", use_es=True, use_novelty=True, beta=0.1),
        Condition("EDER_beta0.2", use_es=True, use_novelty=True, beta=0.2),
        Condition("ES+DQN", use_es=True, use_novelty=False),
    ],
    env_overrides={
        "total_episodes": 100,  # SHORT runs
    },
)

if __name__ == "__main__":
    experiment.main()
