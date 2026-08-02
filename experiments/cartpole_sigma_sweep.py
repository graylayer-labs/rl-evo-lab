"""CartPole-v1 Normal: Diagnostic sweep for optimal es_sigma.

Quick test to find optimal noise level for ES population diversity.

Test 3 sigma values on CartPole Normal with SHORT runs (100 episodes, seed=42):
- σ=0.1 (baseline 0.06 × 1.67)
- σ=0.2 (baseline × 3.33)
- σ=0.3 (baseline × 5)

Measures:
- Which σ creates biggest worker diversity?
- Which σ lets EDER beat or match DQN?
- Trade-off: too high σ might thrash, too low won't explore

Run:
    poetry run python experiments/cartpole_sigma_sweep.py --force --workers 2
"""

from rl_evo_lab.experiment import Condition, Experiment

experiment = Experiment(
    name="cartpole_sigma_sweep",
    env="cartpole",
    seeds=[42],  # single seed for speed
    conditions=[
        Condition("EDER_σ0.1", use_es=True, use_novelty=True, es_sigma=0.1),
        Condition("EDER_σ0.2", use_es=True, use_novelty=True, es_sigma=0.2),
        Condition("EDER_σ0.3", use_es=True, use_novelty=True, es_sigma=0.3),
    ],
    env_overrides={
        "total_episodes": 100,  # short diagnostic run
        "eval_freq": 5,  # evaluate every 5 episodes for faster feedback
    },
)

if __name__ == "__main__":
    experiment.main()
