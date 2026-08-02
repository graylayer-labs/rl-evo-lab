"""CartPole Novelty Ramp Diagnostic: Quick 100-episode tests to identify ramp issue.

Tests 3 novelty_ramp_episodes values to determine if the ramp schedule causes
the collapse observed around ep 100-200 in full runs.

Run:
    python experiments/cartpole_novelty_ramp_diagnostic.py
    python experiments/cartpole_novelty_ramp_diagnostic.py --force
"""

from rl_evo_lab.experiment import Condition, Experiment

experiment = Experiment(
    name="cartpole_novelty_ramp_diagnostic",
    env="cartpole",
    seeds=[42],  # single seed for quick diagnostic
    conditions=[
        Condition("ramp=0 (no ramp)", use_es=True, use_novelty=True, novelty_ramp_episodes=0),
        Condition("ramp=50 (original)", use_es=True, use_novelty=True, novelty_ramp_episodes=50),
        Condition("ramp=200 (delayed)", use_es=True, use_novelty=True, novelty_ramp_episodes=200),
    ],
    env_overrides={
        "total_episodes": 100,  # quick diagnostic only
        "eval_freq": 5,  # eval more frequently on short run
        "novelty_warmup_episodes": 50,  # keep constant
    },
)

if __name__ == "__main__":
    experiment.main()
