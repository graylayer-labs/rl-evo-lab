"""LunarLander-Tough: Real robustness test for precision landing control.

Modified LunarLander with harder requirements to test genuine learning:
- Starting: random position/velocity each episode (not fixed spawn)
  - Position: ±0.3 horizontal, ±0.1 vertical from center
  - Velocity: ±0.5 m/s random perturbation
- Termination: must land within ±0.1 units of pad center (tight zone)
- Episode limit: 2000 steps (vs default 1000) — more time to recover
- Success: episode reward > 250 (successful tight-zone landing)

This is a genuine robustness test: can the policy handle variable starting conditions
and deliver precise control? A policy that learned to exploit a fixed starting state
won't handle arbitrary perturbations or landing precision requirements.

Run:
    python experiments/lunarlander_tough.py
    python experiments/lunarlander_tough.py --force --show
"""

from rl_evo_lab.experiment import Condition, Experiment

experiment = Experiment(
    name="lunarlander_tough",
    env="lunarlander_tough",
    seeds=[7, 42, 123],
    conditions=[
        Condition("EDER", use_es=True, use_novelty=True),
        Condition("ES+DQN", use_es=True, use_novelty=False),
        Condition("DQN", use_es=False, use_novelty=False),
    ],
)

if __name__ == "__main__":
    experiment.main()
