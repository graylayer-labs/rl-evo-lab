"""Does EDER generalise to LunarLander, and does the buffer filter fix forgetting?

LunarLander-v3 is a harder benchmark than CartPole:
  - Longer episodes (~400 steps) give IDN more signal to learn from
  - Partial rewards (leg contact, proximity to pad) provide training signal
  - Diverse failure modes reward exploration-driven variety in the buffer

Observed behaviour (3 seeds each):
  EDER     — solves (peak 235-262) then crashes to -32–80. Forgetting is consistent.
  ES+DQN   — same pattern, less violent (peak 205-269, final 24-151).
  DQN      — solves and holds (peak 243-263, final 202-262). No forgetting.
  EDER-filtered — buffer push filter (alpha=0.5, top_k=7) under test.

The forgetting is ES-driven: post-solve workers flood the buffer with low-quality
transitions via FIFO eviction. EDER-filtered gates which workers can push to the
buffer using a combined fitness+novelty score, with a novelty floor override.

Run:
    python experiments/lunarlander_efficiency.py            # skip completed, run new
    python experiments/lunarlander_efficiency.py --force    # re-run everything
    python experiments/lunarlander_efficiency.py --show     # open plot when done
"""

from rl_evo_lab.experiment import Condition, Experiment

experiment = Experiment(
    name="lunarlander_efficiency",
    env="lunarlander",
    seeds=[7, 42, 123],
    conditions=[
        Condition("EDER", use_es=True, use_novelty=True),
        Condition("EDER-filtered", use_es=True, use_novelty=True,
                  buffer_push_alpha=0.5, buffer_push_top_k=7),
        Condition("ES+DQN", use_es=True, use_novelty=False),
        Condition("DQN", use_es=False, use_novelty=False),
    ],
)

if __name__ == "__main__":
    experiment.main()
