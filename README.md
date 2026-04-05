# rl-evo-lab

Reproduction and extension of **EDER** (Evolutionary Distributed Experience Replay), from the MSc thesis *"Improving Exploration in Evolutionary Reinforcement Learning through Novelty Search"* — NUI Galway, 2021.

EDER replaces epsilon-greedy exploration with an **Evolution Strategy (ES) actor population** that fills the replay buffer with diverse transitions. A **DQN learner** trains purely from that buffer — no env interaction during training. An **intrinsic novelty reward** (KNN over learned state embeddings) keeps the ES population exploring new state regions rather than converging to a local optimum.

---

## Start here

If you want to understand the repo fast and see a result immediately:

1. Read the diagram in [Algorithm](#algorithm).
2. Run `poetry install`.
3. Run `poetry run python experiments/cartpole_efficiency.py --show`.
4. Open `runs/cartpole_efficiency/comparison.png`.

That first experiment compares the three core modes:

- `EDER`: ES actor + DQN learner + novelty
- `ES+DQN`: ES actor + DQN learner, no novelty
- `DQN`: pure epsilon-greedy DQN baseline

---

## What this repo is

This codebase is organized around one simple boundary:

- the **actor** explores
- the **learner** optimizes
- the **replay buffer** is the only interface between them

In ES mode, a population of perturbed policies interacts with the environment and fills the replay buffer. The learner trains only from that buffer. In pure DQN mode, the learner falls back to standard epsilon-greedy data collection.

---

## Setup

```bash
poetry install
poetry run pytest        # verify everything works
```

Requires Python ≥ 3.12. Core dependencies: `torch`, `gymnasium`, `numpy`.

---

## Fastest path to results

**Run the main CartPole comparison:**

```bash
poetry run python experiments/cartpole_efficiency.py --show
```

This will:

- train all missing seeds for `EDER`, `ES+DQN`, and `DQN`
- save per-run CSVs and configs under `runs/cartpole_efficiency/`
- save an aggregate plot to `runs/cartpole_efficiency/comparison.png`
- open the plot window if `--show` is passed

**Re-open the same plot later without retraining:**

```bash
poetry run python experiments/cartpole_efficiency.py --plot-only --show
```

**Run the LunarLander comparison:**

```bash
poetry run python experiments/lunarlander_efficiency.py --show
```

---

## Quick start from Python

**Run a single training job:**

```python
from rl_evo_lab.train import train
from rl_evo_lab.utils.config import make_config

cfg = make_config("lunarlander", seed=42)
train(cfg)
```

Or from the command line using an experiment script (see below).

---

## Experiments

Experiments live in `experiments/`. Each file defines:

- an environment preset
- a list of named conditions
- a fixed seed set
- a reproducible output directory under `runs/<experiment_name>/`

Each experiment runs multiple seeds, caches completed runs, and writes a comparison plot automatically.

**Run any experiment:**

```bash
poetry run python experiments/<name>.py            # run missing conditions, plot
poetry run python experiments/<name>.py --force    # re-run everything from scratch
poetry run python experiments/<name>.py --show     # open plot after saving
poetry run python experiments/<name>.py --workers 4  # limit parallel processes
```

Runs are **idempotent** — already-completed seeds are skipped unless `--force` is passed.

---

## Viewing results

There are two result levels:

**Experiment-level comparison**

After running an experiment, the main artifact is:

```text
runs/<experiment_name>/comparison.png
```

This aggregates all seeds for each condition and is the quickest way to understand the outcome.

**Single-run diagnostics**

Each seed gets its own directory:

```text
runs/<experiment_name>/<condition>__seed<seed>__<hash>/
  config.json
  metrics.csv
  status.json
```

To generate a summary plot for one run:

```bash
poetry run python -m rl_evo_lab.utils.plot runs/<path-to-run>/metrics.csv --show
```

Use this when you want to inspect one seed rather than the mean/std aggregate.

---

### Available experiments

This section is generated from the files in `experiments/`.

<!-- BEGIN AUTO:EXPERIMENTS -->
| Script | Environment | Question |
|---|---|---|
| `cartpole_eder_vs_baseline.py` | CartPole-v1 | Isolated novelty ablation: does IDN novelty help the ES actor? |
| `cartpole_efficiency.py` | CartPole-v1 | Does the ES actor improve sample efficiency vs pure DQN on CartPole? |
| `cartpole_model_size.py` | CartPole-v1 | Does ES diversity compensate for a smaller network? |
| `cartpole_sample_efficiency.py` | CartPole-v1 | Fair sample efficiency comparison: equal env-step budget across conditions. |
| `lunarlander_efficiency.py` | LunarLander-v3 | Does EDER generalise to LunarLander, and does the buffer filter fix forgetting? |
<!-- END AUTO:EXPERIMENTS -->

---

## Read the code in this order

If you want to understand the implementation without bouncing around:

1. `src/rl_evo_lab/train.py`
2. `src/rl_evo_lab/actor/es_actor.py`
3. `src/rl_evo_lab/actor/es_worker.py`
4. `src/rl_evo_lab/learner/dqn.py`
5. `src/rl_evo_lab/intrinsic/inverse_dynamics.py`
6. `src/rl_evo_lab/intrinsic/episodic_novelty.py`
7. `src/rl_evo_lab/experiment.py`

That sequence follows the actual runtime path.

---

### Adding a new experiment

Create a file in `experiments/`. A condition accepts any `EDERConfig` field as a keyword override:

```python
from rl_evo_lab.experiment import Condition, Experiment

experiment = Experiment(
    name="my_experiment",
    env="lunarlander",          # cartpole | lunarlander | acrobot | mountaincar
    seeds=[7, 42, 123],
    conditions=[
        Condition("EDER",          use_es=True,  use_novelty=True),
        Condition("EDER-filtered", use_es=True,  use_novelty=True,
                  buffer_push_alpha=0.5, buffer_push_top_k=7),
        Condition("DQN",           use_es=False, use_novelty=False),
    ],
)

if __name__ == "__main__":
    experiment.main()
```

---

## Algorithm

```
┌─────────────────────────────────────────────────────────┐
│  Actor (Evolution Strategy)                              │
│                                                          │
│  Each training episode:                                  │
│    1. Sample N noisy policies: θᵢ = θ + σεᵢ             │
│    2. Score each on augmented reward: rₐ = rₑ + β·rᵢ    │
│    3. ES gradient update on θ toward best directions     │
│    4. Push selected transitions → Replay Buffer          │
│    5. Periodically sync θ ← learner weights              │
│                                                          │
│  rᵢ = KNN distance in episodic memory of IDN embeddings  │
└─────────────────────┬───────────────────────────────────┘
                      │  shared replay buffer
┌─────────────────────▼───────────────────────────────────┐
│  Learner (DQN)                                           │
│                                                          │
│  - Trains on extrinsic reward only                       │
│  - Never interacts with the env during training          │
│  - Periodically broadcasts weights → Actor               │
└─────────────────────────────────────────────────────────┘
```

The replay buffer is the **only interface** between actor and learner.

---

## Key config options

All options live in `EDERConfig` (`src/rl_evo_lab/utils/config.py`). Use `make_config(env, **overrides)` to build one from an env preset.

This section is generated from `src/rl_evo_lab/utils/config.py`.

<!-- BEGIN AUTO:CONFIG -->
| Parameter | Default | Notes |
|---|---|---|
| `es_sigma` | `0.06` | ES noise std dev. Too small = no diversity; too large = divergence |
| `es_n_workers` | `50` | ES population size before env presets override it |
| `beta` | `0.02` | Intrinsic reward weight |
| `use_novelty` | `True` | False = ES+DQN baseline, no IDN |
| `use_es` | `True` | False = pure DQN with epsilon-greedy |
| `novelty_warmup_episodes` | `50` | Episodes before novelty activates; IDN trains silently |
| `solved_reward` | `475.0` | Reward at which convergence decay begins |
| `novelty_solve_decay` | `True` | Decays beta, sigma, and worker count as learner converges |

**Buffer push filtering**

| Parameter | Default | Notes |
|---|---|---|
| `buffer_push_alpha` | `None` | None = push all workers. 0.5 = equal fitness and novelty weight |
| `buffer_push_top_k` | `None` | Push only top-K workers by combined score |
| `buffer_novelty_floor` | `0.2` | Top fraction by raw novelty always enters the buffer |
<!-- END AUTO:CONFIG -->

---

## Results

The stable source of truth for results is the generated output under `runs/`, especially:

- `runs/<experiment_name>/comparison.png`
- `runs/<experiment_name>/manifest.json`
- per-seed `metrics.csv` files inside each run directory

Qualitatively, the project is centered on two recurring observations:

- On CartPole, the ES-driven variants learn much faster than pure DQN in episode terms.
- On harder environments such as LunarLander, ES improves exploration but can destabilize the replay buffer after solving, which is why the buffer-push filtering path exists.

---

## Repo structure

```
src/rl_evo_lab/
  actor/
    es_actor.py       # ESActor: runs generations, ES update, buffer push filtering
    es_worker.py      # WorkerResult, run_worker_episode
  learner/
    dqn.py            # DQNLearner: train_step, evaluate, collect_episode
    network.py        # QNetwork + FlatParamsMixin
  buffer/
    replay_buffer.py  # ReplayBuffer with diversity_metric()
  intrinsic/
    episodic_novelty.py     # EpisodicNovelty: KNN over embeddings (per-episode)
    inverse_dynamics.py     # InverseDynamicsNetwork: learns controllable-state embeddings
  utils/
    config.py         # EDERConfig dataclass + ENV_PRESETS + make_config()
    logging.py        # RunLogger: CSV + stdout + optional W&B
    seeding.py        # seed_everything()
  experiment.py       # Condition, Experiment: multi-seed parallel runner
  train.py            # train(): single run lifecycle

experiments/          # runnable experiment scripts
runs/                 # per-seed run dirs plus experiment-level comparison plots
tests/                # pytest suite
```

---

## References

- Khadka & Tumer (2018) — [ERL: Evolution-Guided Policy Gradient](https://arxiv.org/abs/1805.07917)
- Salimans et al. (2017) — [ES as a Scalable Alternative to RL](https://arxiv.org/abs/1703.03864)
- Badia et al. (2020) — [Never Give Up](https://arxiv.org/abs/2002.06038)
- Mnih et al. (2015) — [DQN](https://www.nature.com/articles/nature14236)
- Lillicrap et al. (2015) — [DDPG](https://arxiv.org/abs/1509.02971)
- Lehman & Stanley (2011) — [Novelty Search](https://dl.acm.org/doi/10.1145/1830483.1830503)
