# Novelty-Guided Evolutionary RL

## Research Question

**In exploration-stuck environments, can novelty-driven search and/or evolutionary strategies actually improve RL — or are they overhead?**

This thesis evaluates a **2×2 factorial study** on top of Double DQN to see which
techniques (if any) solve the exploration problem:

### Four Methods Under Test
1. **DDQN** — Pure gradient-based learning (ε-greedy baseline)
   - What it tests: Can standard RL solve exploration-stuck tasks?

2. **DDQN + ES** — ES population + DDQN learner, no intrinsic reward
   - What it tests: Does population diversity alone compensate for sparse reward?

3. **DDQN + Novelty** — DDQN learner + novelty-based intrinsic reward, no ES
   - What it tests: Does intrinsic motivation alone improve exploration?

4. **DDQN + ES + Novelty** — full hybrid approach
   - What it tests: Do ES and novelty synergize, or does one dominate?

We test across a spectrum of 5 environments (dense→sparse reward, easy→hard
exploration) specifically chosen to show where standard RL fails and what
techniques succeed. See [Study Design](#study-design) below.

---

## Study Design

| Method | What It Isolates | Expected Pattern |
|---|---|---|
| **DDQN** (baseline) | Pure RL with ε-greedy | Fails on sparse tasks |
| **DDQN + ES** | ES population diversity alone | Helps on sparse? Overhead on dense? |
| **DDQN + Novelty** | Intrinsic motivation alone | Helps discovery without population cost? |
| **DDQN + ES + Novelty** | ES + novelty signal combined | Better than either alone? Synergize? |

Environments: CartPole-v1, LunarLander-v3, CartPole-sparse, Acrobot-v1,
MontezumaRevenge — each run across 3 seeds for all 4 methods (20 experiments
total). All comparisons use **environment steps** (fair compute budget across
methods).

### Status

No experiments have been run yet under this scope — `results/` is currently
empty. All 20 experiments (4 methods × 5 environments × 3 seeds) are pending.

---

## If You Just Want Results

Once experiments have run, see [`results/RESULTS.md`](results/RESULTS.md) for
the full comparison table and [`results/`](results/) for per-environment
comparison plots — this is the tracked, published source of truth (not the
local `runs/` working directory, which holds raw per-seed logs and isn't
checked into the repo). Regenerate `results/` at any point with:

```bash
uv run python scripts/build_results.py
```

---

## Why This Repo Exists

- **Hybrid RL architecture**: Clean actor/learner separation through replay buffer only—neither component depends on the other's internals
- **Environment spectrum**: Not arbitrary benchmarks, but strategically chosen to isolate whether exploration is the bottleneck
- **Fair comparison**: All algorithms compared by environment steps, not episodes (ES gets 50× more interaction per episode; plotting by episode is misleading)
- **Honest results**: What works, what doesn't, what's open — no hand-waving or hidden tuning
- **Reproducibility**: Multi-seed runs, cached reruns, automatic result aggregation

---

## Environment Rationale

| Tier | Environment | Why it's here |
|---|---|---|
| Dense baseline | CartPole-v1 | Trivial exploration; proves the baseline works |
| Dense, precision | LunarLander-v3 | Continuous control; tests whether novelty adds overhead when precision matters |
| Sparse, simple | CartPole-sparse | Zero per-step reward; isolates whether ES/novelty compensate for gradient starvation |
| Sparse, discovery | Acrobot-v1 | Rare-behavior discovery; standard sparse-reward benchmark |
| Hard exploration | Montezuma's Revenge (RAM) | Canonical novelty-search benchmark; falsification test for the whole thesis |

For implementation details, architecture notes, and decision history, see git log and code comments.

---

## Start Here

If you want to understand the repo fast and see a result immediately:

1. Read the diagram in [Algorithm](#algorithm).
2. Run `uv sync`.
3. Run `uv run python experiments/ddqn/baseline.py --env cartpole --plot-only --show`.
4. Open the comparison plot it prints the path to.

That first experiment is the DDQN baseline (ε-greedy, no ES, no novelty) —
the ground truth every other condition is compared against.

---

## The Architecture: What This Repo Is

This codebase is built around a single, clean boundary:

```
Actor (ES population)  →  [Shared Replay Buffer]  →  Learner (DDQN)
```

- The **actor** (ES population of 50 policies) generates diverse experiences via parameter noise
- The **learner** (DDQN) trains purely from the replay buffer, on extrinsic reward only
- The **novelty signal** (optional, episodic KNN bonus) biases which actor experiences enter the buffer

This separation means:
- Actor and learner are independently swappable (ES ↔ A3C, DQN ↔ SAC)
- Novelty is purely internal to the actor — the learner trains on clean extrinsic reward
- We can test ES, novelty, and combinations independently (DDQN, DDQN+ES, DDQN+Novelty, DDQN+ES+Novelty)

---

## Setup

```bash
uv sync
uv run pytest             # verify everything works
```

Requires Python ≥ 3.12. Core dependencies: `torch`, `gymnasium`, `numpy`.

---

## Running Experiments

**Run one condition on one environment**
```bash
uv run python experiments/ddqn/baseline.py --env cartpole --show
uv run python experiments/ddqn/es.py --env cartpole --show
uv run python experiments/ddqn/es_novelty.py --env cartpole --show
```

**Run everything for one environment across conditions**
```bash
uv run python experiments/ddqn/es.py --all
uv run python experiments/ddqn/es_novelty.py --all
```
(This takes 1-2 hours for all seeds/envs. Run in parallel on multiple machines if available.)

**Compare results without retraining**
```bash
uv run python experiments/ddqn/es.py --env cartpole --plot-only --show
uv run python experiments/ddqn/es_novelty.py --env cartpole --plot-only --show
```

Results generate comparison plots automatically (mean ± std across seeds) under `runs/` locally; final published numbers live in `results/`.

---

## Quick start from Python

**Run a single training job:**

```python
from runner.train import train
from infra.config import make_config

cfg = make_config("lunarlander", seed=42)
train(cfg)
```

Or from the command line using an experiment script (see below).

---

## Experiments

Experiments are grouped by base learner: `experiments/ddqn/` holds every
condition built on DDQN. If the base learner is ever swapped or extended
(e.g. SAC), it gets its own sibling directory (`experiments/sac/`) rather than
disturbing this one. Each file defines:

- an environment preset
- a list of named conditions
- a fixed seed set
- a reproducible output directory under `runs/<experiment_name>/`

Each experiment runs multiple seeds, caches completed runs, and writes a comparison plot automatically.

**Run any experiment:**

```bash
uv run python experiments/ddqn/<name>.py            # run missing conditions, plot
uv run python experiments/ddqn/<name>.py --force    # re-run everything from scratch
uv run python experiments/ddqn/<name>.py --show     # open plot after saving
uv run python experiments/ddqn/<name>.py --workers 4  # limit parallel processes
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
uv run python -m infra.plot runs/<path-to-run>/metrics.csv --show
```

Use this when you want to inspect one seed rather than the mean/std aggregate.

---

### Available experiments

| Script | Compares | Environments |
|---|---|---|
| `ddqn/baseline.py` | DDQN alone | All 5 |
| `ddqn/es.py` | DDQN + ES | All 5 |
| `ddqn/novelty.py` | DDQN + Novelty | All 5 |
| `ddqn/es_novelty.py` | DDQN + ES + Novelty | All 5 |
| `ddqn/compare_all.py` | All 4 conditions side-by-side | Any single env or `--all-envs` |

---

## Read the code in this order

If you want to understand the implementation without bouncing around:

1. `src/runner/train.py`
2. `src/actor/es_actor.py`
3. `src/actor/es_worker.py`
4. `src/learner/dqn.py`
5. `src/intrinsic/inverse_dynamics.py`
6. `src/intrinsic/episodic_novelty.py`
7. `src/runner/experiment.py`

That sequence follows the actual runtime path.

---

### Adding a new experiment

Create a file in `experiments/`. A condition accepts any `EDERConfig` field as a keyword override:

```python
from runner.experiment import Condition, Experiment

experiment = Experiment(
    name="my_experiment",
    env="lunarlander",          # cartpole | lunarlander | cartpole_sparse | acrobot | montezuma
    seeds=[7, 42, 123],
    conditions=[
        Condition("EDER",          use_es=True,  use_novelty=True),
        Condition("EDER-filtered", use_es=True,  use_novelty=True,
                  buffer_push_alpha=0.5, buffer_push_top_k=7),
        Condition("DDQN",           use_es=False, use_novelty=False),
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
│  Learner (DDQN)                                          │
│                                                          │
│  - Trains on extrinsic reward only                       │
│  - Never interacts with the env during training          │
│  - Periodically broadcasts weights → Actor               │
└─────────────────────────────────────────────────────────┘
```

The replay buffer is the **only interface** between actor and learner.

---

## Key config options

All options live in `EDERConfig` (`src/infra/config.py`). Use `make_config(env, **overrides)` to build one from an env preset.

This section is generated from `src/infra/config.py`.

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

## Data & Reproducibility

`runs/` is local, working, per-seed training output — not tracked in git:

- `runs/<experiment_name>/comparison.png` — comparison plot (mean ± std across seeds)
- `runs/<experiment_name>/manifest.json` — experiment metadata and run registry
- per-seed `metrics.csv` files — detailed learning curves and diagnostics for each run
- per-seed `config.json` and `policy.pt` — experiment config and trained Q-network weights

Each experiment is fully reproducible: re-running the same script reruns only missing conditions and seeds; already-completed runs are skipped unless `--force` is passed.

`results/` is the tracked, published subset of this data — final numbers only,
regenerated from `runs/` once a set of conditions is complete.

---

## Repo structure

```
src/
  actor/               # ES population — learner-agnostic
    es_actor.py       # ESActor: runs generations, ES update, buffer push filtering
    es_worker.py      # WorkerResult, run_worker_episode
  learner/             # base-learner-specific (currently DDQN only)
    dqn.py            # DQNLearner: implements Double DQN (target/policy net decoupling)
    network.py        # QNetwork + FlatParamsMixin
  buffer/              # replay buffer — learner-agnostic
    replay_buffer.py  # ReplayBuffer with diversity_metric()
  intrinsic/           # novelty signal — learner-agnostic
    episodic_novelty.py     # EpisodicNovelty: KNN over embeddings (per-episode)
    inverse_dynamics.py     # InverseDynamicsNetwork: learns controllable-state embeddings
  envs/
    wrappers.py        # CartPoleSparseWrapper, CartPoleToughWrapper, LunarLanderToughWrapper, make_env_with_config
  runner/              # orchestration
    experiment.py      # Condition, Experiment: multi-seed parallel runner
    train.py           # train(): single run lifecycle
  infra/
    config.py         # EDERConfig dataclass + ENV_PRESETS + make_config()
    logging.py        # RunLogger: CSV + stdout + optional W&B
    seeding.py        # seed_everything()

experiments/
  ddqn/                # all conditions built on DDQN (baseline, es, novelty, es_novelty, compare_all)
scripts/
  build_results.py    # exports runs/ -> results/
results/              # published final numbers (tracked)
runs/                 # per-seed run dirs plus experiment-level comparison plots (local only)
tests/                # pytest suite
```

---

## References

**Foundational (ES + RL hybrid approach)**
- Khadka & Tumer (2018) — [ERL: Evolution-Guided Policy Gradient](https://arxiv.org/abs/1805.07917)
- Salimans et al. (2017) — [ES as a Scalable Alternative to RL](https://arxiv.org/abs/1703.03864)
- Lehman & Stanley (2011) — [Novelty Search](https://dl.acm.org/doi/10.1145/1830483.1830503)

**Core RL algorithms**
- Mnih et al. (2015) — [DQN](https://www.nature.com/articles/nature14236)
- Lillicrap et al. (2015) — [DDPG](https://arxiv.org/abs/1509.02971)

**Intrinsic motivation & exploration (episodic + lifelong)**
- Badia et al. (2020) — [Never Give Up (NGU)](https://arxiv.org/abs/2002.06038) — episodic KNN + RND lifelong; validates our IDN-KNN approach
- Badia et al. (2020) — [Agent57](https://arxiv.org/abs/2003.13350) — population of policies with diverse exploration coefficients
- Ecoffet et al. (2021) — [Go-Explore](https://arxiv.org/abs/2010.04286) — memory + return to promising states

**Quality-Diversity (QD) + RL (next-generation hybrid)**
- Nilsson & Cully (2021) — [PGA-MAP-Elites](https://arxiv.org/abs/2105.01016) — ES maintains behavioral archive, learner improves each cell
- Tjanaka et al. (2023) — [ERL-Re2](https://arxiv.org/abs/2309.11842) — fixes catastrophic forgetting via behavior-level operators
