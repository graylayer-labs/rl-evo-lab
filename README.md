# Novelty-Guided Evolutionary RL

## Research Question

**In exploration-stuck environments, can novelty-driven search and/or evolutionary strategies actually improve RL — or are they overhead?**

This thesis evaluates three algorithmic approaches to see which techniques (if any) solve the exploration problem:
- **Pure RL** (DQN with ε-greedy) — baseline
- **Evolutionary RL** (ES actor + DQN learner, no intrinsic motivation) — isolation of ES
- **Novelty-Guided RL** (ES + novelty-based intrinsic reward) — full hybrid approach

### Three Methods Under Test
1. **DQN** — Pure gradient-based learning (ε-greedy baseline)
   - What it tests: Can standard RL solve exploration-stuck tasks?

2. **Evolutionary RL (ES+DQN)** — ES population + DQN learner, no intrinsic reward
   - What it tests: Does population diversity alone compensate for sparse reward?

3. **Novelty-Guided RL** — ES population + DQN learner + novelty-based intrinsic reward
   - What it tests: Does adding intrinsic motivation improve exploration further?

We test across a spectrum of environments specifically chosen to show where standard RL fails and what techniques succeed.

---

## Phase 2: DQN Baseline Complete ✓ | Phase 3: Ready to Begin

### Baseline Results (DQN Only)

| Environment | Type | DQN Mean | Target | Result | Next Test |
|---|---|---|---|---|---|
| CartPole-v1 | Dense | 151.7 | 475 | ❌ 32% | ES alone? Novelty-guided? |
| LunarLander-v3 | Dense | 215.5 | 200 | ✅ Solved | Does novelty add overhead? |
| CartPole-sparse | Sparse | 0.0 | 475 | ❌ 0% | Does ES bridge the gap? |
| Acrobot-v1 | Sparse | -500 | -100 | ❌ 0% | Can novelty guide discovery? |
| Montezuma's Revenge | Hard | — | — | — | Deferred (after Phase 3) |

### What the Baseline Shows

- **Dense tasks (CartPole, LunarLander)**: DQN struggles or barely succeeds → test if ES/novelty add overhead
- **Sparse tasks (CartPole-sparse, Acrobot)**: DQN fails completely → this is where exploration techniques must prove value
- **Clear baseline**: Ground truth showing exactly where and why RL gets stuck

### Phase 3 Experimental Design

Compare three approaches on the same environment spectrum:

| Method | What It Isolates | Expected Pattern |
|---|---|---|
| **DQN** (baseline) | Pure RL with ε-greedy | Fails on sparse tasks |
| **Evolutionary RL** | ES population diversity alone | Helps on sparse? Overhead on dense? |
| **Novelty-Guided RL** | ES + novelty signal combined | Better than ES alone? Synergize? |

**How to interpret Phase 3 results:**

- **If Evolutionary RL >> DQN on CartPole-sparse**: ES diversity is key; novelty may not be needed
- **If Novelty-Guided RL >> Evolutionary RL on CartPole-sparse**: Novelty guides ES effectively
- **If both fail**: Mechanism doesn't work as theorized; deeper investigation needed
- **If Novelty-Guided RL fails on LunarLander**: Intrinsic motivation hurts precision (adds overhead)

All comparisons use **environment steps** (fair compute budget across methods).

---

## If You Just Want Results

Each environment's comparison plot is the primary artifact:

```
runs/cartpole_baseline_dqn/comparison.png
runs/cartpole_evolutionary_rl/comparison.png
runs/cartpole_novelty_guided_rl/comparison.png
```

Stack these plots side-by-side to see all three methods on CartPole.

See [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md) for detailed hypotheses and rationale.

---

## What Each Algorithm Tests

- **DQN alone fails where?** Identifies the exploration problem
- **ES+DQN solves it?** If yes → ES diversity (population) is sufficient  
- **Need novelty too?** If only EDER succeeds → intrinsic motivation is necessary

This isolates **which technique(s) actually solve exploration-stuck problems**.

---

## Why This Repo Exists

- **Hybrid RL architecture**: Clean actor/learner separation through replay buffer only—neither component depends on the other's internals
- **Environment spectrum**: Not arbitrary benchmarks, but strategically chosen to isolate whether exploration is the bottleneck
- **Fair comparison**: All algorithms compared by environment steps, not episodes (ES gets 50× more interaction per episode; plotting by episode is misleading)
- **Honest results**: What works, what doesn't, what's open — no hand-waving or hidden tuning
- **Reproducibility**: Multi-seed runs, cached reruns, automatic result aggregation

---

## Documentation

**For Users:**
- **[docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md)** — Why this 5-environment spectrum? Testing methodology and explicit hypotheses.

**For Team/Contributors:**
- **[docs/internal/PHASE_STATUS.md](docs/internal/PHASE_STATUS.md)** — Current research phase, progress, findings, ETA
- **[docs/internal/WORK_LOG.md](docs/internal/WORK_LOG.md)** — Session notes, decisions, implementation log
- **[docs/internal/CLAUDE.md](docs/internal/CLAUDE.md)** — Architecture notes, design invariants, AI assistant guidance
- **[docs/internal/CHANGELOG.md](docs/internal/CHANGELOG.md)** — Historical change log

---

## Start Here

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

## The Architecture: What This Repo Is

This codebase is built around a single, clean boundary:

```
Actor (ES population)  →  [Shared Replay Buffer]  →  Learner (DQN)
```

- The **actor** (ES population of 50 policies) generates diverse experiences via parameter noise
- The **learner** (DQN) trains purely from the replay buffer, on extrinsic reward only
- The **novelty signal** (optional, episodic KNN bonus) biases which actor experiences enter the buffer

This separation means:
- Actor and learner are independently swappable (ES ↔ A3C, DQN ↔ SAC)
- Novelty is purely internal to the actor — the learner trains on clean extrinsic reward
- We can test ES, novelty, and combinations independently (DQN, ES+DQN, EDER)

---

## Setup

```bash
poetry install
poetry run pytest        # verify everything works
```

Requires Python ≥ 3.12. Core dependencies: `torch`, `gymnasium`, `numpy`.

---

## Running Phase 3: Complete the Comparison

DQN baselines are done. Now run the other two approaches to compare:

**Option A: Run all three approaches on one environment**
```bash
# CartPole: see how DQN, ES, and novelty compare
python experiments/baseline_dqn.py --env cartpole --plot-only --show
python experiments/evolutionary_rl.py --env cartpole --show
python experiments/novelty_guided_rl.py --env cartpole --show
```

**Option B: Run full Phase 3 across all environments**
```bash
python experiments/evolutionary_rl.py --all
python experiments/novelty_guided_rl.py --all
```
(This takes 1-2 hours for all seeds/envs. Run in parallel on multiple machines if available.)

**Option C: Compare results without retraining**
```bash
python experiments/evolutionary_rl.py --env cartpole --plot-only --show
python experiments/novelty_guided_rl.py --env cartpole --plot-only --show
```

Results generate comparison plots automatically (mean ± std across seeds) under `runs/`.

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
| `acrobot_exploration.py` | Acrobot-v1 | Does EDER shine on hard exploration? Acrobot-v1 trial. |
| `cartpole_beta_diagnostic.py` | CartPole-v1 | CartPole-v1 Beta Diagnostic: Find optimal novelty weight. |
| `cartpole_eder_vs_baseline.py` | CartPole-v1 | Isolated novelty ablation: does IDN novelty help the ES actor? |
| `cartpole_efficiency.py` | CartPole-v1 | Does the ES actor improve sample efficiency vs pure DQN on CartPole? |
| `cartpole_model_size.py` | CartPole-v1 | Does ES diversity compensate for a smaller network? |
| `cartpole_normal.py` | CartPole-v1 | CartPole-v1 Normal: Standard benchmark to validate baseline performance. |
| `cartpole_novelty_ramp_diagnostic.py` | CartPole-v1 | CartPole Novelty Ramp Diagnostic: Quick 100-episode tests to identify ramp issue. |
| `cartpole_sample_efficiency.py` | CartPole-v1 | Fair sample efficiency comparison: equal env-step budget across conditions. |
| `cartpole_sigma_sweep.py` | CartPole-v1 | CartPole-v1 Normal: Diagnostic sweep for optimal es_sigma. |
| `cartpole_tough.py` | CartPole-v1 | CartPole-Tough: Real robustness test for learned control. |
| `diagnostic_phase_123_35.py` | CartPole-v1 | Diagnostic experiment: Phase 1-4 + Phase 3.1+3.2+3.3 smoke test. |
| `lunarlander_efficiency.py` | LunarLander-v3 | Does EDER generalise to LunarLander, and does the buffer filter fix forgetting? |
| `lunarlander_normal.py` | LunarLander-v3 | LunarLander-v3 Normal: Standard benchmark on longer-horizon task. |
| `lunarlander_tough.py` | LunarLander-v3 | LunarLander-Tough: Real robustness test for precision landing control. |
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

## Data & Reproducibility

The source of truth for results is the generated output under `runs/`:

- `runs/<experiment_name>/comparison.png` — comparison plot (mean ± std across seeds)
- `runs/<experiment_name>/manifest.json` — experiment metadata and run registry
- per-seed `metrics.csv` files — detailed learning curves and diagnostics for each run
- per-seed `config.json` and `policy.pt` — experiment config and trained Q-network weights

Each experiment is fully reproducible: re-running the same script reruns only missing conditions and seeds; already-completed runs are skipped unless `--force` is passed.

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
