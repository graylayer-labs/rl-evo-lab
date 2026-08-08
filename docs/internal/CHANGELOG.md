# Changelog

---

## 2026-03-28

### Added — Selective buffer push filtering (`buffer_push_alpha`)

**Problem:** EDER solves LunarLander (peak 235–262) then crashes to -32–80 across all 3 seeds. Post-solve ES workers with noisy perturbations flood the replay buffer with low-quality, repetitive transitions via FIFO eviction, overwriting the high-reward experiences that solved the task. The forgetting is ES-driven — ES+DQN (no novelty) also degrades.

**Fix:** Episode-level selective push — only push a worker's transitions if that episode was sufficiently novel OR sufficiently valuable.

- `push_score = α × fitness_rank + (1-α) × novelty_rank`
- Novelty floor override: top `buffer_novelty_floor` fraction by raw novelty always enters regardless of combined score

**Files changed:**

| File | Change |
|---|---|
| `src/rl_evo_lab/actor/es_worker.py` | Added `mean_novelty: float` to `WorkerResult`; collects per-step intrinsic rewards during episode and computes mean |
| `src/rl_evo_lab/actor/es_actor.py` | Added `_select_workers_to_push()`; replaced unconditional push loop |
| `src/rl_evo_lab/utils/config.py` | Added `buffer_push_alpha`, `buffer_push_top_k`, `buffer_novelty_floor` to `EDERConfig` |
| `tests/test_es.py` | 4 new tests: backward compat, top-K filtering, novelty floor override, balanced alpha |

**Config:** `buffer_push_alpha=None` (default) — push everything, no behaviour change. To enable: `buffer_push_alpha=0.5, buffer_push_top_k=7` (for LunarLander, 10 workers).

**Active experiment:** `EDER-filtered` condition added to `experiments/lunarlander_efficiency.py`. Running now.

---

### Added — `ruff` and `ty` as dev dependencies

- `ruff ^0.15.8` — linting + formatting (replaces flake8/isort/black)
- `ty ^0.0.26` — type checking
- Fixed `pyproject.toml` ty config: `python-version` → `[tool.ty.environment] python-version`
- 9 pre-existing ty errors in `train.py`, `compare.py`, `logging.py`, `dqn.py` — not introduced by this session

---

### Updated — Documentation

- `README.md` — full rewrite: setup, quick start, experiment table, how to add conditions, key config reference (including new filter params), results table (CartPole + LunarLander), repo structure
- All 5 experiment docstrings updated with observed results, hypotheses, and correct run commands
- `experiments/lunarlander_efficiency.py` — added `EDER-filtered` condition

---

## Prior sessions

### Convergence decay (beta/sigma/n_workers)
As learner eval crosses `novelty_decay_start_reward → solved_reward`, all three decay toward their minimum values. Prevents ES population destabilising a solved learner. See `ESActor._convergence_progress()`.

### Global novelty buffer + query/add split
Cross-generation novelty memory. `score()` split into `query()` (read-only, safe for concurrent workers) and `add()` (called after all workers finish). Workers cache embeddings in `WorkerResult.embeddings`.

### Experiment/Condition API
`Experiment` + `Condition` in `src/rl_evo_lab/experiment.py`. Multi-seed parallel runner with idempotent runs, progress display, comparison plots.

### LunarLander baseline runs
3 seeds × 3 conditions (EDER, ES+DQN, DQN). Confirmed catastrophic forgetting in EDER and ES+DQN. DQN holds solution. See `runs/lunarlander_efficiency/`.
