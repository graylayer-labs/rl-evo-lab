# Repo Quality Audit

Date: 2026-04-05
Repo: `rl-evo-lab`

## Verdict

This is a solid research-code repo with a readable architecture, good top-level docs, and a sensible experiment harness. It is buildable as a foundation for further work, but not yet a strong foundation for sustained agent-driven development because the repo has no enforced CI and its advertised lint/type checks are currently red.

## Findings

### 1. High: no enforcement layer for the standards the repo claims to use

Evidence:
- No `.github/workflows/` is present.
- `pyproject.toml` declares `pytest`, `ruff`, and `ty`.
- `README.md` only documents `pytest` as the verification step.
- Local verification result:
  - `pytest -q` passed
  - `ruff check .` failed
  - `ty check` failed

Impact:
- Quality gates are advisory rather than enforced.
- Regressions can accumulate quietly.

### 2. High: static-analysis failures exist in core runtime modules

Evidence:
- `src/rl_evo_lab/train.py`
  - constructed default config in function signature
  - nullable `actor` / `idn` values used without narrowing
  - lambda assigned to `env_fn`
- `src/rl_evo_lab/learner/dqn.py`
  - `float(done)` passed into a `bool` parameter
- `src/rl_evo_lab/utils/compare.py`
  - unresolved `Experiment` annotation
  - minor lint issues around `zip()` and unused loop variables
- `src/rl_evo_lab/utils/logging.py`
  - optional task id passed into progress update without narrowing

Impact:
- Core code is less safe to refactor than the test signal suggests.
- The repo advertises a type/lint culture that is not yet true in practice.

### 3. Medium: collaborator documentation has drifted from reality

Evidence:
- `CLAUDE.md` says the repo does not yet use `rl-core`.
- `pyproject.toml` already includes `rl-core` as a dependency.

Impact:
- Humans and agents can make wrong assumptions from stale guidance.
- Documentation becomes less trustworthy over time.

### 4. Medium: packaging and workflow story is slightly muddled

Evidence:
- `pyproject.toml` mixes PEP 621 `[project]`, `[dependency-groups]`, and `[tool.poetry.group.dev.dependencies]`.
- The repo is documented as straightforward Poetry-managed.

Impact:
- The canonical workflow is less clear than it should be.
- Future dependency maintenance will be noisier than necessary.

### 5. Low: local artifact sprawl is visible, even though git hygiene is okay

Evidence:
- `runs/`, `dist/`, `.pytest_cache/`, and `.ruff_cache/` exist locally.
- `.gitignore` ignores them correctly.
- `git ls-files` shows none of those paths are tracked.

Impact:
- Not a Git problem today.
- Still worth tightening cleanup conventions for long-running experimentation.

## What The Repo Is Doing Well

- Structure is coherent and easy to navigate.
- Architecture boundary is clear: actor, learner, replay buffer.
- File sizes are generally reasonable for a research codebase.
- `README.md` is strong and useful, especially the quick-start path, experiment table, and code-reading order.
- Tests are small but meaningful.
- `pytest -q` passed locally with 17 passing tests.

## Top Cleanup Priorities

1. Add CI that runs `pytest`, `ruff check`, and `ty check`.
2. Fix the current `ruff` and `ty` failures in runtime code.
3. Reconcile `CLAUDE.md`, `README.md`, and `pyproject.toml` so the tooling and dependency story is internally consistent.
4. Add focused tests for experiment runner and plotting/logging edge cases.

## Agent-Readiness

Conditionally yes.

The repo is readable enough for productive agent work, but until CI exists and the static checks are green, agents are more likely to preserve drift than reliably improve the codebase.

## Verification Run For This Audit

- `pytest -q` -> passed
- `ruff check .` -> 11 failures
- `ty check` -> 10 diagnostics
