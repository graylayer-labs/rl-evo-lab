"""Export final numbers from runs/ (local, gitignored) into results/ (tracked).

Reads each experiment's manifest.json + per-seed status.json (written by
RunLogger.close() after training completes), computes the mean/std of the
authoritative post-training final_eval across seeds, and writes:
  - results/results.json      — all numbers, machine-readable
  - results/RESULTS.md        — human-readable summary table
  - results/<env>/<slug>.png  — comparison plots, nested per environment

Run after any batch of experiments completes:
    python scripts/build_results.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from statistics import mean, pstdev

from infra.config import ENV_PRESETS

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"

ENVS = ["cartpole", "lunarlander", "cartpole_sparse", "acrobot", "montezuma"]
CONDITIONS = ["DDQN", "DDQN+ES", "DDQN+Novelty", "DDQN+ES+Novelty"]

# Maps the suffix of an experiment name (env stripped off) to the plot slug
# used under results/<env>/<slug>.png — mirrors the experiments/ddqn/ filenames.
EXPERIMENT_SLUG = {
    "baseline_dqn": "ddqn",
    "evolutionary_rl": "es",
    "novelty_only_rl": "novelty",
    "novelty_guided_rl": "es_novelty",
    "all_methods_comparison": "compare_all",
}


def _final_eval(metrics_csv: Path) -> dict | None:
    status_path = metrics_csv.parent / "status.json"
    if not status_path.exists():
        return None
    status = json.loads(status_path.read_text())
    return status.get("final_eval")


def _collect() -> dict[str, dict[str, dict]]:
    results: dict[str, dict[str, dict]] = {env: {} for env in ENVS}

    for manifest_path in sorted(RUNS_DIR.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        env = manifest["env"]
        if env not in results:
            continue
        solved_reward = ENV_PRESETS[env]["solved_reward"]

        for label, csv_paths in manifest["conditions"].items():
            per_seed_means = []
            n_eval_episodes = None
            all_episode_rewards = []
            for rel_path in csv_paths:
                final_eval = _final_eval(ROOT / rel_path)
                if final_eval is None:
                    continue
                per_seed_means.append(final_eval["mean"])
                n_eval_episodes = final_eval["n_episodes"]
                all_episode_rewards.append(final_eval["episode_rewards"])

            if not per_seed_means:
                continue

            agg_mean = mean(per_seed_means)
            results[env][label] = {
                "mean": round(agg_mean, 1),
                "std": round(pstdev(per_seed_means), 1) if len(per_seed_means) > 1 else 0.0,
                "n_seeds": len(per_seed_means),
                "eval_episodes": n_eval_episodes,
                "solved": agg_mean >= solved_reward,
                "source_experiment": manifest["experiment"],
                "episode_rewards_per_seed": all_episode_rewards,
            }

    return results


def _write_json(results: dict) -> None:
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2) + "\n")


def _write_markdown(results: dict) -> None:
    lines = [
        "# Results\n",
        "Mean ± std of the post-training `final_eval` (fresh held-out episodes,",
        "greedy policy) across seeds. `Solved?` compares the mean against each",
        "environment's `solved_reward`.\n",
    ]
    for env in ENVS:
        lines.append(f"## {env}\n")
        lines.append("| Condition | Mean | Std | Seeds | Eval Episodes | Solved? |")
        lines.append("|---|---|---|---|---|---|")
        for cond in CONDITIONS:
            data = results.get(env, {}).get(cond)
            if data is None:
                lines.append(f"| {cond} | — | — | — | — | not run |")
            else:
                solved = "✓" if data["solved"] else "✗"
                lines.append(
                    f"| {cond} | {data['mean']} | {data['std']} | {data['n_seeds']} | "
                    f"{data['eval_episodes']} | {solved} |"
                )
        lines.append("")
    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(lines) + "\n")


def _copy_plots() -> None:
    for manifest_path in sorted(RUNS_DIR.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        env = manifest["env"]
        if env not in ENVS:
            continue
        plot_src = manifest_path.parent / "comparison.png"
        if not plot_src.exists():
            continue

        suffix = manifest["experiment"].removeprefix(f"{env}_")
        slug = EXPERIMENT_SLUG.get(suffix, suffix)
        env_dir = RESULTS_DIR / env
        env_dir.mkdir(exist_ok=True)
        shutil.copy2(plot_src, env_dir / f"{slug}.png")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    results = _collect()
    _write_json(results)
    _write_markdown(results)
    _copy_plots()
    print(f"Wrote results.json, RESULTS.md, and comparison plots to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
