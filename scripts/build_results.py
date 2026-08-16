"""Export final numbers from runs/ (local, gitignored) into results/ (tracked).

Reads each experiment's manifest.json + per-seed metrics.csv, computes the
mean/std of the final learner_eval_reward across seeds, and writes:
  - results/results.json  — all numbers, machine-readable
  - results/RESULTS.md    — human-readable summary table

Run after any batch of experiments completes:
    python scripts/build_results.py
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"

# Old runs/ data was written before the DQN -> DDQN rename; map old condition
# labels found in existing manifests to the current naming.
LABEL_MAP = {
    "DQN": "DDQN",
    "Evolutionary RL": "DDQN+ES",
    "Novelty-Guided RL": "DDQN+ES+Novelty",
    "DDQN+Novelty": "DDQN+Novelty",
}

ENVS = ["cartpole", "lunarlander", "cartpole_sparse", "acrobot", "montezuma"]
CONDITIONS = ["DDQN", "DDQN+ES", "DDQN+Novelty", "DDQN+ES+Novelty"]


def _final_eval_reward(metrics_csv: Path) -> float | None:
    with open(metrics_csv) as f:
        rows = list(csv.DictReader(f))
    for row in reversed(rows):
        val = row.get("learner_eval_reward", "")
        if val:
            return float(val)
    return None


def _collect() -> dict[str, dict[str, dict]]:
    results: dict[str, dict[str, dict]] = {env: {} for env in ENVS}

    for manifest_path in sorted(RUNS_DIR.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        env = manifest["env"]
        if env not in results:
            continue

        for raw_label, csv_paths in manifest["conditions"].items():
            label = LABEL_MAP.get(raw_label, raw_label)
            rewards = []
            for rel_path in csv_paths:
                reward = _final_eval_reward(ROOT / rel_path)
                if reward is not None:
                    rewards.append(reward)

            if not rewards:
                continue

            results[env][label] = {
                "mean": round(mean(rewards), 1),
                "std": round(pstdev(rewards), 1) if len(rewards) > 1 else 0.0,
                "n_seeds": len(rewards),
                "source_experiment": manifest["experiment"],
            }

    return results


def _write_json(results: dict) -> None:
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2) + "\n")


def _write_markdown(results: dict) -> None:
    lines = ["# Results\n", "Final `learner_eval_reward`, mean ± std across seeds.\n"]
    for env in ENVS:
        lines.append(f"## {env}\n")
        lines.append("| Condition | Mean | Std | Seeds |")
        lines.append("|---|---|---|---|")
        for cond in CONDITIONS:
            data = results.get(env, {}).get(cond)
            if data is None:
                lines.append(f"| {cond} | — | — | not run |")
            else:
                lines.append(f"| {cond} | {data['mean']} | {data['std']} | {data['n_seeds']} |")
        lines.append("")
    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(lines) + "\n")


def _copy_plots(results: dict) -> None:
    for manifest_path in sorted(RUNS_DIR.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        plot_src = manifest_path.parent / "comparison.png"
        if plot_src.exists():
            plot_dst = RESULTS_DIR / f"{manifest['experiment']}.png"
            shutil.copy2(plot_src, plot_dst)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    results = _collect()
    _write_json(results)
    _write_markdown(results)
    _copy_plots(results)
    print(f"Wrote results.json, RESULTS.md, and comparison plots to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
