#!/usr/bin/env python3
"""Aggregate Phase 3 results and generate findings table for README.

This script reads all completed runs (DQN, Evolutionary RL, Novelty-Guided RL)
and generates a markdown table suitable for inserting into README.md.

Usage:
    python scripts/aggregate_phase3_results.py
    python scripts/aggregate_phase3_results.py --verbose
    python scripts/aggregate_phase3_results.py --json  # output as JSON
"""

import json
from pathlib import Path
from typing import Dict, Tuple


def extract_final_reward(run_dir: Path) -> Tuple[float, float, int]:
    """Extract mean and std final eval reward from run directory.

    Returns: (mean, std, count)
    """
    seed_dirs = list(run_dir.glob("*__seed*"))
    final_rewards = []

    for seed_dir in seed_dirs:
        metrics_file = seed_dir / "metrics.csv"
        if not metrics_file.exists():
            continue

        try:
            with open(metrics_file) as f:
                lines = f.readlines()
                if len(lines) > 1:
                    # learner_eval_reward is column 5 (0-indexed)
                    parts = lines[-1].strip().split(',')
                    if len(parts) > 5:
                        reward = float(parts[5])
                        final_rewards.append(reward)
        except (ValueError, IndexError):
            pass

    if not final_rewards:
        return None, None, 0

    mean = sum(final_rewards) / len(final_rewards)
    variance = sum((r - mean)**2 for r in final_rewards) / len(final_rewards)
    std = variance**0.5

    return mean, std, len(final_rewards)


def get_environment_info(env_name: str) -> Dict:
    """Get environment metadata."""
    metadata = {
        "cartpole": {
            "display": "CartPole-v1",
            "type": "Dense baseline",
            "threshold": 475.0,
        },
        "lunarlander": {
            "display": "LunarLander-v3",
            "type": "Dense precision",
            "threshold": 200.0,
        },
        "cartpole_sparse": {
            "display": "CartPole-sparse",
            "type": "Sparse discovery",
            "threshold": 475.0,
        },
        "acrobot": {
            "display": "Acrobot-v1",
            "type": "Sparse discovery",
            "threshold": -100.0,
        },
    }
    return metadata.get(env_name, {})


def format_reward(mean: float, std: float, threshold: float) -> str:
    """Format reward with solved indicator."""
    if mean is None:
        return "—"

    is_solved = (mean >= threshold) if threshold > 0 else (mean <= threshold)
    symbol = "✓" if is_solved else "✗"

    return f"{mean:.1f} ± {std:.1f} {symbol}"


def aggregate_results(verbose: bool = False) -> Dict:
    """Aggregate all Phase 3 results."""

    environments = ["cartpole", "lunarlander", "cartpole_sparse", "acrobot"]
    methods = ["baseline_dqn", "evolutionary_rl", "novelty_guided_rl"]

    results = {}

    for env_name in environments:
        env_info = get_environment_info(env_name)
        threshold = env_info.get("threshold", 0.0)

        results[env_name] = {
            "display": env_info.get("display", env_name),
            "type": env_info.get("type", "Unknown"),
            "threshold": threshold,
            "methods": {},
        }

        for method in methods:
            run_dir = Path(f"runs/{env_name}_{method}")

            if run_dir.exists():
                mean, std, count = extract_final_reward(run_dir)

                if mean is not None:
                    results[env_name]["methods"][method] = {
                        "mean": mean,
                        "std": std,
                        "seeds": count,
                        "formatted": format_reward(mean, std, threshold),
                    }

                    if verbose:
                        print(f"{env_name:20s} {method:25s}: {mean:8.1f} ± {std:6.1f} ({count} seeds)")
                else:
                    results[env_name]["methods"][method] = None

                    if verbose:
                        print(f"{env_name:20s} {method:25s}: INCOMPLETE")
            else:
                results[env_name]["methods"][method] = None

                if verbose:
                    print(f"{env_name:20s} {method:25s}: NOT RUN")

    return results


def generate_markdown_table(results: Dict) -> str:
    """Generate markdown table for README."""

    lines = [
        "| Environment | Type | DQN | Evolutionary RL | Novelty-Guided RL |",
        "|---|---|---|---|---|",
    ]

    for env_name in ["cartpole", "lunarlander", "cartpole_sparse", "acrobot"]:
        if env_name not in results:
            continue

        env_result = results[env_name]
        dqn = env_result["methods"].get("baseline_dqn")
        evo = env_result["methods"].get("evolutionary_rl")
        nov = env_result["methods"].get("novelty_guided_rl")

        dqn_fmt = dqn["formatted"] if dqn else "—"
        evo_fmt = evo["formatted"] if evo else "—"
        nov_fmt = nov["formatted"] if nov else "—"

        line = f"| {env_result['display']} | {env_result['type']} | {dqn_fmt} | {evo_fmt} | {nov_fmt} |"
        lines.append(line)

    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate Phase 3 results and generate findings table."
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed results.")
    parser.add_argument("--json", action="store_true", help="Output as JSON.")
    args = parser.parse_args()

    results = aggregate_results(verbose=args.verbose)

    if args.json:
        # Convert to JSON-serializable format
        json_results = {}
        for env_name, env_data in results.items():
            json_results[env_name] = {
                "display": env_data["display"],
                "type": env_data["type"],
                "threshold": env_data["threshold"],
                "methods": {}
            }
            for method, method_data in env_data["methods"].items():
                if method_data:
                    json_results[env_name]["methods"][method] = {
                        "mean": round(method_data["mean"], 2),
                        "std": round(method_data["std"], 2),
                        "seeds": method_data["seeds"],
                    }
                else:
                    json_results[env_name]["methods"][method] = None

        print(json.dumps(json_results, indent=2))
    else:
        print("\n" + "="*100)
        print("PHASE 3 RESULTS FINDINGS TABLE")
        print("="*100 + "\n")

        table = generate_markdown_table(results)
        print(table)

        print("\n" + "="*100)
        print("Copy the table above into README.md")
        print("="*100)


if __name__ == "__main__":
    main()
