#!/usr/bin/env python3
"""Analyze Phase 3 results: interpret findings and generate thesis narratives.

This script reads completed Phase 3 results and generates interpretative
narratives showing what each environment result means for the thesis.

Usage:
    python scripts/analyze_phase3_results.py --env cartpole
    python scripts/analyze_phase3_results.py --all
    python scripts/analyze_phase3_results.py --thesis-summary
"""

from pathlib import Path
from typing import Dict, Tuple


def extract_detailed_results(env_name: str) -> Dict:
    """Extract detailed results for one environment across all methods."""

    methods = ["baseline_dqn", "evolutionary_rl", "novelty_guided_rl"]
    results = {}

    for method in methods:
        run_dir = Path(f"runs/{env_name}_{method}")
        if not run_dir.exists():
            continue

        seed_data = {}
        seed_dirs = list(run_dir.glob("*__seed*"))

        for seed_dir in seed_dirs:
            seed_num = None
            for part in seed_dir.name.split("__"):
                if part.startswith("seed"):
                    seed_num = part.replace("seed", "")
                    break

            metrics_file = seed_dir / "metrics.csv"
            if not metrics_file.exists():
                continue

            try:
                with open(metrics_file) as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        parts = lines[-1].strip().split(',')
                        eval_reward = float(parts[5]) if len(parts) > 5 else None
                        episodes = int(parts[0]) if len(parts) > 0 else None

                        if seed_num:
                            seed_data[seed_num] = {
                                "eval_reward": eval_reward,
                                "episodes": episodes,
                            }
            except (ValueError, IndexError):
                pass

        if seed_data:
            rewards = [s["eval_reward"] for s in seed_data.values() if s["eval_reward"] is not None]
            mean_reward = sum(rewards) / len(rewards) if rewards else None
            min_reward = min(rewards) if rewards else None
            max_reward = max(rewards) if rewards else None

            results[method] = {
                "mean": mean_reward,
                "min": min_reward,
                "max": max_reward,
                "seeds": seed_data,
            }

    return results


def interpret_environment(env_name: str, threshold: float) -> str:
    """Generate interpretation narrative for environment results."""

    results = extract_detailed_results(env_name)

    if not results:
        return f"No results for {env_name} yet."

    dqn = results.get("baseline_dqn")
    evo = results.get("evolutionary_rl")
    nov = results.get("novelty_guided_rl")

    narrative = f"\n{'='*80}\n{env_name.upper()}\n{'='*80}\n"

    if dqn:
        dqn_solved = (dqn["mean"] >= threshold) if threshold > 0 else (dqn["mean"] <= threshold)
        narrative += f"\nDQN Baseline: {dqn['mean']:.1f} ± ? {'✓ SOLVED' if dqn_solved else '✗ FAILED'}\n"
        narrative += f"  Range: {dqn['min']:.1f} to {dqn['max']:.1f}\n"
        narrative += f"  Seeds: {list(dqn['seeds'].keys())}\n"

    if evo:
        evo_solved = (evo["mean"] >= threshold) if threshold > 0 else (evo["mean"] <= threshold)
        improvement = ((evo["mean"] - dqn["mean"]) / abs(dqn["mean"]) * 100) if dqn and dqn["mean"] != 0 else 0

        narrative += f"\nEvolutionary RL: {evo['mean']:.1f} ± ? {'✓ SOLVED' if evo_solved else '✗ FAILED'}\n"
        narrative += f"  Range: {evo['min']:.1f} to {evo['max']:.1f}\n"
        narrative += f"  vs. DQN: {improvement:+.1f}%\n"

    if nov:
        nov_solved = (nov["mean"] >= threshold) if threshold > 0 else (nov["mean"] <= threshold)
        improvement = ((nov["mean"] - evo["mean"]) / abs(evo["mean"]) * 100) if evo and evo["mean"] != 0 else 0

        narrative += f"\nNovelty-Guided RL: {nov['mean']:.1f} ± ? {'✓ SOLVED' if nov_solved else '✗ FAILED'}\n"
        narrative += f"  Range: {nov['min']:.1f} to {nov['max']:.1f}\n"
        narrative += f"  vs. Evolutionary: {improvement:+.1f}%\n"
    else:
        narrative += f"\nNovelty-Guided RL: (in progress)\n"

    # Interpretation
    if dqn and evo:
        if evo["mean"] > dqn["mean"]:
            narrative += f"\n→ ES population helps on {env_name}: diversity > ε-greedy\n"
        else:
            narrative += f"\n→ ES population hurts on {env_name}: overhead > benefit\n"

    return narrative


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Phase 3 results and generate narratives.")
    parser.add_argument("--env", help="Analyze specific environment.")
    parser.add_argument("--all", action="store_true", help="Analyze all environments.")
    parser.add_argument(
        "--thesis-summary", action="store_true", help="Generate thesis summary from all results."
    )

    args = parser.parse_args()

    env_thresholds = {
        "cartpole": 475.0,
        "lunarlander": 200.0,
        "cartpole_sparse": 475.0,
        "acrobot": -100.0,
    }

    if args.env:
        threshold = env_thresholds.get(args.env, 0.0)
        print(interpret_environment(args.env, threshold))

    elif args.all:
        for env_name, threshold in env_thresholds.items():
            print(interpret_environment(env_name, threshold))

    elif args.thesis_summary:
        print("\n" + "="*80)
        print("THESIS SUMMARY: Which Techniques Solve Exploration-Stuck Tasks?")
        print("="*80)

        envs_analyzed = []
        for env_name, threshold in env_thresholds.items():
            results = extract_detailed_results(env_name)
            if results:
                envs_analyzed.append((env_name, results, threshold))

        if envs_analyzed:
            print("\nFINDINGS BY ENVIRONMENT:\n")
            for env_name, results, threshold in envs_analyzed:
                dqn = results.get("baseline_dqn")
                evo = results.get("evolutionary_rl")
                nov = results.get("novelty_guided_rl")

                print(f"\n{env_name.upper()}")

                if evo and dqn:
                    if evo["mean"] > dqn["mean"]:
                        print(f"  ✓ ES helps: {evo['mean']:.1f} > {dqn['mean']:.1f}")
                    else:
                        print(f"  ✗ ES hurts: {evo['mean']:.1f} < {dqn['mean']:.1f}")

                if nov and evo:
                    if nov["mean"] > evo["mean"]:
                        print(f"  ✓ Novelty helps ES: {nov['mean']:.1f} > {evo['mean']:.1f}")
                    else:
                        print(f"  ✗ Novelty hurts ES: {nov['mean']:.1f} < {evo['mean']:.1f}")
        else:
            print("\nNo results yet. Run Phase 3 experiments first.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
