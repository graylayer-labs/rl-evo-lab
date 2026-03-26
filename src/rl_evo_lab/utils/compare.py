"""Run EDER vs baseline (no novelty) across multiple seeds and plot comparison.

Each seed is run twice: once with use_novelty=True (EDER) and once with
use_novelty=False (ES+DQN baseline). All runs are dispatched in parallel via
ProcessPoolExecutor so seeds × conditions run concurrently.

Usage:
    poetry run python -m rl_evo_lab.utils.compare                        # seeds 42 123 7
    poetry run python -m rl_evo_lab.utils.compare --seeds 1 2 3
    poetry run python -m rl_evo_lab.utils.compare --workers 4            # cap parallelism
    poetry run python -m rl_evo_lab.utils.compare --show
    poetry run python -m rl_evo_lab.utils.compare --plot-only --show     # skip training
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl_evo_lab.train import train
from rl_evo_lab.utils.config import EDERConfig
from rl_evo_lab.utils.logging import _run_id


# EDER = solid lines, baseline = dashed lines
_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


_RUNS_DIR = "runs"


def _maybe_train(cfg: EDERConfig) -> Path:
    run_dir = Path(_RUNS_DIR) / _run_id(cfg)
    csv = run_dir / "metrics.csv"
    if csv.exists():
        print(f"[skip] {_run_id(cfg)} — already exists", flush=True)
    else:
        label = "EDER" if cfg.use_novelty else "baseline"
        print(f"[start] seed={cfg.seed} {label}", flush=True)
        train(cfg, log_dir=_RUNS_DIR)
        print(f"[done]  seed={cfg.seed} {label}", flush=True)
    return csv


def _train_worker(args: tuple[int, bool]) -> Path:
    """Top-level function required for ProcessPoolExecutor pickling."""
    seed, use_novelty = args
    return _maybe_train(EDERConfig(seed=seed, use_novelty=use_novelty))


def _compare_dir(seeds: list[int]) -> Path:
    """Deterministic directory for a given set of seeds."""
    key = "_".join(str(s) for s in sorted(seeds))
    d = Path(_RUNS_DIR) / f"compare__seeds_{key}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_all(seeds: list[int], max_workers: int | None = None) -> dict[str, list[Path]]:
    """Dispatch all seed × condition combinations in parallel."""
    jobs = [(seed, use_novelty) for seed in seeds for use_novelty in (True, False)]

    n_workers = max_workers or min(len(jobs), os.cpu_count() or 1)
    print(f"Launching {len(jobs)} runs across {n_workers} workers …\n")

    results: dict[tuple[int, bool], Path] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_train_worker, job): job for job in jobs}
        for future in as_completed(futures):
            seed, use_novelty = futures[future]
            results[(seed, use_novelty)] = future.result()

    return {
        "eder":     [results[(seed, True)]  for seed in seeds],
        "baseline": [results[(seed, False)] for seed in seeds],
    }


def _smooth(arr: np.ndarray, window: int = 15) -> np.ndarray:
    """Centered rolling mean over a 1-D array."""
    out = np.empty_like(arr)
    half = window // 2
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr), i + half + 1)
        out[i] = arr[lo:hi].mean()
    return out


def _aggregate(csv_list: list[Path], col: str, smooth: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (episodes, mean, std) aggregated across seeds.

    For sparse columns (those with NaN rows) we align on the subset of
    episodes where every seed has a value, then interpolate to full length.
    """
    dfs = [pd.read_csv(p) for p in csv_list]
    # Align all runs to the shortest episode count
    n = min(len(d) for d in dfs)
    dfs = [d.iloc[:n] for d in dfs]
    eps = dfs[0]["episode"].values + 1

    if dfs[0][col].isna().any():
        # Sparse: forward-fill each df then stack
        arrays = [d[col].ffill().bfill().values[:n] for d in dfs]
    else:
        arrays = [d[col].values[:n] for d in dfs]

    stacked = np.stack(arrays)          # (n_seeds, n_episodes)
    mean = stacked.mean(axis=0)
    std  = stacked.std(axis=0)

    if smooth:
        mean = _smooth(mean)
        std  = _smooth(std)

    return eps, mean, std


def compare(paths: dict[str, list[Path]], out_dir: Path | None = None, show: bool = False) -> Path:
    """Plot mean ± std band per condition across all seeds."""
    condition_style = {
        "eder":     {"color": "tab:blue",   "linestyle": "-",  "label": "EDER (novelty)"},
        "baseline": {"color": "tab:orange", "linestyle": "--", "label": "Baseline (no novelty)"},
    }

    panels = [
        ("actor_extrinsic_reward", "Actor extrinsic reward (mean workers)", True),
        ("learner_eval_reward",    "Learner eval reward",                   False),
        ("learner_loss",           "DQN loss",                              True),
        ("buffer_diversity",       "Buffer diversity",                      False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("EDER (novelty) vs Baseline (no novelty) — mean ± std across seeds", fontsize=11)
    ax_map = dict(zip([p[0] for p in panels], axes.flat))

    for condition, csv_list in paths.items():
        style = condition_style.get(condition, {"color": "tab:grey", "linestyle": "-", "label": condition})
        for col, title, do_smooth in panels:
            ax = ax_map[col]
            eps, mean, std = _aggregate(csv_list, col, smooth=do_smooth)
            ax.plot(eps, mean, color=style["color"], linestyle=style["linestyle"],
                    linewidth=1.8, label=style["label"])
            ax.fill_between(eps, mean - std, mean + std,
                            color=style["color"], alpha=0.15)

    for col, title, _ in panels:
        ax = ax_map[col]
        ax.set_title(title)
        ax.set_xlabel("episode")
        ax.legend(fontsize=8)
        if col == "learner_loss":
            ax.set_yscale("log")

    fig.tight_layout()
    dest = out_dir or Path(_RUNS_DIR)
    out_path = dest / "comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    if show:
        plt.show()

    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    parser.add_argument("--workers", type=int, default=None,
                        help="Max parallel processes (default: one per job)")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training, just regenerate the plot from existing runs")
    args = parser.parse_args()

    if args.plot_only:
        paths = {
            "eder":     [Path(_RUNS_DIR) / _run_id(EDERConfig(seed=s, use_novelty=True))  / "metrics.csv" for s in args.seeds],
            "baseline": [Path(_RUNS_DIR) / _run_id(EDERConfig(seed=s, use_novelty=False)) / "metrics.csv" for s in args.seeds],
        }
    else:
        paths = run_all(args.seeds, max_workers=args.workers)

    out_dir = _compare_dir(args.seeds)

    # Write a manifest so the comparison directory is self-documenting
    import json
    manifest = {
        "seeds": args.seeds,
        "conditions": {k: [str(p) for p in v] for k, v in paths.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    compare(paths, out_dir=out_dir, show=args.show)


if __name__ == "__main__":
    main()
