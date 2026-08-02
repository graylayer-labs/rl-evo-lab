"""Plotting utilities and backward-compatible CLI.

The experiment runner has moved to :mod:`rl_evo_lab.experiment`.
Prefer running experiment scripts directly::

    python experiments/cartpole_efficiency.py
    python experiments/lunarlander_efficiency.py --force --show

The CLI below is kept for backward compatibility::

    poetry run python -m rl_evo_lab.utils.compare --experiment efficiency --env cartpole
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl_evo_lab.utils.config import ENV_PRESETS

if TYPE_CHECKING:
    from rl_evo_lab.experiment import Experiment

# ---------------------------------------------------------------------------
# Plot config
# ---------------------------------------------------------------------------

_PALETTE = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
_LINESTYLES = ["-", "--", "-.", ":"]

_SOLVED_THRESHOLDS: dict[str, float] = {
    "CartPole-v1": 475.0,
    "LunarLander-v3": 200.0,
    "Acrobot-v1": -100.0,
    "MountainCar-v0": -110.0,
}

_PANELS_BASE = [
    ("actor_extrinsic_reward", "reward", "ES Worker Return\n(mean across population)", True, True),
    (
        "learner_eval_reward",
        "reward",
        "Learner Eval Reward\n(greedy policy — primary metric)",
        False,
        True,
    ),
    ("learner_loss", "loss", "DQN Loss\n(Huber, log scale)", True, False),
]
_PANEL_BETA = (
    "effective_beta",
    "β",
    "Effective Novelty Weight β\n(zero during warmup, rises as IDN learns)",
    False,
    None,
)
_PANEL_DIVERSITY = (
    "buffer_diversity",
    "diversity",
    "Replay Buffer Diversity\n(mean pairwise distance of sampled obs)",
    False,
    True,
)

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _smooth(arr: np.ndarray, window: int = 15) -> np.ndarray:
    out = np.empty_like(arr)
    half = window // 2
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr), i + half + 1)
        out[i] = arr[lo:hi].mean()
    return out


def _aggregate(
    csv_list: list[Path],
    col: str,
    smooth: bool = False,
    x_col: str = "episode",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate metrics across multiple seed runs using NaN-padding.

    Instead of truncating all seeds to the shortest run, this pads shorter runs
    with NaN and uses np.nanmean/np.nanstd to compute statistics. This preserves
    longer runs and prevents bias toward early-stopped experiments.

    Parameters
    ----------
    csv_list : list[Path]
        List of CSV paths (one per seed run)
    col : str
        Column name to aggregate
    smooth : bool
        Whether to apply smoothing to mean and std
    x_col : str
        X-axis column name ("episode" or "total_env_steps")

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (x, mean, std) arrays where mean and std are computed across seeds
        with NaN padding for runs that ended early.
    """
    dfs = [pd.read_csv(p) for p in csv_list]
    # Find max length to pad all dataframes to
    max_len = max(len(d) for d in dfs)

    # Pad all dataframes to max_len with NaN
    padded_arrays = []
    for d in dfs:
        col_values = d[col].values
        if d[col].isna().any():
            # Fill forward then backward to interpolate missing values
            col_values = d[col].ffill().bfill().values
        # Pad with NaN if shorter than max_len
        if len(col_values) < max_len:
            padded = np.full(max_len, np.nan)
            padded[: len(col_values)] = col_values
            padded_arrays.append(padded)
        else:
            padded_arrays.append(col_values)

    # Extract x-axis (use first dataframe's x values, padded with NaN if needed)
    x_values = dfs[0][x_col].values
    if x_col != "episode":
        # For non-episode columns, pad with NaN
        if len(x_values) < max_len:
            x = np.full(max_len, np.nan)
            x[: len(x_values)] = x_values
        else:
            x = x_values
    else:
        # For episode column, add 1 (episodes are 0-indexed in CSV)
        if len(x_values) < max_len:
            x = np.full(max_len, np.nan)
            x[: len(x_values)] = x_values + 1
        else:
            x = x_values + 1

    # Stack and compute mean/std using NaN-aware functions
    stacked = np.stack(padded_arrays)
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)

    if smooth:
        mean = _smooth(mean)
        std = _smooth(std)

    return x, mean, std


def _any_novelty(paths: dict[str, list[Path]]) -> bool:
    for csv_list in paths.values():
        try:
            df = pd.read_csv(csv_list[0])
            if "effective_beta" in df.columns and df["effective_beta"].max() > 0:
                return True
        except Exception:
            pass
    return False


def _detect_env(csv_list: list[Path]) -> str:
    # Read env_id from the config.json saved alongside each run
    cfg_path = csv_list[0].parent / "config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text()).get("env_id", "")
    return ""


# ---------------------------------------------------------------------------
# compare() — the one public plotting function
# ---------------------------------------------------------------------------


def compare(
    paths: dict[str, list[Path]],
    out_dir: Path | None = None,
    show: bool = False,
    title: str = "",
    x_col: str = "episode",
) -> Path:
    """Plot mean ± std band per condition across seeds.

    Parameters
    ----------
    paths:    ``{label: [csv_path_per_seed]}``
    out_dir:  Directory to save ``comparison.png``. Defaults to ``runs/``.
    x_col:    ``"episode"`` or ``"total_env_steps"``.
    """
    env_id = _detect_env(next(iter(paths.values())))
    solved = _SOLVED_THRESHOLDS.get(env_id)

    fourth = _PANEL_BETA if _any_novelty(paths) else _PANEL_DIVERSITY
    panels = _PANELS_BASE + [fourth]
    x_label = "Env Steps" if x_col == "total_env_steps" else "Episode"

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(
        title or "Condition comparison — mean ± std across seeds", fontsize=12, fontweight="bold"
    )
    ax_map = dict(zip([p[0] for p in panels], axes.flat, strict=False))

    _reward_cols = {"actor_extrinsic_reward", "learner_eval_reward"}
    first_solved_x: dict[str, dict[str, float]] = {}

    for idx, (condition, csv_list) in enumerate(paths.items()):
        color = _PALETTE[idx % len(_PALETTE)]
        ls = _LINESTYLES[idx % len(_LINESTYLES)]
        first_solved_x[condition] = {}
        for col, _ylabel, _subtitle, do_smooth, _higher in panels:
            ax = ax_map[col]
            x, mean, std = _aggregate(csv_list, col, smooth=do_smooth, x_col=x_col)
            ax.plot(x, mean, color=color, linestyle=ls, linewidth=1.8, label=condition)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
            if solved is not None and col in _reward_cols:
                crossed = np.where(mean >= solved)[0]
                if len(crossed):
                    first_solved_x[condition][col] = float(x[crossed[0]])

    for col, ylabel, subtitle, _, higher_is_better in panels:
        ax = ax_map[col]
        ax.set_title(subtitle, fontsize=9, loc="left")
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8, loc="best")
        ax.tick_params(labelsize=8)

        if col == "learner_loss":
            ax.set_yscale("log")
        if higher_is_better is True:
            ax.annotate(
                "▲ better",
                xy=(0.01, 0.97),
                xycoords="axes fraction",
                fontsize=7,
                color="green",
                va="top",
            )
        elif higher_is_better is False:
            ax.annotate(
                "▼ better",
                xy=(0.01, 0.97),
                xycoords="axes fraction",
                fontsize=7,
                color="green",
                va="top",
            )

        if solved is not None and col in _reward_cols:
            ax.axhline(solved, color="black", linewidth=1.0, linestyle=":", alpha=0.6)
            ax.annotate(
                f"solved ({solved:g})",
                xy=(0.01, solved),
                xycoords=("axes fraction", "data"),
                fontsize=7,
                color="black",
                alpha=0.7,
                va="bottom",
            )
            for idx, (_condition, col_map) in enumerate(first_solved_x.items()):
                if col in col_map:
                    ax.axvline(
                        col_map[col],
                        color=_PALETTE[idx % len(_PALETTE)],
                        linestyle=":",
                        linewidth=1.0,
                        alpha=0.7,
                    )
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, min(hi, solved * 1.15) if solved > 0 else max(hi, solved * 1.15))

    fig.tight_layout()
    dest = out_dir or Path("runs")
    out_path = dest / "comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    if show:
        plt.show()
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Backward-compatible CLI
# ---------------------------------------------------------------------------


def _make_registry() -> dict[str, dict[str, Experiment]]:
    """Build the experiment registry lazily to avoid import cycles."""
    from rl_evo_lab.experiment import Condition, Experiment

    _efficiency = [
        Condition("EDER", use_es=True, use_novelty=True),
        Condition("ES+DQN", use_es=True, use_novelty=False),
        Condition("DQN", use_es=False, use_novelty=False),
    ]
    _eder_vs_baseline = [
        Condition("EDER", use_novelty=True),
        Condition("Baseline", use_novelty=False),
    ]
    _model_size = [
        Condition("EDER-64", use_es=True, use_novelty=True, hidden_dim=64),
        Condition("EDER-128", use_es=True, use_novelty=True, hidden_dim=128),
        Condition("DQN-64", use_es=False, use_novelty=False, hidden_dim=64),
        Condition("DQN-128", use_es=False, use_novelty=False, hidden_dim=128),
    ]
    _updates = [
        Condition("EDER-5upd", use_es=True, use_novelty=True, learner_updates_per_episode=5),
        Condition("EDER-20upd", use_es=True, use_novelty=True, learner_updates_per_episode=20),
        Condition("DQN-5upd", use_es=False, use_novelty=False, learner_updates_per_episode=5),
        Condition("DQN-20upd", use_es=False, use_novelty=False, learner_updates_per_episode=20),
    ]
    _sample_efficiency = [
        Condition("EDER", use_es=True, use_novelty=True),
        Condition("ES+DQN", use_es=True, use_novelty=False),
        Condition("DQN", use_es=False, use_novelty=False, total_episodes=10_000),
    ]

    registry: dict[str, dict[str, Experiment]] = {}
    for exp_name, conditions in [
        ("eder_vs_baseline", _eder_vs_baseline),
        ("efficiency", _efficiency),
        ("model_size", _model_size),
        ("updates", _updates),
        ("sample_efficiency", _sample_efficiency),
    ]:
        registry[exp_name] = {
            env: Experiment(
                name=f"{env}_{exp_name}",
                env=env,
                seeds=[42, 123, 7],
                conditions=conditions,
            )
            for env in ENV_PRESETS
        }
    return registry


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run and plot an experiment suite. "
        "Prefer running experiment scripts directly: python experiments/cartpole_efficiency.py"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    parser.add_argument(
        "--experiment",
        default="eder_vs_baseline",
        choices=["eder_vs_baseline", "efficiency", "model_size", "updates", "sample_efficiency"],
    )
    parser.add_argument("--env", default="cartpole", choices=list(ENV_PRESETS))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--x-axis", choices=["episode", "env_steps"], default=None)
    args = parser.parse_args()

    registry = _make_registry()
    exp = registry[args.experiment][args.env]
    exp.seeds = args.seeds

    x_axis = args.x_axis or ("env_steps" if args.experiment == "sample_efficiency" else "episode")

    if args.plot_only:
        exp.plot(show=args.show, x_axis=x_axis)
    else:
        exp.run(force=args.force, show=args.show, workers=args.workers, x_axis=x_axis)


if __name__ == "__main__":
    main()
