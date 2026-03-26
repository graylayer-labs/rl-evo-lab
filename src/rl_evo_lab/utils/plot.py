"""Generate a summary plot for a completed (or in-progress) run.

Usage:
    poetry run python -m rl_evo_lab.utils.plot runs/<run_id>/metrics.csv
    poetry run python -m rl_evo_lab.utils.plot runs/<run_id>/metrics.csv --show
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot(csv_path: Path, show: bool = False) -> Path:
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(csv_path.parent.name, fontsize=10, y=1.01)

    # 1. Actor rewards — augmented vs extrinsic on same axes
    ax = axes[0, 0]
    ax.plot(df["episode"], df["actor_augmented_reward"], label="augmented", alpha=0.8)
    ax.plot(df["episode"], df["actor_extrinsic_reward"], label="extrinsic", alpha=0.8)
    ax.set_title("Actor reward (mean over workers)")
    ax.set_xlabel("episode")
    ax.legend()

    # 2. Learner eval reward (sparse — only logged every eval_freq episodes)
    ax = axes[0, 1]
    eval_df = df.dropna(subset=["learner_eval_reward"])
    ax.plot(eval_df["episode"], eval_df["learner_eval_reward"], marker="o", markersize=3)
    ax.set_title("Learner eval reward")
    ax.set_xlabel("episode")

    # 3. DQN loss
    ax = axes[1, 0]
    # Zero before buffer fills — mask those out
    loss_df = df[df["learner_loss"] > 0]
    ax.plot(loss_df["episode"], loss_df["learner_loss"], alpha=0.7)
    ax.set_title("DQN loss")
    ax.set_xlabel("episode")
    ax.set_yscale("log")

    # 4. Buffer diversity
    ax = axes[1, 1]
    div_df = df.dropna(subset=["buffer_diversity"])
    ax.plot(div_df["episode"], div_df["buffer_diversity"], color="purple", alpha=0.8)
    ax.set_title("Buffer diversity (mean pairwise L2)")
    ax.set_xlabel("episode")

    fig.tight_layout()

    out_path = csv_path.parent / "summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    if show:
        plt.show()

    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    plot(args.csv, show=args.show)


if __name__ == "__main__":
    main()
