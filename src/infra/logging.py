from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass, fields
from multiprocessing import Queue
from pathlib import Path
from typing import Any

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn

from infra.config import EDERConfig


def _condition_label(use_es: bool, use_novelty: bool) -> str:
    if use_es and use_novelty:
        return "EDER"
    if use_es:
        return "DDQN+ES"
    if use_novelty:
        return "DDQN+Novelty"
    return "DDQN"


def _run_hash(cfg: EDERConfig) -> str:
    """6-char HP fingerprint. Same config → same hash. Used as dir suffix for safety."""
    hp = {
        f.name: getattr(cfg, f.name)
        for f in fields(cfg)
        if f.name not in ("use_wandb", "wandb_project")
    }
    return hashlib.sha1(json.dumps(hp, sort_keys=True).encode()).hexdigest()[:6]


def _run_dir(cfg: EDERConfig, results_dir: str | Path = "runs") -> Path:
    """Fallback run directory for standalone train() calls (no experiment context).
    Structure: {results_dir}/{env_id}/seed{N}__{hash}/
    """
    return Path(results_dir) / cfg.env_id / f"seed{cfg.seed}__{_run_hash(cfg)}"


@dataclass
class EpisodeLog:
    episode: int
    total_env_steps: int  # cumulative env steps across all workers/episodes so far
    actor_augmented_reward: float
    actor_extrinsic_reward: float
    learner_loss: float
    learner_eval_reward: float | None
    buffer_diversity: float | None
    idn_loss: float
    effective_beta: float
    buffer_size: int
    wall_clock_seconds: float = 0.0  # elapsed wall-clock time since run start
    sync: bool = False


class RunLogger:
    def __init__(
        self,
        cfg: EDERConfig,
        log_dir: str = "runs",
        verbose: bool = True,
        progress_queue: Queue | None = None,
        run_dir: Path | None = None,
    ) -> None:
        self.cfg = cfg
        run_dir = run_dir if run_dir is not None else _run_dir(cfg, log_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_dir.name  # used for progress queue messages

        self._csv_path = run_dir / "metrics.csv"
        self._csv_file = self._csv_path.open("a", newline="")
        self._writer: csv.DictWriter | None = None
        self._write_header = self._csv_path.stat().st_size == 0

        self._run_dir = run_dir
        self._write_config(run_dir)
        self._write_status("running")
        self._wandb = self._init_wandb() if cfg.use_wandb else None

        # Track wall-clock time for compute summary
        self._start_time = time.time()
        self._last_total_env_steps = 0

        # If a queue is provided, send progress updates to the parent process.
        # Otherwise show a local progress bar when verbose=True.
        self._queue: Queue | None = progress_queue
        self._last_eval: float | None = None
        self._progress: Progress | None = None
        self._task_id = None
        if verbose and progress_queue is None:
            mode = _condition_label(cfg.use_es, cfg.use_novelty)
            desc = f"[cyan]{mode}[/cyan] seed={cfg.seed}"
            self._progress = Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                TextColumn("{task.fields[stats]}"),
            )
            self._task_id = self._progress.add_task(desc, total=cfg.total_episodes, stats="")
            self._progress.start()

    def _write_config(self, run_dir: Path) -> None:
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            cfg_path.write_text(
                json.dumps({f.name: getattr(self.cfg, f.name) for f in fields(self.cfg)}, indent=2)
            )

    def _write_status(self, status: str) -> None:
        (self._run_dir / "status.json").write_text(json.dumps({"status": status}))

    def _init_wandb(self):
        try:
            import wandb

            return wandb.init(
                project=self.cfg.wandb_project,
                name=self.run_id,
                id=self.run_id,
                resume="allow",
                config={f.name: getattr(self.cfg, f.name) for f in fields(self.cfg)},
            )
        except Exception as e:
            print(f"[warn] wandb init failed: {e}")
            return None

    def log(self, entry: EpisodeLog) -> None:
        # Update elapsed wall-clock time from run start
        elapsed = time.time() - self._start_time
        entry.wall_clock_seconds = elapsed
        self._last_total_env_steps = entry.total_env_steps

        row = asdict(entry)

        # CSV
        if self._writer is None:
            self._writer = csv.DictWriter(self._csv_file, fieldnames=row.keys())
            if self._write_header:
                self._writer.writeheader()
        self._writer.writerow(row)
        self._csv_file.flush()

        # wandb
        if self._wandb is not None:
            import wandb

            payload = {k: v for k, v in row.items() if v is not None and k != "episode"}
            wandb.log(payload, step=entry.episode)

        # Progress reporting
        if entry.learner_eval_reward is not None:
            self._last_eval = entry.learner_eval_reward

        if self._queue is not None:
            msg: dict[str, Any] = {
                "run_id": self.run_id,
                "episode": entry.episode,
                "loss": entry.learner_loss,
                "buf": entry.buffer_size,
                "beta": entry.effective_beta if entry.effective_beta > 0 else None,
                "eval": self._last_eval,
                "sync": entry.sync,
            }
            self._queue.put(msg)
        elif self._progress is not None:
            parts = [f"loss={entry.learner_loss:.4f}", f"buf={entry.buffer_size:,}"]
            if entry.effective_beta > 0:
                parts.append(f"β={entry.effective_beta:.4f}")
            if self._last_eval is not None:
                parts.append(f"[green]eval={self._last_eval:.1f}[/green]")
            if entry.sync:
                parts.append("[yellow]sync[/yellow]")
            self._progress.update(self._task_id, advance=1, stats="  ".join(parts))

    def close(self, final_eval_rewards: list[float] | None = None) -> None:
        if self._progress is not None:
            self._progress.stop()
        self._csv_file.close()
        if self._wandb is not None:
            self._wandb.finish()

        # Write compute summary to status.json
        condition = _condition_label(self.cfg.use_es, self.cfg.use_novelty)
        elapsed = time.time() - self._start_time
        compute_summary: dict[str, Any] = {
            "status": "completed",
            "condition": condition,
            "total_env_steps": self._last_total_env_steps,
            "total_wall_clock_seconds": elapsed,
        }
        if final_eval_rewards is not None:
            n = len(final_eval_rewards)
            eval_mean = sum(final_eval_rewards) / n
            variance = sum((r - eval_mean) ** 2 for r in final_eval_rewards) / n
            compute_summary["final_eval"] = {
                "mean": eval_mean,
                "std": variance**0.5,
                "n_episodes": n,
                "episode_rewards": final_eval_rewards,
            }
        (self._run_dir / "status.json").write_text(json.dumps(compute_summary))
