from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from rl_evo_lab.utils.config import EDERConfig


def _run_id(cfg: EDERConfig) -> str:
    """Deterministic run ID derived from config. Same config → same ID."""
    key = {f.name: getattr(cfg, f.name) for f in fields(cfg) if f.name not in ("use_wandb", "wandb_project")}
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]
    return f"{cfg.env_id}__seed{cfg.seed}__b{cfg.beta}__n{cfg.es_n_workers}__{digest}"


@dataclass
class EpisodeLog:
    episode: int
    actor_augmented_reward: float
    actor_extrinsic_reward: float
    learner_loss: float
    learner_eval_reward: float | None
    buffer_diversity: float | None
    idn_loss: float
    buffer_size: int
    sync: bool = False  # whether actor synced from learner this episode


class RunLogger:
    def __init__(self, cfg: EDERConfig, log_dir: str = "runs") -> None:
        self.cfg = cfg
        self.run_id = _run_id(cfg)
        run_dir = Path(log_dir) / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path = run_dir / "metrics.csv"
        # Append mode — idempotent reruns extend rather than overwrite
        self._csv_file = self._csv_path.open("a", newline="")
        self._writer: csv.DictWriter | None = None
        self._write_header = self._csv_path.stat().st_size == 0

        self._write_config(run_dir)
        self._wandb = self._init_wandb() if cfg.use_wandb else None

    def _write_config(self, run_dir: Path) -> None:
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            cfg_path.write_text(
                json.dumps({f.name: getattr(self.cfg, f.name) for f in fields(self.cfg)}, indent=2)
            )

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

        # stdout
        eval_str = f"  eval={entry.learner_eval_reward:6.1f}" if entry.learner_eval_reward is not None else ""
        div_str = f"  div={entry.buffer_diversity:.3f}" if entry.buffer_diversity is not None else ""
        sync_str = "  [sync]" if entry.sync else ""
        print(
            f"ep={entry.episode + 1:4d}"
            f"  aug={entry.actor_augmented_reward:7.2f}"
            f"  ext={entry.actor_extrinsic_reward:7.2f}"
            f"  loss={entry.learner_loss:.4f}"
            f"  buf={entry.buffer_size:6d}"
            f"  idn={entry.idn_loss:.4f}"
            f"{eval_str}{div_str}{sync_str}"
        )

    def close(self) -> None:
        self._csv_file.close()
        if self._wandb is not None:
            self._wandb.finish()
