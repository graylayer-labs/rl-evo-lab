"""Tests for wall-clock time logging and compute summary."""

import json
import tempfile
import time
from pathlib import Path

from infra.config import EDERConfig
from infra.logging import EpisodeLog, RunLogger


def test_wall_clock_logging():
    """Test that wall-clock time is logged and increases over episodes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = EDERConfig(
            seed=42,
            env_id="CartPole-v1",
            use_es=False,
            use_novelty=False,
        )

        logger = RunLogger(cfg, log_dir=tmpdir, verbose=False)

        # Log a few episodes with small delays
        for episode in range(3):
            time.sleep(0.01)  # Small delay to advance wall time
            entry = EpisodeLog(
                episode=episode,
                total_env_steps=100 * (episode + 1),
                actor_augmented_reward=50.0,
                actor_extrinsic_reward=50.0,
                learner_loss=0.1,
                learner_eval_reward=50.0 if episode % 2 == 0 else None,
                buffer_diversity=5.0,
                idn_loss=0.0,
                effective_beta=0.0,
                buffer_size=100 * (episode + 1),
            )
            logger.log(entry)

        logger.close()

        # Read and verify metrics.csv has wall_clock_seconds column
        csv_path = Path(tmpdir) / cfg.env_id / f"seed{cfg.seed}_*"
        import glob

        run_dirs = glob.glob(str(csv_path))
        assert len(run_dirs) == 1, f"Expected 1 run dir, found {len(run_dirs)}"
        run_dir = Path(run_dirs[0])

        metrics_path = run_dir / "metrics.csv"
        assert metrics_path.exists(), f"metrics.csv not found at {metrics_path}"

        with open(metrics_path) as f:
            header = f.readline().strip()
            columns = header.split(",")
            assert "wall_clock_seconds" in columns, (
                f"wall_clock_seconds not in CSV columns: {columns}"
            )

            # Verify wall_clock_seconds increases
            lines = f.readlines()
            wall_times = []
            for line in lines:
                parts = line.strip().split(",")
                col_idx = columns.index("wall_clock_seconds")
                wall_times.append(float(parts[col_idx]))

            # Wall times should be increasing or equal
            for i in range(1, len(wall_times)):
                assert wall_times[i] >= wall_times[i - 1], (
                    f"Wall clock time should be increasing: {wall_times}"
                )

        # Verify status.json has compute summary
        status_path = run_dir / "status.json"
        assert status_path.exists(), f"status.json not found at {status_path}"

        status = json.loads(status_path.read_text())
        assert status["status"] == "completed"
        assert "condition" in status
        assert "total_env_steps" in status
        assert "total_wall_clock_seconds" in status
        assert status["total_env_steps"] == 300  # 100 * 3 episodes
        assert status["total_wall_clock_seconds"] > 0


def test_episode_log_has_wall_clock():
    """Test that EpisodeLog dataclass has wall_clock_seconds field."""
    entry = EpisodeLog(
        episode=0,
        total_env_steps=100,
        actor_augmented_reward=50.0,
        actor_extrinsic_reward=50.0,
        learner_loss=0.1,
        learner_eval_reward=50.0,
        buffer_diversity=5.0,
        idn_loss=0.0,
        effective_beta=0.0,
        buffer_size=100,
        wall_clock_seconds=1.5,
    )

    assert entry.wall_clock_seconds == 1.5
    assert entry.episode == 0
    assert entry.total_env_steps == 100
