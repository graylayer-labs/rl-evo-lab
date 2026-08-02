"""End-to-end test for logging with wall-clock time and compute summary."""

import json
import tempfile
from pathlib import Path

from rl_evo_lab.train import train
from rl_evo_lab.utils.config import EDERConfig


def test_train_logs_compute_metrics():
    """End-to-end test: train a simple model and verify logging output.

    This verifies:
    1. wall_clock_seconds is logged in metrics.csv
    2. total_env_steps is in every row of metrics.csv
    3. status.json contains compute summary with condition, total_env_steps, and wall_clock_seconds
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = EDERConfig(
            env_id="CartPole-v1",
            seed=99,
            total_episodes=5,  # Very short run for testing
            eval_freq=2,  # Evaluate every 2 episodes
            use_es=False,  # Use simple DQN for speed
            use_novelty=False,
        )

        # Run training
        train(cfg, log_dir=tmpdir, verbose=False)

        # Find the run directory
        run_dirs = list(Path(tmpdir).glob("CartPole-v1/seed99__*"))
        assert len(run_dirs) == 1, f"Expected 1 run dir, got {len(run_dirs)}"
        run_dir = run_dirs[0]

        # Verify metrics.csv exists and has correct columns
        metrics_path = run_dir / "metrics.csv"
        assert metrics_path.exists(), f"metrics.csv not found at {metrics_path}"

        with open(metrics_path) as f:
            header = f.readline().strip()
            columns = header.split(",")

            # Check required columns
            assert "episode" in columns, f"episode not in {columns}"
            assert "total_env_steps" in columns, f"total_env_steps not in {columns}"
            assert "wall_clock_seconds" in columns, f"wall_clock_seconds not in {columns}"

            # Parse data rows
            lines = f.readlines()
            assert len(lines) >= 3, f"Expected at least 3 episodes logged, got {len(lines)}"

            for line in lines:
                parts = line.strip().split(",")
                row = dict(zip(columns, parts, strict=True))

                # Every row should have total_env_steps
                assert row["total_env_steps"], f"total_env_steps missing in row: {row}"
                # Every row should have wall_clock_seconds
                assert row["wall_clock_seconds"], f"wall_clock_seconds missing in row: {row}"
                # wall_clock_seconds should be numeric and positive
                wall_time = float(row["wall_clock_seconds"])
                assert wall_time >= 0, f"wall_clock_seconds should be >= 0, got {wall_time}"

        # Verify status.json exists and has compute summary
        status_path = run_dir / "status.json"
        assert status_path.exists(), f"status.json not found at {status_path}"

        status = json.loads(status_path.read_text())

        # Check required fields in status.json
        assert "status" in status, f"status field missing in {status}"
        assert status["status"] == "completed", f"Expected status=completed, got {status['status']}"

        assert "condition" in status, f"condition field missing in {status}"
        assert status["condition"] == "DQN", f"Expected condition=DQN, got {status['condition']}"

        assert "total_env_steps" in status, f"total_env_steps field missing in {status}"
        assert isinstance(status["total_env_steps"], int), (
            f"total_env_steps should be int, got {type(status['total_env_steps'])}"
        )
        assert status["total_env_steps"] > 0, (
            f"total_env_steps should be > 0, got {status['total_env_steps']}"
        )

        assert "total_wall_clock_seconds" in status, (
            f"total_wall_clock_seconds field missing in {status}"
        )
        assert status["total_wall_clock_seconds"] > 0, (
            f"total_wall_clock_seconds should be > 0, got {status['total_wall_clock_seconds']}"
        )
