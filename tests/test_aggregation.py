"""Tests for aggregation and plotting utilities."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from rl_evo_lab.utils.compare import _aggregate


def test_aggregate_with_different_run_lengths():
    """Test that _aggregate pads shorter runs with NaN instead of truncating.

    This tests the fix for Phase 3.4: seed aggregation should preserve
    longer runs rather than truncating all seeds to the shortest run.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create 3 CSV files with different run lengths
        # Seed 1: 10 episodes
        df1 = pd.DataFrame(
            {
                "episode": range(10),
                "total_env_steps": np.arange(10) * 100,
                "learner_eval_reward": np.linspace(0, 100, 10),
            }
        )
        csv1 = tmpdir / "seed1.csv"
        df1.to_csv(csv1, index=False)

        # Seed 2: 8 episodes (shorter)
        df2 = pd.DataFrame(
            {
                "episode": range(8),
                "total_env_steps": np.arange(8) * 100,
                "learner_eval_reward": np.linspace(10, 80, 8),
            }
        )
        csv2 = tmpdir / "seed2.csv"
        df2.to_csv(csv2, index=False)

        # Seed 3: 12 episodes (longer)
        df3 = pd.DataFrame(
            {
                "episode": range(12),
                "total_env_steps": np.arange(12) * 100,
                "learner_eval_reward": np.linspace(-10, 110, 12),
            }
        )
        csv3 = tmpdir / "seed3.csv"
        df3.to_csv(csv3, index=False)

        # Aggregate using episode-based x-axis
        x, mean, std = _aggregate([csv1, csv2, csv3], "learner_eval_reward", x_col="episode")

        # With NaN padding: x should have 12 elements (longest run)
        assert len(x) == 12, f"Expected x length 12 (longest run), got {len(x)}"
        assert len(mean) == 12, f"Expected mean length 12, got {len(mean)}"
        assert len(std) == 12, f"Expected std length 12, got {len(std)}"

        # The first 8 episodes should be valid (all 3 seeds present)
        # Episodes 8-9 should have NaN in seed 2, so std should reflect that
        assert not np.isnan(mean[0:8]).any(), "First 8 episodes should have valid means"

        # Last 2 episodes should have some NaN values (only seed 1 and 3 present)
        # They shouldn't crash or return all NaN
        assert not np.isnan(mean[10:12]).any(), "Mean should use nanmean to handle missing values"


def test_aggregate_with_env_steps_xaxis():
    """Test aggregation using total_env_steps as x-axis.

    This tests that Option B (interpolation) won't break if x_col != "episode".
    For now we're implementing Option A (NaN padding), so this verifies
    that the x-axis is correctly extracted.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create 2 CSV files with different step counts
        df1 = pd.DataFrame(
            {
                "episode": range(5),
                "total_env_steps": [100, 150, 200, 250, 300],
                "learner_eval_reward": [10, 20, 30, 40, 50],
            }
        )
        csv1 = tmpdir / "seed1.csv"
        df1.to_csv(csv1, index=False)

        df2 = pd.DataFrame(
            {
                "episode": range(6),
                "total_env_steps": [100, 155, 210, 265, 320, 375],
                "learner_eval_reward": [15, 25, 35, 45, 55, 65],
            }
        )
        csv2 = tmpdir / "seed2.csv"
        df2.to_csv(csv2, index=False)

        # Aggregate using env steps
        x, mean, std = _aggregate([csv1, csv2], "learner_eval_reward", x_col="total_env_steps")

        # x should be from the first dataframe and have the padded length
        assert len(x) == 6, f"Expected x length 6 (longest run), got {len(x)}"
        # The x values should be the env steps from the first dataframe
        expected_x = df1["total_env_steps"].values
        np.testing.assert_array_equal(x[:5], expected_x)


def test_aggregate_single_seed():
    """Test that aggregation works with a single seed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        df = pd.DataFrame(
            {
                "episode": range(5),
                "total_env_steps": np.arange(5) * 100,
                "learner_eval_reward": [10, 20, 30, 40, 50],
            }
        )
        csv = tmpdir / "seed.csv"
        df.to_csv(csv, index=False)

        x, mean, std = _aggregate([csv], "learner_eval_reward", x_col="episode")

        # With single seed, std should be all zeros
        assert len(x) == 5
        assert len(mean) == 5
        assert len(std) == 5
        np.testing.assert_array_equal(mean, [10, 20, 30, 40, 50])
        np.testing.assert_array_equal(std, [0, 0, 0, 0, 0])
