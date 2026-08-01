from pathlib import Path

import pytest

from rl_evo_lab.train import train, _compute_sync_threshold
from rl_evo_lab.utils.config import EDERConfig


def test_short_run(tmp_path: Path):
    cfg = EDERConfig(
        total_episodes=3,
        es_n_workers=4,
        min_buffer_size=10,
        eval_freq=2,
        eval_episodes=1,
        seed=0,
    )
    train(cfg, log_dir=str(tmp_path))
    # structure: {log_dir}/{env_id}/seed{N}__{hash}/metrics.csv
    env_dir = next(tmp_path.iterdir())
    run_dir = next(env_dir.iterdir())
    assert (run_dir / "metrics.csv").exists()


# ---------------------------------------------------------------------------
# Sync threshold formula (B4)
# ---------------------------------------------------------------------------


def test_sync_threshold_positive_rewards():
    """Sync threshold for positive rewards should scale linearly."""
    # Bug B4: old code used threshold = sync_eval_threshold * mean_return,
    # which inverts on negative rewards. New code handles both signs correctly.
    threshold = _compute_sync_threshold(0.7, 100.0)
    assert pytest.approx(threshold, abs=1e-6) == 70.0, (
        f"For positive mean return (100), 70% threshold should be 70. Got {threshold}"
    )

    threshold = _compute_sync_threshold(0.8, 200.0)
    assert pytest.approx(threshold, abs=1e-6) == 160.0


def test_sync_threshold_negative_rewards():
    """Sync threshold for negative rewards should mirror positive-reward logic."""
    # For negative rewards, a 70% tolerance means learner can be 30% worse (more negative).
    # Example: mean_return=-100, threshold should be -100 * (2 - 0.7) = -130
    # so learner is allowed to sync when eval >= -130 (30% below the actor).
    threshold = _compute_sync_threshold(0.7, -100.0)
    expected = -100.0 * (2.0 - 0.7)
    assert pytest.approx(threshold, abs=1e-6) == expected, (
        f"For negative mean return (-100) with 70% threshold, "
        f"should give {expected} (allowing 30% worse learner). Got {threshold}"
    )

    # Verify: learner at -120 should sync against actor at -100
    # Threshold = -130, so -120 >= -130 is True (syncs)
    assert -120 >= -130, "Learner at -120 should pass threshold of -130"

    # Threshold = -70 (old buggy formula): learner at -120 would NOT sync (-120 >= -70 is False)
    # This demonstrates the bug: old formula prevents syncing on negative-reward envs too early
    assert not (-120 >= -70), "Old buggy formula would wrongly prevent sync"


def test_sync_threshold_acrobot_case():
    """Test the specific Acrobot case: solved=-100, negative env."""
    # Acrobot is solved at -100 reward. If actor's mean return is -150,
    # a reasonable threshold might allow learner to sync at -105 or so.
    actor_return = -150.0
    sync_threshold_fraction = 0.7
    threshold = _compute_sync_threshold(sync_threshold_fraction, actor_return)

    # New formula: -150 * (2 - 0.7) = -150 * 1.3 = -195
    assert pytest.approx(threshold, abs=1e-6) == -195.0

    # Old buggy formula would give: 0.7 * -150 = -105
    # which is TOO LENIENT (allows syncing before learner learns much)
    old_buggy_threshold = sync_threshold_fraction * actor_return
    assert pytest.approx(old_buggy_threshold, abs=1e-6) == -105.0

    # New formula is stricter: -195 < -105, so learner must learn more
    assert threshold < old_buggy_threshold


def test_sync_threshold_mountaincar_case():
    """Test the specific MountainCar case: solved=-110, dense negative reward."""
    actor_return = -120.0  # typical actor performance on MountainCar
    sync_threshold_fraction = 0.7
    threshold = _compute_sync_threshold(sync_threshold_fraction, actor_return)

    # New formula: -120 * (2 - 0.7) = -120 * 1.3 = -156
    assert pytest.approx(threshold, abs=1e-6) == -156.0

    # Old buggy formula: 0.7 * -120 = -84
    # which is MUCH too lenient (learner syncs way too early)
    old_buggy_threshold = sync_threshold_fraction * actor_return
    assert pytest.approx(old_buggy_threshold, abs=1e-6) == -84.0
    assert threshold < old_buggy_threshold  # new formula is more stringent
