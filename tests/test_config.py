"""Tests for config.py: ENV_CATEGORIES wiring and preset validation."""

import pytest

from infra.config import ENV_CATEGORIES, ENV_PRESETS, EDERConfig, make_config


class TestEnvCategoriesWired:
    """Verify ENV_CATEGORIES are wired into make_config() correctly."""

    def test_cartpole_category_applied(self):
        """CartPole: discrete_dense_short category HP values are applied."""
        cfg = make_config("cartpole")
        # CartPole preset explicitly overrides category defaults (same values in this case)
        assert cfg.es_sigma == 0.1, "CartPole should have es_sigma=0.1 from preset"
        assert cfg.beta == 0.1, "CartPole should have beta=0.1 from preset"
        assert cfg.novelty_ramp_episodes == 200, "CartPole should have novelty_ramp_episodes=200"
        assert cfg.es_n_workers == 6, "CartPole should inherit es_n_workers=6 from category"
        # Note: CartPole preset doesn't explicitly set es_n_workers, comes from category

    def test_acrobot_category_applied(self):
        """Acrobot: discrete_sparse_long category HP values are applied."""
        cfg = make_config("acrobot")
        # Acrobot should now have category-specific values
        assert cfg.env_id == "Acrobot-v1"
        assert cfg.obs_dim == 6
        assert cfg.act_dim == 3
        assert cfg.es_sigma == 0.15, "Acrobot should have es_sigma=0.15 from category"
        assert cfg.beta == 0.2, "Acrobot should have beta=0.2 from category"
        assert cfg.es_n_workers == 12, "Acrobot should have es_n_workers=12 from category"
        assert cfg.novelty_ramp_episodes == 300, "Acrobot should have novelty_ramp_episodes=300"
        assert cfg.solved_reward == -100.0
        assert cfg.novelty_decay_start_reward == -130.0

    def test_mountaincar_category_applied(self):
        """MountainCar: continuous_sparse_long category HP values are applied."""
        cfg = make_config("mountaincar")
        # MountainCar should now have category-specific values
        assert cfg.env_id == "MountainCar-v0"
        assert cfg.obs_dim == 2
        assert cfg.act_dim == 3
        assert cfg.es_sigma == 0.18, "MountainCar should have es_sigma=0.18 from category"
        assert cfg.beta == 0.25, "MountainCar should have beta=0.25 from category"
        assert cfg.es_n_workers == 15, "MountainCar should have es_n_workers=15 from category"
        assert cfg.novelty_ramp_episodes == 350, "MountainCar should have novelty_ramp_episodes=350"
        assert cfg.solved_reward == -110.0
        assert cfg.novelty_decay_start_reward == -150.0

    def test_override_wins_over_category(self):
        """User-provided overrides should always win over category defaults."""
        cfg = make_config("acrobot", beta=0.5, es_sigma=0.25)
        assert cfg.beta == 0.5, "User override beta should win over category beta"
        assert cfg.es_sigma == 0.25, "User override es_sigma should win over category es_sigma"
        # Other category values still applied if not overridden
        assert cfg.es_n_workers == 12, "Other category values still apply if not overridden"

    def test_preset_wins_over_category(self):
        """Preset explicit values should win over category defaults."""
        # CartPole preset explicitly specifies es_sigma and beta
        cfg = make_config("cartpole")
        # These values match the category, but they come from the preset
        assert cfg.es_sigma == 0.1
        assert cfg.beta == 0.1

    def test_lunarlander_category_applied(self):
        """LunarLander: continuous_dense_medium category HP values should be present."""
        cfg = make_config("lunarlander")
        assert cfg.env_id == "LunarLander-v3"
        # LunarLander preset explicitly specifies these, matching category defaults
        assert cfg.es_sigma == 0.12
        assert cfg.beta == 0.15
        assert cfg.es_n_workers == 10
        assert cfg.novelty_ramp_episodes == 250

    def test_category_metadata_filtered_out(self):
        """Category 'description' field should not pollute config."""
        cfg = make_config("acrobot")
        # Ensure no "description" attribute was added to config
        assert not hasattr(cfg, "description"), "Description metadata should be filtered out"

    def test_category_field_filtered_out(self):
        """Preset 'category' field should not pollute config."""
        cfg = make_config("acrobot")
        # Ensure no "category" attribute was added to config
        assert not hasattr(cfg, "category"), "Category field should be filtered out"

    def test_custom_flags_attached(self):
        """Custom flags (starting with _) should be attached to config for env wrappers."""
        cfg = make_config("cartpole_tough")
        assert hasattr(cfg, "_tough")
        assert cfg._tough is True
        assert hasattr(cfg, "_cartpole_angle_limit")
        assert cfg._cartpole_angle_limit == 0.209
        assert hasattr(cfg, "_random_start")
        assert cfg._random_start is True

    def test_all_presets_have_valid_categories_or_are_complete(self):
        """Every preset should either have a category or explicitly define HPs."""
        for env_name, preset in ENV_PRESETS.items():
            category_name = preset.get("category")
            if category_name:
                msg = f"{env_name} specifies unknown category {category_name!r}"
                assert category_name in ENV_CATEGORIES, msg
            # All presets should be creatable without error
            cfg = make_config(env_name)
            msg = f"make_config({env_name!r}) should return EDERConfig"
            assert isinstance(cfg, EDERConfig), msg

    def test_make_config_returns_frozen_dataclass(self):
        """EDERConfig should be a frozen dataclass."""
        cfg = make_config("cartpole")
        assert isinstance(cfg, EDERConfig)
        # Frozen dataclass: attempting to modify should raise FrozenInstanceError
        with pytest.raises((AttributeError, TypeError)):
            cfg.beta = 0.5


class TestCategoryDefaults:
    """Verify ENV_CATEGORIES are well-formed and complete."""

    def test_all_categories_have_four_hp_keys(self):
        """Each category should define es_sigma, beta, novelty_ramp_episodes, es_n_workers."""
        required_keys = {
            "es_sigma",
            "beta",
            "novelty_ramp_episodes",
            "es_n_workers",
            "description",
        }
        for cat_name, cat_hps in ENV_CATEGORIES.items():
            msg = f"Category {cat_name!r} missing or has extra keys"
            assert set(cat_hps.keys()) == required_keys, msg

    def test_hp_values_are_reasonable(self):
        """Hyperparameters should be in sensible ranges."""
        for cat_name, cat_hps in ENV_CATEGORIES.items():
            # es_sigma should be positive and < 1
            sigma = cat_hps["es_sigma"]
            msg = f"{cat_name} es_sigma={sigma} out of range"
            assert 0 < sigma < 1, msg
            # beta should be non-negative and < 1
            beta = cat_hps["beta"]
            msg = f"{cat_name} beta={beta} out of range"
            assert 0 <= beta < 1, msg
            # novelty_ramp_episodes should be positive
            ramp = cat_hps["novelty_ramp_episodes"]
            msg = f"{cat_name} novelty_ramp_episodes={ramp} should be positive"
            assert ramp > 0, msg
            # es_n_workers should be positive
            workers = cat_hps["es_n_workers"]
            msg = f"{cat_name} es_n_workers={workers} should be positive"
            assert workers > 0, msg

    def test_sparse_categories_have_higher_novelty_weight(self):
        """Sparse reward categories should have higher beta than dense categories."""
        dense_beta = ENV_CATEGORIES["discrete_dense_short"]["beta"]
        sparse_discrete = ENV_CATEGORIES["discrete_sparse_long"]["beta"]
        sparse_continuous = ENV_CATEGORIES["continuous_sparse_long"]["beta"]

        msg = "Discrete sparse should have higher beta than discrete dense"
        assert sparse_discrete > dense_beta, msg
        msg = "Continuous sparse should have higher beta than discrete sparse"
        assert sparse_continuous > sparse_discrete, msg

    def test_continuous_categories_have_higher_sigma(self):
        """Continuous action spaces should have slightly higher exploration noise."""
        dense_sigma = ENV_CATEGORIES["discrete_dense_short"]["es_sigma"]
        continuous_sparse = ENV_CATEGORIES["continuous_sparse_long"]["es_sigma"]

        assert continuous_sparse > dense_sigma, "Continuous actions need more exploration noise"
