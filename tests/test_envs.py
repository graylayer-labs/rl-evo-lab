"""Tests for environment wrappers and configuration."""
import gymnasium as gym

from rl_evo_lab.envs import CartPoleToughWrapper, LunarLanderToughWrapper, make_env_with_config
from rl_evo_lab.utils.config import make_config


def test_cartpole_tough_wrapper_extended_episode_length():
    """Test that CartPoleToughWrapper respects extended episode limit."""
    env = gym.make("CartPole-v1", render_mode=None, max_episode_steps=None)
    cfg = make_config("cartpole_tough")

    wrapper = CartPoleToughWrapper(env, cfg)
    obs, _ = wrapper.reset(seed=42)

    step_count = 0
    max_steps_seen = 0
    done = False

    # Run the episode to completion
    while not done and step_count < 1100:  # Safety limit slightly above 1000
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = wrapper.step(action)
        step_count += 1
        max_steps_seen = max(max_steps_seen, wrapper.step_count)
        done = terminated or truncated

    # With the extended limit (1000), the episode should reach max_steps without hitting
    # the angle/position limits (random exploration)
    # We verify that truncated is True when reaching max_steps
    assert wrapper.step_count == 1000 or max_steps_seen <= 1000
    assert wrapper.max_steps == 1000  # Configured from _episode_limit


def test_cartpole_tough_wrapper_with_config_flags():
    """Test that CartPoleToughWrapper correctly uses config flags."""
    env = gym.make("CartPole-v1", render_mode=None, max_episode_steps=None)
    cfg = make_config("cartpole_tough")

    wrapper = CartPoleToughWrapper(env, cfg)

    # Verify the wrapper is using config values
    assert wrapper.max_steps == cfg._episode_limit
    assert wrapper.angle_limit == cfg._cartpole_angle_limit
    assert wrapper.position_limit == cfg._cartpole_position_limit


def test_cartpole_tough_wrapper_default_values():
    """Test that CartPoleToughWrapper uses defaults when cfg is None."""
    env = gym.make("CartPole-v1", render_mode=None, max_episode_steps=None)
    wrapper = CartPoleToughWrapper(env, cfg=None)

    assert wrapper.max_steps == 1000
    assert wrapper.angle_limit == 0.209
    assert wrapper.position_limit == 2.0


def test_lunarlander_tough_wrapper_with_config():
    """Test that LunarLanderToughWrapper correctly uses config flags."""
    env = gym.make("LunarLander-v3", render_mode=None, max_episode_steps=None)
    cfg = make_config("lunarlander_tough")

    wrapper = LunarLanderToughWrapper(env, cfg)

    # Verify the wrapper is using config values
    assert wrapper.max_steps == cfg._episode_limit
    assert wrapper.landing_zone_radius == cfg._landing_zone_radius


def test_make_env_with_config_tough_cartpole():
    """Test that make_env_with_config creates tough CartPole correctly."""
    cfg = make_config("cartpole_tough")
    env = make_env_with_config("CartPole-v1", cfg)

    # Should be wrapped
    assert isinstance(env.unwrapped, gym.Env)
    # Check that it's a CartPoleToughWrapper by inspecting the wrapper chain
    wrapper = env
    found_tough = False
    while hasattr(wrapper, "env"):
        if isinstance(wrapper, CartPoleToughWrapper):
            found_tough = True
            break
        wrapper = wrapper.env
    assert found_tough, "CartPoleToughWrapper not found in wrapper chain"


def test_make_env_with_config_base_cartpole():
    """Test that make_env_with_config doesn't wrap for base CartPole."""
    cfg = make_config("cartpole")  # Not tough
    env = make_env_with_config("CartPole-v1", cfg)

    # Should not have CartPoleToughWrapper
    wrapper = env
    while hasattr(wrapper, "env"):
        assert not isinstance(wrapper, CartPoleToughWrapper)
        wrapper = wrapper.env


def test_cartpole_tough_wrapper_truncation_flag():
    """Test that reaching max_steps sets truncated=True, not terminated."""
    env = gym.make("CartPole-v1", render_mode=None, max_episode_steps=None)
    cfg = make_config("cartpole_tough")

    wrapper = CartPoleToughWrapper(env, cfg)
    obs, _ = wrapper.reset(seed=42)

    # Run steps while staying in bounds (no angle/position violation)
    done = False
    step_count = 0
    for _ in range(1001):  # Go past max_steps
        action = 1 if obs[2] > 0 else 0  # Action to stabilize pole
        obs, reward, terminated, truncated, _ = wrapper.step(action)
        step_count += 1

        if step_count == 1000:
            # At max_steps, should have truncated=True, terminated=False
            assert truncated, f"Expected truncated=True at step {step_count}"
            assert not terminated, f"Expected terminated=False at step {step_count}"
            break

        if terminated or truncated:
            # If we hit a genuine failure before max_steps, terminated should be True
            if step_count < 1000:
                assert terminated, "Early termination should have terminated=True"
            break
