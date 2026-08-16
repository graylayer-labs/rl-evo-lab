"""Environment wrappers for tough variants."""

import os
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np

# ale-py doesn't auto-register its envs under the ALE/ namespace on import in
# this gymnasium/ale-py version combo — without this, gym.make("ALE/...")
# raises NamespaceNotFound. Must happen before any gym.make("ALE/...") call.
gym.register_envs(ale_py)

# ale-py looks for ROM .bin files next to itself by default, but AutoROM (the
# `uv sync` dependency that downloads them, since Atari ROMs can't be
# redistributed directly) installs them into its own package instead. Point
# ale-py at AutoROM's install location so `uv run AutoROM --accept-license`
# is the only setup step needed — no manual file copying.
if "ALE_ROMS_DIR" not in os.environ:
    try:
        import AutoROM

        os.environ["ALE_ROMS_DIR"] = str(Path(AutoROM.__file__).parent / "roms")
    except ImportError:
        pass


class CartPoleToughWrapper(gym.Wrapper):
    """CartPole with random starting state, stricter angle limit, and extended episodes."""

    def __init__(self, env, cfg=None):
        super().__init__(env)
        # Use config flags if provided, otherwise use defaults
        if cfg is not None and hasattr(cfg, "_episode_limit"):
            self.max_steps = cfg._episode_limit
        else:
            self.max_steps = 1000
        if cfg is not None and hasattr(cfg, "_cartpole_angle_limit"):
            self.angle_limit = cfg._cartpole_angle_limit
        else:
            self.angle_limit = 0.209  # 12° in radians
        if cfg is not None and hasattr(cfg, "_cartpole_position_limit"):
            self.position_limit = cfg._cartpole_position_limit
        else:
            self.position_limit = 2.0  # slightly tighter than default 2.4
        self.step_count = 0
        # Per-instance RNG to avoid thread-race on global np.random
        self.rng = np.random.Generator(np.random.PCG64(0))

    def reset(self, seed=None, **kwargs):
        # Reset environment first (without passing seed to avoid double-seeding the base env)
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0

        # Create per-instance RNG from seed to avoid thread-race on global np.random
        if seed is not None:
            self.rng = np.random.Generator(np.random.PCG64(seed))

        # Apply random starting state using the instance RNG
        unwrapped_env = self.env.unwrapped
        state = unwrapped_env.state
        state = np.array(state, dtype=np.float32)
        # Random pole angle: [-0.2, +0.2]
        state[2] = self.rng.uniform(-0.2, 0.2)
        # Random cart position: [-0.5, +0.5]
        state[0] = self.rng.uniform(-0.5, 0.5)
        unwrapped_env.state = state

        # Return observation based on modified state
        obs = np.array(state, dtype=np.float32)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        # Custom termination: stricter angle or max steps
        if abs(obs[2]) > self.angle_limit or abs(obs[0]) > self.position_limit:
            terminated = True
        elif self.step_count >= self.max_steps:
            # Time-limit truncation (not a failure): episode ends but reward counts as success
            truncated = True
            terminated = False

        return obs, reward, terminated, truncated, info


class CartPoleSparseWrapper(gym.Wrapper):
    """CartPole with sparse reward: 0 per step, +500 only at success."""

    def __init__(self, env, cfg=None):
        super().__init__(env)
        self.step_count = 0
        self.max_steps = 500  # CartPole-v1 default episode length

    def reset(self, seed=None, **kwargs):
        obs, info = self.env.reset(seed=seed, **kwargs)
        self.step_count = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        # Sparse reward: 0 per step, +500 only if reached step limit without early termination
        if truncated and not terminated:
            # Reached step limit (success)
            reward = 500.0
        else:
            # Any other outcome: early termination or ongoing
            reward = 0.0

        return obs, reward, terminated, truncated, info


class LunarLanderToughWrapper(gym.Wrapper):
    """LunarLander with random starting state, tight landing zone, and extended episodes."""

    def __init__(self, env, cfg=None):
        super().__init__(env)
        # Use config flags if provided, otherwise use defaults
        if cfg is not None and hasattr(cfg, "_episode_limit"):
            self.max_steps = cfg._episode_limit
        else:
            self.max_steps = 2000
        if cfg is not None and hasattr(cfg, "_landing_zone_radius"):
            self.landing_zone_radius = cfg._landing_zone_radius
        else:
            self.landing_zone_radius = 0.1
        self.step_count = 0
        # Per-instance RNG to avoid thread-race on global np.random
        self.rng = np.random.Generator(np.random.PCG64(0))

    def reset(self, seed=None, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0

        # Create per-instance RNG from seed to avoid thread-race on global np.random
        if seed is not None:
            self.rng = np.random.Generator(np.random.PCG64(seed))

        # Apply random starting state using the instance RNG
        unwrapped_env = self.env.unwrapped
        state = (
            np.array(unwrapped_env.state, dtype=np.float32)
            if hasattr(unwrapped_env, "state")
            else obs.copy()
        )
        # Random position perturbation
        state[0] += self.rng.uniform(-0.3, 0.3)  # x position
        state[1] += self.rng.uniform(-0.1, 0.1)  # y position
        # Random velocity perturbation
        state[2] += self.rng.uniform(-0.5, 0.5)  # x velocity
        state[3] += self.rng.uniform(-0.5, 0.5)  # y velocity

        if hasattr(unwrapped_env, "state"):
            unwrapped_env.state = state

        # Return the modified state as observation
        obs = np.array(state, dtype=np.float32)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        # Check if landed
        if terminated and not truncated:
            # Lander crashed or went out of bounds
            landed_x = obs[0]
            if abs(landed_x) < self.landing_zone_radius:
                # Tight landing zone success: boost reward
                reward = max(reward, 250.0)
            else:
                # Tight zone miss: penalize
                reward = min(reward, -100.0)

        elif self.step_count >= self.max_steps:
            # Time-limit truncation (not a failure): episode ends but reward counts as success
            truncated = True
            terminated = False

        return obs, reward, terminated, truncated, info


def make_env_with_config(env_id: str, cfg) -> gym.Env:
    """Create environment, applying tough-variant or sparse wrappers if needed.

    When using tough wrappers, max_episode_steps is disabled (None) to allow
    the wrapper's custom episode limit to take effect.
    """
    # Check for variant flags
    is_tough = hasattr(cfg, "_tough") and cfg._tough
    is_sparse = hasattr(cfg, "_sparse") and cfg._sparse

    make_kwargs = {"render_mode": None}
    if is_tough:
        make_kwargs["max_episode_steps"] = None
    if env_id.startswith("ALE/"):
        # Flat 128-byte RAM observation (not the default image frame).
        make_kwargs["obs_type"] = "ram"

    env = gym.make(env_id, **make_kwargs)

    # Apply variant wrappers based on config flags
    if is_sparse:
        if "CartPole" in env_id:
            env = CartPoleSparseWrapper(env, cfg)
    elif is_tough:
        if "CartPole" in env_id:
            env = CartPoleToughWrapper(env, cfg)
        elif "LunarLander" in env_id:
            env = LunarLanderToughWrapper(env, cfg)

    return env
