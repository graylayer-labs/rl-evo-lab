"""Environment wrappers for tough variants."""
import gymnasium as gym
import numpy as np


class CartPoleToughWrapper(gym.Wrapper):
    """CartPole with random starting state, stricter angle limit, and 1000-step episodes."""

    def __init__(self, env):
        super().__init__(env)
        self.max_steps = 1000
        self.step_count = 0
        self.angle_limit = 0.209  # 12° in radians
        self.position_limit = 2.0  # slightly tighter than default 2.4

    def reset(self, **kwargs):
        # Reset environment first
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0

        # Apply random starting state
        unwrapped_env = self.env.unwrapped
        state = unwrapped_env.state
        state = np.array(state, dtype=np.float32)
        # Random pole angle: [-0.2, +0.2]
        state[2] = np.random.uniform(-0.2, 0.2)
        # Random cart position: [-0.5, +0.5]
        state[0] = np.random.uniform(-0.5, 0.5)
        unwrapped_env.state = state

        obs = unwrapped_env._get_obs()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        # Custom termination: stricter angle or max steps
        if abs(obs[2]) > self.angle_limit or abs(obs[0]) > self.position_limit:
            terminated = True
        if self.step_count >= self.max_steps:
            truncated = False  # Reaching max steps is NOT failure, counts as success
            terminated = False

        return obs, reward, terminated, truncated, info


class LunarLanderToughWrapper(gym.Wrapper):
    """LunarLander with random starting state, tight landing zone, and 2000-step episodes."""

    def __init__(self, env):
        super().__init__(env)
        self.max_steps = 2000
        self.step_count = 0
        self.landing_zone_radius = 0.1

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0

        # Apply random starting state
        unwrapped_env = self.env.unwrapped
        state = list(unwrapped_env.state) if hasattr(unwrapped_env, 'state') else obs[:8]
        # Random position perturbation
        state[0] += np.random.uniform(-0.3, 0.3)  # x position
        state[1] += np.random.uniform(-0.1, 0.1)  # y position
        # Random velocity perturbation
        state[2] += np.random.uniform(-0.5, 0.5)  # x velocity
        state[3] += np.random.uniform(-0.5, 0.5)  # y velocity

        if hasattr(unwrapped_env, 'state'):
            unwrapped_env.state = state
        obs = unwrapped_env._get_obs() if hasattr(unwrapped_env, '_get_obs') else obs

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

        if self.step_count >= self.max_steps:
            truncated = False
            terminated = False

        return obs, reward, terminated, truncated, info


def make_env_with_config(env_id: str, cfg) -> gym.Env:
    """Create environment, applying tough-variant wrappers if needed."""
    env = gym.make(env_id, render_mode=None)

    # Apply tough variant wrappers based on config flags
    if hasattr(cfg, '_tough') and cfg._tough:
        if 'CartPole' in env_id:
            env = CartPoleToughWrapper(env)
        elif 'LunarLander' in env_id:
            env = LunarLanderToughWrapper(env)

    return env
