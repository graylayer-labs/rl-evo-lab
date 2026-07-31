"""Render a trained policy as an animated GIF."""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import torch

from rl_evo_lab.learner.network import QNetwork
from rl_evo_lab.utils.config import make_config


def render_episode(run_dir: str, output_path: str | None = None, fps: int = 30) -> None:
    """Render one greedy episode from a trained policy as a GIF.

    Args:
        run_dir: Path to a run directory containing config.json and policy.pt
        output_path: Where to save the GIF. If None, defaults to docs/img/<env>_solved.gif
        fps: Frames per second for the GIF
    """
    run_path = Path(run_dir)
    config_path = run_path / "config.json"
    policy_path = run_path / "policy.pt"

    if not config_path.exists() or not policy_path.exists():
        raise FileNotFoundError(
            f"Run directory must contain config.json and policy.pt. "
            f"Found: config={config_path.exists()}, policy={policy_path.exists()}"
        )

    with open(config_path) as f:
        config_dict = json.load(f)

    env_id = config_dict.get("env_id", "CartPole-v1")
    env_name = env_id.lower().split("-")[0]
    seed = config_dict.get("seed", 42)

    cfg = make_config(env=env_name, seed=seed)

    device = torch.device("cpu")
    policy_net = QNetwork(cfg.obs_dim, cfg.act_dim, hidden=cfg.hidden_dim).to(device)
    policy_net.load_state_dict(torch.load(policy_path, map_location=device, weights_only=True))
    policy_net.eval()

    env = gym.make(cfg.env_id, render_mode="rgb_array")
    obs, _ = env.reset(seed=cfg.seed)
    frames = []
    done = False
    total_reward = 0.0

    while not done:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            action = policy_net(obs_t).argmax(dim=1).item()

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    env.close()

    if output_path is None:
        output_path = f"docs/img/{cfg.env_id.split('-')[0].lower()}_solved.gif"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if frames:
        imageio.mimsave(str(output_path), frames, fps=fps)
        print(f"Saved GIF to {output_path} (episode reward: {total_reward:.1f}, {len(frames)} frames)")
    else:
        print("No frames rendered (environment did not return RGB data)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a trained policy as a GIF")
    parser.add_argument("run_dir", help="Path to the run directory (containing config.json and policy.pt)")
    parser.add_argument("--output", help="Output GIF path (default: docs/img/<env>_solved.gif)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    args = parser.parse_args()

    render_episode(args.run_dir, args.output, args.fps)
