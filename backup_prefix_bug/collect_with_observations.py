#!/usr/bin/env python3
"""
Re-collect PPO rollout data WITH raw observations for feature visualization.

This is a patched version of the collect_features() from feature_space_analysis.py
that additionally saves the rendered observation images.

Usage:
    python collect_with_observations.py \
        --model_path ppo_doorkey_5x5.zip \
        --env_name MiniGrid-DoorKey-5x5-v0 \
        --n_episodes 800 \
        --save_path ./stage1_outputs/collected_data_with_obs.pt

Then point visualize_rule_features.py at the new file:
    python visualize_rule_features.py \
        --features_path ./stage1_outputs/collected_data_with_obs.pt \
        ...
"""

import argparse
import os
import collections

import numpy as np
import torch
import gymnasium as gym
import minigrid  # noqa: F401 — registers MiniGrid envs
try:
    import ale_py  # noqa: F401 — registers ALE (Atari) envs
except Exception:
    pass
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from tqdm import tqdm


# Simple frame stack wrapper for Atari
class FrameStack(gym.Wrapper):
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)
        # Update observation space for stacked frames
        shape = env.observation_space.shape
        if len(shape) == 3:  # HWC
            self.observation_space = gym.spaces.Box(
                low=env.observation_space.low.min(),
                high=env.observation_space.high.max(),
                shape=(shape[0], shape[1], shape[2] * num_stack),
                dtype=env.observation_space.dtype
            )
        else:  # Already grayscale or other
            self.observation_space = gym.spaces.Box(
                low=env.observation_space.low.min(),
                high=env.observation_space.high.max(),
                shape=(*shape, num_stack),
                dtype=env.observation_space.dtype
            )
    
    def _get_obs(self):
        assert len(self.frames) == self.num_stack
        frames_list = list(self.frames)
        # Stack along last axis
        stacked = np.concatenate(frames_list, axis=-1)
        return stacked
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info


# ============================================================================
# Ground-truth factor extraction (E2)
# Raw / continuous values are stored; binning is done later in E2
# (quantile-based, with threshold sensitivity).
# ============================================================================

def _cell_type(env, pos):
    """Object type at grid position, 'out' if outside the grid, 'empty' if None."""
    x, y = int(pos[0]), int(pos[1])
    if not (0 <= x < env.width and 0 <= y < env.height):
        return "out"
    obj = env.grid.get(x, y)
    return "empty" if obj is None else obj.type


def _find_first(env, obj_type):
    """(position, object) of the first object of obj_type in the grid, else (None, None)."""
    for j in range(env.height):
        for i in range(env.width):
            obj = env.grid.get(i, j)
            if obj is not None and obj.type == obj_type:
                return (i, j), obj
    return None, None


def get_factor_names(env_name):
    if "DoorKey" in env_name:
        return ["has_key", "door_open", "key_in_front", "door_in_front",
                "wall_ahead", "agent_dir", "goal_visible"]
    if "Dynamic-Obstacles" in env_name:
        return ["obstacle_ahead", "obstacle_left", "obstacle_right",
                "agent_dir", "goal_dist"]
    if "PixelCartPole" in env_name:
        return ["cart_pos", "cart_vel", "pole_angle", "pole_ang_vel"]
    raise ValueError(f"No factor definition for {env_name}")


def extract_factors(raw_env, env_name, grid_obs=None):
    """
    Read ground-truth factors from the simulator state at the CURRENT step
    (must be called before vec_env.step, same moment as grid_obs).

    Returns: dict {factor_name: float}, keys ordered as get_factor_names(env_name).
    """
    env = raw_env.unwrapped

    if "DoorKey" in env_name:
        from minigrid.core.constants import OBJECT_TO_IDX
        front = _cell_type(env, env.front_pos)
        _, door = _find_first(env, "door")
        return {
            "has_key": float(env.carrying is not None and env.carrying.type == "key"),
            "door_open": float(door is not None and door.is_open),
            "key_in_front": float(front == "key"),
            "door_in_front": float(front == "door"),
            "wall_ahead": float(front == "wall"),
            "agent_dir": float(env.agent_dir),
            "goal_visible": float(grid_obs is not None
                                  and (grid_obs[..., 0] == OBJECT_TO_IDX["goal"]).any()),
        }

    if "Dynamic-Obstacles" in env_name:
        pos = np.array(env.agent_pos)
        goal_pos, _ = _find_first(env, "goal")
        goal_dist = (np.abs(pos - np.array(goal_pos)).sum()
                     if goal_pos is not None else -1)
        return {
            "obstacle_ahead": float(_cell_type(env, pos + env.dir_vec) == "ball"),
            "obstacle_left": float(_cell_type(env, pos - env.right_vec) == "ball"),
            "obstacle_right": float(_cell_type(env, pos + env.right_vec) == "ball"),
            "agent_dir": float(env.agent_dir),
            "goal_dist": float(goal_dist),
        }

    if "PixelCartPole" in env_name:
        x, x_dot, theta, theta_dot = [float(v) for v in env.state]
        return {"cart_pos": x, "cart_vel": x_dot,
                "pole_angle": theta, "pole_ang_vel": theta_dot}

    raise ValueError(f"No factor definition for {env_name}")


def collect_with_observations(
    model_path: str,
    env_name: str,
    n_episodes: int = 800,
    seed: int = 42,
    tile_size: int = 8,
    save_obs_mode: str = "both",
):
    """
    Collect features, actions, AND observations from a frozen PPO policy.

    Args:
        model_path: Path to PPO .zip model
        env_name: MiniGrid environment name
        n_episodes: Number of episodes to collect
        seed: Random seed
        tile_size: Pixel size per grid cell for rendered images
        save_obs_mode: What to save:
            "grid"   — raw grid encoding (H, W, 3) with object/color/state channels
            "pixel"  — rendered RGB pixel images
            "both"   — save both (recommended, ~2x storage but most flexible)

    Returns:
        dict with keys: features, actions, observations_grid, observations_pixel
    """
    print(f"Loading PPO model from {model_path}...")
    model = PPO.load(model_path)

    # We need TWO envs:
    #   1. A wrapped env for the PPO policy (VecTransposeImage expects CHW)
    #   2. Access to the raw MiniGrid env for grid observations
    # 
    # The trick: DummyVecEnv wraps a single env, so we can reach into it.

    raw_env_holder = [None]  # mutable container for the raw env reference

    def make_env():
        def _init():
            from utils_env import make_env_by_name
            env = make_env_by_name(env_name, seed=seed)
            raw_env_holder[0] = env
            return env
        return _init

    vec_env = DummyVecEnv([make_env()])
    # Apply VecTransposeImage for both MiniGrid and Atari (they output HWC format)
    vec_env = VecTransposeImage(vec_env)
    vec_env.seed(seed)

    is_minigrid = "MiniGrid" in env_name
    factor_names = get_factor_names(env_name)

    features_list = []
    actions_list = []
    obs_grid_list = []
    obs_pixel_list = []
    factors_list = []
    episode_ids = []
    step_ids = []
    ep_step = 0

    print(f"Collecting {n_episodes} episodes from {env_name}...")
    obs = vec_env.reset()
    episode_count = 0
    step_count = 0

    with torch.no_grad():
        pbar = tqdm(total=n_episodes, desc="Episodes")
        while episode_count < n_episodes:
            # --- Get raw observation from the underlying MiniGrid env ---
            raw_env = raw_env_holder[0]

            grid_now = (raw_env.unwrapped.gen_obs()['image'].copy()
                        if is_minigrid else None)
            f = extract_factors(raw_env, env_name, grid_now)
            factors_list.append([f[k] for k in factor_names])
            episode_ids.append(episode_count)
            step_ids.append(ep_step)

            if is_minigrid and save_obs_mode in ("grid", "both"):
                # Grid encoding: (H, W, 3) with channels [object_type, color, state]
                # This is what ImgObsWrapper returns before any transposing
                # Access directly from the underlying env
                try:
                    grid_obs = raw_env.unwrapped.gen_obs()['image']
                except AttributeError:
                    # Fallback: grab from the vec env's buffer (already HWC)
                    grid_obs = vec_env.get_attr('get_wrapper_attr', indices=[0])
                    grid_obs = raw_env.observation(raw_env.unwrapped.gen_obs())
                obs_grid_list.append(grid_obs.copy())

            if save_obs_mode in ("pixel", "both"):
                # Rendered pixel image
                try:
                    # MiniGrid's get_obs_render gives a pixel rendering of the agent's view
                    pixel_obs = raw_env.unwrapped.get_obs_render(
                        raw_env.unwrapped.gen_obs()['image'], tile_size=tile_size
                    )
                except (AttributeError, TypeError):
                    # Fallback: full render
                    try:
                        pixel_obs = raw_env.unwrapped.render()
                        if pixel_obs is None:
                            pixel_obs = np.zeros((tile_size * 7, tile_size * 7, 3), dtype=np.uint8)
                    except Exception:
                        pixel_obs = np.zeros((tile_size * 7, tile_size * 7, 3), dtype=np.uint8)
                obs_pixel_list.append(pixel_obs)

            # --- Get features from PPO's feature extractor ---
            action, _ = model.predict(obs, deterministic=True)
            obs_tensor = torch.as_tensor(obs).float().to(model.device)
            features = model.policy.features_extractor(obs_tensor)

            features_list.append(features.cpu())
            actions_list.append(torch.tensor(action))

            # --- Step ---
            obs, rewards, dones, infos = vec_env.step(action)
            step_count += 1
            ep_step += 1

            if dones[0]:
                episode_count += 1
                ep_step = 0
                pbar.update(1)

        pbar.close()

    vec_env.close()

    # Stack everything
    result = {
        'features': torch.cat(features_list, dim=0),
        'actions': torch.cat(actions_list, dim=0),
        'factors': torch.tensor(np.array(factors_list), dtype=torch.float32),
        'factor_names': factor_names,
        'episode_ids': torch.tensor(episode_ids, dtype=torch.long),
        'step_ids': torch.tensor(step_ids, dtype=torch.long),
        'seed': seed,
    }

    if obs_grid_list:
        result['observations'] = torch.tensor(np.stack(obs_grid_list))
        print(f"  Grid observations: {result['observations'].shape}")

    if obs_pixel_list:
        result['observations_pixel'] = torch.tensor(np.stack(obs_pixel_list))
        print(f"  Pixel observations: {result['observations_pixel'].shape}")

    print(f"  Features: {result['features'].shape}")
    print(f"  Actions: {result['actions'].shape}")
    print(f"  Factors: {result['factors'].shape} {factor_names}")
    print(f"  Total steps: {step_count}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Collect PPO rollout data with raw observations"
    )
    parser.add_argument("--model_path", type=str, default="ppo_doorkey_5x5.zip",
                        help="Path to trained PPO model")
    parser.add_argument("--env_name", type=str, default="MiniGrid-DoorKey-5x5-v0",
                        help="MiniGrid environment name")
    parser.add_argument("--n_episodes", type=int, default=800,
                        help="Number of episodes to collect")
    parser.add_argument("--seed", type=int, default=1000,
                        help="Use a seed different from the training collection (0)")
    parser.add_argument("--tile_size", type=int, default=8,
                        help="Pixel size per grid cell for rendered images")
    parser.add_argument("--save_obs_mode", type=str, default="both",
                        choices=["none", "grid", "pixel", "both"],
                        help="What observation format to save")
    parser.add_argument("--save_path", type=str,
                        default="./stage1_outputs/collected_data_with_obs.pt")
    args = parser.parse_args()

    data = collect_with_observations(
        model_path=args.model_path,
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        seed=args.seed,
        tile_size=args.tile_size,
        save_obs_mode=args.save_obs_mode,
    )

    dir_name = os.path.dirname(args.save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    torch.save(data, args.save_path)
    print(f"\nSaved to {args.save_path}")
    print(f"  File size: {os.path.getsize(args.save_path) / 1e6:.1f} MB")

    # Verify the observations look right
    fac = data['factors']
    for i, name in enumerate(data['factor_names']):
        col = fac[:, i]
        print(f"  {name:>15}: mean={col.mean():.3f} min={col.min():.3f} max={col.max():.3f}")

    if 'observations' in data:
        obs = data['observations']
        print(f"\n  Grid obs sample [0,0]: obj_type={obs[0, 0, 0, 0].item()}, "
              f"color={obs[0, 0, 0, 1].item()}, state={obs[0, 0, 0, 2].item()}")
        print(f"  Grid obs range: [{obs.min().item()}, {obs.max().item()}]")
        print(f"  Unique object types: {torch.unique(obs[:, :, :, 0]).tolist()}")


if __name__ == "__main__":
    main()