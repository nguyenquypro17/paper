"""
Stage 1: Feature Collect
================================
Supports: MiniGrid, CartPole, Atari (Boxing, Pong, etc.)
"""

import argparse
import os
import json
import numpy as np
import torch
import matplotlib

import ale_py
import gymnasium as gym

# Register Atari environments for Gymnasium
gym.register_envs(ale_py)
from utils_env import make_env_by_name
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Feature normalization stats
# ---------------------------------------------------------------------------

def compute_normalization_stats(X: np.ndarray):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.maximum(std, 1e-6)

    print(f"\n{'='*60}")
    print(f"NORMALIZATION STATS")
    print(f"{'='*60}")
    print(f"  Feature dimension          : {X.shape[1]}")
    print(f"  Mean range                 : [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  Std range                  : [{std.min():.4f}, {std.max():.4f}]")
    print(f"  Near-zero std dims (<1e-4) : {(std < 1e-4).sum()}")
    print(f"  High-variance dims (>10)   : {(std > 10).sum()}")

    return {"mean": mean, "std": std}


# ---------------------------------------------------------------------------
# Action distribution analysis
# ---------------------------------------------------------------------------

def action_distribution_analysis(actions: np.ndarray, env_name: str = ""):
    n_actions = int(actions.max()) + 1
    
    # Dynamically assign labels based on the environment type
    if "MiniGrid" in env_name and n_actions <= 7:
        action_names = ["TurnLeft", "TurnRight", "Forward", "Pickup", "Drop", "Toggle", "Done"]
        n_actions = 7
    else:
        # Generic labels for Atari or CartPole
        action_names = [f"Action {i}" for i in range(n_actions)]

    counts = np.bincount(actions, minlength=n_actions)
    freqs = counts / counts.sum()

    print(f"\n{'='*60}")
    print(f"ACTION DISTRIBUTION ({env_name})")
    print(f"{'='*60}")
    print(f"  Total samples: {len(actions)}")
    for a in range(n_actions):
        bar = "█" * int(freqs[a] * 50)
        print(f"  {action_names[a]:10s}: {counts[a]:6d} ({freqs[a]*100:5.1f}%) {bar}")

    freqs_nonzero = freqs[freqs > 0]
    entropy = -np.sum(freqs_nonzero * np.log2(freqs_nonzero))
    max_entropy = np.log2(n_actions) if n_actions > 1 else 1.0
    print(f"\n  Entropy: {entropy:.3f} / {max_entropy:.3f} (max)")
    
    return {"counts": counts, "freqs": freqs, "entropy": entropy, "names": action_names}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_diagnostics(norm_stats, action_stats, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 1. Feature Std Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.suptitle("Stage 1: Feature Space Analysis", fontsize=14, fontweight="bold")
    std = norm_stats["std"]
    ax.hist(std, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Per-dimension std"); ax.set_ylabel("Count")
    ax.set_title("Feature Std Distribution"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "stage1_diagnostics.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Action Distribution Plot
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    action_names = action_stats["names"]
    freqs = action_stats["freqs"]
    
    bars = ax2.bar(action_names, freqs * 100, color="steelblue", alpha=0.8, edgecolor="black")
    ax2.set_ylabel("Frequency (%)"); ax2.set_title("Action Distribution in Rollout Data")
    ax2.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")
    
    for bar, f in zip(bars, freqs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{f*100:.1f}%", ha="center", va="bottom", fontsize=8)
    
    plt.tight_layout()
    action_plot_path = os.path.join(save_dir, "action_distribution.png")
    plt.savefig(action_plot_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_stage1_outputs(norm_stats, action_stats, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    stage1_data = {
        "feature_mean": torch.from_numpy(norm_stats["mean"]).float(),
        "feature_std": torch.from_numpy(norm_stats["std"]).float(),
        "action_counts": torch.from_numpy(action_stats["counts"]).long(),
        "action_freqs": torch.from_numpy(action_stats["freqs"]).float(),
        "action_entropy": action_stats["entropy"],
    }
    save_path = os.path.join(save_dir, "stage1_outputs.pt")
    torch.save(stage1_data, save_path)

    summary = {
        "feature_mean_range": [float(norm_stats["mean"].min()), float(norm_stats["mean"].max())],
        "feature_std_range": [float(norm_stats["std"].min()), float(norm_stats["std"].max())],
        "action_entropy": float(action_stats["entropy"]),
    }
    summary_path = os.path.join(save_dir, "stage1_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return save_path

def load_stage1_outputs(path):
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Data collection (Supports MiniGrid, Atari, CartPole)
# ---------------------------------------------------------------------------

def collect_features(model_path, env_name, n_episodes=800, seed=42, tile_size=8):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
    from stable_baselines3.common.env_util import make_atari_env
    from tqdm import tqdm

    print(f"Loading PPO model from {model_path}...")
    model = PPO.load(model_path)

    # 1. ENVIRONMENT DETECTION AND INITIALIZATION
    is_minigrid = "MiniGrid" in env_name
    is_atari = "NoFrameskip" in env_name or "Boxing" in env_name or "Pong" in env_name

    raw_env_ref = [None]

    if is_atari:
        print(f"[*] Detected Atari environment: {env_name}")
        env = make_atari_env(env_name, n_envs=1, seed=seed)
        env = VecFrameStack(env, n_stack=4)
        env = VecTransposeImage(env)
    
    elif is_minigrid:
        print(f"[*] Detected MiniGrid environment: {env_name}")
        import minigrid
        from minigrid.wrappers import ImgObsWrapper
        def make_env():
            def _init():
                e = gym.make(env_name, render_mode="rgb_array")
                raw_env_ref[0] = e
                e = ImgObsWrapper(e)
                e.reset(seed=seed)
                return e
            return _init
        env = DummyVecEnv([make_env()])
        env = VecTransposeImage(env)
        
    else:
        print(f"[*] Detected Classic/Other environment: {env_name}")

        def make_env():
            def _init():
                e = make_env_by_name(
                    env_name,
                    render_mode="rgb_array",
                    seed=seed
                )
                raw_env_ref[0] = e
                return e

            return _init
        env = DummyVecEnv([make_env()])
        env = VecTransposeImage(env)

    # 2. DATA COLLECTION LOOP
    features_list, actions_list = [], []
    obs_grid_list, obs_pixel_list = [], []

    print(f"Collecting {n_episodes} episodes (Deterministic: True)...")
    obs = env.reset()
    episode_count = 0

    with torch.no_grad():
        pbar = tqdm(total=n_episodes)
        while episode_count < n_episodes:
            
            # Extract observation specific to the environment type
            if is_minigrid:
                raw_env = raw_env_ref[0].unwrapped
                try:
                    grid_obs = raw_env.gen_obs()['image']
                    obs_grid_list.append(grid_obs.copy())
                    pixel_obs = raw_env.get_obs_render(grid_obs, tile_size=tile_size)
                    obs_pixel_list.append(pixel_obs)
                except Exception:
                    obs_grid_list.append(np.zeros((7, 7, 3), dtype=np.uint8))
                    obs_pixel_list.append(np.zeros((7 * tile_size, 7 * tile_size, 3), dtype=np.uint8))
            
            elif is_atari:
                # Save raw Atari tensor directly (Shape: 4, 84, 84)
                obs_grid_list.append(obs[0].copy())
                obs_pixel_list.append(obs[0].copy())
                
            else:
                obs_grid_list.append(obs[0].copy())
                try:
                    pixel_obs = raw_env_ref[0].render()
                    obs_pixel_list.append(pixel_obs if pixel_obs is not None else obs[0].copy())
                except:
                    obs_pixel_list.append(obs[0].copy())

            # Model prediction (strictly deterministic for consistent feature extraction)
            action, _ = model.predict(obs, deterministic=True)
            obs_tensor = torch.as_tensor(obs).float().to(model.device)
            
            # Safely extract latent features from the CNN policy
            features = model.policy.features_extractor(obs_tensor)
            
            features_list.append(features.cpu())
            actions_list.append(torch.tensor(action))
            
            obs, rewards, dones, infos = env.step(action)
            
            if dones[0]:
                episode_count += 1
                pbar.update(1)
                
        pbar.close()

    env.close()

    # 3. AGGREGATE DATA
    features = torch.cat(features_list, dim=0)
    actions = torch.cat(actions_list, dim=0)
    
    observations = torch.tensor(np.stack(obs_grid_list))
    
    try:
        observations_pixel = torch.tensor(np.stack(obs_pixel_list))
    except ValueError:
        print("Warning: Pixel observations have inconsistent shapes. Returning raw list.")
        observations_pixel = obs_pixel_list

    print(f"Collected {len(features)} samples, feature dim = {features.shape[1]}")
    
    return features, actions, observations, observations_pixel


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_stage1(features: torch.Tensor, actions: torch.Tensor, env_name: str,
               save_dir: str = "./stage1_outputs"):
    X = features.numpy()
    A = actions.numpy().astype(int)

    print(f"\n{'#'*60}")
    print(f"  STAGE 1: FEATURE SPACE ANALYSIS")
    print(f"  N = {X.shape[0]}, d = {X.shape[1]}")
    print(f"{'#'*60}")

    norm_stats = compute_normalization_stats(X)
    action_stats = action_distribution_analysis(A, env_name)
    plot_diagnostics(norm_stats, action_stats, save_dir)
    save_path = save_stage1_outputs(norm_stats, action_stats, save_dir)

    print(f"\n{'='*60}")
    print(f"RECOMMENDATIONS FOR STAGE 2 (SAE TRAINING)")
    print(f"{'='*60}")
    print(f"  → Normalize features with saved mean/std before SAE training\n")

    return load_stage1_outputs(save_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Feature Space Analysis (Multi-env Support)")
    parser.add_argument("--features_path", type=str, default=None,
                        help="Path to pre-collected features .pt file")
    parser.add_argument("--model_path", type=str, default="ppo_doorkey_6x6.zip",
                        help="PPO model path")
    parser.add_argument("--env_name", type=str, default="MiniGrid-DoorKey-6x6-v0",
                        help="Environment name (MiniGrid, PixelCartPole-v0, PongNoFrameskip-v4, etc.)")
    parser.add_argument("--n_episodes", type=int, default=800,
                        help="Number of episodes to collect")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="./stage1_outputs")

    args = parser.parse_args()

    if args.features_path is not None:
        print(f"Loading features from {args.features_path}...")
        data = torch.load(args.features_path, map_location="cpu", weights_only=False)
        features = data["features"]
        actions = data["actions"]
    else:
        features, actions, observations, observations_pixel = collect_features(
            args.model_path, args.env_name, args.n_episodes, args.seed
        )
        os.makedirs(args.save_dir, exist_ok=True)
        raw_path = os.path.join(args.save_dir, "collected_data.pt")
        
        torch.save({
            "features": features,
            "actions": actions,
            "observations": observations if isinstance(observations, torch.Tensor) else None,
            "observations_pixel": observations_pixel if isinstance(observations_pixel, torch.Tensor) else None,
        }, raw_path)
        print(f"\nRaw data saved: {raw_path}")

    run_stage1(features, actions, env_name=args.env_name, save_dir=args.save_dir)


if __name__ == "__main__":
    main()