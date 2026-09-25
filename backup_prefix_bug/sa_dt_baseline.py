"""
State-Action Decision Tree (SA-DT) Baseline Implementation
(Behavioral Cloning via Standard Decision Tree)

This script trains a standard Decision Tree to imitate a frozen PPO feature extractor.
Unlike VIPER, it does not use DAgger or cost-sensitive resampling.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import torch

import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

import minigrid
import stable_baselines3
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="State-Action Decision Tree (SA-DT) Baseline")

    parser.add_argument(
        "--data_path",
        type=str,
        default="stage1_outputs/collected_data.pt",
        help="Path to the collected data containing features and actions from the CNN.",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Maximum depth of the Decision Tree.",
    )
    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        default=1,
        help="Minimum samples per leaf.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="experiments/sa_dt/results/",
        help="Directory to save outputs (model and metrics).",
    )
    parser.add_argument(
        "--ppo_path",
        type=str,
        default="ppo_doorkey_6x6.zip",
        help="Path to the trained PPO model for feature extraction.",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="MiniGrid-DoorKey-6x6-v0",
        help="Gym environment name to evaluate on.",
    )
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=100,
        help="Number of episodes for game evaluation.",
    )
    parser.add_argument(
        "--max_steps", 
        type=int, 
        default=27000, 
        help="Max steps per episode (27000 recommended for Atari Pong/Boxing)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for environment evaluation.",
    )
    parser.add_argument(
        "--only-test",
        action="store_true",
        help="Skip training and only load/test existing model.",
    )
    parser.add_argument(
        "--multi-seed",
        action="store_true",
        help="Run evaluation on fixed 5 seeds (42,43,44,45,46) with 1/5 episodes each.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the trained SA-DT model (.pkl file). If provided with --only-test, loads from this path instead of save_dir.",
    )

    return parser.parse_args()


def make_eval_env(env_name: str, seed: int = 42):
    """Universal Environment Factory for Atari, MiniGrid, and CartPole."""
    is_atari = "NoFrameskip" in env_name or "Boxing" in env_name or "Pong" in env_name
    is_minigrid = "MiniGrid" in env_name

    if is_atari:
        env = make_atari_env(env_name, n_envs=1, seed=seed, wrapper_kwargs={"clip_reward": False})
        env = VecFrameStack(env, n_stack=4)
        return VecTransposeImage(env)
    elif is_minigrid:
        def _init():
            e = gym.make(env_name, render_mode="rgb_array")
            e = ImgObsWrapper(e)
            e.reset(seed=seed)
            return e
        env = DummyVecEnv([_init])
        return VecTransposeImage(env)
    else:
        def _init():
            import sys
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
            from utils_env import make_env_by_name
            return make_env_by_name(env_name, render_mode="rgb_array", seed=seed)
        env = DummyVecEnv([_init])
        if len(env.observation_space.shape) == 3:
            env = VecTransposeImage(env)
        return env


def calculate_tree_metrics(clf: DecisionTreeClassifier) -> Dict[str, Any]:
    """Calculate interpretability metrics from the scikit-learn DecisionTree."""
    tree = clf.tree_
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right

    rule_set_cardinality = clf.get_n_leaves()
    binarized_concepts = n_nodes - rule_set_cardinality

    total_literals = 0
    stack = [(0, 0)]  

    while stack:
        node_id, depth = stack.pop()
        is_split_node = children_left[node_id] != -1
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            total_literals += depth

    metrics = {
        "Rule Set Cardinality": int(rule_set_cardinality),
        "Binarized Concepts": int(binarized_concepts),
        "Total Literals": int(total_literals),
        "Tree Depth": int(clf.get_depth()),
    }

    return metrics


class SADTAgent:
    """Wrapper to interact with the environment using a frozen PPO feature extractor and a SA-DT for actions."""
    def __init__(self, dt_model: DecisionTreeClassifier, ppo_model: PPO, device: str = "cpu"):
        self.dt_model = dt_model
        self.ppo_model = ppo_model
        self.device = device
        
    def predict(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).float().to(self.device)
            features = self.ppo_model.policy.features_extractor(obs_tensor)
        action = self.dt_model.predict(features.cpu().numpy())
        return action


def evaluate_on_env(dt_model: DecisionTreeClassifier, args: argparse.Namespace) -> Dict[str, Any]:
    """Evaluate the DecisionTreeClassifier directly in the Gym Environment."""
    print(f"\nEvaluating SA-DT policy on {args.n_eval_episodes} episodes of {args.env_name}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        ppo_model = PPO.load(args.ppo_path, device=device)
    except Exception as e:
        print(f"Standard PPO load failed, retrying with custom feature extractors (error: {e})")
        import torch.nn as nn
        class MinigridFeaturesExtractor(BaseFeaturesExtractor):
            def __init__(self, observation_space: gym.Space, features_dim: int = 128):
                super().__init__(observation_space, features_dim)
                n_input_channels = observation_space.shape[0]
                self.cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                if '6x6' in args.env_name:
                    self.cnn = nn.Sequential(
                        nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Flatten(),
                    )
                with torch.no_grad():
                    sample = torch.as_tensor(observation_space.sample()[None]).float()
                    n_flatten = self.cnn(sample).shape[1]
                self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
            def forward(self, observations: torch.Tensor) -> torch.Tensor:
                return self.linear(self.cnn(observations.float()))
        
        custom_objects = {
            "policy_kwargs": {
                "features_extractor_class": MinigridFeaturesExtractor,
                "features_extractor_kwargs": {"features_dim": 128},
                "net_arch": {"pi": [128, 128], "vf": [128, 128]},
            }
        }
        ppo_model = PPO.load(args.ppo_path, device=device, custom_objects=custom_objects)
        
    env = make_eval_env(args.env_name, args.seed)
    agent = SADTAgent(dt_model, ppo_model, device)
    
    successes = 0
    returns = []
    lengths = []
    
    for _ in range(args.n_eval_episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0
        ep_len = 0
        
        while not done and ep_len < args.max_steps:
            action = agent.predict(obs)
            obs, rewards, dones, infos = env.step(action)
            ep_return += float(rewards[0])
            ep_len += 1
            done = bool(dones[0])

            if done and "episode" in infos[0]:
                ep_return = float(infos[0]["episode"]["r"])
            
        is_atari = "NoFrameskip" in args.env_name or "Boxing" in args.env_name or "Pong" in args.env_name
        if "CartPole" in args.env_name:
            is_success = (ep_len >= args.max_steps)
        elif is_atari:
            is_success = ep_return > 0
        else:
            info = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else (infos[0] if infos else {})
            is_success = bool(info.get("is_success", False)) or ep_return > 0
            
        if is_success:
            successes += 1
            
        returns.append(ep_return)
        lengths.append(ep_len)
        
    env.close()
    
    return {
        "Game Success Rate": successes / args.n_eval_episodes,
        "Game Avg Return": float(np.mean(returns)),
        "Game Avg Length": float(np.mean(lengths)),
    }


def main() -> None:
    args = parse_args()
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if args.only_test:
        print(f"[Only-Test Mode] Loading existing model...")
        if args.model_path:
            model_path = Path(args.model_path)
        else:
            model_path = save_dir / "sa_dt_policy.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Train first or provide correct path.")
        clf = joblib.load(model_path)
        print(f"Loaded SA-DT model from {model_path}")
    else:
        print(f"Loading data from {args.data_path}...")
        data = torch.load(args.data_path, map_location="cpu", weights_only=False)

        if isinstance(data, dict):
            if "features" in data and "actions" in data:
                features = data["features"]
                actions = data["actions"]
            else:
                raise KeyError("The .pt file is a dictionary but lacks 'features' or 'actions' keys.")
        elif isinstance(data, tuple) and len(data) == 2:
            features, actions = data
        else:
            raise ValueError("Unsupported data format in .pt file. Expected a dict or tuple.")

        features_np = features.numpy().astype(np.float32)
        actions_np = actions.numpy().astype(np.int64)

        print(f"Features shape: {features_np.shape}, Actions shape: {actions_np.shape}")

        print("Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            features_np, actions_np, test_size=0.20, random_state=42
        )

        print("Training State-Action Decision Tree (SA-DT)...")
        clf = DecisionTreeClassifier(
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=42,
        )
        clf.fit(X_train, y_train)

        print("Calculating metrics...")
        y_pred = clf.predict(X_test)
        action_fidelity = accuracy_score(y_test, y_pred)

        metrics = calculate_tree_metrics(clf)
        metrics["Action Fidelity"] = float(action_fidelity)

        model_path = save_dir / "sa_dt_policy.pkl"
        print(f"Saving model to {model_path}...")
        joblib.dump(clf, model_path)

        metrics_path = save_dir / "sa_dt_metrics.json"
        print(f"Saving metrics to {metrics_path}...")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

    print("Evaluating on environment...")
    game_metrics = evaluate_on_env(clf, args)

    if args.multi_seed:
        print(f"\nMulti-Seed Evaluation: Running on seeds [42,43,44,45,46] with {args.n_eval_episodes//5} episodes each...")
        all_results = {}
        for seed in [42, 43, 44, 45, 46]:
            args.seed = seed
            game_metrics = evaluate_on_env(clf, args)
            all_results[seed] = game_metrics
        
        print(f"\n{'='*70}")
        print("MULTI-SEED AGGREGATED RESULTS")
        print(f"{'='*70}")
        for key in game_metrics.keys():
            values = [all_results[seed][key] for seed in [42,43,44,45,46]]
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            print(f"{key:25s}: {mean_val:.4f} ± {std_val:.4f}")
        print(f"{'='*70}\n")
        
        if not args.only_test:
            metrics_multiseed_path = save_dir / "sa_dt_metrics_multiseed.json"
            multiseed_summary = {}
            for key in game_metrics.keys():
                values = [all_results[seed][key] for seed in [42,43,44,45,46]]
                multiseed_summary[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "per_seed": {str(seed): float(all_results[seed][key]) for seed in [42,43,44,45,46]}
                }
            print(f"Saving multi-seed metrics to {metrics_multiseed_path}...")
            with open(metrics_multiseed_path, "w") as f:
                json.dump(multiseed_summary, f, indent=4)
    else:
        print("\n--- Metrics ---")
        if not args.only_test:
            metrics = {"Game Avg Return": game_metrics.get("Game Avg Return"), "Game Avg Length": game_metrics.get("Game Avg Length")}
            for key, val in metrics.items():
                if isinstance(val, float):
                    print(f"{key}: {val:.4f}")
                else:
                    print(f"{key}: {val}")
        else:
            for key, val in game_metrics.items():
                if isinstance(val, float):
                    print(f"{key}: {val:.4f}")
                else:
                    print(f"{key}: {val}")
        print("---------------\n")

    print("Success! SA-DT baseline execution completed.")


if __name__ == "__main__":
    main()