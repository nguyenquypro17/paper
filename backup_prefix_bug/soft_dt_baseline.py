"""
Sparse Soft Decision Tree (SDT) Baseline Implementation
Inspired by Frosst & Hinton, 2017 - Enhanced with L1 Regularization for Interpretability.

This script trains a Sparse Soft Decision Tree to approximate a frozen PPO policy.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sparse Soft Decision Tree (SDT) Baseline")
    parser.add_argument(
        "--data_path",
        type=str,
        default="stage1_outputs/collected_data.pt",
        help="Path to the collected data containing continuous features and discrete actions.",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=4,
        help="Depth of the Soft Decision Tree.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--l1_lambda",
        type=float,
        default=1e-3,
        help="L1 regularization coefficient to make the tree sparse.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Training batch size.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="experiments/soft_dt/results/",
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
        help="Max steps per episode (27000 recommended for Atari)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for evaluation.",
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
        help="Path to the trained Soft DT model (.pth file). If provided with --only-test, loads from this path instead of save_dir.",
    )
    return parser.parse_args()


class SoftDecisionTree(nn.Module):
    """A PyTorch Sparse Soft Decision Tree model."""
    def __init__(self, input_dim: int, output_dim: int, depth: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        
        self.num_internal_nodes = (2 ** depth) - 1
        self.num_leaves = 2 ** depth

        self.routing = nn.Linear(self.input_dim, self.num_internal_nodes)
        self.leaf_logits = nn.Parameter(torch.randn(self.num_leaves, self.output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        p = torch.sigmoid(self.routing(x))
        
        node_probs = [None] * (self.num_internal_nodes + self.num_leaves)
        node_probs[0] = torch.ones(batch_size, device=x.device)
        
        for i in range(self.num_internal_nodes):
            current_prob = node_probs[i]
            prob_right = p[:, i]
            prob_left = 1.0 - prob_right
            
            left_child = 2 * i + 1
            right_child = 2 * i + 2
            
            node_probs[left_child] = current_prob * prob_left
            node_probs[right_child] = current_prob * prob_right
            
        leaf_probs = torch.stack(node_probs[self.num_internal_nodes:], dim=1)
        leaf_dists = F.softmax(self.leaf_logits, dim=-1)
        output_probs = torch.matmul(leaf_probs, leaf_dists)
        
        return output_probs

    def get_metrics(self) -> Dict[str, Any]:
        rule_set_cardinality = self.num_leaves
        binarized_concepts = self.num_internal_nodes
        
        epsilon = 1e-3
        non_zero_weights = (torch.abs(self.routing.weight) > epsilon).float()
        total_literals_in_model = int(non_zero_weights.sum().item())
        
        avg_literals_per_node = total_literals_in_model / max(1, self.num_internal_nodes)
        total_literals = int(avg_literals_per_node * self.depth * self.num_leaves)
        
        return {
            "Rule Set Cardinality": rule_set_cardinality,
            "Binarized Concepts": binarized_concepts,
            "Total Literals (Global)": total_literals_in_model,
            "Total Literals (DNF Equivalent)": total_literals,
            "Tree Depth": self.depth
        }


class SoftDTAgent:
    """Wrapper to interact with the environment using a frozen PPO feature extractor and a Soft DT for actions."""
    def __init__(self, sdt_model: SoftDecisionTree, ppo_model: PPO, device: torch.device):
        self.sdt_model = sdt_model
        self.sdt_model.eval()
        self.ppo_model = ppo_model
        self.device = device
        
    def predict(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).float().to(self.device)
            features = self.ppo_model.policy.features_extractor(obs_tensor)
            action_probs = self.sdt_model(features)
            action = torch.argmax(action_probs, dim=1)
        return action.cpu().numpy()


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


def evaluate_on_env(sdt_model: SoftDecisionTree, args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    print(f"\nEvaluating SDT policy on {args.n_eval_episodes} episodes of {args.env_name}...")
    
    try:
        ppo_model = PPO.load(args.ppo_path, device=device)
    except Exception as e:
        print(f"Standard PPO load failed, retrying with custom feature extractors (error: {e})")
        class MinigridFeaturesExtractor(BaseFeaturesExtractor):
            def __init__(self, observation_space: gym.Space, features_dim: int = 128, env_name: str = ""):
                super().__init__(observation_space, features_dim)
                n_input_channels = observation_space.shape[0]
                layers = [
                    nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                ]
                if '6x6' in env_name:
                    layers.extend([
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU()
                    ])
                layers.append(nn.Flatten())
                self.cnn = nn.Sequential(*layers)
                
                with torch.no_grad():
                    sample = torch.as_tensor(observation_space.sample()[None]).float()
                    n_flatten = self.cnn(sample).shape[1]
                self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
                
            def forward(self, observations: torch.Tensor) -> torch.Tensor:
                return self.linear(self.cnn(observations.float()))
        
        custom_objects = {
            "policy_kwargs": {
                "features_extractor_class": lambda obs_space, feat_dim: MinigridFeaturesExtractor(obs_space, feat_dim, args.env_name),
                "features_extractor_kwargs": {"features_dim": 128},
                "net_arch": {"pi": [128, 128], "vf": [128, 128]},
            }
        }
        ppo_model = PPO.load(args.ppo_path, device=device, custom_objects=custom_objects)
        
    env = make_eval_env(args.env_name, args.seed)
    agent = SoftDTAgent(sdt_model, ppo_model, device)
    
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
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    actions = actions.long().view(-1)
    
    # [FIX] Environment-based Action Labeling to ensure correct output_dim
    n_actions = int(actions.max().item()) + 1
    if "Pong" in args.env_name:
        n_actions = max(n_actions, 6)
    elif "MiniGrid" in args.env_name:
        n_actions = max(n_actions, 7)
    elif "CartPole" in args.env_name:
        n_actions = max(n_actions, 2)
    elif "Boxing" in args.env_name:
        n_actions = max(n_actions, 18)
        
    # Clamp actions safely
    actions = torch.clamp(actions, min=0, max=n_actions - 1)
    
    input_dim = features.shape[1]
    output_dim = n_actions
    
    if args.only_test:
        print(f"[Only-Test Mode] Loading existing Soft DT model with depth={args.max_depth}...")
        if args.model_path:
            model_path = Path(args.model_path)
        else:
            model_path = save_dir / "soft_dt_model.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Train first or provide correct path.")
        
        model = SoftDecisionTree(input_dim=input_dim, output_dim=output_dim, depth=args.max_depth).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"Loaded Soft DT model from {model_path}")
    else:
        print(f"Initializing Sparse Soft Decision Tree (L1={args.l1_lambda}) with {output_dim} outputs...")
        
        dataset = TensorDataset(features, actions)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(args.seed)
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model = SoftDecisionTree(input_dim=input_dim, output_dim=output_dim, depth=args.max_depth).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        print(f"Training for {args.epochs} epochs...")
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                output_probs = model(batch_x)
                
                log_probs = torch.log(output_probs + 1e-8)
                nll_loss = F.nll_loss(log_probs, batch_y)
                
                l1_penalty = torch.norm(model.routing.weight, p=1)
                loss = nll_loss + args.l1_lambda * l1_penalty
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_x.size(0)
                
            train_loss /= len(train_dataset)
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch [{epoch}/{args.epochs}] -> Loss: {train_loss:.4f}")

        print("Calculating metrics...")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output_probs = model(batch_x)
                preds = torch.argmax(output_probs, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        action_fidelity = correct / total

        metrics = model.get_metrics()
        metrics["Action Fidelity"] = float(action_fidelity)
        
        model_path = save_dir / "soft_dt_model.pth"
        print(f"Saving model to {model_path} (depth={args.max_depth})...")
        torch.save(model.state_dict(), model_path)
        
        metrics_path = save_dir / "soft_dt_metrics.json"
        print(f"Saving metrics to {metrics_path}...")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

    print("Evaluating on environment...")
    game_metrics = evaluate_on_env(model, args, device)

    if args.multi_seed:
        print(f"\nMulti-Seed Evaluation: Running on seeds [42,43,44,45,46] with {args.n_eval_episodes//5} episodes each...")
        all_results = {}
        for seed in [42, 43, 44, 45, 46]:
            args_copy = type('obj', (object,), vars(args))()
            args_copy.seed = seed
            args_copy.n_eval_episodes = args.n_eval_episodes // 5
            game_metrics_seed = evaluate_on_env(model, args_copy, device)
            all_results[seed] = game_metrics_seed
        
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
            metrics_multiseed_path = save_dir / "soft_dt_metrics_multiseed.json"
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
        print("\n--- Game Evaluation Metrics ---")
        for key, val in game_metrics.items():
            if isinstance(val, float):
                print(f"{key}: {val:.4f}")
            else:
                print(f"{key}: {val}")
        print("--------------------------------\n")
    
    print("Success! Sparse Soft DT baseline execution completed.")

if __name__ == "__main__":
    main()