"""
Entmax Decision Tree (SDT) Baseline Implementation
Based on Peters et al., 2019 (Entmax) - Modern Sparse Decision Trees.

Requires: pip install entmax
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import gymnasium as gym
import minigrid
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

# Nhập thư viện entmax chính chủ (PyTorch)
try:
    from entmax import entmax15
except ImportError:
    raise ImportError("Please install entmax package: pip install entmax")

def entmoid15(x: torch.Tensor) -> torch.Tensor:
    stacked = torch.stack([x, torch.zeros_like(x)], dim=-1)
    return entmax15(stacked, dim=-1)[..., 0]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entmax Decision Tree Baseline")
    parser.add_argument("--data_path", type=str, default="stage1_outputs/collected_data.pt")
    parser.add_argument("--max_depth", type=int, default=4, help="Depth of the DT.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for Entmax.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="experiments/entmax_dt/results/")
    parser.add_argument("--ppo_path", type=str, default="ppo_doorkey_6x6.zip")
    parser.add_argument("--env_name", type=str, default="MiniGrid-DoorKey-6x6-v0")
    parser.add_argument("--n_eval_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class SubtractiveEntmaxDense(nn.Module):
    """
    Mô phỏng chính xác SubtractiveEntmaxDense từ JAX sang PyTorch.
    Dùng entmax15 để tạo ra feature selection thưa thớt (sparse weights).
    """
    def __init__(self, input_dim: int, output_dim: int, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        # Khởi tạo giống bản JAX: weight ~ normal(0.1), bias ~ normal(1.0)
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim) * 0.1)
        self.bias = nn.Parameter(torch.randn(output_dim) + 1.0)

    def forward(self, x: torch.Tensor, max_path: bool = False) -> torch.Tensor:
        if max_path:
            # Chế độ Hard Inference: Không dùng Entmax cho weights nữa
            w = self.weight
        else:
            # Chế độ Soft Training: Ép tính thưa lên ma trận trọng số
            # PyTorch entmax15 hoạt động trên dim cuối cùng, nên ta phải transpose
            w = entmax15(self.weight.t() / self.temperature, dim=-1).t()

        # Tính Dense Output
        out = torch.matmul(x, w) - self.bias

        if max_path:
            # Ngắt cứng luồng đi thành 0 hoặc 1 với temperature cực nhỏ
            return entmoid15(out / 0.0001)
        else:
            return entmoid15(out / self.temperature)


class EntmaxDecisionTree(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, depth: int, temperature: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.temperature = temperature
        
        self.num_internal_nodes = (2 ** depth) - 1
        self.num_leaves = 2 ** depth

        # Module điều hướng chứa các node nội tại
        self.routing = SubtractiveEntmaxDense(self.input_dim, self.num_internal_nodes, self.temperature)
        
        # Node lá
        self.leaf_logits = nn.Parameter(torch.randn(self.num_leaves, self.output_dim) * 0.1)

    def forward(self, x: torch.Tensor, max_path: bool = False) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Lấy xác suất điều hướng từ SubtractiveEntmaxDense
        p = self.routing(x, max_path=max_path)
        
        node_probs = [None] * (self.num_internal_nodes + self.num_leaves)
        node_probs[0] = torch.ones(batch_size, device=x.device)
        
        # Tính toán xác suất đường đi
        for i in range(self.num_internal_nodes):
            current_prob = node_probs[i]
            prob_right = p[:, i]
            prob_left = 1.0 - prob_right
            
            left_child = 2 * i + 1
            right_child = 2 * i + 2
            
            node_probs[left_child] = current_prob * prob_left
            node_probs[right_child] = current_prob * prob_right
            
        leaf_probs = torch.stack(node_probs[self.num_internal_nodes:], dim=1)
        
        # Nhân xác suất chạm lá với phân phối tại lá đó
        leaf_dists = F.softmax(self.leaf_logits, dim=-1)
        output_probs = torch.matmul(leaf_probs, leaf_dists)
        
        return output_probs

    def get_metrics(self) -> Dict[str, Any]:
        """Metrics cho Entmax Tree - Không cần dùng Epsilon nữa vì Entmax tạo số 0 tuyệt đối"""
        # Lấy ma trận trọng số thưa
        w_sparse = entmax15(self.routing.weight.t() / self.temperature, dim=-1).t()
        
        # Entmax đẩy các giá trị không quan trọng về đúng 0. 
        # Ta dùng 1e-4 chỉ để phòng tránh sai số dấu phẩy động của GPU.
        non_zero_weights = (torch.abs(w_sparse) > 1e-4).float()
        
        total_literals_in_model = int(non_zero_weights.sum().item())
        avg_literals_per_node = total_literals_in_model / max(1, self.num_internal_nodes)
        total_literals_dnf = int(avg_literals_per_node * self.depth * self.num_leaves)
        
        # Chỉ số giải thích nâng cao: Có bao nhiêu Features TỔNG CỘNG được dùng?
        # Tổng theo hàng: Nếu hàng > 0 tức là feature đó có mặt ít nhất ở 1 node.
        active_features = int((non_zero_weights.sum(dim=1) > 0).sum().item())
        
        return {
            "Rule Set Cardinality": self.num_leaves,
            "Binarized Concepts (Nodes)": self.num_internal_nodes,
            "Unique Active Features": active_features,
            "Total Literals (Global)": total_literals_in_model,
            "Total Literals (DNF Equivalent)": total_literals_dnf,
            "Tree Depth": self.depth
        }


class EntmaxDTAgent:
    def __init__(self, dt_model: EntmaxDecisionTree, ppo_model: PPO, device: torch.device):
        self.dt_model = dt_model
        self.dt_model.eval()
        self.ppo_model = ppo_model
        self.device = device
        
    def predict(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).float().to(self.device)
            features = self.ppo_model.policy.features_extractor(obs_tensor)
            
            # CHÚ Ý QUAN TRỌNG: Inference trong game phải BẬT max_path=True 
            # để cây trở thành Hard Decision Tree (Bậc thang 0-1)
            action_probs = self.dt_model(features, max_path=True)
            action = torch.argmax(action_probs, dim=1)
        return action.cpu().numpy()


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


def evaluate_on_env(dt_model: EntmaxDecisionTree, args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    print(f"\nEvaluating Entmax DT on {args.n_eval_episodes} episodes of {args.env_name}...")
    try:
        ppo_model = PPO.load(args.ppo_path, device=device)
    except Exception as e:
        custom_objects = {
            "policy_kwargs": {
                "features_extractor_class": lambda obs_space, feat_dim: MinigridFeaturesExtractor(obs_space, feat_dim, args.env_name),
                "features_extractor_kwargs": {"features_dim": 128},
            }
        }
        ppo_model = PPO.load(args.ppo_path, device=device, custom_objects=custom_objects)
        
    def _init():
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
        from utils_env import make_env_by_name
        return make_env_by_name(args.env_name, seed=args.seed)
        
    env = VecTransposeImage(DummyVecEnv([_init]))
    env.seed(args.seed)
    agent = EntmaxDTAgent(dt_model, ppo_model, device)
    
    successes = 0
    returns = []
    lengths = []
    
    for _ in range(args.n_eval_episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0
        ep_len = 0
        
        while not done and ep_len < 1000:
            action = agent.predict(obs)
            obs, rewards, dones, infos = env.step(action)
            ep_return += float(rewards[0])
            ep_len += 1
            done = bool(dones[0])
            
        info = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else (infos[0] if infos else {})
        is_success = bool(info.get("is_success", False)) or float(ep_return) > 0
        if is_success: successes += 1
            
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

    # 1. Load data
    data = torch.load(args.data_path, map_location="cpu")
    if isinstance(data, dict):
        features, actions = data["features"], data["actions"]
    else:
        features, actions = data

    dataset = TensorDataset(features, actions)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    input_dim = features.shape[1]
    output_dim = len(torch.unique(actions))

    # 2. Model Training (KHÔNG CẦN L1 PENALTY NỮA)
    print(f"Initializing Entmax Decision Tree (Temp={args.temperature})...")
    model = EntmaxDecisionTree(input_dim=input_dim, output_dim=output_dim, depth=args.max_depth, temperature=args.temperature).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            # Trong lúc train, dùng Soft Routing (max_path = False)
            output_probs = model(batch_x, max_path=False)
            
            log_probs = torch.log(output_probs + 1e-8)
            loss = F.nll_loss(log_probs, batch_y)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{args.epochs}] -> Loss: {train_loss:.4f}")

    # 3. Offline Metrics Calculation
    print("\nCalculating metrics...")
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # Evaluate offline metric với Hard Routing
            output_probs = model(batch_x, max_path=True)
            preds = torch.argmax(output_probs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    action_fidelity = correct / total
    metrics = model.get_metrics()
    metrics["Action Fidelity"] = float(action_fidelity)

    # 4. Game Evaluation
    game_metrics = evaluate_on_env(model, args, device)
    metrics.update(game_metrics)

    print("\n--- Metrics ---")
    for key, val in metrics.items():
        if isinstance(val, float): print(f"{key}: {val:.4f}")
        else: print(f"{key}: {val}")
    print("---------------\n")

    # 5. Saving Artifacts
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), save_dir / "entmax_dt_model.pth")
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Saved artifacts to {save_dir}")

if __name__ == "__main__":
    main()