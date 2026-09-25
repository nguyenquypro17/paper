"""
VIPER Baseline Implementation
(Verifiable Reinforcement Learning via Policy Extraction - Bastani et al. 2018)

ACTUAL VIPER IMPLEMENTATION:
1. Interactive DAgger (Dataset Aggregation) Loop.
2. Cost-Sensitive Learning via Original Resampling (Weighted by Teacher's Action Probability Gap).
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Tuple

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACTUAL VIPER Baseline (DAgger + Original Resampling)")

    parser.add_argument("--ppo_path", type=str, default="ppo_doorkey_6x6.zip",
                        help="Path to the trained PPO Teacher model.")
    parser.add_argument("--env_name", type=str, default="MiniGrid-DoorKey-6x6-v0",
                        help="Gym environment name.")
    parser.add_argument("--save_dir", type=str, default="experiments/dt/results/",
                        help="Directory to save the best model and metrics.")
    
    # VIPER specific hyperparameters
    parser.add_argument("--n_dagger_iters", type=int, default=15,
                        help="Number of DAgger iterations.")
    parser.add_argument("--episodes_per_iter", type=int, default=20,
                        help="Number of episodes to rollout per DAgger iteration.")
    parser.add_argument("--max_depth", type=int, default=None,
                        help="Maximum depth of the Decision Tree.")
    parser.add_argument("--min_samples_leaf", type=int, default=5,
                        help="Minimum samples per leaf to prevent overfitting.")
    parser.add_argument("--n_eval_episodes", type=int, default=1000,
                        help="Number of episodes for the FINAL rigorous evaluation.")
    parser.add_argument("--max_steps", type=int, default=27000,
                        help="Max steps per episode (27000 recommended for Atari).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only-test", action="store_true",
                        help="Skip DAgger training and only load/test existing model.")
    parser.add_argument("--multi-seed", action="store_true",
                        help="Run evaluation on fixed 5 seeds (42,43,44,45,46) with 1/5 episodes each.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the trained VIPER model (.pkl file). If provided with --only-test, loads from this path instead of save_dir.")
    parser.add_argument("--data_path", type=str, default="stage1_outputs/collected_data.pt",
                        help="Path to offline collected data for action fidelity calculation.")

    return parser.parse_args()


# ============================================================================
# AGENT WRAPPERS
# ============================================================================

class DTAgent:
    """Student Agent: Uses a Decision Tree on top of PPO's frozen CNN."""
    def __init__(self, dt_model: DecisionTreeClassifier, ppo_model: PPO, device: str):
        self.dt_model = dt_model
        self.ppo_model = ppo_model
        self.device = device
        
    def predict(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).float().to(self.device)
            features = self.ppo_model.policy.features_extractor(obs_tensor)
        
        action = self.dt_model.predict(features.cpu().numpy())
        return action

class PPOTeacherAgent:
    """Teacher Agent: Original PPO policy."""
    def __init__(self, ppo_model: PPO):
        self.ppo_model = ppo_model
        
    def predict(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.ppo_model.predict(obs, deterministic=True)
        return action


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


# ============================================================================
# CORE VIPER LOGIC (DAgger + Labeling + Weighting)
# ============================================================================

def query_teacher(ppo_model: PPO, obs: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs).float().to(device)
        features = ppo_model.policy.features_extractor(obs_tensor)
        latent_pi, _ = ppo_model.policy.mlp_extractor(features)
        logits = ppo_model.policy.action_net(latent_pi)
        probs = torch.softmax(logits, dim=-1)
        best_actions = logits.argmax(dim=-1).cpu().numpy()
        weights = (probs.max(dim=-1)[0] - probs.min(dim=-1)[0]).cpu().numpy()
    return features.cpu().numpy(), best_actions, weights


def collect_rollouts(env, agent, teacher_ppo: PPO, n_episodes: int, device: str, max_steps: int):
    all_features = []
    all_actions = []
    all_weights = []
    
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_len = 0
        
        while not done and ep_len < max_steps:
            feats, teacher_acts, weights = query_teacher(teacher_ppo, obs, device)
            
            all_features.append(feats[0])
            all_actions.append(teacher_acts[0])
            all_weights.append(weights[0])
            
            action = agent.predict(obs)
            obs, _, dones, _ = env.step(action)
            
            done = bool(dones[0])
            ep_len += 1
            
    return all_features, all_actions, all_weights


def evaluate_agent(env, agent, n_episodes: int, env_name: str, max_steps: int) -> Tuple[float, float, float]:
    successes = 0
    returns = []
    lengths = []
    
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0
        ep_len = 0
        
        while not done and ep_len < max_steps:
            action = agent.predict(obs)
            obs, rewards, dones, infos = env.step(action)
            ep_return += float(rewards[0])
            done = bool(dones[0])
            ep_len += 1
            
            if done and "episode" in infos[0]:
                ep_return = float(infos[0]["episode"]["r"])
            
        is_atari = "NoFrameskip" in env_name or "Boxing" in env_name or "Pong" in env_name
        if "CartPole" in env_name:
            is_success = (ep_len >= max_steps)
        elif is_atari:
            is_success = ep_return > 0
        else:
            info = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else (infos[0] if infos else {})
            is_success = bool(info.get("is_success", False)) or ep_return > 0
            
        if is_success:
            successes += 1
            
        returns.append(ep_return)
        lengths.append(ep_len)
            
    return successes / n_episodes, float(np.mean(returns)), float(np.mean(lengths))


# ============================================================================
# METRICS & UTILS
# ============================================================================

def calculate_tree_metrics(clf: DecisionTreeClassifier) -> Dict[str, Any]:
    tree = clf.tree_
    rule_set_cardinality = clf.get_n_leaves()
    binarized_concepts = tree.node_count - rule_set_cardinality

    total_literals = 0
    stack = [(0, 0)] 
    while stack:
        node_id, depth = stack.pop()
        if tree.children_left[node_id] != -1:
            stack.append((tree.children_left[node_id], depth + 1))
            stack.append((tree.children_right[node_id], depth + 1))
        else:
            total_literals += depth

    return {
        "Rule Set Cardinality": int(rule_set_cardinality),
        "Binarized Concepts": int(binarized_concepts),
        "Total Literals": int(total_literals),
        "Tree Depth": int(clf.get_depth()),
    }


def calculate_offline_action_fidelity(clf: DecisionTreeClassifier, data_path: str, ppo_model: PPO, device: str) -> Tuple[float, Dict[str, Any]]:
    print(f"Loading offline collected data from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    
    if isinstance(data, dict):
        X = data["features"].numpy() if isinstance(data["features"], torch.Tensor) else data["features"]
        y = data["actions"].numpy() if isinstance(data["actions"], torch.Tensor) else data["actions"]
    else:
        X = data[0].numpy() if isinstance(data[0], torch.Tensor) else data[0]
        y = data[1].numpy() if isinstance(data[1], torch.Tensor) else data[1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    y_pred = clf.predict(X_test)
    action_fidelity = accuracy_score(y_test, y_pred)
    
    fidelity_metrics = {
        "Action Fidelity (Offline Test)": float(action_fidelity),
        "Test Set Size": int(len(X_test)),
        "Train Set Size": int(len(X_train))
    }
    
    return action_fidelity, fidelity_metrics


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    env = make_eval_env(args.env_name, args.seed)
    eval_env = make_eval_env(args.env_name, args.seed)
    
    print(f"Loading PPO Teacher from {args.ppo_path}...")
    try:
        ppo_teacher = PPO.load(args.ppo_path, device=device)
    except Exception as e:
        print(f"Fallback loading custom PPO... Error: {e}")
        class MinigridFeaturesExtractor(BaseFeaturesExtractor):
            def __init__(self, observation_space, features_dim=128):
                super().__init__(observation_space, features_dim)
                c = observation_space.shape[0]
                self.cnn = torch.nn.Sequential(
                    torch.nn.Conv2d(c, 32, 3, 1, 1), torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 64, 3, 1, 1), torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 64, 3, 1, 1), torch.nn.ReLU(),
                    torch.nn.Flatten(),
                )
                with torch.no_grad():
                    n_flat = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
                self.linear = torch.nn.Sequential(torch.nn.Linear(n_flat, features_dim), torch.nn.ReLU())
            def forward(self, obs): return self.linear(self.cnn(obs.float()))
        ppo_teacher = PPO.load(args.ppo_path, device=device, custom_objects={
            "policy_kwargs": {"features_extractor_class": MinigridFeaturesExtractor, "features_extractor_kwargs": {"features_dim": 128}}
        })

    if args.only_test:
        print(f"[Only-Test Mode] Loading existing VIPER model...")
        if args.model_path:
            model_path = Path(args.model_path)
        else:
            model_path = save_dir / "viper_policy.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Train first or provide correct path.")
        best_clf = joblib.load(model_path)
        print(f"Loaded VIPER model from {model_path}")
        
        if Path(args.data_path).exists():
            offline_af, fidelity_metrics = calculate_offline_action_fidelity(best_clf, args.data_path, ppo_teacher, device)
            print(f"\nAction Fidelity on Offline Data (test split): {offline_af:.4f}")
            print(f"  (Note: Model was trained on DAgger data, not offline data. Lower AF is expected.)")
            
            metrics_path = save_dir / "viper_metrics_only_test.json"
            print(f"Saving only-test metrics to {metrics_path}...")
            with open(metrics_path, "w") as f:
                json.dump(fidelity_metrics, f, indent=4)
        else:
            print(f"Warning: Data path {args.data_path} not found. Skipping offline action fidelity.")
    else:
        print(f"Starting VIPER with DAgger. Device: {device}")

        teacher_agent = PPOTeacherAgent(ppo_teacher)
        
        dataset_features = []
        dataset_actions = []
        dataset_weights = []
        
        best_success_rate = -1.0
        best_clf = None
        
        print("\n" + "="*50)
        print("VIPER ITERATION 0: TEACHER DEMONSTRATIONS")
        print("="*50)
        f, a, w = collect_rollouts(env, teacher_agent, ppo_teacher, args.episodes_per_iter * 2, device, args.max_steps)
        dataset_features.extend(f)
        dataset_actions.extend(a)
        dataset_weights.extend(w)
        
        for i in range(1, args.n_dagger_iters + 1):
            print(f"\n" + "="*50)
            print(f"VIPER ITERATION {i}/{args.n_dagger_iters}")
            print("="*50)
            
            X = np.vstack(dataset_features)
            y = np.array(dataset_actions)
            w = np.array(dataset_weights)
            
            w_sum = np.sum(w)
            if w_sum > 0:
                p = w / w_sum
            else:
                p = np.ones_like(w) / len(w)
                
            n_samples = len(y)
            print(f"  -> Resampling {n_samples} transitions based on Teacher's probability gap...")
            
            rng = np.random.default_rng(args.seed + i)
            indices = rng.choice(n_samples, size=n_samples, replace=True, p=p)
            
            X_resampled = X[indices]
            y_resampled = y[indices]
            
            print(f"  -> Training DT on resampled dataset...")
            clf = DecisionTreeClassifier(
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                ccp_alpha=0.001,
                random_state=args.seed + i,
            )
            clf.fit(X_resampled, y_resampled) 
            
            student_agent = DTAgent(clf, ppo_teacher, device)
            eval_eps = 20
            print(f"  -> Evaluating Student on {eval_eps} episodes...")
            success_rate, _, _ = evaluate_agent(eval_env, student_agent, eval_eps, args.env_name, args.max_steps)
            print(f"  -> Student Success Rate: {success_rate * 100:.1f}%")
            
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_clf = clf
                print("  -> [*] New Best Model!")
                
            if i < args.n_dagger_iters:
                print(f"  -> Student exploring the environment to find edge cases...")
                f, a, w = collect_rollouts(env, student_agent, ppo_teacher, args.episodes_per_iter, device, args.max_steps)
                dataset_features.extend(f)
                dataset_actions.extend(a)
                dataset_weights.extend(w)
                print(f"  -> Added {len(a)} new samples to dataset.")
        
        model_path = save_dir / "viper_policy.pkl"
        print(f"Saving best VIPER model to {model_path}...")
        joblib.dump(best_clf, model_path)
        
        metrics = calculate_tree_metrics(best_clf)
        
        X_full = np.vstack(dataset_features)
        y_full = np.array(dataset_actions)
        y_pred = best_clf.predict(X_full)
        dagger_af = accuracy_score(y_full, y_pred)
        metrics["Action Fidelity (DAgger Dataset)"] = float(dagger_af)
        print(f"\nAction Fidelity on DAgger Data (training dataset): {dagger_af:.4f}")
        
        if Path(args.data_path).exists():
            offline_af, fidelity_metrics = calculate_offline_action_fidelity(best_clf, args.data_path, ppo_teacher, device)
            metrics.update(fidelity_metrics)
            print(f"Action Fidelity on Offline Data (test split): {offline_af:.4f}")
            print(f"  (Note: Model was trained on DAgger data, not offline data. Lower AF is expected.)")
        else:
            print(f"Warning: Data path {args.data_path} not found. Skipping offline action fidelity.")
        
        metrics_path = save_dir / "viper_metrics.json"
        print(f"Saving metrics to {metrics_path}...")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

    print(f"Running evaluation on {args.n_eval_episodes} episodes...")
    
    if args.multi_seed:
        print(f"\nMulti-Seed Evaluation: Running on seeds [42,43,44,45,46] with {args.n_eval_episodes//5} episodes each...")
        eval_eps = args.n_eval_episodes // 5
        all_results = {}
        for seed in [42, 43, 44, 45, 46]:
            args.seed = seed
            eval_env_seed = make_eval_env(args.env_name, seed)
            student_agent = DTAgent(best_clf, ppo_teacher, device)
            success_rate, avg_return, avg_length = evaluate_agent(eval_env_seed, student_agent, eval_eps, args.env_name, args.max_steps)
            all_results[seed] = {"Success Rate": success_rate, "Avg Return": avg_return, "Avg Length": avg_length}
        
        print(f"\n{'='*70}")
        print("MULTI-SEED AGGREGATED RESULTS")
        print(f"{'='*70}")
        for key in ["Success Rate", "Avg Return", "Avg Length"]:
            values = [all_results[seed][key] for seed in [42,43,44,45,46]]
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            print(f"{key:25s}: {mean_val:.4f} ± {std_val:.4f}")
        print(f"{'='*70}\n")
        
        if not args.only_test:
            metrics_multiseed_path = save_dir / "viper_metrics_multiseed.json"
            multiseed_summary = {}
            for key in ["Success Rate", "Avg Return", "Avg Length"]:
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
        final_student = DTAgent(best_clf, ppo_teacher, device)
        final_success_rate, final_avg_return, final_avg_length = evaluate_agent(eval_env, final_student, args.n_eval_episodes, args.env_name, args.max_steps)
        
        print("\n--- Final Evaluation Metrics ---")
        print(f"Success Rate: {final_success_rate:.4f}")
        print(f"Avg Return: {final_avg_return:.4f}")
        print(f"Avg Length: {final_avg_length:.4f}")
        print("--------------------------------\n")
    
    print("Success! VIPER baseline execution completed.")

if __name__ == "__main__":
    main()