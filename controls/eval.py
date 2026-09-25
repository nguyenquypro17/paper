"""
controls/eval.py
Evaluation shared by every arm: offline action fidelity to the teacher on the
held-out split, and online rollout of the surrogate driving the real env
through the frozen PPO CNN.

Every agent here has the same contract:
    predict(obs) -> (np.ndarray([action_idx]), {"triggered": bool})
so `rollout` never needs to know which arm it is running.
"""
import math
import os
import re
import sys
from typing import Callable, Dict, List

import numpy as np
import torch
from tqdm import tqdm

_ROOT = os.environ.get("LUCID_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_EVAL_DIR = os.path.join(_ROOT, 'experiments', 'lucid')
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

def make_eval_env(*a, **kw):  # lazily imported: only needed for rollouts
    from evaluate_lucid_metrics import make_eval_env as _f
    return _f(*a, **kw)


# ===========================================================================
# Teacher CNN
# ===========================================================================

def load_ppo_cnn(ppo_path: str, device: str):
    """Return the frozen feature extractor of the PPO teacher, in eval mode."""
    from stable_baselines3 import PPO
    ppo = PPO.load(ppo_path, device=device)
    cnn = ppo.policy.features_extractor
    cnn.eval()
    for p in cnn.parameters():
        p.requires_grad_(False)
    return cnn


# ===========================================================================
# Offline fidelity
# ===========================================================================

def offline_fidelity(predict_fn: Callable[[np.ndarray], np.ndarray],
                     Z: np.ndarray, a_teacher: np.ndarray) -> float:
    """Percentage of held-out states on which the surrogate matches the teacher."""
    pred = predict_fn(Z)
    return float((pred == a_teacher).mean() * 100.0)


def dnf_soft_predict(model, Z: torch.Tensor, device: str, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Z), batch_size):
            out.append(model(Z[i:i + batch_size].to(device)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def parse_rules(rules: Dict[str, List[str]]):
    """Parse extract_rules() output into (action -> [(pos, neg, sigmoid_bias)])."""
    parsed = {}
    for act, clauses in rules.items():
        parsed[act] = []
        for clause in clauses:
            if clause.startswith("(False)") or clause == "(no active clauses)":
                continue
            pos = [int(x) for x in re.findall(r"(?<![¬])f_(\d+)", clause)]
            neg = [int(x) for x in re.findall(r"¬f_(\d+)", clause)]
            m = re.search(r"\[bias=([-\d.]+)\]", clause)
            bias = float(m.group(1)) if m else 0.0
            parsed[act].append((pos, neg, 1.0 / (1.0 + math.exp(-bias))))
    return parsed


def dnf_hard_predict(parsed, z_bin: np.ndarray, action_names: List[str],
                     fallback_idx: int, hard_threshold: float = 0.66):
    """Vectorized hard-rule policy. Returns (predictions, coverage_mask)."""
    z_bool = z_bin > hard_threshold
    n, A = len(z_bool), len(action_names)
    scores = np.zeros((n, A))
    covered = np.zeros(n, dtype=bool)

    for ai, act in enumerate(action_names):
        for pos, neg, w in parsed.get(act, []):
            fires = np.ones(n, dtype=bool)
            for f in pos:
                fires &= z_bool[:, f]
            for f in neg:
                fires &= ~z_bool[:, f]
            scores[:, ai] += fires * w
            covered |= fires

    pred = np.where(covered, scores.argmax(axis=1), fallback_idx)
    return pred, covered


# ===========================================================================
# Rollout agents
# ===========================================================================

class _BaseAgent:
    """Holds the teacher CNN + representation; subclasses supply the head."""

    def __init__(self, ppo_cnn, repr_fn, device: str):
        self.ppo_cnn = ppo_cnn
        self.repr_fn = repr_fn          # standardized features -> representation
        self.device = device

    @torch.no_grad()
    def _features(self, obs) -> torch.Tensor:
        obs_t = torch.as_tensor(obs).float().to(self.device)
        return self.repr_fn(self.ppo_cnn(obs_t))


class DNFSoftAgent(_BaseAgent):
    def __init__(self, ppo_cnn, repr_fn, dnf, device):
        super().__init__(ppo_cnn, repr_fn, device)
        self.dnf = dnf

    @torch.no_grad()
    def predict(self, obs):
        logits = self.dnf(self._features(obs))
        return logits.argmax(1).cpu().numpy(), {"triggered": True}


class DNFHardAgent(_BaseAgent):
    def __init__(self, ppo_cnn, repr_fn, dnf, rules, action_names,
                 fallback_idx, hard_threshold=0.66, device="cpu"):
        super().__init__(ppo_cnn, repr_fn, device)
        self.dnf = dnf
        self.parsed = parse_rules(rules)
        self.action_names = action_names
        self.fallback_idx = fallback_idx
        self.hard_threshold = hard_threshold

    @torch.no_grad()
    def predict(self, obs):
        z = self._features(obs)
        z_bin = self.dnf.binarize(z).cpu().numpy()
        pred, covered = dnf_hard_predict(
            self.parsed, z_bin, self.action_names, self.fallback_idx, self.hard_threshold
        )
        return pred, {"triggered": bool(covered[0])}


class SklearnAgent(_BaseAgent):
    """CART / sparse-linear head. Always fires, so trigger rate is 100%."""

    def __init__(self, ppo_cnn, repr_fn, clf, device):
        super().__init__(ppo_cnn, repr_fn, device)
        self.clf = clf

    @torch.no_grad()
    def predict(self, obs):
        z = self._features(obs).cpu().numpy()
        return self.clf.predict(z).astype(int), {"triggered": True}


# ===========================================================================
# Live rollout
# ===========================================================================

def rollout(agent, env_name: str, n_episodes: int, seed: int = 42,
            max_steps: int = 27000, desc: str = "") -> Dict[str, float]:
    env = make_eval_env(env_name, seed=seed)
    successes, rewards = 0, []
    triggered_steps = total_steps = 0

    for _ in tqdm(range(n_episodes), desc=desc or f"{env_name} seed {seed}", leave=False):
        obs = env.reset()
        ep_reward, ep_steps, done = 0.0, 0, False
        while not done and ep_steps < max_steps:
            action, info = agent.predict(obs)
            triggered_steps += int(info["triggered"])
            total_steps += 1
            ep_steps += 1
            obs, reward, dones, infos = env.step(action)
            ep_reward += float(reward[0])
            done = bool(dones[0])
            if done and "episode" in infos[0]:
                ep_reward = float(infos[0]["episode"]["r"])
        successes += int(ep_reward > 0)
        rewards.append(ep_reward)
    env.close()

    return {
        "success_rate": successes / n_episodes * 100.0,
        "score": float(np.mean(rewards)),
        "score_std": float(np.std(rewards)),
        "trigger_rate": triggered_steps / max(total_steps, 1) * 100.0,
    }


def rollout_multi_seed(agent, env_name: str, episodes_per_seed: int,
                       seeds=(42, 43, 44, 45, 46), max_steps: int = 27000) -> Dict[str, Dict[str, float]]:
    """Aggregate over evaluation seeds; the unit of the std is the eval seed."""
    per_seed = {s: rollout(agent, env_name, episodes_per_seed, seed=s, max_steps=max_steps) for s in seeds}
    agg = {}
    for metric in ("success_rate", "score", "trigger_rate"):
        vals = [per_seed[s][metric] for s in seeds]
        agg[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    agg["per_seed"] = {str(s): per_seed[s] for s in seeds}
    return agg
