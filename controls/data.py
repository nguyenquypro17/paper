"""
controls/data.py
Offline dataset loading + a single canonical train/val/test split shared by every
control arm, so all arms see exactly the same data and the same normalization.
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Action-space resolution (mirrors train_joint.main so labels stay consistent)
# ---------------------------------------------------------------------------

_MINIGRID_ACTIONS = ["TurnLeft", "TurnRight", "Forward", "Pickup", "Drop", "Toggle", "Done"]
_PONG_ACTIONS = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]
_CARTPOLE_ACTIONS = ["Left", "Right"]
_BOXING_ACTIONS = [
    "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT",
    "DOWNRIGHT", "DOWNLEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE",
    "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE",
]


def resolve_actions(env_name: str, actions: torch.Tensor) -> Tuple[int, List[str]]:
    """Return (n_actions, action_names) for an environment name."""
    if "Pong" in env_name:
        return 6, list(_PONG_ACTIONS)
    if "MiniGrid" in env_name:
        return 7, list(_MINIGRID_ACTIONS)
    if "CartPole" in env_name:
        return 2, list(_CARTPOLE_ACTIONS)
    if "Boxing" in env_name:
        return 18, list(_BOXING_ACTIONS)
    n = int(actions.max().item()) + 1
    return n, [f"Action_{i}" for i in range(n)]


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------

@dataclass
class OfflineData:
    X_tr: torch.Tensor
    a_tr: torch.Tensor
    X_val: torch.Tensor
    a_val: torch.Tensor
    X_te: torch.Tensor
    a_te: torch.Tensor
    feat_mean: torch.Tensor
    feat_std: torch.Tensor
    n_actions: int
    action_names: List[str]

    @property
    def input_dim(self) -> int:
        return int(self.X_tr.shape[1])

    def numpy(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        X = {"train": self.X_tr, "val": self.X_val, "test": self.X_te}[split]
        a = {"train": self.a_tr, "val": self.a_val, "test": self.a_te}[split]
        return X.cpu().numpy(), a.cpu().numpy()

    def fallback_action(self) -> int:
        """Most frequent teacher action on train — same convention as LUCID."""
        return int(np.bincount(self.a_tr.cpu().numpy()).argmax())


def load_offline(
    features_path: str,
    env_name: str,
    seed: int = 42,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    stage1_path: str = None,
) -> OfflineData:
    """
    Load the same `.pt` file that train_joint.py consumes and split it once.

    Normalization matches train_joint: per-dimension standardization using the
    Stage-1 statistics when available, otherwise statistics of the full file.
    The split is driven only by `seed`, so every arm run with the same seed
    trains, tunes and is tested on identical rows.
    """
    data = torch.load(features_path, map_location="cpu", weights_only=False)
    features = data["features"].float()
    actions = data["actions"].long().view(-1)

    n_actions, action_names = resolve_actions(env_name, actions)
    actions = torch.clamp(actions, min=0, max=n_actions - 1)

    if stage1_path:
        s1 = torch.load(stage1_path, map_location="cpu", weights_only=False)
        feat_mean, feat_std = s1["feature_mean"].float(), s1["feature_std"].float()
    else:
        feat_mean = features.mean(0)
        feat_std = features.std(0).clamp(min=1e-6)
    features = (features - feat_mean) / feat_std

    n = len(features)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_te = int(test_frac * n)
    n_val = int(val_frac * n)
    te_idx, val_idx, tr_idx = idx[:n_te], idx[n_te:n_te + n_val], idx[n_te + n_val:]

    return OfflineData(
        X_tr=features[tr_idx], a_tr=actions[tr_idx],
        X_val=features[val_idx], a_val=actions[val_idx],
        X_te=features[te_idx], a_te=actions[te_idx],
        feat_mean=feat_mean, feat_std=feat_std,
        n_actions=n_actions, action_names=action_names,
    )


def load_offline_from_lucid(run_dir: str, env_name: str, seed: int = 42,
                            val_frac: float = 0.1) -> OfflineData:
    """
    Reuse the exact split of a trained LUCID run.

    test  = shuffle_indices[n_train:]            (LUCID's held-out set)
    train = shuffle_indices[:n_train] minus the last `val_frac` of it
    val   = that last `val_frac`, used only for control model selection
    """
    import os
    d = torch.load(os.path.join(run_dir, "training_data.pt"), map_location="cpu", weights_only=False)
    features = d["features"].float()
    actions = d["actions"].long().view(-1)
    feat_mean = d["feature_mean"].float().view(-1)
    feat_std = d["feature_std"].float().view(-1).clamp(min=1e-6)
    if not bool(d.get("pre_normalized", False)):
        features = (features - feat_mean) / feat_std

    n_actions, action_names = resolve_actions(env_name, actions)
    actions = torch.clamp(actions, min=0, max=n_actions - 1)

    idx = torch.as_tensor(d["shuffle_indices"]).long()
    n_train = int(d["n_train"])
    train_all, te_idx = idx[:n_train], idx[n_train:]

    g = torch.Generator().manual_seed(seed)
    train_all = train_all[torch.randperm(len(train_all), generator=g)]
    n_val = int(val_frac * len(train_all))
    val_idx, tr_idx = train_all[:n_val], train_all[n_val:]

    return OfflineData(
        X_tr=features[tr_idx], a_tr=actions[tr_idx],
        X_val=features[val_idx], a_val=actions[val_idx],
        X_te=features[te_idx], a_te=actions[te_idx],
        feat_mean=feat_mean, feat_std=feat_std,
        n_actions=n_actions, action_names=action_names,
    )
