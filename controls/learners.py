"""
controls/learners.py
The rule-learner axis of the control grid.

  * BottleneckDNF   — SigmoidBottleneck + ProductTNormLogicLayer, i.e. LUCID's
                      head detached from the SAE. Fed with raw CNN features it
                      gives arm A3 (DNF without SAE); fed with a frozen
                      recon-only SAE's concepts it gives arm A4 (two-stage).
  * fit_cart        — sklearn CART on whatever representation it is handed.
  * fit_sparse_linear — L1 multinomial logistic regression (sparse linear rules).
  * fit_matched_tree  — CART constrained to a target complexity budget, for the
                      complexity-matched comparison the reviewers asked for.
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_ROOT = os.environ.get("LUCID_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from train_joint import SigmoidBottleneck, ProductTNormLogicLayer  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402


# ===========================================================================
# DNF head without the SAE in the loop
# ===========================================================================

@dataclass
class DNFConfig:
    n_features: int = 300
    n_actions: int = 7
    n_clauses_per_action: int = 10
    initial_alpha: float = 1.0
    l0_penalty_weight: float = 1e-4
    entropy_weight: float = 0.005
    bimodal_max: float = 0.3
    bimodal_warmup: int = 30
    bimodal_ramp: int = 80
    beta_action: float = 5.0
    logic_lr: float = 3e-3
    bottleneck_lr: float = 1e-3
    n_epochs: int = 400
    batch_size: int = 256
    max_grad_norm: float = 5.0
    seed: int = 42
    action_class_weights: Tuple[float, ...] = field(default_factory=tuple)


class BottleneckDNF(nn.Module):
    """Binarization bottleneck + product t-norm DNF layer, no SAE attached."""

    def __init__(self, cfg: DNFConfig, device: str = "cpu"):
        super().__init__()
        self.config = cfg
        self.device = device
        self.bottleneck = SigmoidBottleneck(cfg.n_features, cfg.initial_alpha).to(device)
        self.logic_layer = ProductTNormLogicLayer(
            n_features=cfg.n_features,
            n_actions=cfg.n_actions,
            n_clauses_per_action=cfg.n_clauses_per_action,
            l0_penalty_weight=cfg.l0_penalty_weight,
        ).to(device)
        self.register_buffer("z_mean", torch.zeros(cfg.n_features))
        self.register_buffer("z_std", torch.ones(cfg.n_features))

    def normalize_z(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.z_mean) / self.z_std

    def binarize(self, z: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(self.normalize_z(z))

    def forward(self, z: torch.Tensor, return_features: bool = False):
        z_bin = self.binarize(z)
        logits = self.logic_layer(z_bin)
        if return_features:
            return logits, {"z_binary": z_bin}
        return logits

    def extract_rules(self, action_names: List[str] = None, threshold: float = 0.5) -> Dict[str, List[str]]:
        return self.logic_layer.extract_rules(action_names=action_names, threshold=threshold)


def train_bottleneck_dnf(Z_tr: torch.Tensor, a_tr: torch.Tensor,
                         Z_val: torch.Tensor, a_val: torch.Tensor,
                         cfg: DNFConfig, device: str, log_every: int = 25):
    """
    Same objective as LUCID's joint loss minus the SAE terms:
        CE + ramped bimodality + L0 + ramped selector entropy.

    Model selection is by validation accuracy, ties broken by training loss —
    identical to train_joint.train_logic, so the comparison is like-for-like.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = BottleneckDNF(cfg, device=device).to(device)
    opt = torch.optim.Adam([
        {"params": model.bottleneck.parameters(), "lr": cfg.bottleneck_lr},
        {"params": model.logic_layer.parameters(), "lr": cfg.logic_lr},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_epochs, eta_min=1e-5)

    weights = cfg.action_class_weights or tuple(1.0 for _ in range(cfg.n_actions))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    loader = DataLoader(TensorDataset(Z_tr, a_tr), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Z_val, a_val), batch_size=cfg.batch_size, shuffle=False)

    best_acc, best_loss, best_state, history = 0.0, float("inf"), None, []

    for epoch in range(cfg.n_epochs):
        model.train()
        info = []
        for z_batch, a_batch in loader:
            z_batch = z_batch.to(device)
            a_batch = a_batch.long().view(-1).to(device)

            logits, feats = model(z_batch, return_features=True)
            action_loss = F.cross_entropy(logits, a_batch, weight=class_weights)

            z_bin = feats["z_binary"]
            if epoch < cfg.bimodal_warmup:
                w_bim = w_ent = 0.0
            else:
                prog = min(1.0, (epoch - cfg.bimodal_warmup) / max(cfg.bimodal_ramp, 1))
                w_bim, w_ent = cfg.bimodal_max * prog, cfg.entropy_weight * prog

            bimodal = w_bim * (z_bin * (1.0 - z_bin)).mean()
            l0 = model.logic_layer.complexity_penalty()
            ent = model.logic_layer.entropy_penalty()
            loss = cfg.beta_action * action_loss + bimodal + l0 + w_ent * ent

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()

            info.append({
                "total_loss": loss.item(),
                "action_loss": action_loss.item(),
                "acc": (logits.argmax(1) == a_batch).float().mean().item(),
                "near_binary_frac": ((z_bin < 0.05) | (z_bin > 0.95)).float().mean().item(),
            })
        sched.step()

        val_acc = dnf_accuracy(model, val_loader, device)
        avg = {k: float(np.mean([d[k] for d in info])) for k in info[0]}
        avg.update({"val_acc": val_acc, "epoch": epoch})
        history.append(avg)

        better = val_acc > best_acc or (val_acc == best_acc and avg["total_loss"] < best_loss)
        if better:
            best_acc, best_loss = val_acc, avg["total_loss"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if log_every and (epoch + 1) % log_every == 0:
            print(f"  [DNF] epoch {epoch+1}/{cfg.n_epochs} | loss {avg['total_loss']:.4f} "
                  f"| train {avg['acc']:.3f} | val {val_acc:.3f} | nearbin {avg['near_binary_frac']:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_acc, history


@torch.no_grad()
def dnf_accuracy(model: BottleneckDNF, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    for z_batch, a_batch in loader:
        z_batch = z_batch.to(device)
        a_batch = a_batch.long().view(-1).to(device)
        correct += (model(z_batch).argmax(1) == a_batch).sum().item()
        total += a_batch.size(0)
    return correct / max(total, 1)


# ===========================================================================
# Classical learners
# ===========================================================================

def fit_cart(Z: np.ndarray, a: np.ndarray, max_depth: int = None,
             ccp_alpha: float = 0.0, max_leaf_nodes: int = None,
             seed: int = 42) -> DecisionTreeClassifier:
    """CART on any representation. Knobs used for the Pareto sweep."""
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        ccp_alpha=ccp_alpha,
        max_leaf_nodes=max_leaf_nodes,
        random_state=seed,
    )
    clf.fit(Z, a)
    return clf


def fit_sparse_linear(Z: np.ndarray, a: np.ndarray, C: float = 1.0,
                      seed: int = 42, max_iter: int = 2000) -> LogisticRegression:
    """
    L1 multinomial logistic regression: the 'sparse linear surrogate' control.

    sklearn >= 1.8 deprecates `penalty` in favour of `l1_ratio`, so pick the
    argument the installed version actually wants.
    """
    import inspect
    params = inspect.signature(LogisticRegression.__init__).parameters
    kwargs = dict(C=C, solver="saga", max_iter=max_iter, random_state=seed, n_jobs=-1)
    if "penalty" in params and sklearn_version() < (1, 8):
        kwargs["penalty"] = "l1"
    else:
        kwargs["l1_ratio"] = 1.0
    clf = LogisticRegression(**kwargs)
    clf.fit(Z, a)
    return clf


def sklearn_version() -> Tuple[int, ...]:
    import sklearn
    return tuple(int(x) for x in sklearn.__version__.split(".")[:2])


def fit_matched_tree(Z: np.ndarray, a: np.ndarray, target_tests: int,
                     seed: int = 42, max_leaf_cap: int = 512) -> Tuple[DecisionTreeClassifier, int]:
    """
    Grow a CART whose number of internal nodes (= feature tests, our common
    complexity unit) is as close as possible to `target_tests` without
    exceeding it. Returns (tree, achieved_tests).

    A tree with L leaves has exactly L-1 internal nodes, so we sweep
    max_leaf_nodes directly instead of guessing depths.
    """
    best, best_tests = None, -1
    for leaves in range(2, min(target_tests + 2, max_leaf_cap) + 1):
        clf = DecisionTreeClassifier(max_leaf_nodes=leaves, random_state=seed)
        clf.fit(Z, a)
        tests = int((clf.tree_.children_left != -1).sum())
        if tests <= target_tests and tests > best_tests:
            best, best_tests = clf, tests
    if best is None:  # target smaller than the smallest possible tree
        best = DecisionTreeClassifier(max_leaf_nodes=2, random_state=seed).fit(Z, a)
        best_tests = int((best.tree_.children_left != -1).sum())
    return best, best_tests
