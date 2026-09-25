"""
controls/representations.py
The representation axis of the control grid.

Two representations are exposed:
  * `raw`  — the teacher's standardized CNN features, unchanged.
  * `sae`  — an overcomplete TopK SAE trained with reconstruction + L1 ONLY,
             i.e. without any imitation signal. This is the honest "SAE as a
             representation" control: it isolates what the dictionary buys us
             from what joint training with the logic layer buys us.
"""
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_ROOT = os.environ.get("LUCID_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sparse_concept_autoencoder import OvercompleteSAE  # noqa: E402


@dataclass
class SAEReconConfig:
    hidden_dim: int = 300
    k: int = 50
    lambda_sparsity: float = 5e-3
    lr: float = 1e-3
    n_epochs: int = 200
    batch_size: int = 256
    seed: int = 42


def fit_sae_recon_only(X_tr: torch.Tensor, cfg: SAEReconConfig, device: str,
                       X_val: torch.Tensor = None, log_every: int = 25) -> OvercompleteSAE:
    """
    Train an overcomplete TopK SAE on standardized features with
    `L = MSE(x, x_hat) + lambda * mean|h|` and nothing else.

    Returns the trained SAE in eval mode. Decoder columns are re-normalized
    after every step, identically to the joint trainer.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    sae = OvercompleteSAE(input_dim=X_tr.shape[1], hidden_dim=cfg.hidden_dim, k=cfg.k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_epochs, eta_min=1e-5)
    loader = DataLoader(TensorDataset(X_tr), batch_size=cfg.batch_size, shuffle=True)

    for epoch in range(cfg.n_epochs):
        sae.train()
        running = []
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            z_sparse, z_pre = sae.encode(batch_x)
            x_recon = sae.decode(z_sparse)

            recon = F.mse_loss(x_recon, batch_x)
            sparsity = cfg.lambda_sparsity * z_pre.abs().mean()
            loss = recon + sparsity

            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                sae._normalize_decoder()
            running.append((loss.item(), recon.item()))
        sched.step()

        if log_every and (epoch + 1) % log_every == 0:
            tot, rec = np.mean([r[0] for r in running]), np.mean([r[1] for r in running])
            msg = f"  [SAE recon-only] epoch {epoch+1}/{cfg.n_epochs} | loss {tot:.4f} | recon {rec:.4f}"
            if X_val is not None:
                msg += f" | val_recon {reconstruction_error(sae, X_val, device):.4f}"
            print(msg)

    sae.eval()
    return sae


@torch.no_grad()
def reconstruction_error(sae: OvercompleteSAE, X: torch.Tensor, device: str, batch_size: int = 512) -> float:
    sae.eval()
    tot, n = 0.0, 0
    for i in range(0, len(X), batch_size):
        b = X[i:i + batch_size].to(device)
        z, _ = sae.encode(b)
        tot += F.mse_loss(sae.decode(z), b, reduction="sum").item()
        n += b.numel()
    return tot / max(n, 1)


@torch.no_grad()
def encode(sae: OvercompleteSAE, X: torch.Tensor, device: str, batch_size: int = 512) -> torch.Tensor:
    """Map standardized features to sparse concept activations z (on CPU)."""
    sae.eval()
    out = []
    for i in range(0, len(X), batch_size):
        z, _ = sae.encode(X[i:i + batch_size].to(device))
        out.append(z.cpu())
    return torch.cat(out, dim=0)


def live_dims(Z: torch.Tensor, threshold: float = 1e-3) -> np.ndarray:
    """Indices of concepts that fire at all — reported alongside complexity."""
    return np.flatnonzero((Z.abs() > threshold).any(dim=0).cpu().numpy())


class Representation:
    """Uniform interface so the rollout agent does not care which arm it serves."""

    def __init__(self, kind: str, sae: OvercompleteSAE = None, device: str = "cpu"):
        assert kind in ("raw", "sae")
        self.kind = kind
        self.sae = sae
        self.device = device

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.kind == "raw":
            return X
        return encode(self.sae, X, self.device)

    @torch.no_grad()
    def transform_online(self, x: torch.Tensor) -> torch.Tensor:
        """Single standardized feature batch, kept on device (rollout path)."""
        if self.kind == "raw":
            return x
        z, _ = self.sae.encode(x)
        return z

    @property
    def dim(self) -> int:
        return self.sae.hidden_dim if self.kind == "sae" else None
