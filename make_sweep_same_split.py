"""
make_sweep_same_split.py
Sinh lệnh chạy control trên ĐÚNG split của từng run LUCID (lucid_model_<env>_<seed>).
hidden_dim / k / n_clauses / epochs / bimodal_ramp / entropy lấy thẳng từ config
checkpoint LUCID của env đó, để control và LUCID dùng cùng cấu hình.

  python make_sweep_same_split.py > sweep_same.sh
  xargs -P 12 -I{} bash -c '{}' < sweep_same.sh 2>&1 | grep -E "saved|Error|Traceback"
"""
import os
import sys

import torch

ENVS = {
    "doorkey": "MiniGrid-DoorKey-6x6-v0",
    "dyobs": "MiniGrid-Dynamic-Obstacles-5x5-v0",
    "cartpole": "PixelCartPole-v1",
    "pong": "PongNoFrameskip-v4",
    "boxing": "BoxingNoFrameskip-v4",
}
SEEDS = (42, 43, 44)
CCP = (0.0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03)
CS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
OUT = "results/controls_same"


def lucid_cfg(run_dir):
    ck = torch.load(os.path.join(run_dir, "sae_logic_joint_model.pt"),
                    map_location="cpu", weights_only=False)
    c = ck.get("config", {})
    return {
        "hidden_dim": c.get("hidden_dim", 300),
        "k": c.get("k", 50),
        "n_clauses_per_action": c.get("n_clauses_per_action", 20),
        "dnf_epochs": c.get("n_epochs", 600),
        "bimodal_ramp": c.get("bimodal_ramp", 120),
        "entropy_weight": c.get("entropy_weight", 0.005),
    }


def main():
    n = 0
    for tag, env in ENVS.items():
        for s in SEEDS:
            d = f"outputs/lucid_model_{tag}_{s}"
            if not os.path.isfile(os.path.join(d, "training_data.pt")):
                print(f"# SKIP {d}: thiếu training_data.pt", file=sys.stderr)
                continue
            c = lucid_cfg(d)
            base = (f"python run_controls.py --env_name {env} --lucid_split_dir {d} "
                    f"--hidden_dim {c['hidden_dim']} --k {c['k']} --seed {s} "
                    f"--episodes 0 --out {OUT}")
            dnf = (f"--n_clauses_per_action {c['n_clauses_per_action']} "
                   f"--dnf_epochs {c['dnf_epochs']} --bimodal_ramp {c['bimodal_ramp']} "
                   f"--entropy_weight {c['entropy_weight']} --save_rules")
            # DNF arms trước (nặng nhất) để xargs bắt đầu chúng sớm
            for arm in ("A3", "A4"):
                print(f"{base} --arm {arm} {dnf} --tag {tag}_s{s}"); n += 1
            for a in CCP:
                print(f"{base} --arm A1 --ccp_alpha {a} --tag {tag}_ccp{a}_s{s}"); n += 1
            for cc in CS:
                print(f"{base} --arm A2 --C {cc} --tag {tag}_C{cc}_s{s}"); n += 1
            print(f"# {tag} seed {s}: {c}", file=sys.stderr)
    print(f"# tổng {n} lệnh", file=sys.stderr)


if __name__ == "__main__":
    main()
