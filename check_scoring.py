"""
check_scoring.py
Hai số để trả lời "LUCID score-based thì so literal với cây có công bằng không?"

  1. Conflict rate   : % state có >= 2 action cùng có clause fire.
                       Thấp  -> tầng arbitration gần như không hoạt động.
  2. Unweighted ablation : bỏ hết bias, argmax theo SỐ clause fire (tie -> fallback).
                       Fidelity rụng ít -> scoring không phải chỗ tạo ra fidelity,
                       nên so 16 literal vs 44 literal là sạch.

Chạy:
  python check_scoring.py --run_dir outputs/lucid_model_doorkey2 \
      --env_name MiniGrid-DoorKey-6x6-v0 --hard_threshold 0.66
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (_ROOT, os.path.join(_ROOT, "experiments", "lucid")):
    if p not in sys.path:
        sys.path.insert(0, p)

from train_joint import SAELogicAgentV3, SAELogicConfig  # noqa: E402
from controls.data import resolve_actions  # noqa: E402
from controls.eval import parse_rules  # noqa: E402


def fire_matrix(parsed, z_bool, action_names):
    """
    Trả về (fires, weights):
      fires[a]  : (n, n_clauses_a) bool - từng clause của action a có fire không
      weights[a]: (n_clauses_a,) float  - sigmoid(bias) của từng clause
    """
    n = len(z_bool)
    fires, weights = {}, {}
    for act in action_names:
        clauses = parsed.get(act, [])
        F = np.zeros((n, len(clauses)), dtype=bool)
        W = np.zeros(len(clauses), dtype=float)
        for ci, (pos, neg, w) in enumerate(clauses):
            m = np.ones(n, dtype=bool)
            for f in pos:
                m &= z_bool[:, f]
            for f in neg:
                m &= ~z_bool[:, f]
            F[:, ci] = m
            W[ci] = w
        fires[act], weights[act] = F, W
    return fires, weights


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = args.run_dir

    ckpt = torch.load(os.path.join(run, "sae_logic_joint_model.pt"),
                      map_location=device, weights_only=False)
    model = SAELogicAgentV3(SAELogicConfig(**ckpt["config"]), device=device)
    model.load_state_dict(ckpt["model_state"])
    model.set_normalization(ckpt["feature_mean"].to(device), ckpt["feature_std"].to(device))
    if "z_mean" in ckpt:
        model.z_mean.copy_(ckpt["z_mean"].to(device))
        model.z_std.copy_(ckpt["z_std"].to(device))
    model.to(device).eval()

    rules = json.load(open(os.path.join(run, "learned_rules.json")))
    data = torch.load(os.path.join(run, "training_data.pt"), weights_only=False)
    feats = data["features"].float()
    acts = data["actions"].long().view(-1).numpy()
    pre_normalized = bool(data.get("pre_normalized", False))

    # tên action lấy từ chính file rules (thứ tự key = thứ tự index)
    action_names = list(rules.keys())
    n_actions = len(action_names)
    env_names = resolve_actions(args.env_name, torch.as_tensor(acts))[1]
    if action_names != env_names:
        print(f'[warn] rules dùng tên {action_names[:3]}..., env dùng {env_names[:3]}... '
              f'-> theo file rules')

    # --- chọn split ---
    if args.split == "val" and "shuffle_indices" in data:
        idx = data["shuffle_indices"][data["n_train"]:].numpy()
    else:
        idx = np.arange(len(feats))
    feats, acts = feats[idx], acts[idx]

    # --- nhị phân hoá concept ---
    z_bin = []
    with torch.no_grad():
        for i in range(0, len(feats), 512):
            b = feats[i:i + 512].to(device)
            if not pre_normalized:
                b = model.normalize_input(b)
            z, _ = model.sae.encode(b)
            z_bin.append(model.bottleneck(model.normalize_z(z)).cpu())
    z_bin = torch.cat(z_bin).numpy()
    z_bool = z_bin > args.hard_threshold

    parsed = parse_rules(rules)
    n_true = {a: sum(1 for c in v if not c[0] and not c[1]) for a, v in parsed.items()}
    n_real = {a: sum(1 for c in v if c[0] or c[1]) for a, v in parsed.items()}
    print('clause (True) mỗi action :', n_true)
    print('clause thực  mỗi action :', n_real)
    if args.drop_true:
        parsed = {a: [c for c in v if c[0] or c[1]] for a, v in parsed.items()}
        print('-> đã bỏ clause (True), chỉ giữ logic thuần\n')
    fires, weights = fire_matrix(parsed, z_bool, action_names)

    n = len(z_bool)
    act_fires = np.stack([fires[a].any(axis=1) for a in action_names], axis=1)   # (n, A)
    covered = act_fires.any(axis=1)
    n_fired = act_fires.sum(axis=1)
    fallback = int(np.bincount(acts).argmax())

    # --- chấm điểm: có trọng số vs không trọng số ---
    scores_w = np.stack([(fires[a] * weights[a]).sum(axis=1) for a in action_names], axis=1)
    scores_u = np.stack([fires[a].sum(axis=1).astype(float) for a in action_names], axis=1)

    pred_w = np.where(covered, scores_w.argmax(axis=1), fallback)

    # unweighted: hoà thì rơi về fallback, không được argmax ngầm phá hoà
    top_u = scores_u.max(axis=1, keepdims=True)
    n_tied = (scores_u == top_u).sum(axis=1)
    pred_u = np.where(covered & (n_tied == 1), scores_u.argmax(axis=1), fallback)

    fid_w = (pred_w == acts).mean() * 100
    fid_u = (pred_u == acts).mean() * 100

    out = {
        "run_dir": run,
        "split": args.split,
        "n_states": int(n),
        "coverage_pct": float(covered.mean() * 100),
        "conflict_rate_pct": float((n_fired >= 2).mean() * 100),
        "conflict_rate_among_covered_pct": float((n_fired[covered] >= 2).mean() * 100) if covered.any() else 0.0,
        "mean_actions_fired": float(n_fired.mean()),
        "multi_clause_same_action_pct": float(
            np.mean([fires[a].sum(axis=1).max() > 1 for a in action_names]) * 100),
        "fidelity_weighted_pct": float(fid_w),
        "fidelity_unweighted_pct": float(fid_u),
        "fidelity_drop_pct": float(fid_w - fid_u),
        "tie_rate_unweighted_pct": float((covered & (n_tied > 1)).mean() * 100),
        "agreement_weighted_vs_unweighted_pct": float((pred_w == pred_u).mean() * 100),
        "fallback_action": action_names[fallback],
    }
    print(json.dumps(out, indent=2))

    with open(os.path.join(run, "scoring_check.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {os.path.join(run, 'scoring_check.json')}")

    # --- diễn giải ---
    print("\n--- diễn giải ---")
    if out["conflict_rate_pct"] < 5:
        print(f"Conflict {out['conflict_rate_pct']:.2f}% : tầng arbitration hầu như không dùng đến.")
    else:
        print(f"Conflict {out['conflict_rate_pct']:.2f}% : scoring CÓ hoạt động, phải nêu rõ trong paper.")
    if out["fidelity_drop_pct"] < 1:
        print(f"Bỏ trọng số chỉ rụng {out['fidelity_drop_pct']:.2f}% : luật tự nó quyết định, "
              f"so literal với cây là sạch.")
    else:
        print(f"Bỏ trọng số rụng {out['fidelity_drop_pct']:.2f}% : fidelity phụ thuộc linear head "
              f"-> baseline đúng để so là A2 (SAE + L1 linear), không phải cây.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--env_name", required=True)
    p.add_argument("--hard_threshold", type=float, default=0.66)
    p.add_argument("--split", default="val", choices=["val", "all"])
    p.add_argument("--drop_true", action="store_true",
                   help="bỏ clause (True) để đo logic thuần, không có prior")
    main(p.parse_args())
