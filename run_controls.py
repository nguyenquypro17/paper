"""
run_controls.py
One arm, one environment, one seed -> one JSON record.

Arms
----
  A0  SAE (joint)          + DNF   : LUCID itself, re-evaluated under this protocol
  A1  SAE (recon-only)     + CART  : is the gain the representation or the learner?
  A2  SAE (recon-only)     + L1-linear : does DNF buy anything over a sparse linear head?
  A3  raw CNN              + DNF   : what does the SAE contribute?
  A4  SAE (recon-only, frozen) + DNF : joint training vs two-stage
  A5  raw CNN              + CART matched to LUCID's literal budget

Every arm shares: the same offline file, the same split, the same teacher CNN,
the same complexity unit, the same offline/online evaluation.

Example
-------
python run_controls.py --arm A1 --env_name MiniGrid-DoorKey-6x6-v0 \
    --features_path data/doorkey/features.pt --ppo_path models/ppo_doorkey.zip \
    --seed 42 --episodes 200 --out results/controls
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_ROOT = os.environ.get("LUCID_ROOT", os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from controls.data import load_offline, load_offline_from_lucid
from controls.representations import SAEReconConfig, Representation, fit_sae_recon_only, live_dims
from controls.learners import (DNFConfig, fit_cart, fit_matched_tree, fit_sparse_linear,
                               train_bottleneck_dnf)
from controls.complexity import count_any
from controls.eval import (DNFHardAgent, DNFSoftAgent, SklearnAgent, dnf_hard_predict,
                           dnf_soft_predict, load_ppo_cnn, offline_fidelity, parse_rules,
                           rollout_multi_seed)

ARMS = {
    "A0": ("sae_joint", "dnf"),
    "A1": ("sae_recon", "tree"),
    "A2": ("sae_recon", "linear"),
    "A3": ("raw", "dnf"),
    "A4": ("sae_recon_frozen", "dnf"),
    "A5": ("raw", "tree_matched"),
}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_representation(arm: str, data, args, device):
    """Return (Representation, Z_tr, Z_val, Z_te, repr_meta)."""
    repr_kind = ARMS[arm][0]
    meta = {"representation": repr_kind}

    if repr_kind == "raw":
        rep = Representation("raw", device=device)
        return rep, data.X_tr, data.X_val, data.X_te, meta

    if repr_kind == "sae_joint":
        from train_joint import SAELogicAgentV3, SAELogicConfig
        ckpt = torch.load(args.lucid_ckpt, map_location=device, weights_only=False)
        lucid = SAELogicAgentV3(SAELogicConfig(**ckpt["config"]), device=device)
        lucid.load_state_dict(ckpt["model_state"])
        lucid.to(device).eval()
        rep = Representation("sae", sae=lucid.sae, device=device)
        meta["lucid_ckpt"] = args.lucid_ckpt
        Zs = [rep.transform(X) for X in (data.X_tr, data.X_val, data.X_te)]
        return rep, Zs[0], Zs[1], Zs[2], meta

    # sae_recon / sae_recon_frozen: identical object, different downstream use
    sae_cfg = SAEReconConfig(
        hidden_dim=args.hidden_dim, k=args.k, lambda_sparsity=args.lambda_sparsity,
        lr=args.sae_lr, n_epochs=args.sae_epochs, batch_size=args.batch_size, seed=args.seed,
    )
    print(f"\n[{arm}] training recon-only SAE ({sae_cfg.hidden_dim} dims, k={sae_cfg.k})...")
    sae = fit_sae_recon_only(data.X_tr, sae_cfg, device, X_val=data.X_val)
    rep = Representation("sae", sae=sae, device=device)
    Zs = [rep.transform(X) for X in (data.X_tr, data.X_val, data.X_te)]
    meta["sae_live_dims"] = int(len(live_dims(Zs[0])))
    meta["sae_config"] = sae_cfg.__dict__
    return rep, Zs[0], Zs[1], Zs[2], meta


def run(args):
    device = pick_device()
    arm = args.arm.upper()
    assert arm in ARMS, f"unknown arm {arm}"
    learner_kind = ARMS[arm][1]
    t0 = time.time()

    print(f"=== arm {arm} | {args.env_name} | seed {args.seed} | device {device} ===")
    if args.lucid_split_dir:
        data = load_offline_from_lucid(args.lucid_split_dir, args.env_name, seed=args.seed)
        print(f"  split = LUCID held-out of {args.lucid_split_dir}")
    else:
        data = load_offline(args.features_path, args.env_name, seed=args.seed,
                            stage1_path=args.stage1_path or None)
    print(f"  train/val/test = {len(data.X_tr)}/{len(data.X_val)}/{len(data.X_te)}"
          f" | d={data.input_dim} | A={data.n_actions}")

    rep, Z_tr, Z_val, Z_te, meta = build_representation(arm, data, args, device)
    fallback = data.fallback_action()
    record = {
        "arm": arm, "env_name": args.env_name, "seed": args.seed,
        "learner": learner_kind, "lucid_split_dir": args.lucid_split_dir, **meta,
    }

    # -------------------- fit the learner --------------------
    if learner_kind == "dnf":
        dnf_cfg = DNFConfig(
            n_features=Z_tr.shape[1], n_actions=data.n_actions,
            n_clauses_per_action=args.n_clauses_per_action,
            l0_penalty_weight=args.l0_penalty, entropy_weight=args.entropy_weight,
            n_epochs=args.dnf_epochs, batch_size=args.batch_size,
            beta_action=args.beta_action, seed=args.seed,
            bimodal_warmup=args.bimodal_warmup, bimodal_ramp=args.bimodal_ramp,
        )
        print(f"\n[{arm}] training bottleneck+DNF on {Z_tr.shape[1]} features...")
        model, val_acc, _ = train_bottleneck_dnf(Z_tr, data.a_tr, Z_val, data.a_val, dnf_cfg, device)
        rules = model.extract_rules(action_names=data.action_names, threshold=args.tau)

        soft_pred = dnf_soft_predict(model, Z_te, device)
        z_bin_te = model.binarize(Z_te.to(device)).detach().cpu().numpy()
        hard_pred, covered = dnf_hard_predict(parse_rules(rules), z_bin_te,
                                              data.action_names, fallback, args.hard_threshold)
        a_te = data.a_te.numpy()
        parsed_pure = {k: [c for c in v if c[0] or c[1]] for k, v in parse_rules(rules).items()}
        pure_pred, pure_cov = dnf_hard_predict(parsed_pure, z_bin_te, data.action_names,
                                               fallback, args.hard_threshold)

        record.update({
            "val_acc": val_acc,
            "fidelity_soft": float((soft_pred == a_te).mean() * 100),
            "fidelity_hard": float((hard_pred == a_te).mean() * 100),
            "coverage_offline": float(covered.mean() * 100),
            "fidelity_hard_pure": float((pure_pred == a_te).mean() * 100),
            "coverage_pure": float(pure_cov.mean() * 100),
            "complexity": count_any(model, "dnf", rules=rules),
            "knob": {"l0_penalty": args.l0_penalty, "n_clauses_per_action": args.n_clauses_per_action,
                     "tau": args.tau},
            "rules": rules if args.save_rules else None,
        })
        head_for_rollout = ("dnf", model, rules)

    else:
        Z_tr_np, a_tr_np = Z_tr.numpy(), data.a_tr.numpy()
        Z_te_np, a_te_np = Z_te.numpy(), data.a_te.numpy()

        if learner_kind == "tree":
            clf = fit_cart(Z_tr_np, a_tr_np, max_depth=args.max_depth,
                           ccp_alpha=args.ccp_alpha, seed=args.seed)
            knob = {"max_depth": args.max_depth, "ccp_alpha": args.ccp_alpha}
            kind = "tree"
        elif learner_kind == "tree_matched":
            assert args.match_literals > 0, "A5 needs --match_literals (LUCID's literal count)"
            clf, achieved = fit_matched_tree(Z_tr_np, a_tr_np, args.match_literals, seed=args.seed)
            knob = {"match_literals": args.match_literals, "achieved_tests": achieved}
            kind = "tree"
        else:
            clf = fit_sparse_linear(Z_tr_np, a_tr_np, C=args.C, seed=args.seed)
            knob = {"C": args.C}
            kind = "linear"

        record.update({
            "val_acc": float(clf.score(Z_val.numpy(), data.a_val.numpy())),
            "fidelity_soft": offline_fidelity(clf.predict, Z_te_np, a_te_np),
            "fidelity_hard": offline_fidelity(clf.predict, Z_te_np, a_te_np),
            "coverage_offline": 100.0,
            "complexity": count_any(clf, kind),
            "knob": knob,
        })
        head_for_rollout = ("sklearn", clf, None)

    # -------------------- online rollout --------------------
    if args.ppo_path and args.episodes > 0:
        print(f"\n[{arm}] rolling out {args.episodes} episodes x {len(args.eval_seeds)} seeds...")
        ppo_cnn = load_ppo_cnn(args.ppo_path, device)

        def repr_fn(feats):
            x = (feats - data.feat_mean.to(device)) / data.feat_std.to(device)
            return rep.transform_online(x)

        kind, model_obj, rules = head_for_rollout
        if kind == "dnf":
            soft_agent = DNFSoftAgent(ppo_cnn, repr_fn, model_obj, device)
            hard_agent = DNFHardAgent(ppo_cnn, repr_fn, model_obj, rules, data.action_names,
                                      fallback, args.hard_threshold, device)
            record["online_soft"] = rollout_multi_seed(soft_agent, args.env_name, args.episodes,
                                                       seeds=tuple(args.eval_seeds), max_steps=args.max_steps)
            record["online_hard"] = rollout_multi_seed(hard_agent, args.env_name, args.episodes,
                                                       seeds=tuple(args.eval_seeds), max_steps=args.max_steps)
        else:
            agent = SklearnAgent(ppo_cnn, repr_fn, model_obj, device)
            record["online_hard"] = rollout_multi_seed(agent, args.env_name, args.episodes,
                                                       seeds=tuple(args.eval_seeds), max_steps=args.max_steps)

    record["runtime_sec"] = round(time.time() - t0, 1)

    os.makedirs(args.out, exist_ok=True)
    env_tag = args.env_name.replace("/", "_")
    knob_tag = args.tag or "default"
    path = os.path.join(args.out, f"{arm}_{env_tag}_seed{args.seed}_{knob_tag}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    print(f"\nsaved -> {path}")
    print(json.dumps({k: v for k, v in record.items()
                      if k in ("fidelity_soft", "fidelity_hard", "complexity", "coverage_offline")}, indent=2))
    return record


def build_parser():
    p = argparse.ArgumentParser(description="LUCID control-arm ablations")
    p.add_argument("--arm", required=True, choices=list(ARMS) + [a.lower() for a in ARMS])
    p.add_argument("--env_name", required=True)
    p.add_argument("--features_path", default="")
    p.add_argument("--lucid_split_dir", default="",
                   help="dùng đúng split train/held-out của run LUCID này")
    p.add_argument("--stage1_path", default="")
    p.add_argument("--ppo_path", default="")
    p.add_argument("--lucid_ckpt", default="", help="A0 only: trained LUCID checkpoint")
    p.add_argument("--seed", type=int, default=42)

    # representation
    p.add_argument("--hidden_dim", type=int, default=300)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--lambda_sparsity", type=float, default=5e-3)
    p.add_argument("--sae_lr", type=float, default=1e-3)
    p.add_argument("--sae_epochs", type=int, default=200)

    # DNF learner
    p.add_argument("--n_clauses_per_action", type=int, default=10)
    p.add_argument("--l0_penalty", type=float, default=1e-4)
    p.add_argument("--entropy_weight", type=float, default=0.005)
    p.add_argument("--beta_action", type=float, default=5.0)
    p.add_argument("--dnf_epochs", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--bimodal_warmup", type=int, default=30)
    p.add_argument("--bimodal_ramp", type=int, default=80)
    p.add_argument("--hard_threshold", type=float, default=0.66)

    # classical learners
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--ccp_alpha", type=float, default=0.0)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--match_literals", type=int, default=0)

    # evaluation
    p.add_argument("--episodes", type=int, default=200, help="episodes PER eval seed; 0 disables rollout")
    p.add_argument("--eval_seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--max_steps", type=int, default=27000)

    p.add_argument("--out", default="results/controls")
    p.add_argument("--tag", default="", help="knob tag for the output filename")
    p.add_argument("--save_rules", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.arm = args.arm.upper()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run(args)
