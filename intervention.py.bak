#!/usr/bin/env python3
"""
E1 — Intervention test.

Flip concepts and measure whether the decision changes, at three levels
(concept / clause / action) and for three policies:
    hard    : DNF rules on binarized concepts
    soft    : LUCID logic layer on continuous z_binary
    teacher : PPO head on h_hat = h + W_d (z' - z) * feature_std   (clamp >= 0 and raw)

Baselines (paired, one row per intervention row):
    matched : flip concepts OUTSIDE the winning clauses with the same on/off state
              (coverage can be low: reported, and compared only on the paired subset)
    dirb    : dictionary-direction baseline -- the same dz values applied to decoder
              directions of random alive concepts outside the winning clauses
              (always available; teacher only)
    noise   : Gaussian dz with the same ||dz||

Intervention strength (--strengths): ON/OFF values are percentiles of z_sparse over
held-out samples in that state (50 = median = primary). All baselines are recomputed
at each strength with the same random choices, so the curve is comparable.

Usage:
    python intervention.py \
        --model_path ./outputs/lucid_model_doorkey/sae_logic_joint_model.pt \
        --data_path ./held_out_doorkey_data.pt \
        --ppo_path ./ppo_doorkey_6x6.zip \
        --hard_threshold 0.66 \
        --groups_path ./experiments/e2/results_doorkey/groups.json \
        --out_dir ./experiments/e1/results_doorkey
"""

import argparse
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LEVELS = ["concept", "clause", "action"]
MAX_FLIPS = 5


# ============================================================================
# Loading
# ============================================================================

def load_lucid(model_path, device, hard_threshold=None):
    from train_joint import SAELogicConfig, SAELogicAgentV3
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = SAELogicConfig(**ckpt["config"])
    model = SAELogicAgentV3(config, device=device)
    model.load_state_dict(ckpt["model_state"])
    model.set_normalization(ckpt["feature_mean"].to(device), ckpt["feature_std"].to(device))
    if "z_mean" in ckpt:
        model.z_mean.copy_(ckpt["z_mean"].to(device))
        model.z_std.copy_(ckpt["z_std"].to(device))
    model.to(device).eval()
    if hard_threshold is not None:
        config.hard_threshold = hard_threshold
    return model, config, ckpt["rules"]


def load_teacher(ppo_path, device):
    from stable_baselines3 import PPO
    try:
        return PPO.load(ppo_path, device=device).policy.eval()
    except Exception as e:
        print(f"  plain PPO.load failed ({e}); retrying with MiniGrid custom_objects")
        from test_ppo_doorkey_6x6 import load_model
        return load_model(ppo_path, device).policy.eval()


@torch.no_grad()
def teacher_probs(policy, h):
    """h (M,d) raw CNN features -> softmax probs (M,A)"""
    latent = policy.mlp_extractor.forward_actor(h)
    return F.softmax(policy.action_net(latent), dim=-1)


# ============================================================================
# Rules (numpy)
# ============================================================================

def parse_rules(rules, n_features):
    """
    -> dict(Pm (C,D) bool, Nm (C,D) bool, act (C,), w (C,), n_false, n_true, uniq (C,))
    Action index = insertion order of the rules dict. '(False)' clauses are dropped.
    '(True)' clauses have empty Pm/Nm -> always fire (they only add score).
    uniq[c] = id of the distinct (action, literal set) clause c belongs to.
    """
    Pm, Nm, act, w = [], [], [], []
    n_false = n_true = 0
    for a, clauses in enumerate(rules.values()):
        for c in clauses:
            if c.startswith("(False)"):
                n_false += 1
                continue
            if c.startswith("(True)"):
                n_true += 1
            p = np.zeros(n_features, bool); n = np.zeros(n_features, bool)
            for neg, idx in re.findall(r"(¬?)f_(\d+)", c.split("[")[0]):
                (n if neg else p)[int(idx)] = True
            m = re.search(r"\[bias=([-\d.]+)\]", c)
            b = float(m.group(1)) if m else 0.0
            Pm.append(p); Nm.append(n); act.append(a); w.append(1 / (1 + np.exp(-b)))
    Pm, Nm = np.array(Pm), np.array(Nm)
    keys = [(a, p.tobytes(), n.tobytes()) for a, p, n in zip(act, Pm, Nm)]
    ids = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    return dict(Pm=Pm, Nm=Nm, act=np.array(act), w=np.array(w), n_actions=len(rules),
                n_false=n_false, n_true=n_true, uniq=np.array([ids[k] for k in keys]),
                is_lit=(Pm.any(1) | Nm.any(1)))


def blocker_count(Z, R):
    """Z (N,D) bool -> (N,C) number of unsatisfied literals per clause"""
    Zi = Z.astype(np.int32)
    return (1 - Zi) @ R["Pm"].T.astype(np.int32) + Zi @ R["Nm"].T.astype(np.int32)


def scores_from_fire(Fire, R, fallback):
    """Fire (N,C) bool -> action (N,), same tie-breaking as evaluate_lucid_metrics (first max)"""
    W = np.zeros((len(R["act"]), R["n_actions"]))
    W[np.arange(len(R["act"])), R["act"]] = R["w"]
    S = Fire.astype(float) @ W
    a = S.argmax(1)
    a[~Fire.any(1)] = fallback
    return a


def hard_policy(Z, R, fallback):
    """-> action (N,), Fire (N,C)"""
    Fire = blocker_count(Z, R) == 0
    return scores_from_fire(Fire, R, fallback), Fire


def literals_of(R, c):
    return np.where(R["Pm"][c] | R["Nm"][c])[0]


def concept_rule_stats(R):
    """Per concept: number of distinct clauses / actions containing it."""
    lit = R["Pm"] | R["Nm"]
    first = {}
    for c, u in enumerate(R["uniq"]):
        first.setdefault(u, c)
    U = np.array(sorted(first.values()))
    n_cl = lit[U].sum(0)
    n_act = np.array([len(set(R["act"][U][lit[U][:, j]])) for j in range(lit.shape[1])])
    return n_cl, n_act


# ============================================================================
# Intervention construction (numpy)
# ============================================================================

def winning_clauses(Fire_i, a0, R):
    """Distinct literal clauses of a0 that fire (one representative index each)."""
    idx = np.where(Fire_i & (R["act"] == a0) & R["is_lit"])[0]
    seen, out = set(), []
    for c in idx:
        if R["uniq"][c] not in seen:
            seen.add(R["uniq"][c]); out.append(c)
    return out


def necessity_targets(z, Fire_i, a0, R):
    """Satisfied literals in the winning clauses (all literals of a firing clause are satisfied)."""
    wc = winning_clauses(Fire_i, a0, R)
    lits = sorted(set(j for c in wc for j in literals_of(R, c)))
    return wc, lits


def sufficiency_targets(z, Fire_i, a0, R):
    """[(action a != a0, clause c, blocking concept j)] for clauses blocked by exactly ONE literal."""
    blk = (R["Pm"] & ~z) | (R["Nm"] & z)
    cnt = blk.sum(1)
    out, seen = [], set()
    for c in np.where((cnt == 1) & (R["act"] != a0) & R["is_lit"])[0]:
        if R["uniq"][c] in seen:
            continue
        seen.add(R["uniq"][c])
        out.append((int(R["act"][c]), int(c), int(np.where(blk[c])[0][0])))
    return out


def matched_baseline(z, flips, excluded, pool, rng):
    """
    For each concept in `flips`, pick a concept in `pool` \\ excluded with the same on/off state.
    -> list or None if impossible
    """
    avail = pool & ~excluded
    out = []
    for j in flips:
        cand = np.where(avail & (z == z[j]))[0]
        if len(cand) == 0:
            return None
        b = int(rng.choice(cand)); out.append(b); avail[b] = False
    return out


def predicted_after_single_flip(Zrows, flip_j, B0rows, R, fallback):
    """
    RPCR: predict the hard action after flipping one concept by reading the rules
    (update blocker counts symbolically), independent of recomputing from scratch.
    """
    lit = R["Pm"] | R["Nm"]
    zj = Zrows[np.arange(len(flip_j)), flip_j]
    was_blocker = (R["Pm"][:, flip_j].T & ~zj[:, None]) | (R["Nm"][:, flip_j].T & zj[:, None])
    contains = lit[:, flip_j].T
    B1 = B0rows - was_blocker + (contains & ~was_blocker)
    return scores_from_fire(B1 == 0, R, fallback)


def minimal_counterfactual(z, a0, R, fallback, max_flips=MAX_FLIPS):
    """Fewest concept flips (via one runner-up clause) that change the hard action. NaN if none."""
    blk = (R["Pm"] & ~z) | (R["Nm"] & z)
    cnt = blk.sum(1)
    cand = np.where((cnt >= 1) & (cnt <= max_flips) & (R["act"] != a0) & R["is_lit"])[0]
    for c in cand[np.argsort(cnt[cand], kind="stable")]:
        z2 = z.copy(); z2[blk[c]] = ~z2[blk[c]]
        a1, _ = hard_policy(z2[None], R, fallback)
        if a1[0] != a0:
            return int(cnt[c])
    return np.nan


# ============================================================================
# SAE-level evaluation (torch)
# ============================================================================

@torch.no_grad()
def onoff_values(model, zs, zb, thr, q=50.0, p_on=0.9, p_off=0.1):
    """
    Intervention strength q (percentile):
      ON  value of concept j = q-th percentile of z_sparse_j over held-out samples where j is ON
      OFF value              = (100-q)-th percentile over samples where j is OFF
    q=50 -> median. Fallback when a state never occurs: invert the bottleneck (p_on / p_off).
    -> numpy on, off (D,), flippable_on, flippable_off (D,) bool
    """
    D = zs.shape[1]
    on = torch.empty(D, device=zs.device); off = torch.empty(D, device=zs.device)
    alpha = model.bottleneck.get_sharpness(); beta = model.bottleneck.beta

    def inv(p):
        zn = beta + torch.logit(torch.tensor(p, device=zs.device)) / alpha
        return (zn * model.z_std + model.z_mean).clamp(min=0)
    inv_on, inv_off = inv(p_on), inv(p_off)
    mask = zb > thr
    for j in range(D):
        m = mask[:, j]
        on[j] = torch.quantile(zs[m, j], q / 100.0) if m.any() else inv_on[j]
        off[j] = torch.quantile(zs[~m, j], 1 - q / 100.0) if (~m).any() else inv_off[j]
    bz = lambda v: model.bottleneck(model.normalize_z(v[None]))[0]
    return (on.cpu().numpy(), off.cpu().numpy(),
            (bz(on) > thr).cpu().numpy(), (bz(off) <= thr).cpu().numpy())


@torch.no_grad()
def sae_eval(model, policy, H, ZS, DZ, thr, batch=4096):
    """
    H (M,d) raw features, ZS (M,D) z_sparse, DZ (M,D) change in z_sparse.
    -> dict of numpy: hard_z (M,D) bool, soft (M,), t_clamp, t_raw (M,), p_clamp, p_raw (M,A),
       dh_norm (M,), neg_frac (M,)
    """
    Wd, std = model.sae.W_d, model.feature_std
    out = {k: [] for k in ["hard_z", "soft", "p_clamp", "p_raw", "dh_norm", "neg_frac"]}
    for s in range(0, len(H), batch):
        h, zs, dz = H[s:s + batch], ZS[s:s + batch], DZ[s:s + batch]
        zb = model.bottleneck(model.normalize_z(zs + dz))
        out["hard_z"].append((zb > thr).cpu())
        out["soft"].append(model.logic_layer(zb).argmax(1).cpu())
        hh = h + (dz @ Wd.T) * std
        out["neg_frac"].append((hh < 0).float().mean(1).cpu())
        hc = hh.clamp(min=0)
        out["dh_norm"].append((hc - h).norm(dim=1).cpu())
        out["p_clamp"].append(teacher_probs(policy, hc).cpu())
        out["p_raw"].append(teacher_probs(policy, hh).cpu())
    res = {k: torch.cat(v).numpy() for k, v in out.items()}
    res["t_clamp"] = res["p_clamp"].argmax(1)
    res["t_raw"] = res["p_raw"].argmax(1)
    return res


# ============================================================================
# Statistics
# ============================================================================

def bootstrap_diff(x, y, groups, B=2000, seed=0):
    """Mean(x - y) with a cluster (episode) bootstrap 95% CI."""
    x = np.asarray(x, float); y = np.asarray(y, float); g = np.asarray(groups)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y, g = x[ok], y[ok], g[ok]
    if len(x) == 0:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, n=0, x_mean=np.nan, y_mean=np.nan)
    ug, inv = np.unique(g, return_inverse=True)
    sx = np.bincount(inv, x - y); cnt = np.bincount(inv)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        k = rng.integers(0, len(ug), len(ug))
        bs.append(sx[k].sum() / cnt[k].sum())
    return dict(mean=float((x - y).mean()), lo=float(np.percentile(bs, 2.5)),
                hi=float(np.percentile(bs, 97.5)), n=int(len(x)),
                x_mean=float(x.mean()), y_mean=float(y.mean()))


def rate(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


# ============================================================================
# Main
# ============================================================================

CONDS = ["int", "mb_any", "mb_rule", "dirb", "noise"]
KEYS = ["hard_rule", "hard_sae", "soft", "t_clamp", "t_raw", "dp_clamp", "dh_norm", "neg_frac"]
DIFF_KEYS = ["hard_rule", "soft", "t_clamp", "t_raw", "dp_clamp"]
TEACHER_ONLY = {"dirb"}          # conditions where rule/soft outputs are not meaningful


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--ppo_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--hard_threshold", type=float, default=None)
    ap.add_argument("--groups_path", default=None, help="E2 groups.json")
    ap.add_argument("--train_data_path", default=None,
                    help="Training collected_data.pt, only used for the fallback action (mode)")
    ap.add_argument("--max_states", type=int, default=5000)
    ap.add_argument("--strengths", type=float, nargs="+", default=[50, 75, 90, 95, 99],
                    help="ON/OFF percentiles; 50 (median) is the primary setting")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    strengths = sorted(set(float(x) for x in args.strengths))
    q_primary = 50.0 if 50.0 in strengths else strengths[0]

    model, config, rules = load_lucid(args.model_path, device, args.hard_threshold)
    thr = config.hard_threshold
    policy = load_teacher(args.ppo_path, device)
    d = torch.load(args.data_path, map_location="cpu", weights_only=False)
    H_all = d["features"].float()
    act_data = d["actions"].long().view(-1).numpy()
    ep_all = d["episode_ids"].numpy()
    D = config.hidden_dim

    R = parse_rules(rules, D)
    assert R["n_actions"] == config.n_actions, (R["n_actions"], config.n_actions)
    src = torch.load(args.train_data_path, map_location="cpu", weights_only=False)["actions"] \
        if args.train_data_path else d["actions"]
    fallback = int(np.bincount(src.long().view(-1).numpy()).argmax())
    groups = json.load(open(args.groups_path))["concepts"] if args.groups_path else {}

    # ---------------- base pass on all held-out states ----------------
    with torch.no_grad():
        Hd = H_all.to(device)
        zs_l, zb_l = [], []
        for s in range(0, len(Hd), 4096):
            zs, _ = model.sae.encode(model.normalize_input(Hd[s:s + 4096]))
            zs_l.append(zs); zb_l.append(model.bottleneck(model.normalize_z(zs)))
        ZS_all, ZB_all = torch.cat(zs_l), torch.cat(zb_l)
    Zbool_all = (ZB_all > thr).cpu().numpy()
    alive = ZS_all.std(0).cpu().numpy() > 1e-8
    rule_concepts = (R["Pm"] | R["Nm"]).any(0)
    onoff = {q: onoff_values(model, ZS_all, ZB_all, thr, q) for q in strengths}
    _, _, flip_on_ok, flip_off_ok = onoff[q_primary]

    base = sae_eval(model, policy, Hd, ZS_all, torch.zeros_like(ZS_all), thr)
    a_hard_all, Fire_all = hard_policy(Zbool_all, R, fallback)

    sanity = dict(
        n_states=len(H_all), n_clauses=len(R["act"]), n_distinct_clauses=int(len(set(R["uniq"]))),
        n_false_clauses=R["n_false"], n_true_clauses=R["n_true"], fallback_action=fallback,
        fallback_rate=float((~Fire_all.any(1)).mean()),
        teacher_vs_data=float((base["t_clamp"] == act_data).mean()),
        teacher_raw_vs_data=float((base["t_raw"] == act_data).mean()),
        hard_fidelity=float((a_hard_all == act_data).mean()),
        soft_fidelity=float((base["soft"] == act_data).mean()),
        hard_z_equals_bool=float((base["hard_z"] == Zbool_all).all()),
        zero_intervention_max_dh=float(base["dh_norm"].max()),
        concepts_in_rules=int(rule_concepts.sum()),
        rule_concepts_dead=[int(j) for j in np.where(rule_concepts & ~alive)[0]],
        rule_concepts_unflippable_on=[int(j) for j in np.where(rule_concepts & ~flip_on_ok)[0]],
        rule_concepts_unflippable_off=[int(j) for j in np.where(rule_concepts & ~flip_off_ok)[0]],
        strengths=strengths, primary_strength=q_primary,
        onoff_values_rule_concepts={
            str(q): {int(j): [float(onoff[q][0][j]), float(onoff[q][1][j])]
                     for j in np.where(rule_concepts)[0]} for q in strengths},
    )
    print(json.dumps({k: v for k, v in sanity.items() if k != "onoff_values_rule_concepts"}, indent=2))

    # ---------------- state subset & intervention rows (strength-independent) ----------------
    S = rng.choice(len(H_all), size=min(args.max_states, len(H_all)), replace=False)
    Z, Fire, a0h = Zbool_all[S], Fire_all[S], a_hard_all[S]
    B0 = blocker_count(Z, R)
    alive_idx = np.where(alive)[0]

    def pick_dirb(excl_list, n):
        excl = np.zeros(D, bool); excl[excl_list] = True
        pool = np.where(alive & ~excl)[0]
        return [int(b) for b in rng.choice(pool, size=n, replace=False)] if len(pool) >= n else None

    rows, min_flips, n_no_winner = [], [], 0
    for p in range(len(S)):
        z = Z[p]
        wc, lits = necessity_targets(z, Fire[p], a0h[p], R)
        if not wc:
            n_no_winner += 1
        else:
            excl = np.zeros(D, bool); excl[lits] = True
            specs = [("concept", [int(j)], int(j)) for j in lits]
            specs += [("clause", [int(j) for j in literals_of(R, c)], int(c)) for c in wc]
            specs += [("action", [int(j) for j in lits], None)]
            for level, flips, info in specs:
                rows.append(dict(
                    p=p, flips=flips, level=level, info=info, kind="nec", excl=list(lits),
                    mb_any=matched_baseline(z, flips, excl, alive.copy(), rng),
                    mb_rule=matched_baseline(z, flips, excl, (alive & rule_concepts).copy(), rng),
                    dirb=pick_dirb(lits, len(flips))))
        for a, c, j in sufficiency_targets(z, Fire[p], a0h[p], R):
            cl = literals_of(R, c)
            excl = np.zeros(D, bool); excl[cl] = True
            rows.append(dict(
                p=p, flips=[int(j)], level="concept", info=int(j), target=a, clause=c, kind="suf",
                excl=list(cl), mb_any=matched_baseline(z, [j], excl, alive.copy(), rng),
                mb_rule=None, dirb=pick_dirb(cl, 1)))
        min_flips.append(minimal_counterfactual(z, a0h[p], R, fallback))
    N = len(rows)
    row_p = np.array([r["p"] for r in rows], int)
    frac_to_on = np.array([np.mean([not Z[r["p"], j] for j in r["flips"]]) for r in rows]) if N else np.array([])
    level = np.array([r["level"] for r in rows]); kind = np.array([r["kind"] for r in rows])
    tgt = np.array([r.get("target", -1) for r in rows])
    eps = ep_all[S][row_p] if N else np.array([])
    print(f"Rows: {N} | states without a winning literal clause: {n_no_winner}/{len(S)}")

    # fixed Gaussian direction per row (same across strengths)
    noise_dirs = rng.standard_normal((N, D)).astype(np.float32)
    noise_dirs /= np.linalg.norm(noise_dirs, axis=1, keepdims=True)

    # ---------------- rule level (strength-independent) ----------------
    def flip_bool(zrow, flips):
        z2 = zrow.copy(); z2[flips] = ~z2[flips]; return z2

    rule_hard = {}
    for c in ["int", "mb_any", "mb_rule"]:
        idx = np.array([i for i, r in enumerate(rows)
                        if (r["flips"] if c == "int" else r[c]) is not None], int)
        if len(idx):
            zb = np.stack([flip_bool(Z[rows[i]["p"]], rows[i]["flips"] if c == "int" else rows[i][c])
                           for i in idx])
            rule_hard[c] = (idx, hard_policy(zb, R, fallback)[0])

    # ---------------- SAE level, per strength ----------------
    ZS_S = ZS_all[torch.as_tensor(S, device=device)].cpu().numpy()

    def evaluate_strength(q):
        on_np, off_np, fon, foff = onoff[q]

        def build_dz(p, flips):
            dz = np.zeros(D, np.float32)
            for j in flips:
                to_on = not Z[p, j]
                if (to_on and not fon[j]) or (not to_on and not foff[j]):
                    return None
                dz[j] = (on_np[j] if to_on else off_np[j]) - ZS_S[p, j]
            return dz

        dz_int = [build_dz(r["p"], r["flips"]) for r in rows]
        results = {}
        for c in CONDS:
            idx, dzs = [], []
            for i, r in enumerate(rows):
                d0 = dz_int[i]
                if c == "int":
                    dd = d0
                elif c in ("mb_any", "mb_rule"):
                    dd = build_dz(r["p"], r[c]) if r[c] is not None else None
                elif c == "dirb":
                    if d0 is None or r["dirb"] is None:
                        dd = None
                    else:
                        dd = np.zeros(D, np.float32)
                        for j, b in zip(r["flips"], r["dirb"]):
                            dd[b] += d0[j]
                else:  # noise
                    dd = None if d0 is None else noise_dirs[i] * np.linalg.norm(d0)
                if dd is not None:
                    idx.append(i); dzs.append(dd)
            res = dict(idx=np.array(idx, int))
            if idx:
                St = torch.as_tensor(S[row_p[res["idx"]]], device=device)
                DZ = torch.as_tensor(np.stack(dzs), device=device)
                ev = sae_eval(model, policy, Hd[St], ZS_all[St], DZ, thr)
                ev["hard_sae"], _ = hard_policy(ev.pop("hard_z"), R, fallback)
                res["sae"] = ev
            results[c] = res
        return results

    base_S = {k: base[k][S] for k in ["soft", "t_clamp", "t_raw"]}
    p0c = base["p_clamp"][S, base["t_clamp"][S]]

    def col(results, c, key):
        out = np.full(N, np.nan)
        if key == "hard_rule":
            if c in rule_hard:
                idx, a = rule_hard[c]
                out[idx] = a != a0h[row_p[idx]]
            return out
        res = results[c]
        if "sae" not in res or (c in TEACHER_ONLY and key in ("hard_sae", "soft")):
            return out
        sel = res["idx"]; p = row_p[sel]; ev = res["sae"]
        if key == "hard_sae":
            out[sel] = ev["hard_sae"] != a0h[p]
        elif key == "soft":
            out[sel] = ev["soft"] != base_S["soft"][p]
        elif key in ("t_clamp", "t_raw"):
            out[sel] = ev[key] != base_S[key][p]
        elif key == "dp_clamp":
            out[sel] = p0c[p] - ev["p_clamp"][np.arange(len(p)), base_S["t_clamp"][p]]
        elif key in ("dh_norm", "neg_frac"):
            out[sel] = ev[key]
        return out

    def col_target(results, c, key):
        out = np.full(N, np.nan)
        if key == "hard_rule":
            if c in rule_hard:
                idx, a = rule_hard[c]
                out[idx] = a == tgt[idx]
            return out
        res = results[c]
        if "sae" in res and not (c in TEACHER_ONLY and key == "soft"):
            sel = res["idx"]
            out[sel] = res["sae"][key] == tgt[sel]
        return out

    def level_summary(T):
        out = {}
        for lv in LEVELS:
            m = (level == lv) & (kind == "nec")
            e = {"n_rows": int(m.sum())}
            for c in CONDS:
                e[c] = {k: rate(T[c][k][m]) for k in KEYS}
                e[c]["coverage"] = float(np.isfinite(T[c]["t_clamp"][m]).mean()) if m.any() else np.nan
            for b in ["mb_any", "mb_rule", "dirb", "noise"]:
                e[f"diff_vs_{b}"] = {k: bootstrap_diff(T["int"][k][m], T[b][k][m], eps[m], seed=args.seed)
                                     for k in DIFF_KEYS}
            out[lv] = e
        return out

    strength_summary, results, T = {}, None, None
    for q in strengths:
        res_q = evaluate_strength(q)
        T_q = {c: {k: col(res_q, c, k) for k in KEYS} for c in CONDS}
        strength_summary[str(q)] = level_summary(T_q)
        ms = kind == "suf"
        suf_x = col_target(res_q, "int", "t_clamp")[ms]
        strength_summary[str(q)]["sufficiency"] = dict(
            n_rows=int(ms.sum()),
            teacher_int=rate(suf_x),
            teacher_dirb=rate(col_target(res_q, "dirb", "t_clamp")[ms]),
            teacher_noise=rate(col_target(res_q, "noise", "t_clamp")[ms]),
            teacher_changed_int=rate(T_q["int"]["t_clamp"][ms]),
            dh_norm=rate(T_q["int"]["dh_norm"][ms]),
            **{f"diff_vs_{b}": bootstrap_diff(suf_x,
                                              col_target(res_q, b, "t_clamp")[ms],
                                              eps[ms], seed=args.seed)
               for b in ["dirb", "noise"]})
        if q == q_primary:
            results, T = res_q, T_q
        e = strength_summary[str(q)]["concept"]
        sf = strength_summary[str(q)]["sufficiency"]
        sd = sf["diff_vs_dirb"]
        print(f"  strength q={q:g}: suff. teacher->target int={sf['teacher_int']:.3f} "
              f"dirb={sf['teacher_dirb']:.3f} noise={sf['teacher_noise']:.3f} "
              f"diff={sd['mean']:+.3f} [{sd['lo']:+.3f},{sd['hi']:+.3f}] "
              f"||dh||={sf['dh_norm']:.3f} | "
              f"concept nec teacher int={e['int']['t_clamp']:.3f} "
              f"dirb={e['dirb']['t_clamp']:.3f} noise={e['noise']['t_clamp']:.3f} "
              f"||dh||={e['int']['dh_norm']:.3f}")

    # ---------------- sanity on rows (primary strength) ----------------
    both = ~np.isnan(T["int"]["hard_rule"]) & ~np.isnan(T["int"]["hard_sae"])
    sanity["flip_consistency_rule_vs_sae"] = float(
        (T["int"]["hard_rule"][both] == T["int"]["hard_sae"][both]).mean()) if both.any() else np.nan
    single = np.where(level == "concept")[0]
    if len(single):
        js = np.array([rows[k]["flips"][0] for k in single])
        pred = predicted_after_single_flip(Z[row_p[single]], js, B0[row_p[single]], R, fallback)
        idx, a = rule_hard["int"]
        actual = a[np.searchsorted(idx, single)]
        sanity["RPCR"] = float((pred == actual).mean())
    sanity["rows_dropped_unflippable"] = int(np.isnan(T["int"]["hard_sae"]).sum())
    sanity["mean_neg_frac_before_clamp"] = rate(T["int"]["neg_frac"])

    # ---------------- primary summary ----------------
    summary = strength_summary[str(q_primary)]
    m = kind == "suf"
    sk = ["hard_rule", "soft", "t_clamp"]
    summary["sufficiency"] = {
        "n_rows": int(m.sum()),
        "int": {k: rate(col_target(results, "int", k)[m]) for k in sk},
        "mb_any": {k: rate(col_target(results, "mb_any", k)[m]) for k in sk},
        "dirb": {"t_clamp": rate(col_target(results, "dirb", "t_clamp")[m])},
        "int_changed": {k: rate(T["int"][k][m]) for k in sk},
    }
    mf = np.array(min_flips, float)
    summary["minimal_counterfactual"] = dict(
        mean=rate(mf), median=float(np.nanmedian(mf)) if np.isfinite(mf).any() else np.nan,
        frac_found=float(np.isfinite(mf).mean()),
        hist={int(k): int((mf == k).sum()) for k in range(1, MAX_FLIPS + 1)})

    # ---------------- per-concept table (primary strength) ----------------
    n_cl, n_act = concept_rule_stats(R)
    flip0 = np.array([r["flips"][0] if len(r["flips"]) == 1 else -1 for r in rows])
    table = []
    for j in np.where(rule_concepts)[0]:
        mj = (level == "concept") & (kind == "nec") & (flip0 == j)
        ms = (kind == "suf") & (flip0 == j)
        g = groups.get(str(j), {})
        table.append(dict(
            concept=int(j), n_nec=int(mj.sum()), n_suf=int(ms.sum()),
            nec_hard=rate(T["int"]["hard_rule"][mj]), nec_soft=rate(T["int"]["soft"][mj]),
            nec_teacher=rate(T["int"]["t_clamp"][mj]), nec_teacher_raw=rate(T["int"]["t_raw"][mj]),
            dp_teacher=rate(T["int"]["dp_clamp"][mj]),
            base_any_teacher=rate(T["mb_any"]["t_clamp"][mj]),
            base_any_coverage=float(np.isfinite(T["mb_any"]["t_clamp"][mj]).mean()) if mj.any() else np.nan,
            base_rule_teacher=rate(T["mb_rule"]["t_clamp"][mj]),
            dirb_teacher=rate(T["dirb"]["t_clamp"][mj]),
            noise_teacher=rate(T["noise"]["t_clamp"][mj]),
            suf_hard=rate(col_target(results, "int", "hard_rule")[ms]),
            suf_teacher=rate(col_target(results, "int", "t_clamp")[ms]),
            n_clauses_affected=int(n_cl[j]), n_actions_affected=int(n_act[j]),
            occurs_once=bool(n_cl[j] == 1), alive=bool(alive[j]),
            e2_group=g.get("group"), e2_best_factor=g.get("best_factor"), e2_auc=g.get("auc_max")))

    ok = [t for t in table if t["n_nec"] > 0 and np.isfinite(t["nec_teacher"])
          and np.isfinite(t["dirb_teacher"])]
    if len(ok) >= 2:
        u = mannwhitneyu([t["nec_teacher"] for t in ok], [t["dirb_teacher"] for t in ok],
                         alternative="greater")
        summary["per_concept_mwu_teacher_vs_dirb"] = dict(p=float(u.pvalue), n=len(ok))

    # ---------------- save ----------------
    json.dump(dict(sanity=sanity, summary=summary, strength_summary=strength_summary),
              open(os.path.join(args.out_dir, "levels.json"), "w"), indent=2, default=float)
    json.dump(table, open(os.path.join(args.out_dir, "per_concept.json"), "w"), indent=2, default=float)
    write_latex(table, os.path.join(args.out_dir, "table_per_concept.tex"))
    plot_concept_hist(table, os.path.join(args.out_dir, "fig1_concept_vs_baseline.png"))
    plot_levels(summary, os.path.join(args.out_dir, "fig2_levels.png"))
    plot_minflips(mf, os.path.join(args.out_dir, "fig3_min_flips.png"))
    plot_strength(strength_summary, strengths, os.path.join(args.out_dir, "fig4_strength.png"))

    # ---------------- print ----------------
    extra = {k: sanity[k] for k in ["RPCR", "flip_consistency_rule_vs_sae",
                                     "mean_neg_frac_before_clamp", "rows_dropped_unflippable"]}
    print(f"\nRow sanity: {extra}")
    print("\nShare of flips that turn a concept ON (only these depend on --strengths):")
    for lv in LEVELS:
        mm = (level == lv) & (kind == "nec")
        print(f"  necessity {lv:>7}: {rate(frac_to_on[mm]):.3f}")
    print(f"  sufficiency       : {rate(frac_to_on[kind == 'suf']):.3f}")
    print(f"\n== Levels at primary strength q={q_primary:g} (necessity; teacher change rate) ==")
    for lv in LEVELS:
        e = summary[lv]
        cov = " ".join(f"{b}={e[b]['coverage']:.2f}" for b in ["mb_any", "mb_rule", "dirb", "noise"])
        print(f"  {lv:>7} n={e['n_rows']:>6} | int={e['int']['t_clamp']:.3f} "
              f"dirb={e['dirb']['t_clamp']:.3f} noise={e['noise']['t_clamp']:.3f} "
              f"| raw int={e['int']['t_raw']:.3f} soft={e['int']['soft']:.3f} hard={e['int']['hard_rule']:.3f} "
              f"| ||dh|| int={e['int']['dh_norm']:.3f} dirb={e['dirb']['dh_norm']:.3f} "
              f"noise={e['noise']['dh_norm']:.3f}")
        print(f"          coverage: {cov}")
        for b in ["dirb", "noise", "mb_any"]:
            dd = e[f"diff_vs_{b}"]["t_clamp"]
            tag = " (paired subset only)" if b == "mb_any" else ""
            print(f"          int - {b:<6}: {dd['mean']:+.3f} [{dd['lo']:+.3f}, {dd['hi']:+.3f}] "
                  f"n={dd['n']} | int={dd['x_mean']:.3f} vs {b}={dd['y_mean']:.3f}{tag}")

    print("\n== Strength curve (teacher change rate; int - dirb with 95% CI) ==")
    for lv in LEVELS:
        for q in strengths:
            e = strength_summary[str(q)][lv]
            dd = e["diff_vs_dirb"]["t_clamp"]; dn = e["diff_vs_noise"]["t_clamp"]
            print(f"  {lv:>7} q={q:>4g} | int={e['int']['t_clamp']:.3f} dirb={e['dirb']['t_clamp']:.3f} "
                  f"noise={e['noise']['t_clamp']:.3f} | int-dirb {dd['mean']:+.3f} "
                  f"[{dd['lo']:+.3f}, {dd['hi']:+.3f}] | int-noise {dn['mean']:+.3f} "
                  f"[{dn['lo']:+.3f}, {dn['hi']:+.3f}] | ||dh||={e['int']['dh_norm']:.3f}")

    s = summary["sufficiency"]
    print(f"\n== Sufficiency n={s['n_rows']} | switch-to-target: hard={s['int']['hard_rule']:.3f} "
          f"soft={s['int']['soft']:.3f} teacher={s['int']['t_clamp']:.3f} "
          f"(dirb teacher={s['dirb']['t_clamp']:.3f}, matched teacher={s['mb_any']['t_clamp']:.3f})")
    mc = summary["minimal_counterfactual"]
    print(f"== Minimal counterfactual: mean={mc['mean']:.2f} median={mc['median']} "
          f"found={mc['frac_found']:.2f} hist={mc['hist']}")
    if "per_concept_mwu_teacher_vs_dirb" in summary:
        print(f"== Per-concept MWU (teacher int > dirb): "
              f"p={summary['per_concept_mwu_teacher_vs_dirb']['p']:.3g}")
    print("\n== Per concept (primary strength) ==")
    for t in table:
        print(f"  c{t['concept']:>3} [{t['e2_group']},{t['e2_best_factor']}] n={t['n_nec']:>5} "
              f"nec hard={fmt(t['nec_hard'])} teacher={fmt(t['nec_teacher'])} "
              f"(dirb {fmt(t['dirb_teacher'])}, noise {fmt(t['noise_teacher'])}, "
              f"matched {fmt(t['base_any_teacher'])} cov {fmt(t['base_any_coverage'])}) "
              f"suf teacher={fmt(t['suf_teacher'])} clauses={t['n_clauses_affected']} "
              f"actions={t['n_actions_affected']}")
    print(f"\nSaved to {args.out_dir}")


# ============================================================================
# Outputs
# ============================================================================

def fmt(x):
    return "--" if x is None or not np.isfinite(x) else f"{x:.2f}"


def write_latex(table, path):
    lines = [r"\begin{tabular}{lcccccccc}", r"\toprule",
             r"Concept & E2 & Nec.\ (hard) & Nec.\ (teacher) & Dict.-dir. & Noise & Suff.\ (teacher) & \#clauses & \#actions \\",
             r"\midrule"]
    for t in table:
        lines.append(f"$c_{{{t['concept']}}}$ & {t['e2_group'] or '--'} & {fmt(t['nec_hard'])} & "
                     f"{fmt(t['nec_teacher'])} & {fmt(t['dirb_teacher'])} & {fmt(t['noise_teacher'])} & "
                     f"{fmt(t['suf_teacher'])} & {t['n_clauses_affected']} & {t['n_actions_affected']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines))


def plot_concept_hist(table, path):
    get = lambda k: [t[k] for t in table if t[k] is not None and np.isfinite(t[k])]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bins = np.linspace(0, 1, 21)
    ax.hist(get("nec_teacher"), bins=bins, alpha=0.6, label="concepts in winning clause")
    ax.hist(get("dirb_teacher"), bins=bins, alpha=0.6, label="dictionary-direction baseline")
    ax.set_xlabel("teacher action change rate"); ax.set_ylabel("# concepts"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_levels(summary, path):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(LEVELS)); wdt = 0.27
    for k, (c, lab) in enumerate([("int", "intervention"), ("dirb", "dictionary-direction"),
                                  ("noise", "norm-matched noise")]):
        vals = [summary[lv][c]["t_clamp"] for lv in LEVELS]
        ax.bar(x + (k - 1) * wdt, vals, wdt, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(LEVELS); ax.set_ylabel("teacher action change rate")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_minflips(mf, path):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    f = mf[np.isfinite(mf)]
    ax.hist(f, bins=np.arange(0.5, MAX_FLIPS + 1.5), rwidth=0.8)
    ax.set_xlabel("minimal # concept flips to change the rule decision"); ax.set_ylabel("# states")
    ax.set_title(f"not found (> {MAX_FLIPS}): {np.isnan(mf).mean():.1%}")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_strength(strength_summary, strengths, path):
    fig, axes = plt.subplots(1, len(LEVELS), figsize=(4 * len(LEVELS), 3.3), sharey=True)
    for ax, lv in zip(axes, LEVELS):
        for c, lab in [("int", "intervention"), ("dirb", "dictionary-direction"), ("noise", "noise")]:
            ax.plot(strengths, [strength_summary[str(q)][lv][c]["t_clamp"] for q in strengths],
                    marker="o", label=lab)
        ax.set_title(lv); ax.set_xlabel("ON/OFF percentile")
    axes[0].set_ylabel("teacher action change rate"); axes[0].legend()
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
