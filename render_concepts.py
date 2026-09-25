#!/usr/bin/env python3
"""
render_concepts.py -- Step 0 of the VLM pipeline (replaces auto_label_concepts.py).

Rolls out the TEACHER (not LUCID), records LUCID hard concepts at every step, splits
episodes into a naming set N and a test set T (disjoint, deduplicated), and writes:

    <out_dir>/naming/C{c}/on_{j}.png, off_{j}.png   single views          -> vlm_name.py
    <out_dir>/naming/C{c}.png          ON/OFF panel with IG (figure for the paper only)
    <out_dir>/test/C{c}/{i:03d}.png    single view of one test state, NO heatmap
    <out_dir>/test/C{c}/targets.json   y (1 = hard concept ON) and z for each test image
    <out_dir>/test/_factors/{f}/...    positive control: ground-truth factor statements
    <out_dir>/meta.json                args, split, sanity stats, per-concept counts,
                                       concept x factor AUC on all unique states
Single view = MiniGrid: the agent's ego-centric view (the concept's only input);
              CartPole / Atari: the full rendered frame.

Differences from auto_label_concepts.py:
    features via policy.extract_features (same path as collect_with_observations.py)
    concepts = literals of ckpt["rules"] (the rules used in the paper / E1)
    ON = z_binary > hard_threshold (0.66), not > 0.1
    actions from the teacher, not from LUCID soft
    states deduplicated, then unique states split at random into N and T

Usage:
    python render_concepts.py \
        --model_path outputs/lucid_model_doorkey_42/sae_logic_joint_model.pt \
        --ppo_path ppo_doorkey_6x6.zip --env_name MiniGrid-DoorKey-6x6-v0 \
        --n_episodes 200 --seed 2000 --out_dir experiments/vlm/doorkey_42
Atari: --n_episodes 6 --stride 4 (frames are large).
"""

import argparse
import json
import os
import re

import numpy as np
import torch
import gymnasium as gym
import minigrid  # noqa: F401
try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

from intervention import load_lucid, load_teacher
from vlm_common import factor_tests

PANEL = 200


def is_atari(env_name):
    return ("NoFrameskip" in env_name) or ("ALE/" in env_name)


def font(size=15):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ============================================================================
# Environment (same construction as collect_with_observations.py)
# ============================================================================

def make_env(env_name, seed):
    """-> (vec_env, raw_env)"""
    if is_atari(env_name):
        vec_env = make_atari_env(env_name, n_envs=1, seed=seed)
        vec_env = VecFrameStack(vec_env, n_stack=4)
        vec_env = VecTransposeImage(vec_env)
    else:
        from utils_env import make_env_by_name
        vec_env = DummyVecEnv([lambda: make_env_by_name(env_name, seed=seed)])
        vec_env = VecTransposeImage(vec_env)
        vec_env.seed(seed)
    raw = vec_env.unwrapped.envs[0].unwrapped
    return vec_env, raw


def god_frame(raw, env_name, tile=16):
    if "MiniGrid" in env_name:
        return raw.get_frame(highlight=True, tile_size=tile)
    if is_atari(env_name):
        return raw.ale.getScreenRGB().copy()
    return raw.render()


def model_view(raw, obs_chw, env_name, tile=16):
    """What the teacher sees: MiniGrid ego-centric render, else the last stacked frame."""
    if "MiniGrid" in env_name:
        return raw.get_pov_render(tile_size=tile)
    last = obs_chw[-1]
    return np.stack([last] * 3, axis=-1).astype(np.uint8)


# ============================================================================
# Collection
# ============================================================================

# Ground-truth factors: copied verbatim from collect_with_observations.py (E2 version),
# so this script does not depend on which copy of that file is on the server.

def _cell_type(env, pos):
    """Object type at grid position, 'out' if outside the grid, 'empty' if None."""
    x, y = int(pos[0]), int(pos[1])
    if not (0 <= x < env.width and 0 <= y < env.height):
        return "out"
    obj = env.grid.get(x, y)
    return "empty" if obj is None else obj.type


def _find_first(env, obj_type):
    """(position, object) of the first object of obj_type in the grid, else (None, None)."""
    for j in range(env.height):
        for i in range(env.width):
            obj = env.grid.get(i, j)
            if obj is not None and obj.type == obj_type:
                return (i, j), obj
    return None, None


def get_factor_names(env_name):
    if "DoorKey" in env_name:
        return ["has_key", "door_open", "key_in_front", "door_in_front",
                "wall_ahead", "agent_dir", "goal_visible"]
    if "Dynamic-Obstacles" in env_name:
        return ["obstacle_ahead", "obstacle_left", "obstacle_right",
                "agent_dir", "goal_dist"]
    if "PixelCartPole" in env_name:
        return ["cart_pos", "cart_vel", "pole_angle", "pole_ang_vel"]
    raise ValueError(f"No factor definition for {env_name}")


def extract_factors(raw_env, env_name, grid_obs=None):
    """
    Read ground-truth factors from the simulator state at the CURRENT step
    (must be called before vec_env.step, same moment as grid_obs).

    Returns: dict {factor_name: float}, keys ordered as get_factor_names(env_name).
    """
    env = raw_env.unwrapped

    if "DoorKey" in env_name:
        from minigrid.core.constants import OBJECT_TO_IDX
        front = _cell_type(env, env.front_pos)
        _, door = _find_first(env, "door")
        return {
            "has_key": float(env.carrying is not None and env.carrying.type == "key"),
            "door_open": float(door is not None and door.is_open),
            "key_in_front": float(front == "key"),
            "door_in_front": float(front == "door"),
            "wall_ahead": float(front == "wall"),
            "agent_dir": float(env.agent_dir),
            "goal_visible": float(grid_obs is not None
                                  and (grid_obs[..., 0] == OBJECT_TO_IDX["goal"]).any()),
        }

    if "Dynamic-Obstacles" in env_name:
        pos = np.array(env.agent_pos)
        goal_pos, _ = _find_first(env, "goal")
        goal_dist = (np.abs(pos - np.array(goal_pos)).sum()
                     if goal_pos is not None else -1)
        return {
            "obstacle_ahead": float(_cell_type(env, pos + env.dir_vec) == "ball"),
            "obstacle_left": float(_cell_type(env, pos - env.right_vec) == "ball"),
            "obstacle_right": float(_cell_type(env, pos + env.right_vec) == "ball"),
            "agent_dir": float(env.agent_dir),
            "goal_dist": float(goal_dist),
        }

    if "PixelCartPole" in env_name:
        x, x_dot, theta, theta_dot = [float(v) for v in env.state]
        return {"cart_pos": x, "cart_vel": x_dot,
                "pole_angle": theta, "pole_ang_vel": theta_dot}

    raise ValueError(f"No factor definition for {env_name}")



def used_concepts(rules):
    """concept ids appearing as literals in the rule set (ckpt['rules'])"""
    out = set()
    for clauses in rules.values():
        for c in clauses:
            out.update(int(i) for i in re.findall(r"f_(\d+)", c.split("[")[0]))
    return sorted(out)


@torch.no_grad()
def get_features(policy, obs):
    """obs (B,C,H,W) np/tensor -> (B,d) CNN features, SB3 preprocessing included"""
    obs_t = torch.as_tensor(obs, device=policy.device)
    return policy.extract_features(obs_t, policy.features_extractor)


@torch.no_grad()
def collect_states(policy, lucid, env_name, n_episodes, seed, stride=1, max_steps=300_000):
    """
    Teacher rollouts. Every `stride`-th step is stored.
    -> dict obs (list CHW uint8), god (list), view (list), feat (N,d), z (N,D),
            ep (N,), hash (list), stats
    """
    vec_env, raw = make_env(env_name, seed)
    obs = vec_env.reset()
    S = dict(obs=[], god=[], view=[], feat=[], z=[], ep=[], hash=[], factors=[])
    fnames = []
    if not is_atari(env_name):
        fnames = get_factor_names(env_name)
    ep = t = 0
    agree = n_pred = 0
    norms = []
    pbar = tqdm(total=n_episodes, desc="Episodes")
    while ep < n_episodes and t < max_steps:
        feat = get_features(policy, obs)
        action, _ = policy.predict(obs, deterministic=True)
        logits, f = lucid(feat, normalize_input=True, return_features=True)
        agree += int(logits.argmax(1).item() == int(action[0])); n_pred += 1
        norms.append(feat.norm(dim=1).item())

        if t % stride == 0:
            o = np.asarray(obs[0]).copy()
            S["obs"].append(o)
            S["god"].append(god_frame(raw, env_name))
            S["view"].append(model_view(raw, o, env_name))
            S["feat"].append(feat[0].cpu().numpy())
            S["z"].append(f["z_binary"][0].cpu().numpy())
            S["ep"].append(ep)
            S["hash"].append(hash(o.tobytes()))
            if fnames:
                grid = raw.gen_obs()["image"] if "MiniGrid" in env_name else None
                fd = extract_factors(raw, env_name, grid)
                S["factors"].append([fd[k] for k in fnames])

        obs, _, dones, _ = vec_env.step(action)
        t += 1
        if dones[0]:
            ep += 1
            pbar.update(1)
    pbar.close()
    vec_env.close()

    S["feat"] = np.stack(S["feat"]); S["z"] = np.stack(S["z"]); S["ep"] = np.array(S["ep"])
    S["factors"] = np.array(S["factors"], dtype=float) if fnames else np.zeros((len(S["ep"]), 0))
    S["factor_names"] = fnames
    S["stats"] = dict(n_steps=t, n_stored=len(S["obs"]), n_episodes=ep,
                      feat_mean_norm=float(np.mean(norms)),
                      soft_teacher_agreement=agree / max(n_pred, 1))
    return S


def split_unique(hashes, frac=0.3, seed=0):
    """
    -> idx_N, idx_T (int arrays). Deduplicate all stored states (first occurrence of each
    observation), then split the unique states at random. No observation is in both sets.
    """
    seen, uniq = set(), []
    for i, h in enumerate(hashes):
        if h not in seen:
            seen.add(h); uniq.append(i)
    uniq = np.random.default_rng(seed).permutation(np.array(uniq, dtype=int))
    n = max(1, int(round(frac * len(uniq))))
    return np.sort(uniq[:n]), np.sort(uniq[n:])


def _medoids(idx, feat, k, seed):
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min
    if len(idx) <= k:
        return list(idx)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(feat[idx])
    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, feat[idx])
    return [int(idx[j]) for j in dict.fromkeys(closest.tolist())]


def select_naming(S, idx_N, c, k=8, theta=0.66, cap=1000, seed=0):
    """
    -> on_idx, off_idx (lists). ON: strongest activations above theta, diversified by
    k-means medoids on CNN features. OFF: random typical states at or below theta,
    diversified the same way (not only the lowest activations).
    """
    rng = np.random.default_rng(seed)
    z = S["z"][idx_N, c]
    on = idx_N[z > theta]
    on = on[np.argsort(-S["z"][on, c])][:cap]
    off = idx_N[z <= theta]
    if len(off) > cap:
        off = rng.choice(off, cap, replace=False)
    return _medoids(on, S["feat"], k, seed), _medoids(off, S["feat"], k, seed)


def sample_test(S, idx_T, c, m=20, theta=0.66, seed=0):
    """-> idx (list), y (list of 0/1); balanced, n = min(m, #ON, #OFF) per class"""
    rng = np.random.default_rng(seed)
    z = S["z"][idx_T, c]
    on, off = idx_T[z > theta], idx_T[z <= theta]
    n = min(m, len(on), len(off))
    if n == 0:
        return [], []
    idx = list(rng.choice(on, n, replace=False)) + list(rng.choice(off, n, replace=False))
    y = [1] * n + [0] * n
    perm = rng.permutation(len(idx))
    return [int(idx[i]) for i in perm], [y[i] for i in perm]


def _auc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0) + 0.5 * (d == 0)).mean())


def factor_labels(S, env_name):
    """-> {factor: bool array (N,)} for the visible factors of FACTOR_TESTS"""
    out = {}
    for f, (_, fn) in factor_tests(env_name).items():
        if f in S["factor_names"]:
            col = S["factors"][:, S["factor_names"].index(f)]
            out[f] = np.array([fn(v) for v in col], bool)
    return out


def concept_factor_auc(S, idx, concepts, env_name):
    """-> {C{c}: {factor: AUC of continuous z_c against the binarized factor}} on idx"""
    F = factor_labels(S, env_name)
    return {f"C{c}": {f: _auc(y[idx], S["z"][idx, c]) for f, y in F.items()} for c in concepts}


def sample_factor_test(S, idx_T, y_all, m=20, seed=0):
    """y_all (N,) bool -> idx (list), y (list); balanced, n = min(m, #pos, #neg) per class"""
    rng = np.random.default_rng(seed)
    pos, neg = idx_T[y_all[idx_T]], idx_T[~y_all[idx_T]]
    n = min(m, len(pos), len(neg))
    if n == 0:
        return [], []
    idx = list(rng.choice(pos, n, replace=False)) + list(rng.choice(neg, n, replace=False))
    y = [1] * n + [0] * n
    perm = rng.permutation(len(idx))
    return [int(idx[i]) for i in perm], [y[i] for i in perm]


# ============================================================================
# Integrated Gradients
# ============================================================================

class ConceptIG(torch.nn.Module):
    def __init__(self, policy, lucid, c):
        super().__init__()
        self.policy, self.lucid, self.c = policy, lucid, c

    def forward(self, x):
        h = self.policy.extract_features(x, self.policy.features_extractor)
        _, f = self.lucid(h, normalize_input=True, return_features=True)
        return f["z_binary"][:, self.c]


def compute_ig(policy, lucid, c, obs_chw, env_name, n_steps=50):
    """-> (H,W) attribution in [0,1] aligned with model_view, or None if all zero"""
    from captum.attr import IntegratedGradients
    x = torch.as_tensor(obs_chw, device=policy.device).float().unsqueeze(0)
    if "MiniGrid" in env_name:
        base = torch.zeros_like(x)
        for ch in range(x.shape[1]):
            base[:, ch] = torch.mode(x[0, ch].flatten().cpu()).values.item()
    else:
        base = torch.full_like(x, 255.0)
    try:
        a = IntegratedGradients(ConceptIG(policy, lucid, c)).attribute(x, baselines=base, n_steps=n_steps)
    except Exception as e:
        print(f"  IG failed for C{c}: {e}")
        return None
    a = a[0].abs().max(0).values.detach().cpu().numpy()
    if "MiniGrid" in env_name:
        a = a.T                       # obs is (x, y); image is (row = y, col = x)
    if a.max() <= 1e-8:
        return None
    return (a - a.min()) / (a.max() - a.min())


def heat_image(a):
    if a is None:
        img = Image.new("RGB", (PANEL, PANEL), "black")
        ImageDraw.Draw(img).text((55, 90), "zero attribution", fill="white", font=font())
        return img
    rgb = (plt.get_cmap("jet")(a)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize((PANEL, PANEL), Image.NEAREST)


def to_panel(arr):
    return Image.fromarray(np.asarray(arr).astype(np.uint8)).resize((PANEL, PANEL), Image.NEAREST)


# ============================================================================
# Rendering
# ============================================================================

def render_panel(S, i, heat=None):
    """-> PIL row [global | model view] (+ [IG heatmap] if heat is not False)"""
    parts = [to_panel(S["god"][i]), to_panel(S["view"][i])]
    if heat is not False:
        parts.append(heat_image(heat))
    row = Image.new("RGB", (len(parts) * PANEL + (len(parts) - 1) * 10, PANEL), "white")
    for j, p in enumerate(parts):
        row.paste(p, (j * (PANEL + 10), 0))
    return row


def naming_image(S, on_idx, off_idx, heats, c, env_name):
    view = "Agent View" if "MiniGrid" in env_name else "Model Input"
    titles = [f"Global (ON)", f"{view} (ON)", "IG Heatmap (ON)",
              f"Global (OFF)", f"{view} (OFF)", "IG Heatmap (OFF)"]
    rows = []
    for r in range(max(len(on_idx), len(off_idx))):
        halves = []
        for lst in (on_idx, off_idx):
            if r < len(lst):
                halves.append(render_panel(S, lst[r], heats.get(lst[r])))
            else:
                halves.append(Image.new("RGB", (3 * PANEL + 20, PANEL), "black"))
        row = Image.new("RGB", (6 * PANEL + 50, PANEL), "white")
        row.paste(halves[0], (0, 0)); row.paste(halves[1], (3 * PANEL + 30, 0))
        rows.append(row)
    W, head = rows[0].width, 60
    canvas = Image.new("RGB", (W, head + len(rows) * (PANEL + 12)), "white")
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), f"Concept C{c}: {len(on_idx)} ON rows vs {len(off_idx)} OFF rows", fill="black", font=font(17))
    for j, t in enumerate(titles):
        x = j * (PANEL + 10) + (20 if j >= 3 else 0)
        d.text((x + 10, 36), t, fill=(0, 0, 200) if j < 3 else (200, 0, 0), font=font())
    for r, row in enumerate(rows):
        canvas.paste(row, (0, head + r * (PANEL + 12)))
    return canvas


def single_view(S, i, env_name, size=336):
    """MiniGrid: ego-centric view; CartPole / Atari: full rendered frame. Aspect kept."""
    arr = np.asarray(S["view"][i] if "MiniGrid" in env_name else S["god"][i]).astype(np.uint8)
    img = Image.fromarray(arr)
    s = size / max(img.size)
    return img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.NEAREST)


def save_set(S, idx, y, env_name, d, extra=None):
    os.makedirs(d, exist_ok=True)
    for j, i in enumerate(idx):
        single_view(S, i, env_name).save(os.path.join(d, f"{j:03d}.png"))
    js = dict(y=y, images=[f"{j:03d}.png" for j in range(len(idx))], state_idx=[int(i) for i in idx])
    js.update(extra or {})
    json.dump(js, open(os.path.join(d, "targets.json"), "w"), indent=1)


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--ppo_path", required=True)
    ap.add_argument("--env_name", required=True)
    ap.add_argument("--n_episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2000, help="differs from train (0) and held-out (1000)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--hard_threshold", type=float, default=0.66)
    ap.add_argument("--k", type=int, default=8, help="ON/OFF rows per naming panel")
    ap.add_argument("--m", type=int, default=20, help="test images per class per concept")
    ap.add_argument("--frac_naming", type=float, default=0.3)
    ap.add_argument("--min_test", type=int, default=5, help="min test images per class")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lucid, config, rules = load_lucid(args.model_path, device, args.hard_threshold)
    policy = load_teacher(args.ppo_path, device)
    theta = config.hard_threshold
    concepts = used_concepts(rules)
    print(f"[+] {len(concepts)} concepts in rules: {concepts}")

    S = collect_states(policy, lucid, args.env_name, args.n_episodes, args.seed, args.stride)
    st = S["stats"]
    print(f"[sanity] feat_mean_norm={st['feat_mean_norm']:.2f}  "
          f"soft_teacher_agreement={st['soft_teacher_agreement']:.4f}  stored={st['n_stored']}")

    idx_N, idx_T = split_unique(S["hash"], args.frac_naming, args.seed)
    print(f"[split] naming states={len(idx_N)}  test states={len(idx_T)} (deduplicated, disjoint)")

    os.makedirs(os.path.join(args.out_dir, "naming"), exist_ok=True)
    per_concept, skipped = {}, []
    for c in tqdm(concepts, desc="Concepts"):
        on_idx, off_idx = select_naming(S, idx_N, c, args.k, theta, seed=args.seed)
        t_idx, y = sample_test(S, idx_T, c, args.m, theta, seed=args.seed + c)
        per_concept[f"C{c}"] = dict(
            on_rate_N=float((S["z"][idx_N, c] > theta).mean()) if len(idx_N) else 0.0,
            n_on_rows=len(on_idx), n_off_rows=len(off_idx), n_test=len(t_idx))
        if not on_idx or not off_idx or len(t_idx) < 2 * args.min_test:
            skipped.append(f"C{c}")
            continue

        heats = {i: compute_ig(policy, lucid, c, S["obs"][i], args.env_name) for i in on_idx + off_idx}
        naming_image(S, on_idx, off_idx, heats, c, args.env_name).save(
            os.path.join(args.out_dir, "naming", f"C{c}.png"))
        ndir = os.path.join(args.out_dir, "naming", f"C{c}")
        os.makedirs(ndir, exist_ok=True)
        for tag, lst in (("on", on_idx), ("off", off_idx)):
            for j, i in enumerate(lst):
                single_view(S, i, args.env_name).save(os.path.join(ndir, f"{tag}_{j}.png"))

        save_set(S, t_idx, y, args.env_name, os.path.join(args.out_dir, "test", f"C{c}"),
                 dict(concept=f"C{c}", z=[float(S["z"][i, c]) for i in t_idx]))

    # positive control: ground-truth factor statements on the same kind of test images
    ctrl = {}
    for f, yf in factor_labels(S, args.env_name).items():
        f_idx, f_y = sample_factor_test(S, idx_T, yf, args.m, seed=args.seed + 7)
        ctrl[f] = len(f_idx)
        if len(f_idx) >= 2 * args.min_test:
            save_set(S, f_idx, f_y, args.env_name, os.path.join(args.out_dir, "test", "_factors", f),
                     dict(factor=f, statement=factor_tests(args.env_name)[f][0]))

    idx_all = np.concatenate([idx_N, idx_T])
    from vlm_common import manifest
    meta = dict(manifest=manifest(args), args=vars(args), hard_threshold=theta, stats=st,
                n_naming_states=int(len(idx_N)), n_test_states=int(len(idx_T)),
                concepts=per_concept, skipped=skipped, factor_control_n=ctrl,
                concept_factor_auc=concept_factor_auc(S, idx_all, concepts, args.env_name))
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)
    print(f"[done] {len(concepts) - len(skipped)} concepts rendered, skipped {skipped}")


if __name__ == "__main__":
    main()
