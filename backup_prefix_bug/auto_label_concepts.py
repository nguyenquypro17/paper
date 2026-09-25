#!/usr/bin/env python3
"""
auto_label_concepts.py
Automatically collects Top-K frames and draws IG Heatmaps for Used Concepts.
Features CONTRASTIVE output (ON vs OFF).
UPDATED: Uses PPO CNN features for semantically accurate K-Means Clustering 
and Hash-based deduplication to ensure true visual diversity.
"""

import os
import re
import argparse
import numpy as np
import torch
import gymnasium as gym
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from tqdm import tqdm

from stable_baselines3 import PPO

try:
    from captum.attr import IntegratedGradients
except ImportError:
    raise ImportError("Please install captum: pip install captum")

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min
except ImportError:
    raise ImportError("Please install scikit-learn for clustering: pip install scikit-learn")

from train_joint import SAELogicAgentV3, SAELogicConfig
from check_success_rules import ACTION_NAMES, make_vec_env

# ============================================================
# RENDERING HELPERS
# ============================================================
COLOR_MAP = {
    0: (100, 100, 100),   1: (0, 200, 0),       2: (0, 0, 200),
    3: (200, 0, 200),     4: (200, 200, 0),     5: (100, 100, 100),
}

OBJECT_BASE_COLORS = {
    0: np.array([40, 40, 40]),       1: np.array([220, 220, 220]),
    2: np.array([100, 100, 100]),    3: np.array([180, 180, 160]),
    4: np.array([150, 75, 0]),       5: np.array([255, 215, 0]),
    6: np.array([200, 50, 50]),      7: np.array([160, 82, 45]),
    8: np.array([0, 200, 50]),       9: np.array([255, 69, 0]),
    10: np.array([30, 144, 255]),
}

def render_minigrid_obs(obs: np.ndarray, cell_size: int = 16) -> np.ndarray:
    if obs.ndim == 1:
        n = obs.shape[0]
        for side in [7, 5, 6, 8, 11, 13]:
            if n == side * side * 3:
                obs = obs.reshape(side, side, 3)
                break
        else:
            return np.full((cell_size * 4, cell_size * 4, 3), 128, dtype=np.uint8)

    H, W = obs.shape[0], obs.shape[1]
    img = np.zeros((H * cell_size, W * cell_size, 3), dtype=np.uint8)

    for r in range(H):
        for c in range(W):
            obj_type = int(obs[r, c, 0])
            color_id = int(obs[r, c, 1])
            state = int(obs[r, c, 2])

            base = OBJECT_BASE_COLORS.get(obj_type, np.array([128, 128, 128]))
            if obj_type in (4, 5, 6, 7) and color_id in COLOR_MAP:
                crgb = np.array(COLOR_MAP[color_id], dtype=np.float32)
                base = (base.astype(np.float32) * 0.3 + crgb * 0.7).astype(np.uint8)

            if obj_type == 4 and state == 2:
                base = (base * 0.5).astype(np.uint8)
            elif obj_type == 4 and state == 0:
                base = np.minimum(base.astype(np.int32) + 60, 255).astype(np.uint8)

            y0, y1 = r * cell_size, (r + 1) * cell_size
            x0, x1 = c * cell_size, (c + 1) * cell_size
            img[y0:y1, x0:x1] = base
            img[y0, x0:x1] = np.minimum(base.astype(np.int32) - 30, 0).clip(0).astype(np.uint8)
            img[y0:y1, x0] = np.minimum(base.astype(np.int32) - 30, 0).clip(0).astype(np.uint8)
    return img

def render_image_obs(obs: np.ndarray) -> np.ndarray:
    if obs.dtype in (np.float32, np.float64):
        if obs.max() <= 1.0: obs = (obs * 255).astype(np.uint8)
        else: obs = obs.astype(np.uint8)
    return obs

def smart_render(obs: np.ndarray, cell_size: int = 16) -> np.ndarray:
    if obs.ndim == 3 and obs.shape[0] > 20 and obs.shape[1] > 20 and obs.shape[2] == 3:
        if obs.max() > 15: return render_image_obs(obs)
    
    is_grid = False
    if obs.ndim == 3 and obs.shape[2] == 3:
        if obs[:, :, 0].max() <= 15:
            if len(np.unique(obs[:, :, 0])) < 16: is_grid = True
            
    if is_grid:
        try:
            from minigrid.core.grid import Grid
            grid, vis_mask = Grid.decode(obs)
            return grid.render(cell_size, agent_pos=None, agent_dir=None, highlight_mask=vis_mask)
        except: pass
        return render_minigrid_obs(obs, cell_size)
    
    if obs.ndim == 3 and obs.shape[0] in (3, 1): obs = np.transpose(obs, (1, 2, 0))
    if obs.ndim == 3 and obs.shape[2] == 1: obs = np.repeat(obs, 3, axis=2)
    if obs.ndim == 2: obs = np.stack([obs] * 3, axis=-1)
    return render_image_obs(obs)

# ============================================================

def normalize_to_0_1(x, eps=1e-8):
    x_min, x_max = x.min(), x.max()
    denom = (x_max - x_min) if (x_max - x_min) > eps else eps
    return (x - x_min) / denom

def heatmap_apply_colormap(cam: np.ndarray, cmap_name="jet"):
    cmap = plt.get_cmap(cmap_name)
    colored = cmap(cam)[:, :, :3]
    return (colored * 255).astype(np.uint8)

def get_used_concepts(model):
    rules = model.extract_rules(action_names=None,threshold=0.5)
    used_concepts = set()
    pattern = re.compile(r'(?:f|z)_(\d+)')
    for _, clauses in rules.items():
        for clause in clauses:
            matches = pattern.findall(clause)
            used_concepts.update(int(m) for m in matches)
    return sorted(list(used_concepts))

class ConceptIGWrapper(torch.nn.Module):
    def __init__(self, ppo_cnn, logic_model, concept_idx):
        super().__init__()
        self.ppo_cnn = ppo_cnn
        self.logic_model = logic_model
        self.concept_idx = concept_idx

    def forward(self, x):
        x = x.contiguous()
        features = self.ppo_cnn(x)
        features = self.logic_model.normalize_input(features) 
        z_sparse, _ = self.logic_model.sae.encode(features)
        z_normed = self.logic_model.normalize_z(z_sparse)
        z_binary = self.logic_model.bottleneck(z_normed)
        return z_binary[:, self.concept_idx]

# ============================================================
def get_cluster_medoids(items, top_k, reverse=True):
    if len(items) <= top_k:
        return sorted(items, key=lambda x: x['val'], reverse=reverse)
    
    X = np.array([item['feat'] for item in items])
    kmeans = KMeans(n_clusters=top_k, random_state=42, n_init=10)
    kmeans.fit(X)
    
    closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    medoids = [items[idx] for idx in closest_indices]
    
    medoids.sort(key=lambda x: x['val'], reverse=reverse)
    return medoids


def collect_top_k_frames(ppo_cnn, logic_model, env_name, used_concepts, num_episodes=50, top_k=5, device="cpu"):
    env = make_vec_env(env_name, seed=43)
    obs = env.reset()
    
    history = {c: {'act': [], 'inact': []} for c in used_concepts}
    seen_hashes = {c: {'act': set(), 'inact': set()} for c in used_concepts}
    
    print(f"\n[1] Collecting rollout data over {num_episodes} episodes...")
    max_steps = num_episodes * 500
    
    for step_idx in tqdm(range(max_steps), desc="Rollout Steps"):
        is_minigrid = 'MiniGrid' in env_name
        if is_minigrid:
            try:
                rendered_frame = env.envs[0].unwrapped.get_frame()
                agent_dir = env.envs[0].unwrapped.agent_dir
            except:
                rendered_frame = env.render()
                agent_dir = 0
        else:
            rendered_frame = env.render()
            agent_dir = None

        obs_tensor = torch.as_tensor(obs).float().to(device)
        
        with torch.no_grad():
            features = ppo_cnn(obs_tensor)
            logits, feats = logic_model(features, normalize_input=True, return_features=True)
            z_bin = feats['z_binary'][0].cpu().numpy()
            feat_np = features[0].cpu().numpy()
            action = logits.argmax(dim=1).cpu().numpy()
        
        for c in used_concepts:
            val = float(z_bin[c])
            obs_hash = hash(obs[0].tobytes()) 
            
            item = {
                'val': val,
                'obs': obs[0].copy(),
                'frame': rendered_frame.copy(),
                'agent_dir': agent_dir,
                'is_minigrid': is_minigrid,
                'feat': feat_np 
            }
            
            if val > 0.1: 
                if obs_hash not in seen_hashes[c]['act']:
                    seen_hashes[c]['act'].add(obs_hash)
                    history[c]['act'].append(item)
            else: 
                if obs_hash not in seen_hashes[c]['inact']:
                    seen_hashes[c]['inact'].add(obs_hash)
                    history[c]['inact'].append(item)
                
        if step_idx > 0 and step_idx % 1000 == 0:
            for c in used_concepts:
                if len(history[c]['act']) > 1000:
                    history[c]['act'].sort(key=lambda x: x['val'], reverse=True)
                    history[c]['act'] = history[c]['act'][:1000]
                
                if len(history[c]['inact']) > 1000:
                    history[c]['inact'].sort(key=lambda x: x['val'])
                    history[c]['inact'] = history[c]['inact'][:1000]
                
        obs, _, dones, _ = env.step(action)
        if dones[0]: obs = env.reset()
            
    env.close()
    
    top_frames_dict = {}
    print("\n[2] Applying K-Means clustering to find diverse representative frames...")
    
    for c in tqdm(used_concepts, desc="Clustering"):
        history[c]['act'].sort(key=lambda x: x['val'], reverse=True)
        history[c]['inact'].sort(key=lambda x: x['val'])
        
        act_medoids = get_cluster_medoids(history[c]['act'], top_k, reverse=True)
        inact_medoids = get_cluster_medoids(history[c]['inact'], top_k, reverse=False)
            
        top_frames_dict[c] = {'act': act_medoids, 'inact': inact_medoids}
        
    return top_frames_dict

def generate_concept_summary(ppo_cnn, logic_model, concept_idx, top_frames_dict, save_dir, device, env_name):
    top_act = top_frames_dict['act']
    top_inact = top_frames_dict['inact']
    
    if not top_act: return None
        
    wrapper = ConceptIGWrapper(ppo_cnn, logic_model, concept_idx).to(device)
    wrapper.eval()
    ig = IntegratedGradients(wrapper)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        header_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()
        header_font = ImageFont.load_default()

    is_minigrid_global = 'MiniGrid' in env_name

    def process_single_item(item):
        obs_np = item['obs'] 
        if obs_np.ndim == 3 and obs_np.shape[0] in [3, 4]:
            obs_t = torch.from_numpy(obs_np).float().unsqueeze(0).to(device).contiguous()
            obs_vis = np.stack((obs_np[-1],) * 3, axis=-1) if obs_np.shape[0] == 4 else np.transpose(obs_np, (1, 2, 0)) 
        elif obs_np.ndim == 3 and obs_np.shape[-1] in [3, 4]:
            obs_t = torch.from_numpy(np.transpose(obs_np, (2, 0, 1))).float().unsqueeze(0).to(device).contiguous()
            obs_vis = np.stack((obs_np[..., -1],) * 3, axis=-1) if obs_np.shape[-1] == 4 else obs_np
        else:
            obs_t = torch.from_numpy(obs_np).float().unsqueeze(0).to(device).contiguous()
            obs_vis = obs_np
            
        baseline = torch.zeros_like(obs_t).contiguous() 
        if item.get('is_minigrid', False):
                for c in range(obs_t.shape[1]):
                    mode_val = torch.mode(obs_t[0, c].cpu().flatten()).values.item()
                    baseline[:, c, :, :] = mode_val
        else:
            white_val = 1.0 if obs_t.max() <= 1.0 else 255.0
            baseline = torch.ones_like(obs_t).contiguous() * white_val
        try:
            attributions = ig.attribute(obs_t, baselines=baseline, n_steps=50)
            attr_np = attributions.squeeze(0).cpu().detach().numpy()
            if attr_np.ndim == 3: attr_np = np.abs(attr_np).max(axis=0) 
        except:
            attr_np = np.zeros((obs_t.shape[-2], obs_t.shape[-1]))

        W, H = 200, 200
        
        if attr_np.max() > 1e-8:
            if item['is_minigrid']:
                attr_np = attr_np.T
                
            attr_np = normalize_to_0_1(attr_np)
            heatmap_rgb = heatmap_apply_colormap(attr_np, "jet")
            heatmap_img = Image.fromarray(heatmap_rgb).resize((W, H), Image.NEAREST)
        else:
            heatmap_img = Image.new("RGB", (W, H), color="black")
            draw_heat = ImageDraw.Draw(heatmap_img)
            draw_heat.text((60, 90), "Zero Attrib.", fill="white", font=font)

        render_img = Image.fromarray(item['frame']).resize((W, H), Image.LANCZOS)
        
        if item['is_minigrid']:
            rendered_obs = smart_render(obs_vis, cell_size=32)
            model_input_img = Image.fromarray(rendered_obs).resize((W, H), Image.NEAREST)
            
            # --- HIGHLIGHT AGENT POSITION (RED BOX) ---
            draw_agent = ImageDraw.Draw(model_input_img)
            cw, ch = W / 7.0, H / 7.0
            x0, y0 = 3 * cw, 6 * ch
            x1, y1 = 4 * cw, 7 * ch
            draw_agent.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
            # ------------------------------------------
        else:
            obs_vis_scaled = (obs_vis * 255).astype(np.uint8) if obs_vis.max() <= 1.0 else obs_vis.astype(np.uint8)
            model_input_img = Image.fromarray(obs_vis_scaled).resize((W, H), Image.LANCZOS)
            
        return render_img, model_input_img, heatmap_img, item['val']

    row_images = []
    num_rows = max(len(top_act), len(top_inact))
    
    for i in range(num_rows):
        if i < len(top_act):
            act_god, act_agent, act_heat, act_val = process_single_item(top_act[i])
        else:
            act_god = act_agent = act_heat = Image.new("RGB", (200, 200), "black")
            act_val = 0.0
            
        if i < len(top_inact):
            inact_god, inact_agent, inact_heat, inact_val = process_single_item(top_inact[i])
        else:
            inact_god = inact_agent = inact_heat = Image.new("RGB", (200, 200), "black")
            inact_val = 0.0

        W, H = 200, 200
        spacing = 10
        row_w = W * 6 + spacing * 5
        row_canvas = Image.new("RGB", (row_w, H), color=(255, 255, 255))
        
        panels = [act_god, act_agent, act_heat, inact_god, inact_agent, inact_heat]
        x_offset = 0
        for panel in panels:
            row_canvas.paste(panel, (x_offset, 0))
            x_offset += W + spacing
            
        draw = ImageDraw.Draw(row_canvas)
        draw.rectangle([0, 0, 85, 25], fill=(0, 0, 0, 180))
        draw.text((5, 5), f"Act: {act_val:.2f}", fill=(0, 255, 0), font=font) 
        
        inact_x_start = (W + spacing) * 3
        draw.rectangle([inact_x_start, 0, inact_x_start + 85, 25], fill=(0, 0, 0, 180))
        draw.text((inact_x_start + 5, 5), f"Act: {inact_val:.2f}", fill=(255, 100, 100), font=font) 
        
        row_images.append(row_canvas)
        
    if not row_images: return None
        
    row_w = row_images[0].width
    spacing_y = 15
    header_h = 70
    total_h = header_h + (200 + spacing_y) * len(row_images)
    
    final_canvas = Image.new("RGB", (row_w, total_h), color="white")
    draw = ImageDraw.Draw(final_canvas)
    
    title_text = f"Concept C{concept_idx} - Contrastive Visual Grounding (Top {len(top_act)} ON vs {len(top_inact)} OFF)"
    draw.text((10, 10), title_text, fill="black", font=header_font)

    blue = (0, 0, 255)
    red = (200, 0, 0)
    
    labels = [
        "[God View (ON)]" if is_minigrid_global else "[Full Render (ON)]",
        "[Agent View (ON)]" if is_minigrid_global else "[Model Input (ON)]",
        "[IG Heatmap (ON)]",
        "[God View (OFF)]" if is_minigrid_global else "[Full Render (OFF)]",
        "[Agent View (OFF)]" if is_minigrid_global else "[Model Input (OFF)]",
        "[IG Heatmap (OFF)]"
    ]
    colors = [blue, blue, blue, red, red, red]
    
    for idx, (text, color) in enumerate(zip(labels, colors)):
        x_center = idx * (200 + 10) + 100
        text_w = draw.textsize(text, font=font)[0] if hasattr(draw, 'textsize') else 80
        draw.text((x_center - text_w//2, 40), text, fill=color, font=font)
    
    y_offset = header_h
    for r_img in row_images:
        final_canvas.paste(r_img, (0, y_offset))
        y_offset += 200 + spacing_y
        
    out_path = os.path.join(save_dir, f"C{concept_idx}_summary.png")
    final_canvas.save(out_path)
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to Logic/SAE model checkpoint (.pt)")
    parser.add_argument("--ppo_path", type=str, required=True, help="Path to base PPO model zip (MUST MATCH ENVIRONMENT)")
    parser.add_argument("--env_name", type=str, required=True, help="Gym environment name (e.g., PixelCartPole-v0)")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes for rollout")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top activating frames to extract per concept")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading PPO model from {args.ppo_path}...")
    ppo_model = PPO.load(args.ppo_path, device=device)
    ppo_cnn = ppo_model.policy.features_extractor
    ppo_cnn.eval()

    print(f"Loading Logic model from {args.model_path}...")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    config = SAELogicConfig(**ckpt['config'])
    logic_model = SAELogicAgentV3(config, device=device)
    logic_model.load_state_dict(ckpt['model_state'])
    logic_model.set_normalization(ckpt['feature_mean'].to(device), ckpt['feature_std'].to(device))
    logic_model.z_mean.copy_(ckpt['z_mean'])
    logic_model.z_std.copy_(ckpt['z_std'])
    logic_model.to(device) 
    logic_model.eval()

    env_clean_name = args.env_name.replace("-", "_")
    save_dir = f"./concept_grounding/{env_clean_name}"
    os.makedirs(save_dir, exist_ok=True)

    used_concepts = get_used_concepts(logic_model)
    print(f"\n[+] Analyzed {len(used_concepts)} used concepts: {used_concepts}")

    top_frames = collect_top_k_frames(
        ppo_cnn, logic_model, args.env_name, used_concepts, 
        num_episodes=args.episodes, top_k=args.top_k, device=device
    )

    print(f"\n[3] Computing Integrated Gradients and generating summary images (Top {args.top_k} clustered frames)...")
    for c in tqdm(used_concepts, desc="Generating Images"):
        generate_concept_summary(ppo_cnn, logic_model, c, top_frames[c], save_dir, device, args.env_name)
        
    print(f"\n✓ Complete! All summary images saved to: {save_dir}")

if __name__ == "__main__":
    main()